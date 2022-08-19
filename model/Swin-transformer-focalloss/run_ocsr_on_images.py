import sys
sys.path.append("..")
import os
import argparse
from tqdm import tqdm
import deepsmiles
from typing import Any, cast, Callable, List, Tuple, Union
from PIL import Image

import torch
import torchvision
import torchvision.transforms as transforms

from config import get_config
from eval import Greedy_decode
from models import build_model
from pre_transformer import Transformer


class FocalLossModelInference:
    """
    Inference Class
    """
    def __init__(self):
        # Load dictionary that maps tokens to integers
        word_map_path = '../../Data/500wan/500wan_shuffle_DeepSMILES_word_map'
        self.word_map = torch.load(word_map_path)
        self.inv_word_map = {v: k for k, v in self.word_map.items()}

        # Define device, load models and weights
        self.dev = "cuda" if torch.cuda.is_available() else "cpu"
        self.args, config = self.get_inference_config()
        self.encoder = build_model(config, tag=False)
        self.decoder = self.build_decoder()
        self.load_checkpoint(os.path.join(os.path.split(__file__)[0],
                                          "swin_transform_focalloss.pth"))
        self.decoder = self.decoder.to(self.dev).eval()
        self.encoder = self.encoder.to(self.dev).eval()

    def load_checkpoint(self, checkpoint_path):
        """
        Load checkpoint and update encoder and decoder accordingly

        Args:
            checkpoint_path (str): path of checkpoint file
        """
        print(f"=====> Resuming from {checkpoint_path} <=====")
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        encoder_msg = self.encoder.load_state_dict(checkpoint['encoder'],
                                                   strict=False)
        decoder_msg = self.decoder.load_state_dict(checkpoint['decoder'],
                                                   strict=False)
        print(f"Encoder: {encoder_msg}")
        print(f"Decoder: {decoder_msg}")
        del checkpoint
        torch.cuda.empty_cache()

    def build_decoder(self):
        """
        This method builds the Transformer decoder and returns it
        """
        self.decoder_dim = 256  # dimension of decoder RNN
        self.ff_dim = 2048
        self.num_head = 8
        self.dropout = 0.1
        self.encoder_num_layer = 6
        self.decoder_num_layer = 6
        self.max_len = 277
        self.decoder_lr = 5e-4
        self.best_acc = 0.
        return Transformer(dim=self.decoder_dim,
                           ff_dim=self.ff_dim,
                           num_head=self.num_head,
                           encoder_num_layer=self.encoder_num_layer,
                           decoder_num_layer=self.decoder_num_layer,
                           vocab_size=len(self.word_map),
                           max_len=self.max_len,
                           drop_rate=self.dropout,
                           tag=False)

    def get_inference_config(self):
        parser = argparse.ArgumentParser('Swin Transformer Inference script',
                                         add_help=False)
        parser.add_argument('--cfg',
                            default='../configs/swin_large_patch4_window7_224.yaml',
                            type=str,
                            metavar="FILE",
                            help='path to config file', )
        parser.add_argument(
            "--opts",
            help="Modify config options by adding 'KEY VALUE' pairs. ",
            default=None,
            nargs='+',
        )

        # easy config modification
        parser.add_argument('--batch-size', type=int, help="batch size for single GPU")
        parser.add_argument('--data-path', type=str, help='path to image directory')
        parser.add_argument('--zip', action='store_true', help='use zipped dataset instead of folder dataset')
        parser.add_argument('--cache-mode', type=str, default='part', choices=['no', 'full', 'part'],
                            help='no: no cache, '
                                 'full: cache all data, '
                                 'part: sharding the dataset into nonoverlapping pieces and only cache one piece')
        parser.add_argument('--resume', help='resume from checkpoint',
                            default=os.path.join(os.path.split(__file__)[0],
                                                 'swin_transform_focalloss.pth'))
        parser.add_argument('--accumulation-steps', type=int, help="gradient accumulation steps")
        parser.add_argument('--use-checkpoint', action='store_true',
                            help="whether to use gradient checkpointing to save memory")
        parser.add_argument('--amp-opt-level', type=str, default='O1', choices=['O0', 'O1', 'O2'],
                            help='mixed precision opt level, if O0, no amp is used')
        parser.add_argument('--output', default='output', type=str, metavar='PATH',
                            help='root of output folder, the full path is <output>/<model_name>/<tag> (default: output)')
        parser.add_argument('--tag', help='tag of experiment')
        parser.add_argument('--eval', action='store_true', help='Perform evaluation only')
        parser.add_argument('--throughput', action='store_true', help='Test throughput only')

        # distributed training
        parser.add_argument("--local_rank", type=int, required=False, help='local rank for DistributedDataParallel')
        parser.add_argument("--test_dir", default='../../Data/500wan/500wan_shuffle_DeepSMILES_test.pkl', type=str,
                            help='direction for eval_dataset')
        args, _ = parser.parse_known_args()
        args.eval = True
        config = get_config(args)
        return args, config

    def run_inference_on_images(self, img_dir: str) -> List[str]:
        """
        Run SwinOCSR model on images in given directory and return SMILES

        Args:
            img_dir (str): root directory that contains a directory with images

        Returns:
            List[str]: DeepSMILES representations of images in sub-directory
                       of img_dir
        """
        data_loader = SwinOCSRInferenceImageFolder(img_dir)
        deep_smiles = []
        with torch.no_grad():
            for imgs, im_name in tqdm(data_loader, desc="EVALUATING INPUT IMAGES"):
                # Add dummy dimension
                imgs = imgs[None, :]
                imgs = imgs.to(self.dev)

                imgs = self.encoder(imgs)

                _, logits = Greedy_decode(self.decoder, imgs, max_len=102,
                                          start_symbol=self.word_map['<start>'],
                                          end_symbol=self.word_map['<end>'])

                _, preds = torch.max(logits, dim=-1)
                for r in range(preds.shape[0]):
                    pre_list = []
                    for j in preds[r].tolist():
                        if j != self.word_map['<end>']:
                            pre_list.append(self.inv_word_map[j])
                        else:
                            break
                    deep_smiles.append((im_name, ''.join(pre_list)))
        return deep_smiles

    def deepsmiles2smiles(self, deep_smiles: str) -> str:
        """
        This method takes a DeepSMILES str and returns a SMILES str.
        If the given deepsmiles cannot be parsed, False is returned

        Args:
            deepsmiles (str): DeepSMILES representation of molecule

        Returns:
            str: SMILES representation of molecule
        """
        converter = deepsmiles.Converter(rings=True, branches=True)
        try:
            smiles = converter.decode(deep_smiles)
        except deepsmiles.DecodeError:
            smiles = False
        return smiles


class SwinOCSRInferenceImageFolder(torchvision.datasets.ImageFolder):
    """
    Modified version of torchvision's ImageFolder.
    Modifications:
    - The root directory is the image directory, and not a
      subdirectory of the image directoy
    - Every sample contains the image and the image name.
    - Transformations are pre-defined to match SwinOCSR images
    """
    def __init__(
        self,
        image_dir: str,
    ):
        self.root = image_dir
        # Define image normalisation steps
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])
        self.samples = self.make_dataset(self.root)
        self.loader = self.pil_loader
        self.targets = [s[1] for s in self.samples]

    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        """
        Args:
            index (int): Index

        Returns:
            tuple: (sample, target) image, filename
        """
        path, _ = self.samples[index]
        target = os.path.split(path)[-1]
        sample = self.loader(path)
        if self.transform is not None:
            sample = self.transform(sample)
        return sample, target

    def __len__(self) -> int:
        return len(self.samples)

    def pil_loader(self, path: str) -> Image.Image:
        with open(path, "rb") as f:
            img = Image.open(f)
            return img.convert("RGB")

    def has_file_allowed_extension(self, filename: str, extensions: Union[str, Tuple[str, ...]]) -> bool:
        """Checks if a file is an allowed extension.

        Args:
            filename (string): path to a file
            extensions (tuple of strings): extensions to consider (lowercase)

        Returns:
            bool: True if the filename ends with one of given extensions
        """
        return filename.lower().endswith(extensions if isinstance(extensions, str) else tuple(extensions))

    def make_dataset(
        self,
        directory: str,
    ) -> List[Tuple[str, int]]:
        """Generates a list of samples of a form (path_to_sample, 0).

        The samples are created based on all images in a given directory.

        """
        extensions = (".jpg", ".jpeg", ".png", ".ppm", ".bmp", ".pgm",
                      ".tif", ".tiff", ".webp")
        directory = os.path.expanduser(directory)

        def is_valid_file(x: str) -> bool:
            return self.has_file_allowed_extension(x, extensions)

        is_valid_file = cast(Callable[[str], bool], is_valid_file)

        instances = []
        for im_name in os.listdir(directory):
            im_path = os.path.join(directory, im_name)
            if is_valid_file(im_path):
                item = im_path, 0
                instances.append(item)
        return instances


def main():
    inference = FocalLossModelInference()
    img_dir = inference.args.data_path
    deep_smiles_list = inference.run_inference_on_images(img_dir)
    smiles_list = [(im_name, inference.deepsmiles2smiles(deep_smiles))
                   for im_name, deep_smiles in deep_smiles_list]
    output_file_path = os.path.join(img_dir, 'smiles_output.tsv')
    with open(output_file_path, 'w') as output_file:
        for im_name, smiles in smiles_list:
            output_file.write(f"{im_name}\t{smiles}\n")


if __name__ == '__main__':
    main()
