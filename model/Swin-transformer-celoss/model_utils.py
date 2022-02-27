# --------------------------------------------------------
# Swin Transformer
# Copyright (c) 2021 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ze Liu
# --------------------------------------------------------

import os
import torch
import torch.distributed as dist
import numpy as np
import h5py
import json
#from scipy.misc import imread, imresize
from tqdm import tqdm
from collections import Counter
from random import seed, choice, sample
import datetime,time

try:
    # noinspection PyUnresolvedReferences
    from apex import amp
except ImportError:
    amp = None


def load_checkpoint(config, encoder, encoder_optimizer, decoder, decoder_optimizer, encoder_lr_scheduler, decoder_lr_scheduler, logger):
    logger.info(f"==============> Resuming form {config.MODEL.RESUME}....................")
    if config.MODEL.RESUME.startswith('https'):
        checkpoint = torch.hub.load_state_dict_from_url(
            config.MODEL.RESUME, map_location='cpu', check_hash=True)
    else:
        checkpoint = torch.load(config.MODEL.RESUME, map_location='cpu')
    # for key in list(checkpoint['encoder'].keys()):
    #     if ('relative_position_bias_table' in key) or ('relative_position_index' in key) or ('attn_mask' in key):
    #         checkpoint['encoder'].pop(key)
    encoder_msg = encoder.load_state_dict(checkpoint['encoder'], strict=False)
    decoder_msg = decoder.load_state_dict(checkpoint['decoder'], strict=False)
    logger.info(encoder_msg)
    logger.info(decoder_msg)
    max_accuracy = 0.0
    if not config.EVAL_MODE :
        encoder_optimizer.load_state_dict(checkpoint['encoder_optimizer'])
        decoder_optimizer.load_state_dict(checkpoint['decoder_optimizer'])
        encoder_lr_scheduler.load_state_dict(checkpoint['encoder_lr_scheduler'])
        decoder_lr_scheduler.load_state_dict(checkpoint['decoder_lr_scheduler'])
        config.defrost()
        config.TRAIN.START_EPOCH = checkpoint['epoch'] + 1
        config.freeze()
        # if 'amp' in checkpoint and config.AMP_OPT_LEVEL != "O0" and checkpoint['config'].AMP_OPT_LEVEL != "O0":
        #     amp.load_state_dict(checkpoint['amp'])
        logger.info(f"=> loaded successfully '{config.MODEL.RESUME}' (epoch {checkpoint['epoch']})")
        if 'max_accuracy' in checkpoint:
            max_accuracy = checkpoint['max_accuracy']
    acc = checkpoint['acc']
    del checkpoint
    torch.cuda.empty_cache()
    return max_accuracy, acc

# def load_checkpoint(config, model, optimizer, lr_scheduler, logger):
#     logger.info(f"==============> Resuming form {config.MODEL.RESUME}....................")
#     if config.MODEL.RESUME.startswith('https'):
#         checkpoint = torch.hub.load_state_dict_from_url(
#             config.MODEL.RESUME, map_location='cpu', check_hash=True)
#     else:
#         checkpoint = torch.load(config.MODEL.RESUME, map_location='cpu')
#     msg = model.load_state_dict(checkpoint['model'], strict=False)
#     logger.info(msg)
#     max_accuracy = 0.0
#     if not config.EVAL_MODE and 'optimizer' in checkpoint and 'lr_scheduler' in checkpoint and 'epoch' in checkpoint:
#         # optimizer.load_state_dict(checkpoint['optimizer'])
#         # lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
#         config.defrost()
#         config.TRAIN.START_EPOCH = checkpoint['epoch'] + 1
#         config.freeze()
#         # if 'amp' in checkpoint and config.AMP_OPT_LEVEL != "O0" and checkpoint['config'].AMP_OPT_LEVEL != "O0":
#         #     amp.load_state_dict(checkpoint['amp'])
#         logger.info(f"=> loaded successfully '{config.MODEL.RESUME}' (epoch {checkpoint['epoch']})")
#         if 'max_accuracy' in checkpoint:
#             max_accuracy = checkpoint['max_accuracy']
#
#     del checkpoint
#     torch.cuda.empty_cache()
#     return max_accuracy



def save_checkpoint(config, epoch, encoder, encoder_optimizer, decoder, decoder_optimizer,  max_accuracy, encoder_lr_scheduler, decoder_lr_scheduler,  logger, acc, is_best):
    save_state = {'encoder': encoder.state_dict(),
                  'encoder_optimizer': encoder_optimizer.state_dict(),
                  'decoder': decoder.state_dict(),
                  'decoder_optimizer': decoder_optimizer.state_dict(),
                  'encoder_lr_scheduler': encoder_lr_scheduler.state_dict(),
                  'decoder_lr_scheduler': decoder_lr_scheduler.state_dict(),
                  'max_accuracy': max_accuracy,
                  'epoch': epoch,
                  'config': config,
                  'acc': acc}
    # if config.AMP_OPT_LEVEL != "O0":
    #     save_state['amp'] = amp.state_dict()

    save_path = os.path.join(config.OUTPUT, f'swin_transform_celoss_{epoch}.pth')
    logger.info(f"{save_path} saving......")
    torch.save(save_state, save_path)
    logger.info(f"{save_path} saved !!!")
    if is_best:
        # torch.save(state,base_dir+ 'BEST_'+ filename)
        torch.save(save_state, config.OUTPUT+'BEST.pth')


def get_grad_norm(parameters, norm_type=2):
    if isinstance(parameters, torch.Tensor):
        parameters = [parameters]
    parameters = list(filter(lambda p: p.grad is not None, parameters))
    norm_type = float(norm_type)
    total_norm = 0
    for p in parameters:
        param_norm = p.grad.data.norm(norm_type)
        total_norm += param_norm.item() ** norm_type
    total_norm = total_norm ** (1. / norm_type)
    return total_norm


def auto_resume_helper(output_dir):
    checkpoints = os.listdir(output_dir)
    checkpoints = [ckpt for ckpt in checkpoints if ckpt.endswith('pth')]
    print(f"All checkpoints founded in {output_dir}: {checkpoints}")
    if len(checkpoints) > 0:
        latest_checkpoint = max([os.path.join(output_dir, d) for d in checkpoints], key=os.path.getmtime)
        print(f"The latest checkpoint founded: {latest_checkpoint}")
        resume_file = latest_checkpoint
    else:
        resume_file = None
    return resume_file


def reduce_tensor(tensor):
    rt = tensor.clone()
    dist.all_reduce(rt, op=dist.ReduceOp.SUM)
    rt /= dist.get_world_size()
    return rt
def create_input_files(dataset, karpathy_json_path, image_folder, captions_per_image, min_word_freq, output_folder,
                       max_len=100):
    """
    Creates input files for training, validation, and test data.

    :param dataset: name of dataset, one of 'coco', 'flickr8k', 'flickr30k'
    :param karpathy_json_path: path of Karpathy JSON file with splits and captions
    :param image_folder: folder with downloaded images
    :param captions_per_image: number of captions to sample per image
    :param min_word_freq: words occuring less frequently than this threshold are binned as <unk>s
    :param output_folder: folder to save files
    :param max_len: don't sample captions longer than this length
    """

    assert dataset in {'coco', 'flickr8k', 'flickr30k'}

    # Read Karpathy JSON
    with open(karpathy_json_path, 'r') as j:
        data = json.load(j)

    # Read image paths and captions for each image
    train_image_paths = []
    train_image_captions = []
    val_image_paths = []
    val_image_captions = []
    test_image_paths = []
    test_image_captions = []
    word_freq = Counter()

    for img in data['images']:
        captions = []
        for c in img['sentences']:
            # Update word frequency
            word_freq.update(c['tokens'])
            if len(c['tokens']) <= max_len:
                captions.append(c['tokens'])

        if len(captions) == 0:
            continue

        path = os.path.join(image_folder, img['filepath'], img['filename']) if dataset == 'coco' else os.path.join(
            image_folder, img['filename'])

        if img['split'] in {'train', 'restval'}:
            train_image_paths.append(path)
            train_image_captions.append(captions)
        elif img['split'] in {'val'}:
            val_image_paths.append(path)
            val_image_captions.append(captions)
        elif img['split'] in {'test'}:
            test_image_paths.append(path)
            test_image_captions.append(captions)

    # Sanity check
    assert len(train_image_paths) == len(train_image_captions)
    assert len(val_image_paths) == len(val_image_captions)
    assert len(test_image_paths) == len(test_image_captions)

    # Create word map
    words = [w for w in word_freq.keys() if word_freq[w] > min_word_freq]
    word_map = {k: v + 1 for v, k in enumerate(words)}
    word_map['<unk>'] = len(word_map) + 1
    word_map['<start>'] = len(word_map) + 1
    word_map['<end>'] = len(word_map) + 1
    word_map['<pad>'] = 0

    # Create a base/root name for all output files
    base_filename = dataset + '_' + str(captions_per_image) + '_cap_per_img_' + str(min_word_freq) + '_min_word_freq'

    # Save word map to a JSON
    now = time.strftime("%Y-%m-%d_%H-%M-%S")
    with open(os.path.join(output_folder, 'WORDMAP_' + base_filename +'.json'), 'w') as j:

        json.dump(word_map, j)

    # Sample captions for each image, save images to HDF5 file, and captions and their lengths to JSON files
    seed(123)
    for impaths, imcaps, split in [(train_image_paths, train_image_captions, 'TRAIN'),
                                   (val_image_paths, val_image_captions, 'VAL'),
                                   (test_image_paths, test_image_captions, 'TEST')]:

        with h5py.File(os.path.join(output_folder, split + '_IMAGES_' + base_filename + '.hdf5'), 'a') as h:
            # Make a note of the number of captions we are sampling per image
            h.attrs['captions_per_image'] = captions_per_image

            # Create dataset inside HDF5 file to store images
            images = h.create_dataset('images', (len(impaths), 3, 300, 300), dtype='uint8')

            print("\nReading %s images and captions, storing to file...\n" % split)

            enc_captions = []
            caplens = []

            for i, path in enumerate(tqdm(impaths)):

                # Sample captions
                if len(imcaps[i]) < captions_per_image:
                    captions = imcaps[i] + [choice(imcaps[i]) for _ in range(captions_per_image - len(imcaps[i]))]
                else:
                    captions = sample(imcaps[i], k=captions_per_image)

                # Sanity check
                assert len(captions) == captions_per_image

                # Read images
                try:
                    img = imread(impaths[i])
                    if len(img.shape) == 2:
                        img = img[:, :, np.newaxis]
                        img = np.concatenate([img, img, img], axis=2)
                    img = imresize(img, (300, 300))
                    img = img.transpose(2, 0, 1)
                    assert img.shape == (3, 300, 300)
                    assert np.max(img) <= 255

                    # Save image to HDF5 file
                    images[i] = img

                    for j, c in enumerate(captions):
                        # Encode captions
                        enc_c = [word_map['<start>']] + [word_map.get(word, word_map['<unk>']) for word in c] + [
                            word_map['<end>']] + [word_map['<pad>']] * (max_len - len(c))

                        # Find caption lengths
                        c_len = len(c) + 2

                        enc_captions.append(enc_c)
                        caplens.append(c_len)
                except:
                    continue

            # Sanity check
            # assert images.shape[0] * captions_per_image == len(enc_captions) == len(caplens)

            now = time.strftime("%Y-%m-%d_%H-%M-%S")
            # Save encoded captions and their lengths to JSON files
            with open(os.path.join(output_folder, split + '_CAPTIONS_' + base_filename + '.json'), 'w') as j:
                json.dump(enc_captions, j)

            with open(os.path.join(output_folder, split + '_CAPLENS_' + base_filename + '.json'), 'w') as j:
                json.dump(caplens, j)


def init_embedding(embeddings):
    """
    Fills embedding tensor with values from the uniform distribution.

    :param embeddings: embedding tensor
    """
    bias = np.sqrt(3.0 / embeddings.size(1))
    torch.nn.init.uniform_(embeddings, -bias, bias)


def load_embeddings(emb_file, word_map):
    """
    Creates an embedding tensor for the specified word map, for loading into the model.

    :param emb_file: file containing embeddings (stored in GloVe format)
    :param word_map: word map
    :return: embeddings in the same order as the words in the word map, dimension of embeddings
    """

    # Find embedding dimension
    with open(emb_file, 'r') as f:
        emb_dim = len(f.readline().split(' ')) - 1

    vocab = set(word_map.keys())

    # Create tensor to hold embeddings, initialize
    embeddings = torch.FloatTensor(len(vocab), emb_dim)
    init_embedding(embeddings)

    # Read embedding file
    print("\nLoading embeddings...")
    for line in open(emb_file, 'r'):
        line = line.split(' ')

        emb_word = line[0]
        embedding = list(map(lambda t: float(t), filter(lambda n: n and not n.isspace(), line[1:])))

        # Ignore word if not in train_vocab
        if emb_word not in vocab:
            continue

        embeddings[word_map[emb_word]] = torch.FloatTensor(embedding)

    return embeddings, emb_dim


def clip_gradient(optimizer, grad_clip):
    """
    Clips gradients computed during backpropagation to avoid explosion of gradients.

    :param optimizer: optimizer with the gradients to be clipped
    :param grad_clip: clip value
    """
    for group in optimizer.param_groups:
        for param in group['params']:
            if param.grad is not None:
                param.grad.data.clamp_(-grad_clip, grad_clip)






def adjust_learning_rate(optimizer, shrink_factor):
    """
    Shrinks learning rate by a specified factor.

    :param optimizer: optimizer whose learning rate must be shrunk.
    :param shrink_factor: factor in interval (0, 1) to multiply learning rate with.
    """

    print("\nDECAYING learning rate.")
    for param_group in optimizer.param_groups:
        param_group['lr'] = param_group['lr'] * shrink_factor
    print("The new learning rate is %f\n" % (optimizer.param_groups[0]['lr'],))


def print_model_parm_nums(model):            #得到模型参数总量
    total = sum([param.nelement() for param in model.parameters()])
    print('  + Number of params: %.2fM' % (total / 1e6))     #每一百万为一个单位


def accuracy(scores, targets, k):
    """
    Computes top-k accuracy, from predicted and true labels.

    :param scores: scores from the model
    :param targets: true labels
    :param k: k in top-k accuracy
    :return: top-k accuracy
    """

    batch_size = targets.size(0)
    _, ind = scores.topk(k, 1, True, True)
    correct = ind.eq(targets.view(-1, 1).expand_as(ind))
    correct_total = correct.view(-1).float().sum()  # 0D tensor
    return correct_total.item() * (100.0 / batch_size)