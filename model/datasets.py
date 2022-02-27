import torch
from torch.utils.data import Dataset
import h5py
import json
import os
from PIL import Image
import pandas as pd
import  numpy as np
class CaptionDataset(Dataset):
    """
    A PyTorch Dataset class to be used in a PyTorch DataLoader to create batches.
    """

    def __init__(self, data_folder, data_name, split, transform=None):
        """
        :param data_folder: folder where data files are stored
        :param data_name: base name of processed datasets
        :param split: split, one of 'TRAIN', 'VAL', or 'TEST'
        :param transform: image transform pipeline
        """
        self.split = split

        assert self.split in {'TRAIN', 'TRAIN_DOMAIN', 'TRAIN_TARGET', 'VAL', 'TEST'}

        # Open hdf5 file where images are stored
        self.h = h5py.File(os.path.join(data_folder, self.split + '_IMAGES_' + data_name + '.hdf5'), 'r')
        self.imgs = self.h['images']
        # print(self.imgs[0])

        # Captions per image
        self.cpi = self.h.attrs['captions_per_image']
        # print(self.cpi)

        # Load encoded captions (completely into memory)
        with open(os.path.join(data_folder, self.split + '_CAPTIONS_' + data_name + '.json'), 'r') as j:
            self.captions = json.load(j)

        # Load caption lengths (completely into memory)
        with open(os.path.join(data_folder, self.split + '_CAPLENS_' + data_name + '.json'), 'r') as j:
            self.caplens = json.load(j)

        # PyTorch transformation pipeline for the image (normalizing, etc.)
        self.transform = transform

        # Total number of datapoints
        self.dataset_size = len(self.captions)

    def __getitem__(self, i):
        # Remember, the Nth caption corresponds to the (N // captions_per_image)th image
        img = torch.FloatTensor(self.imgs[i // self.cpi] / 255.)
        if self.transform is not None:
            img = self.transform(img)

        caption = torch.LongTensor(self.captions[i])

        caplen = torch.LongTensor([self.caplens[i]])
        return img, caption, caplen


    def __len__(self):
        return self.dataset_size

class CaptionDataset_500wan(Dataset):
    """
    A PyTorch Dataset class to be used in a PyTorch DataLoader to create batches.
    """

    def __init__(self, dir, split, transform=None):
        """
        :param data_folder: folder where data files are stored
        :param data_name: base name of processed datasets
        :param split: split, one of 'TRAIN', 'VAL', or 'TEST'
        :param transform: image transform pipeline
        """
        self.split = split

        df = pd.read_pickle(dir+'_'+split+'.pkl')


        self.imgs = df['image_path'].values.astype(np.str)

        # Captions per image

        # Load encoded captions (completely into memory)
        self.captions = df['label_map'].values

        # Load caption lengths (completely into memory)

        self.caplens = df['label_length'].values

        # PyTorch transformation pipeline for the image (normalizing, etc.)
        self.transform = transform
        self.dataset_size = len(self.captions)



    def __getitem__(self, i):
        # Remember, the Nth caption corresponds to the (N // captions_per_image)th image
        img_file = '../../Data/500wan/500wanBinarizationPNG/'+self.imgs[i]+'.png'
        img = Image.open(img_file).convert('RGB')
        if self.transform is not None:
            img = self.transform(img)
        #print(self.captions[i])
        caption = torch.LongTensor(self.captions[i])

        caplen = torch.LongTensor([self.caplens[i]])

        return img, caption, caplen


    def __len__(self):
        return self.dataset_size

class CaptionDataset_500wan_test(Dataset):
    """
    A PyTorch Dataset class to be used in a PyTorch DataLoader to create batches.
    """

    def __init__(self, dir, transform=None):
        """
        :param data_folder: folder where data files are stored
        :param data_name: base name of processed datasets
        :param split: split, one of 'TRAIN', 'VAL', or 'TEST'
        :param transform: image transform pipeline
        """

        df = pd.read_pickle(dir)


        self.imgs = df['image_path'].values.astype(np.str)

        # Captions per image

        # Load encoded captions (completely into memory)
        self.captions = df['label_map'].values

        # Load caption lengths (completely into memory)

        self.caplens = df['label_length'].values

        # PyTorch transformation pipeline for the image (normalizing, etc.)
        self.transform = transform
        self.dataset_size = len(self.captions)



    def __getitem__(self, i):
        # Remember, the Nth caption corresponds to the (N // captions_per_image)th image
        img_file = '../../Data/500wan/500wanBinarizationPNG/'+self.imgs[i]+'.png'
        img = Image.open(img_file).convert('RGB')
        if self.transform is not None:
            img = self.transform(img)

        caption = torch.LongTensor(self.captions[i])

        caplen = torch.LongTensor([self.caplens[i]])

        return img, caption, caplen


    def __len__(self):
        return self.dataset_size