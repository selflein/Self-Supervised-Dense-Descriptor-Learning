# contains the class SiameseDataset which loads images for the use in a neural net
# contains the class SiameseDatasetEdgeDetection which loads images for the use in a neural net and samples points only from edge points
 

import random
from pathlib import Path

import numpy as np
import h5py
import torch
from torchvision import transforms
from torch.utils.data import Dataset
from visual_descriptor.dataloader import dataset_helper
import os
import itertools
from functools import partial


class SiameseDataset(Dataset):

    def __init__(self,
                 image_folder='/storage/remote/atcremers51/w0020/visual_descriptor/frames',
                 num_points=100):
                 
        self.img_list = list(Path(image_folder).glob('*.jpg'))
        self.num_points = num_points

        self.process_pipline = [dataset_helper.brightness_change, dataset_helper.get_original]
        permuations = list(itertools.permutations([0, 1, 2]))
        # self.process_pipline.extend(partial(dataset_helper.channel_flipping,*permuation) for permuation in permuations)

        self.transform = transforms.Compose(
            [transforms.ToTensor(),
             transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
            ]
        )

    def __len__(self):
        return len(self.img_list)

    def __getitem__(self, idx):
        orig, orig_points = dataset_helper.get_image_from_list(self.img_list, idx, self.num_points)

        img_1, img_1_points = dataset_helper.create_augmented_image(orig, orig_points, self.process_pipline)
        img_1 = self.transform(img_1)
        img_2, img_2_points = dataset_helper.create_augmented_image(orig, orig_points, self.process_pipline)
        img_2 = self.transform(img_2)

        img_1_points = np.array(img_1_points)
        img_2_points = np.array(img_2_points)

        return (
            img_1, img_2,
            torch.from_numpy(img_1_points),
            torch.from_numpy(img_2_points)
        ) 


class SiameseDatasetEdgeDetection(Dataset):

    def __init__(self,
                 image_folder='/storage/remote/atcremers51/w0020/visual_descriptor/frames',
                 num_points=100):

        self.img_list = list(Path(image_folder).glob('*.jpg'))
        self.num_points = num_points

        self.process_pipline = [dataset_helper.brightness_change, dataset_helper.get_original]
        permuations = list(itertools.permutations([0, 1, 2]))
        self.process_pipline.extend(partial(dataset_helper.channel_flipping,*permuation) for permuation in permuations)

        self.transform = transforms.Compose(
            [transforms.ToTensor(),
             # transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
            ]
        )

    def __len__(self):
        return len(self.img_list)

    def __getitem__(self, idx):
        orig, orig_points = dataset_helper.get_image_from_list(self.img_list, idx, self.num_points, True)

        img_1, img_1_points = dataset_helper.create_augmented_image(orig, orig_points, self.process_pipline)
        img_1 = self.transform(img_1)
        img_2, img_2_points = dataset_helper.create_augmented_image(orig, orig_points, self.process_pipline)
        img_2 = self.transform(img_2)

        img_1_points = np.array(img_1_points)
        img_2_points = np.array(img_2_points)

        return (
            img_1, img_2,
            torch.from_numpy(np.array(img_1_points)),
            torch.from_numpy(np.array(img_2_points))
        )


class SiameseDatasetAllAugmentations(Dataset):

    def __init__(self,
                 image_folder='/storage/remote/atcremers51/w0020/visual_descriptor/frames',
                 num_points=100):
                 
        self.img_list = list(Path(image_folder).glob('*.jpg'))
        self.num_points = num_points

        self.process_pipline = [dataset_helper.brightness_change, dataset_helper.get_original, dataset_helper.crop_out_of_the_other, dataset_helper.crop_with_overlap]
        permuations = list(itertools.permutations([0, 1, 2]))
        self.process_pipline.extend(partial(dataset_helper.channel_flipping,*permuation) for permuation in permuations)
        self.process_pipline.extend(partial(dataset_helper.rotating, degree) for degree in [90,180,-90])

        self.transform = transforms.Compose(
            [transforms.ToTensor(),
             # transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
            ]
        )

    def __len__(self):
        return len(self.img_list)

    def __getitem__(self, idx):
        orig, orig_points = dataset_helper.get_image_from_list(self.img_list, idx, self.num_points, True)

        img_1, img_1_points = dataset_helper.create_augmented_image(orig, orig_points, self.process_pipline)
        img_1 = self.transform(img_1)
        img_2, img_2_points = dataset_helper.create_augmented_image(orig, orig_points, self.process_pipline)
        img_2 = self.transform(img_2)

        img_1_points = np.array(img_1_points)
        img_2_points = np.array(img_2_points)

        return (
            img_1, img_2,
            torch.from_numpy(np.array(img_1_points)),
            torch.from_numpy(np.array(img_2_points))
        )
