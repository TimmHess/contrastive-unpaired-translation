"""Dataset class template

This module provides a template for users to implement custom datasets.
You can specify '--dataset_mode template' to use this dataset.
The class name should be consistent with both the filename and its dataset_mode option.
The filename should be <dataset_mode>_dataset.py
The class name should be <Dataset_mode>Dataset.py
You need to implement the following functions:
    -- <modify_commandline_options>:　Add dataset-specific options and rewrite default values for existing options.
    -- <__init__>: Initialize this dataset class.
    -- <__getitem__>: Return a data point and its metadata information.
    -- <__len__>: Return the number of images.
"""
from email.policy import default
from data.base_dataset import BaseDataset, get_transform
from data.image_folder import make_dataset_from_csv
from PIL import Image
import util.util as util

import random

class UERoboCupDataset(BaseDataset):
    """A template dataset class for you to implement custom datasets."""
    @staticmethod
    def modify_commandline_options(parser, is_train):
        """Add new dataset-specific options, and rewrite default values for existing options.

        Parameters:
            parser          -- original option parser
            is_train (bool) -- whether training phase or test phase. You can use this flag to add training-specific or test-specific options.

        Returns:
            the modified parser.
        """
        parser.add_argument('--data_dirs_map_A', type=str, default=None, help='csv file listing all paths to load data from for train set A')
        parser.add_argument('--data_dirs_map_B', type=str, default=None, help='csv file listing all paths to load data from for train set B')
        
        return parser

    def __init__(self, opt):
        """Initialize this dataset class.

        Parameters:
            opt (Option class) -- stores all the experiment flags; needs to be a subclass of BaseOptions

        A few things can be done here.
        - save the options (have been done in BaseDataset)
        - get image paths and meta information of the dataset.
        - define the image transformation.
        """
        # save the option and dataset root
        BaseDataset.__init__(self, opt)

        # Sanity check that directory 'maps' for A and B are provided
        assert(opt.data_dirs_map_A is not None)
        assert(opt.data_dirs_map_B is not None)

        # get the image paths of your dataset;
        self.A_paths = []  # You can call sorted(make_dataset(self.root, opt.max_dataset_size)) to get all the image paths under the directory self.root
        self.A_paths = make_dataset_from_csv(opt.dataroot, opt.data_dirs_map_A)
        self.A_size = len(self.A_paths)

        self.B_paths = []
        self.B_paths = make_dataset_from_csv(opt.dataroot, opt.data_dirs_map_B)        
        self.B_size = len(self.B_paths)


        # define the default transform function. You can use <base_dataset.get_transform>; You can also define your custom transform function
        self.transform = get_transform(opt)

    def __getitem__(self, index):
        """Return a data point and its metadata information.

        Parameters:
            index -- a random integer for data indexing

        Returns:
            a dictionary of data with their names. It usually contains the data itself and its metadata information.

        Step 1: get a random image path: e.g., path = self.image_paths[index]
        Step 2: load your data from the disk: e.g., image = Image.open(path).convert('RGB').
        Step 3: convert your data to a PyTorch tensor. You can use helpder functions such as self.transform. e.g., data = self.transform(image)
        Step 4: return a data point as a dictionary.
        """
        A_path = self.A_paths[index % self.A_size]  # make sure index is within then range
        if self.opt.serial_batches:   # make sure index is within then range
            index_B = index % self.B_size
        else:   # randomize the index for domain B to avoid fixed pairs.
            index_B = random.randint(0, self.B_size - 1)
        B_path = self.B_paths[index_B]
        A_img = Image.open(A_path).convert('RGB')
        B_img = Image.open(B_path).convert('RGB')

        #import cv2
        #import numpy as np
        #cv2.imshow("img_A", np.asarray(A_img))
        #cv2.imshow("img_B", np.asarray(B_img))
        #cv2.waitKey()
        #import sys
        #sys.exit()

        is_finetuning = self.opt.isTrain and self.current_epoch > self.opt.n_epochs
        modified_opt = util.copyconf(self.opt, load_size=self.opt.crop_size if is_finetuning else self.opt.load_size)
        transform = get_transform(modified_opt)
        A = transform(A_img)
        B = transform(B_img)

        return {'A': A, 'B': B, 'A_paths': A_path, 'B_paths': B_path}

    def __len__(self):
        """Return the total number of images."""
        return max(self.A_size, self.B_size)
