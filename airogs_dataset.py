import torchvision
import pandas as pd
import glob
import os
from PIL import Image
import cv2
import numpy as np
from PIL import Image
from skimage.exposure import equalize_adapthist
from skimage.transform import warp_polar

def polar(image):
    return warp_polar(image, radius=(max(image.shape) // 2), multichannel=True)

class Airogs(torchvision.datasets.VisionDataset):

    def __init__(self, split='train', path='', images_dir_name='train',transforms=None,polar_transforms=False,apply_clahe=False):
        self.split = split
        self.path = path
        self.images_dir_name = images_dir_name        
        self.df_files = pd.read_csv(os.path.join(self.path, self.split + ".csv"))    ## columns = ['challenge_id', 'class', 'referable', 'gradable']
        self.transforms = transforms
        self.polar_transforms = polar_transforms
        self.apply_clahe = apply_clahe
        print("{} size: {}".format(split, len(self.df_files)))

    def __getitem__(self, index):
        file_name = self.df_files.loc[index, 'challenge_id']
        path_mask = os.path.join(self.path, self.images_dir_name,"*" ,file_name + '.jpg')
        image_path = glob.glob(path_mask)[0]
        image = Image.open(image_path)
        
        label = self.df_files.loc[index, 'class']
        label = 0 if label == 'NRG' else 1

        if self.polar_transforms:
            image = image = np.array(image, dtype=np.float64)
            image = polar(image)

        if self.apply_clahe:
            image = np.array(image, dtype=np.float64) / 255.0
            image = equalize_adapthist(image)
            image = (image*255).astype('uint8')

        assert(self.transforms != None)   
        image = self.transforms(image)
        return image, label
        
    def __len__(self):
        return len(self.df_files)

