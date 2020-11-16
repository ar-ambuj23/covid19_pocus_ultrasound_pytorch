import os
from torch.utils.data import Dataset
from PIL import Image

class PocovidDataset(Dataset):
    """Subclass of Dataset for POCOVID-Net data"""
  
    def __init__(self, root_dir, transform=None):
        """
        Args:
          root_dir (string): Directory with all the images.
          transform (callable, optional): Optional transform to be applied
              on a sample.
        """
        self.__transform = transform
        covid_dir = root_dir + '/' + 'covid'
        pneu_dir = root_dir + '/' + 'pneumonia'
        regular_dir = root_dir + '/' + 'regular'

        self.__covid_class = 0
        self.__pneu_class = 1
        self.__regular_class = 2

        # Modified code snippet from Daniel Stutzbach:
        # https://stackoverflow.com/a/2632251
        dir_items = lambda d: [d + "/" + name for name in os.listdir(d) if os.path.isfile(d + "/" + name)]

        covid_items = dir_items(covid_dir)
        pneu_items = dir_items(pneu_dir)
        regular_items = dir_items(regular_dir)
      
        num_covid = len(covid_items)
        num_pneu = len(pneu_items)
        num_regular = len(regular_items)
      
        self.__img_info = []
        for covid_filename in covid_items:
            self.__img_info.append((covid_filename,self.__covid_class))
        for pneu_filename in pneu_items:
            self.__img_info.append((pneu_filename,self.__pneu_class))
        for regular_filename in regular_items:
            self.__img_info.append((regular_filename,self.__regular_class)) 

        self.__transform = transform
        self.__num_images = num_covid + num_pneu + num_regular
  
    def __len__(self):
        return self.__num_images
  
    def __getitem__(self,idx):
        img_name, img_class = self.__img_info[idx]
        image = Image.open(img_name)
        sample = {'image': image, 'class': img_class} 

        if self.__transform:
            sample['image'] = self.__transform(image)

        return sample