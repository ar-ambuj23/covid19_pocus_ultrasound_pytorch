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
        self.root_dir = root_dir
        self.transform = transform
        self.covid_dir = root_dir + '/' + 'covid'
        self.pneu_dir = root_dir + '/' + 'pneumonia'
        self.regular_dir = root_dir + '/' + 'regular'

        self.covid_class = 0
        self.pneu_class = 1
        self.regular_class = 2

        # Modified code snippet from Daniel Stutzbach:
        # https://stackoverflow.com/a/2632251
        dir_items = lambda d: [d + "/" + name for name in os.listdir(d) if os.path.isfile(d + "/" + name)]

        covid_items = dir_items(self.covid_dir)
        pneu_items = dir_items(self.pneu_dir)
        regular_items = dir_items(self.regular_dir)
      
        num_covid = len(covid_items)
        num_pneu = len(pneu_items)
        num_regular = len(regular_items)
      
        self.img_info = []
        for covid_filename in covid_items:
            self.img_info.append((covid_filename,self.covid_class))
        for pneu_filename in pneu_items:
            self.img_info.append((pneu_filename,self.pneu_class))
        for regular_filename in regular_items:
            self.img_info.append((regular_filename,self.regular_class)) 

        self.transform = transform
        self.num_images = num_covid + num_pneu + num_regular
  
    def __len__(self):
        return self.num_images
  
    def __getitem__(self,idx):
        img_name, img_class = self.img_info[idx]
        image = Image.open(img_name)
        sample = {'image': image, 'class': img_class} 

        if self.transform:
            sample['image'] = self.transform(image)

        return sample