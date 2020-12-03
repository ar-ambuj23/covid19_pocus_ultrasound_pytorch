import os
from torch.utils.data import Dataset
from PIL import Image
from torchvision.transforms import ToTensor

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
        
        print(covid_items)

        num_covid = 0
        num_pneu = 0
        num_regular = 0
      
        # FIXME: 
        #  1) Avoid calling the ToTensor() transformation and instead use information from PIL
        #  to determine the number of channels in an image. (This should increase speed.)
        #  2) Right now we are dropping images that do not have exactly 3 channels. In the future,
        #  we may want to support 4 channel images, e.g. by removing the alpha channel or compositing 
        #  against a black (or maybe white?) background. 
        
        self.__img_info = []
        for covid_filename in covid_items: 
            image = Image.open(covid_filename)
            to_tensor_tsfm = ToTensor()
            im_tensor = to_tensor_tsfm(image)
            if im_tensor.shape[0] == 3:
                self.__img_info.append((covid_filename,self.__covid_class))
                num_covid += 1
            
        for pneu_filename in pneu_items:
            image = Image.open(pneu_filename)
            to_tensor_tsfm = ToTensor()
            im_tensor = to_tensor_tsfm(image)
            if im_tensor.shape[0] == 3:
                self.__img_info.append((pneu_filename,self.__pneu_class))
                num_pneu += 1
                
        for regular_filename in regular_items:
            image = Image.open(regular_filename)
            to_tensor_tsfm = ToTensor()
            im_tensor = to_tensor_tsfm(image)
            if im_tensor.shape[0] == 3:
                self.__img_info.append((regular_filename,self.__regular_class)) 
                num_regular += 1

        self.__transform = transform
        self.__num_images = num_covid + num_pneu + num_regular
  
    def __len__(self):
        return self.__num_images
  
    def __getitem__(self,idx):
        img_name, img_class = self.__img_info[idx]
        image = Image.open(img_name)
#         sample = {'image': image, 'class': img_class} 
        sample = [image, img_class]

        if self.__transform:
#             sample['image'] = self.__transform(image)
            sample = [self.__transform(image), img_class]

        return sample
    
    def get_covid_class_idx(self):
        return self.__covid_class
    
    def get_pneu_class_idx(self):
        return self.__pneu_class
    
    def get_regular_class_idx(self):
        return self.__regular_class