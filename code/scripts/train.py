#!/usr/bin/env python
# coding: utf-8

# Import required libraries

import torch
import torchvision
import torchvision.transforms as transforms
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
from torchsummary import summary

import sys
sys.path.append('../models')
from vgg import VGG16_model

import sys
sys.path.append('../scripts')
from pocovid_dataset import PocovidDataset

import os
import random
from imutils import paths
from collections import defaultdict 
import numpy as np
import time
import argparse

from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt


        
# ### Fixing Random Seeds
SEED = 1234
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.backends.cudnn.deterministic = True

# Construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument('-d', '--data_dir', type=str, default='../data/cross_validation', help='path to input cross-validation dataset')
ap.add_argument('-m', '--model_name', type=str, default='vgg16')
ap.add_argument('-s', '--model_save_dir', type=str, default='../trained_models')
ap.add_argument('-f', '--fold', type=int, default='0', help='fold to take as test data')
ap.add_argument('-lr', '--learning_rate', type=float, default=1e-4)
ap.add_argument('-ep', '--epochs', type=int, default=20)
ap.add_argument('-bs', '--batch_size', type=int, default=16)
ap.add_argument('-iw', '--img_width', type=int, default=224)
ap.add_argument('-ih', '--img_height', type=int, default=224)
args = vars(ap.parse_args())


# ### Initializing parameters
CROSS_VAL_DIR = args['data_dir']
MODEL_NAME = args['model_name']
MODEL_SAVE_DIR = args['model_save_dir']
FOLD = args['fold']
LR = args['learning_rate']
N_EPOCHS = args['epochs']
BATCH_SIZE = args['batch_size']
IMG_WIDTH = args['img_width']
IMG_HEIGHT = args['img_height']

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")



# ### The Metrics Class
class Metrics():
    
    def __init__(self, images, true_labels, pred_labels, pred_probs, classes):

        self.images = images
        self.true_labels = true_labels
        self.pred_labels = pred_labels
        self.pred_probs = pred_probs
        self.classes = classes
        
    def plot_confusion_matrix(self):
        fig = plt.figure(figsize = (10, 10));
        ax = fig.add_subplot(1, 1, 1);
        cm = self.get_confusion_matrix()
        cm = ConfusionMatrixDisplay(cm, display_labels = self.classes);
        cm.plot(values_format = 'd', cmap = 'Blues', ax = ax)
        plt.xticks(rotation = 20)
    
    def get_confusion_matrix(self):
        cm = confusion_matrix(self.true_labels.cpu().numpy(), self.pred_labels.cpu().numpy())
        return cm
        
    def get_classification_report(self):
        cr = classification_report(self.true_labels.cpu().numpy(), self.pred_labels.cpu().numpy(), target_names=self.classes)
        return cr

# ### The Trainer Class
class Trainer():
    
    def __init__(self, model_name=MODEL_NAME, lr=LR, n_epochs=N_EPOCHS, batch_size=BATCH_SIZE, 
                 image_width=IMG_WIDTH, image_height=IMG_HEIGHT, cross_val_dir=CROSS_VAL_DIR,
                model_save_dir=MODEL_SAVE_DIR, fold=FOLD):
        
        if(model_name=='vgg16'):
            self.model = VGG16_model().to(device)
        elif(model_name=='resnet50'):
            self.model = RESNET50_model().to(device)
        else:
            print('Select models from the following:\n 1) vgg16\n 2) resnet50')
                    
        self.lr = lr
        self.n_epochs = n_epochs
        self.batch_size = batch_size
        self.fold = fold
        self.cross_val_dir = cross_val_dir
        
        self.image_width = image_width
        self.image_height = image_height
        
        self.criterion = nn.CrossEntropyLoss().to(device)
        self.optimizer = optim.Adam(params = self.model.parameters(), lr=self.lr) #experiment with weigth_decay
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, 1, gamma=0.95) # use scheduler
        
        self.model_save_dir = model_save_dir
        self.model_name = model_name
        
        self.train_loader = None
        self.test_loader = None
        
        self.classes = None
        self.class_map = None
        
        
    def get_train_test_info(self):
        """
        Get information dictionaries for train and test data
        """
    
        imagePaths = list(paths.list_images(self.cross_val_dir))

        train_path_info = defaultdict(list)
        test_path_info = defaultdict(list)

        for imagePath in imagePaths:
            path_parts = imagePath.split(os.path.sep)
            fold_number = path_parts[-3][-1]
            label = path_parts[-2]
            if(fold_number==str(self.fold)):
                test_path_info['path_list'].append(imagePath)
                test_path_info['label_list'].append(label)
            else:
                train_path_info['path_list'].append(imagePath)
                train_path_info['label_list'].append(label)

        return train_path_info, test_path_info
    
    
    def get_train_test_loaders(self, num_workers=2):
        
        """
        Get the train and test data according to the fold
        """
        print('Loading the image data...')
        
        train_path_info, test_path_info = self.get_train_test_info()

        train_transform = transforms.Compose([transforms.Resize((self.image_width, self.image_height)),
                                           transforms.RandomAffine(10,translate=(0.1,0.1)),
                                           transforms.ToTensor()])

        test_transform = transforms.Compose([transforms.Resize((self.image_width, self.image_height)),
                                           transforms.ToTensor()])

        trainset = PocovidDataset(train_path_info, transform = train_transform)
        testset = PocovidDataset(test_path_info, transform = test_transform)
        
        self.class_map = trainset.get_class_map()
        self.classes = [self.class_map[key] for key in sorted(self.class_map)]

        train_loader = torch.utils.data.DataLoader(trainset, num_workers=num_workers, shuffle=True,
                                          batch_size=self.batch_size, drop_last=True)

        test_loader = torch.utils.data.DataLoader(testset, num_workers=num_workers, shuffle=True,
                                        batch_size=self.batch_size)
        
        print('Image data is loaded with fold {} as the test data'.format(self.fold))
        print('Number of training images:', len(trainset))
        print('Number of testing images:', len(testset))
        print('*'*100)
        print('The classes are:', self.classes)
        print('*'*100)
        
        return train_loader, test_loader
    
    def train(self, iterator):
        """
        The train function
        """
    
        self.model.train()

        epoch_loss = 0

        for i, batch in enumerate(iterator):

            inputs, labels = batch[0].to(device), batch[1].to(device)

            self.optimizer.zero_grad()

            outputs = self.model(inputs)

            loss = self.criterion(outputs, labels)

            loss.backward()

            self.optimizer.step()

            epoch_loss += loss.item()

        return epoch_loss / len(iterator)

    def evaluate(self, iterator):
        """
        The eval function
        """
    
        self.model.eval()

        epoch_loss = 0

        with torch.no_grad():    
            for i, batch in enumerate(iterator):    

                inputs, labels = batch[0].to(device), batch[1].to(device)

                outputs = self.model(inputs)

                loss = self.criterion(outputs, labels)

                epoch_loss += loss.item()

        return epoch_loss / len(iterator)
    
    def epoch_time(self, start_time, end_time):
        """
        The utility function to measure the time taken by an epoch to run
        """
        elapsed_time = end_time - start_time
        elapsed_mins = int(elapsed_time / 60)
        elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
        return elapsed_mins, elapsed_secs
    
    def training(self):
        """
        The training function which does the training by calling train and eval functions
        """
    
        best_valid_loss = np.inf
        c = 0
        
        self.train_loader, self.test_loader = self.get_train_test_loaders()
        
        print('Training the {} model with the following architecture:'.format(self.model_name))
        print(summary(self.model, (3, self.image_width, self.image_height)))
        print('*'*100)
        print('Starting the training...')
        print('*'*100)
        
        # Create the model save dir if it already doesn't exist
        if not os.path.exists(self.model_save_dir):
            os.makedirs(self.model_save_dir)
        
        for epoch in range(self.n_epochs):

            print(f'Epoch: {epoch+1:02}')

            start_time = time.time()

            train_loss = self.train(self.train_loader)
            valid_loss = self.evaluate(self.test_loader)

            epoch_mins, epoch_secs = self.epoch_time(start_time, time.time())

            c+=1
            if valid_loss < best_valid_loss:
                best_valid_loss = valid_loss
                torch.save(self.model.state_dict(), os.path.join(self.model_save_dir, '{}_trained.pt'.format(self.model_name)))
                c=0

            if c>4:
                #decrease lr if loss does not decrease after 5 steps
                self.scheduler.step()
                c=0

            print(f'Time: {epoch_mins}m {epoch_secs}s') 
            print(f'Train Loss: {train_loss:.3f}')
            print(f'Val   Loss: {valid_loss:.3f}')
            print('-'*60)
        print('The best validation loss is', best_valid_loss)
        print('*'*100)

        
    def evaluate_model(self, iterator=None, proba=False, one_batch=False):
        
        if iterator is None:
            iterator = self.test_loader
    
        self.model.eval()

        images = []
        true_labels = []
        pred_labels = []
        pred_probs = []

        with torch.no_grad():    
            for i, batch in enumerate(iterator):    

                inputs, labels = batch[0].to(device), batch[1].to(device)

                outputs = self.model(inputs)

                y_prob = F.softmax(outputs, dim = -1)

                top_preds = y_prob.argmax(1, keepdim = True)

                images.append(inputs.to(device))
                true_labels.append(labels.to(device))
                pred_labels.append(top_preds.to(device))
                pred_probs.append(y_prob.to(device))

                if(one_batch):
                    break

        images = torch.cat(images, dim=0)
        true_labels = torch.cat(true_labels, dim=0)
        pred_labels = torch.cat(pred_labels, dim=0)
        pred_probs = torch.cat(pred_probs, dim=0)

        if(proba):
            return images, true_labels, pred_labels, pred_probs

        return images, true_labels, pred_labels
    
    def visualize_test_samples(self):
    
        images, true_labels, pred_labels, pred_probs = self.evaluate_model(proba=True, one_batch=True)

        true_labels = true_labels.cpu().numpy()
        pred_labels = pred_labels.cpu().numpy()
        pred_probs = pred_probs.cpu().numpy()


        rows = int(np.sqrt(len(images)))
        cols = int(np.sqrt(len(images)))

        fig = plt.figure(figsize = (25, 20))

        for i in range(rows*cols):

            ax = fig.add_subplot(rows, cols, i+1)

            image, true_label, pred_label, pred_prob = images[i], true_labels[i], pred_labels[i], pred_probs[i]
            image = image.permute(1, 2, 0)
            ax.imshow(image.cpu().numpy())
            ax.set_title(f'true label: {self.class_map[true_label]}\n' \
                        f'pred label: {self.class_map[pred_label[0]]} (Prob: {max(pred_prob):.3f})',
                        color = ('green' if true_label==pred_label[0] else 'red'))
            ax.axis('off')

        fig.subplots_adjust(hspace = 0.4)

        plt.show()
        
    def start_training(self):
        """
        To start trainig, evaluate the trained model and print the metrics
        """
        self.training()
        
        images, true_labels, pred_labels, pred_probs = self.evaluate_model(proba=True)
        
        metrics = Metrics(images, true_labels, pred_labels, pred_probs, self.classes)

        cm = metrics.get_confusion_matrix()
        print('The confusion matrix is:\n', cm)
        print('*'*100)
        
        cr = metrics.get_classification_report()
        print('The classification report is:\n', cr)
        print('*'*100)
        
# ### The Trained Model Class
        
class TrainedModel():
    
    def __init__(self, model_name='vgg16'):
        """
        To get the details of the pre-trained model
        """
        trainer = Trainer(model_name=model_name)
        self.model = trainer.model
        self.model_save_dir = trainer.model_save_dir
        self.model_name = model_name
        
    def loadModel(self):
        """
        To load the pre trained model
        """
        self.model.load_state_dict(torch.load(os.path.join(self.model_save_dir, '{}_trained.pt'.format(self.model_name)), map_location=torch.device(device)))
        return self.model
    
    def countParameters(self):
        """
        To get the number of trainable parameters of the model
        """
        return sum(p.numel() for p in self.model.parameters() if p.requires_grad)
    
    def printModel(self):
        """
        To print the network architecture
        """
        print(self.model)

# Start the training
trainer = Trainer()
print('*'*100)
print('Training {} model with parameters: {}'.format(MODEL_NAME, args))
print('*'*100)
trainer.start_training()

# End of script