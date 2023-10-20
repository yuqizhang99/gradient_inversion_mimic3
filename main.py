"""Run reconstruction in a terminal prompt.
Optional arguments can be found in inversefed/options.py

This CLI can recover the baseline experiments.
"""

import torch
import torchvision
import torch.nn as nn
from torch.utils.data import Dataset
from torch.nn import BCELoss

import numpy as np

from collections import defaultdict
import datetime
import time
import os

import matplotlib.pyplot as plt

from readers import InHospitalMortalityReader
from preprocessing import Discretizer, Normalizer
import utils
from model import LSTM
import gradient_inversion as gi

#Model Configuration
DIM =  16 
DEPTH = 2
DROPOUT = 0.3
#Inversion Configuration
num_exp=1
num_images=4
restarts=3

# detect GPU
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

class MortalityDataset(Dataset):
    def __init__(self, dataframe):
        self.features, self.labels = dataframe
    
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        features = torch.tensor(self.features[idx,:,:], dtype=torch.float32)
        label = torch.tensor(self.labels[idx], dtype=torch.int64)
        
        return features, label
    
def get_dataloader():
    train_reader = InHospitalMortalityReader(dataset_dir=os.path.join("in-hospital-mortality", 'train'),
                                         listfile=os.path.join("in-hospital-mortality", 'train_listfile.csv'),
                                         period_length=48.0)

    val_reader = InHospitalMortalityReader(dataset_dir=os.path.join("in-hospital-mortality", 'train'),
                                        listfile=os.path.join("in-hospital-mortality", 'val_listfile.csv'),
                                        period_length=48.0)

    discretizer = Discretizer(timestep=float(1.0),
                            store_masks=True,
                            impute_strategy='previous',
                            start_time='zero')
    
    discretizer_header = discretizer.transform(train_reader.read_example(0)["X"])[1].split(',')
    cont_channels = [i for (i, x) in enumerate(discretizer_header) if x.find("->") == -1]

    normalizer = Normalizer(fields=cont_channels)  # choose here which columns to standardize
    
    normalizer_state = None
    if normalizer_state is None:
        normalizer_state = 'ihm_ts{}.input_str-{}.start_time-zero.normalizer'.format(1.0, "previous")
        normalizer_state = os.path.join(os.path.dirname(__file__), normalizer_state)
    normalizer.load_params(normalizer_state)
    
    train_raw = utils.load_data(train_reader, discretizer, normalizer, False)
    val_raw = utils.load_data(val_reader, discretizer, normalizer, False)

    train_dataset = MortalityDataset(train_raw)
    
    val_dataset = MortalityDataset(val_raw)
    
    trainloader = torch.utils.data.DataLoader(train_dataset, drop_last=True)
    validloader = torch.utils.data.DataLoader(val_dataset, drop_last=True)
    return trainloader, validloader

def train_model(criterion, optimizer, model, train_loader, val_loader, num_epochs=10):

    # Training loop
    for epoch in range(10):  # Adjust the number of epochs
        model.train()
        total_loss = 0.0

        for batch in train_loader:
            inputs, labels = batch
            inputs = inputs.to(DEVICE)
            labels = labels.to(DEVICE)
            optimizer.zero_grad()
            
            outputs = model(inputs)
            loss = criterion(outputs.view(-1), labels.float())
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        print(f"Epoch {epoch+1}, Loss: {total_loss/len(train_loader)}")

        model.eval()
        total_val_loss = 0.0
        with torch.no_grad():
            for batch in val_loader:
                inputs, labels = batch
                inputs = inputs.to(DEVICE)
                labels = labels.to(DEVICE)
                outputs = model(inputs)
                val_loss = criterion(outputs.view(-1).float(), labels.float())
                total_val_loss += val_loss.item()

        print(f"Validation Loss after Epoch {epoch+1}: {total_val_loss/len(val_loader)}")

if __name__ == "__main__":
    # Choose GPU device and print status information:
    # setup = inversefed.utils.system_startup(args)
    start_time = time.time()

    # Prepare for training

    # Get data:
    trainloader, validloader = get_dataloader()

    criterion = BCELoss()
    # if find model.pt, load it
    if os.path.isfile('model.pt'):
        model = LSTM(DIM,DROPOUT,depth=DEPTH)
        model.load_state_dict(torch.load('model.pt'))
    else:
        model = LSTM(DIM,DROPOUT,depth=DEPTH)
        model.to(DEVICE)
        #train model
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001, betas=(0.9, 0.999))  # Adjust learning rate and betas

        train_model(criterion, optimizer, model, trainloader, validloader, num_epochs=10)
        torch.save(model.state_dict(), 'model.pt')
    
    model.to(DEVICE)
    model.eval()
    
    #gradient inversion
    for i in range(num_exp):
        # here we just try to reconstruct one data point
        ground_truth, labels = validloader.dataset[0] # get the first one (48,76) (1)
        
        ground_truth = ground_truth.unsqueeze(0).to(DEVICE) # (1,48,76)
        labels = labels.view(-1).to(DEVICE)
        
        target_loss = criterion(model(ground_truth).view(-1), labels.float())
        input_gradient = torch.autograd.grad(target_loss, model.parameters())
        input_gradient = [grad.detach() for grad in input_gradient]

        rec_machine = gi.GradientReconstructor(model)

        output, stats = rec_machine.reconstruct(input_gradient, labels.reshape(-1), data_shape=(48,76))

        # Compute stats and save to a table:

        feat_mse = (model(output) - model(ground_truth)).pow(2).mean().item()
        print(f"Rec. loss: {stats['opt']:2.4f} | FMSE: {feat_mse:2.4e} |")

        plt.subplot(1, 2, 1)
        plt.imshow(ground_truth[0], cmap='viridis', aspect='auto')
        plt.colorbar()
        plt.title("Ground Truth")

        # Plot predictions
        plt.subplot(1, 2, 2)
        plt.imshow(output[0], cmap='viridis', aspect='auto')
        plt.colorbar()
        plt.title("Predicted")

        # Show the plots
        plt.show()
        
        #draw heatmap to compare ground truth and output
        

    # Print final timestamp
    print(datetime.datetime.now().strftime("%A, %d. %B %Y %I:%M%p"))
    print('---------------------------------------------------')
    print(f'Finished computations with time: {str(datetime.timedelta(seconds=time.time() - start_time))}')
    print('-------------Job finished.-------------------------')
