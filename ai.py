# AI for Doom



# Importing the libraries
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable

# Importing the packages for OpenAI and Doom
import gym
from gym.wrappers import SkipWrapper
from ppaquette_gym_doom.wrappers.action_space import ToDiscrete

# Importing the other Python files
import experience_replay, image_preprocessing



# Part 1 - Building the AI

# Making the brain

class CNN(nn.Module):
    
    def __init__(self, number_actions):
        super(CNN, self).__init__() #inherit from parent function
        self.convolution1 = nn.Conv2d(in_channels = 1, out_channels = 32, kernel_size = 5) #convolutional layer 1
        self.convolution2 = nn.Conv2d(in_channels = 32, out_channels = 32, kernel_size = 3) #convolutional layer 2
        self.convolution3 = nn.Conv2d(in_channels = 32, out_channels = 64, kernel_size = 2) #convolutional layer 3
        self.fc1 = nn.Linear(in_features = self.count_neurons((1, 80, 80)), out_features = 40) #full connections between vector and hidden layer
        self.fc2 = nn.Linear(in_features = 40, out_features = number_actions) #full connections between hidden layer and output layer

    def count_neurons(self, image_dim): #arguments of input images: reduce images by 1 x 80 x 80
        x = Variable(torch.rand(1, *image_dim)) # set random pxiels 
        x = F.relu(F.max_pool2d(self.convolution1(x), 3, 2)) #propagate image into convolutional layer 1
        x = F.relu(F.max_pool2d(self.convolution2(x), 3, 2)) #propagate image into convolutional layer 2
        x = F.relu(F.max_pool2d(self.convolution3(x), 3, 2)) #propagate image into convolutional layer 3
        return x.data.view(1, -1).size(1) #flattening layer: flatten pixels

    def forward(self, x): #x - input images and will be updated as it is propagated into nn
        x = F.relu(F.max_pool2d(self.convolution1(x), 3, 2)) #propagation of signal in convolutional layer1
        x = F.relu(F.max_pool2d(self.convolution2(x), 3, 2)) #propagation of signal in convolutional layer2
        x = F.relu(F.max_pool2d(self.convolution3(x), 3, 2)) #propagation of signal in convolutional layer3
        x = x.view(x.size(0), -1) #flatten several channels in convo layer 
        x = F.relu(self.fc1(x)) #breaking linearity
        x = self.fc2(x) #apply to neural of the hidden layer (x)
        return x #return output neuron


