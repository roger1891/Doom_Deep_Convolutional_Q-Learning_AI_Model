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

# Making the body

class SoftmaxBody(nn.Module):
    
    def __init__(self, T):
        super(SoftmaxBody, self).__init__()
        self.T = T #the higher exploration the less exoploration

    def forward(self, outputs):
        probs = F.softmax(outputs * self.T) #probability of different outputs
        actions = probs.multinomial() #sample the actions according to distribution of probabilities
        return actions #return action

# Making the AI

class AI:

    def __init__(self, brain, body):
        self.brain = brain
        self.body = body

    def __call__(self, inputs):
        input = Variable(torch.from_numpy(np.array(inputs, dtype = np.float32)))
        output = self.brain(input)
        actions = self.body(output)
        return actions.data.numpy()



# Part 2 - Training the AI with Deep Convolutional Q-Learning

# Getting the Doom environment
doom_env = image_preprocessing.PreprocessImage(SkipWrapper(4)(ToDiscrete("minimal")(gym.make("ppaquette/DoomCorridor-v0"))), width = 80, height = 80, grayscale = True)
doom_env = gym.wrappers.Monitor(doom_env, "videos", force = True)
number_actions = doom_env.action_space.n

# Building an AI
cnn = CNN(number_actions)
softmax_body = SoftmaxBody(T = 1.0)
ai = AI(brain = cnn, body = softmax_body)

# Setting up Experience Replay
n_steps = experience_replay.NStepProgress(env = doom_env, ai = ai, n_step = 10) #learning every 10 steps
memory = experience_replay.ReplayMemory(n_steps = n_steps, capacity = 10000) #use last 10000 steps
    
# Implementing Eligibility Trace
def eligibility_trace(batch):
    gamma = 0.99
    inputs = []
    targets = []
    for series in batch:
        input = Variable(torch.from_numpy(np.array([series[0].state, series[-1].state], dtype = np.float32)))
        output = cnn(input)
        cumul_reward = 0.0 if series[-1].done else output[1].data.max() #if last transition of series is done
        for step in reversed(series[:-1]):
            cumul_reward = step.reward + gamma * cumul_reward
        state = series[0].state
        target = output[0].data #Q-value
        target[series[0].action] = cumul_reward
        inputs.append(state)
        targets.append(target)
    return torch.from_numpy(np.array(inputs, dtype = np.float32)), torch.stack(targets)

# Making the moving average on 100 steps
class MA:
    def __init__(self, size): #size of list rewards
        self.list_of_rewards = []
        self.size = size
    def add(self, rewards): #add cumulative reward (10 steps) and not simple reward
        if isinstance(rewards, list): #if rewards are into a list
            self.list_of_rewards += rewards #add rewards to list
        else:
            self.list_of_rewards.append(rewards) #append rewards to list
        while len(self.list_of_rewards) > self.size: #if more than 100 elements...
            del self.list_of_rewards[0] #delete 1st element in list of rewards
    def average(self): 
        return np.mean(self.list_of_rewards) #get average and return
ma = MA(100) #moving average object and 100 because we want it for 100 steps

# Training the AI
loss = nn.MSELoss()
optimizer = optim.Adam(cnn.parameters(), lr = 0.001) #connection between optimizer and brain
nb_epochs = 100
for epoch in range(1, nb_epochs + 1):
    memory.run_steps(200) #200 runs at 10 steps
    for batch in memory.sample_batch(128):
        inputs, targets = eligibility_trace(batch)
        inputs, targets = Variable(inputs), Variable(targets)
        predictions = cnn(inputs)
        loss_error = loss(predictions, targets)
        optimizer.zero_grad()
        loss_error.backward()
        optimizer.step()
    rewards_steps = n_steps.rewards_steps()
    ma.add(rewards_steps)
    avg_reward = ma.average()
    print("Epoch: %s, Average Reward: %s" % (str(epoch), str(avg_reward)))
    if avg_reward >= 1500:#if AI reaches 1500 we can be sure that the ai reaches the goal and is good to go
        print("Congratulations, your AI wins")
        break

# Closing the Doom environment
doom_env.close()
