# =========================================
# SCRIPT NAME: models.py
# PURPOSE:     Defines the neural network architecture  
#              the training script uses (train_policy.py)
# AUTHOR:      Keone Leao
# DATE:        04/21/26
# DEPENDENCIES:torch, torch.nn
# =========================================

## Imports
import torch
import torch.nn as nn


# control policy
class MLPPolicy(nn.Module):
    def __init__(self, state_dim, hidden_dim, action_dim):
        super(MLPPolicy, self).__init__()

        ## define policy
        # fully connected multi-layer perceptron (MLP)
        # three linear layers
        self.pi_1 = nn.Linear(state_dim, hidden_dim)
        self.pi_2 = nn.Linear(hidden_dim, hidden_dim)
        self.pi_3 = nn.Linear(hidden_dim, action_dim)

        ## helper functions
        # relu activation function
        self.relu = nn.ReLU()

        self.dropout = nn.Dropout(p=0.2) # prevents the model from memorizing exact states        # loss function (The loss function isn't ever used)
        # self.mse_loss = nn.CrossEntropyLoss()(this belongs in training script) # apparently this is better for learning small differences 
                                              # (ex: no ambiguity between 0.2 & 0.3)
        # self.mse_loss = nn.MSELoss() # this is used when numbers have physical meaning instead of choices (like they are used here)

    ## execute robot policy
    # input state output action
    def forward(self, state):
        x = self.relu(self.pi_1(state))
        x = self.dropout(x)
        x = self.relu(self.pi_2(x))
        x = self.dropout(x)
        return self.pi_3(x)  # logits, discrete decisions (NO tanh)
        # return 0.1 * torch.tanh(self.pi_3(x)) # this is a continous output between -0.1 & +0.1