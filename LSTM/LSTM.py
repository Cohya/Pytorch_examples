import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim import Adam


class MyModel(nn.Module):
  def __init__(self):
    super(MyModel, self).__init__()
    self.layer_dims = 1
    self.hidden_dim = 256
    self.input_features = 6
    self.lstm0 = nn.LSTM(self.input_features,self.hidden_dim,self.layer_dims)
    self.linear = nn.Linear(in_features = 256, out_features = 4)
    
  def forward(self,x):
    h0 = torch.zeros(self.layer_dims, x.shape(0), self.hidden_dim)#.requires_grad_()
    # Initialize cell state
    c0 = torch.zeros(self.layer_dims,x.shape(0), self.hidden_dim)#.requires_grad_()
    lstm_output, (hn, cn) = self.lstm0(x, (h0,c0))
    print(lstm_output.shape)
    lstm_output = F.relu(lstm_output)
    output = self.linear(lstm_output)
    output = F.log_softmax(output, dim=1)
    return output