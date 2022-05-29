import torch
from torch import nn

class ELM(nn.Module):
    
    def __init__(self, input_size, hidden_size, activation, device):
        
        super().__init__()
        
        if activation == 'tanh':      self.act = torch.tanh
        elif activation == 'sigmoid': self.act = torch.sigmoid
        elif activation == 'relu':    self.act = torch.relu
            
        # Initialise weights and biases for the hidden layer
        self.W = torch.empty(size = (input_size, hidden_size), device = device)
        self.b = torch.empty(size = (1, hidden_size), device = device)
        
        nn.init.uniform_(self.W, a = -1.0, b = 1.0)
        nn.init.uniform_(self.b, a = -1.0, b = 1.0)
    
    
    def fit(self, X, y):
        
        # Hidden layer nodes
        H = self.act(torch.mm(X, self.W) + self.b)
        
        # Moore-penrose pseudoinverse
        H = torch.pinverse(H)
        
        # Output weights
        self.betas = torch.mm(H, y)
        
        return
        
    def predict(self, X):
        
        # Hidden layer nodes
        H = self.act(torch.mm(X, self.W) + self.b)
        y = torch.mm(H, self.betas)
        
        return y
    