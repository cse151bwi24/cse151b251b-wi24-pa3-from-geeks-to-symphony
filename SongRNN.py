import torch
import torch.nn as nn
from torch.autograd import Variable
from constants import *


class SongRNN(nn.Module):
    def __init__(self, input_size, output_size, config, device):
        """
        Initialize the SongRNN model.
        """
        super(SongRNN, self).__init__()

        HIDDEN_SIZE = config["hidden_size"]
        NUM_LAYERS = config["no_layers"]
        MODEL_TYPE = config["model_type"]
        DROPOUT_P = config["dropout"]
        self.device = device
        self.h_n = torch.zeros(1, HIDDEN_SIZE, device = "cuda")
        #print('h_n device_init', self.h_n.device)
        self.c_n = torch.zeros(1, HIDDEN_SIZE, device = "cuda")
        
        self.model_type = MODEL_TYPE
        self.input_size = input_size
        self.hidden_size = HIDDEN_SIZE
        self.output_size = output_size
        self.num_layers = NUM_LAYERS
        self.dropout = DROPOUT_P
        
        """
        Complete the code

        TODO:
        (i) Initialize embedding layer with input_size and hidden_size
        (ii) Initialize the recurrent layer based on model type (i.e., LSTM or RNN) using hidden size and num_layers
        (iii) Initialize linear output layer using hidden size and output size
        (iv) Initialize dropout layer with dropout probability
        """
        self.encoder = nn.Embedding(self.input_size, self.hidden_size)
        if(self.model_type == 'lstm'):
            self.recurrentLayer = nn.LSTM(self.hidden_size, self.hidden_size, num_layers = self.num_layers)
        else:
            self.recurrentLayer = nn.RNN(self.hidden_size, self.hidden_size, num_layers = self.num_layers)
        self.decoder = nn.Linear(self.hidden_size, self.output_size)
        self.DROPOUT = nn.Dropout(p=self.dropout)
        
        
    def init_hidden(self):
        """
        Initializes the hidden state for the recurrent neural network.

        TODO:
        Check the model type to determine the initialization method:
        (i) If model_type is LSTM, initialize both cell state and hidden state.
        (ii) If model_type is RNN, initialize the hidden state only.

        Initialise with zeros.
        """
        self.h_n = torch.zeros(1, self.hidden_size, device = self.device)
        self.c_n = torch.zeros(1, self.hidden_size, device = self.device)
        
    def forward(self, seq):
        """
        Forward pass of the SongRNN model.
        (Hint: In teacher forcing, for each run of the model, input will be a single character
        and output will be pred-probability vector for the next character.)

        Parameters:
        - seq (Tensor): Input sequence tensor of shape (seq_length)

        Returns:
        - output (Tensor): Output tensor of shape (output_size)
        - activations (Tensor): Hidden layer activations to plot heatmap values


        TODOs:
        (i) Embed the input sequence
        (ii) Forward pass through the recurrent layer
        (iii) Apply dropout (if needed)
        (iv) Pass through the linear output layer
        """
        seq = seq.to(self.device)
        e1 = self.encoder(seq)
        e1 = torch.unsqueeze(e1, 0)

        if(self.model_type == 'lstm'):
            h_out, hc = self.recurrentLayer(e1, (self.h_n, self.c_n))
            self.h_n = hc[0]
            self.c_n = hc[1]
        else:
            h_out, self.h_n = self.recurrentLayer(e1, self.h_n)
        dr1 = self.DROPOUT(h_out)
        de1 = self.decoder(dr1)

#         prob = torch.nn.functional.softmax(de1, dim=1)
#         output = torch.squeeze(torch.multinomial(prob, 1))
        
        
#         return torch.squeeze(prob), self.h_n
        return torch.squeeze(de1), self.h_n, torch.Tensor(dr1)