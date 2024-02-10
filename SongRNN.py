import torch
import torch.nn as nn
from torch.autograd import Variable
from constants import *


class SongRNN(nn.Module):
    def __init__(self, input_size, output_size, config):
        """
        Initialize the SongRNN model.
        """
        super(SongRNN, self).__init__()

        HIDDEN_SIZE = config["hidden_size"]
        NUM_LAYERS = config["no_layers"]
        MODEL_TYPE = config["model_type"]
        DROPOUT_P = config["dropout"]

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
        
    def init_hidden(self):
        """
        Initializes the hidden state for the recurrent neural network.

        TODO:
        Check the model type to determine the initialization method:
        (i) If model_type is LSTM, initialize both cell state and hidden state.
        (ii) If model_type is RNN, initialize the hidden state only.

        Initialise with zeros.
        """

        raise NotImplementedError
        
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

        raise NotImplementedError