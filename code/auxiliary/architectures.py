import torch
import torch.nn as nn
  
                 
class CentroidPredictionLSTM(nn.Module):
    """Long-short-term-memory network for supervised prediction of centroid positions. 

    Args:
        input_size (int): number of features at each time step
        hidden_size (int): number of features of LSTM hidden state
        num_layers (int): number of LSTM hidden layers
        batch_size (int): number of data patterns to be fed to network simultaneously
        seq_length_in (int): length of input window
        seq_length_out (int): length of predicted window
        dropout (float): probability of dropout [0,1] in dropout layer  
        bi (bool, optional): if True, becomes a bidirectional LSTM
        gpu_usage (bool, optional): whether to use GPU or not
        device (string, optional): which GPU device to use
    """
    def __init__(self, input_size, hidden_size, num_layers, 
                 seq_length_in, seq_length_out, 
                 dropout=0, bi=False, gpu_usage=True, device=None):
        super(CentroidPredictionLSTM, self).__init__()
        
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.seq_length_in = seq_length_in
        self.seq_length_out = seq_length_out
        self.bi = bi
        
        self.gpu_usage = gpu_usage
        if self.gpu_usage:
            self.device = device
        
        # construct lstm 
        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size,
                            num_layers=num_layers, dropout=dropout, 
                            bidirectional=self.bi, batch_first=True)
        
        # construct fully-connected layer
        if self.bi is False:
            self.fc = nn.Linear(in_features=hidden_size, out_features=seq_length_out)
        if self.bi:
            self.fc = nn.Linear(in_features=hidden_size * 2, out_features=seq_length_out)        
        
    def reset_h_c_states(self, batch_size=1):
        "Reset the hidden state and the cell state of the LSTM."
        
        # tensors containing the initial hidden state and initial cell state
        # with shape (num_layers * num_directions, batch_size, hidden_size)
        if self.bi is False:
            self.h_c = (torch.zeros(self.num_layers, batch_size, self.hidden_size),
                        torch.zeros(self.num_layers, batch_size, self.hidden_size))
        if self.bi:
            self.h_c = (torch.zeros(self.num_layers * 2, batch_size, self.hidden_size),
                        torch.zeros(self.num_layers * 2, batch_size, self.hidden_size))           
            

        # move states to GPU
        if self.gpu_usage:
            self.h_c = (self.h_c[0].to(self.device), self.h_c[1].to(self.device))
            

    def forward(self, input_batch):
        """Compute forward pass through network.

        Args:
            input_batch (array): input array with shape (batch_size, seq_length_in) 
                                    containing current batch of data        

        Returns:
            predictions (array): output array with shape (batch_size, seq_length_out) 
                                    containing predicted time sequence
        """
        # get batch size from current input batch
        batch_size = input_batch.shape[0]
        
        # reset hidden state and cell state for current batch of data (--> stateless LSTM)
        self.reset_h_c_states(batch_size=batch_size)
 
        # print(f'Shape of input_batch: {input_batch.shape} ')  
        # -> torch.Size([batch_size, seq_length_in]) 
               
        # propagate input of shape=(batch, seq_len, input_size) through LSTM
        lstm_out, self.h_c = self.lstm(input_batch.view(batch_size, self.seq_length_in, -1), 
                                       self.h_c)
        
        # print(f'Shape of lstm_out: {lstm_out.shape} ')  
        # -> torch.Size([batch_size, seq_length_in, hidden_size]) 

        # only take the output from the final timetep of LSTM
        # (can pass on the entirety of lstm_out to the next layer if it is a seq2seq prediction)
        # and reshape to torch.Size([batch_size, num_directions * hidden_size])
        predictions = self.fc(lstm_out[:, -1, :].view(batch_size, -1))
        
        # print(f'Shape of predictions: {predictions.shape} ')   
        # -> torch.Size([batch_size, seq_lenght_out])
        
        # return output windows of batch
        return predictions

