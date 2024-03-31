from square_data_generation import generate_sequences
from model_trainer import ModelTrainer
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, random_split, TensorDataset
import torch.optim as optim
import torch.nn.functional as F

train_points, train_directions = generate_sequences(n=256)
test_points, test_directions = generate_sequences(n=1024)
train_data = TensorDataset(torch.as_tensor(train_points).float(), 
                           torch.as_tensor(train_directions).view(-1, 1).float())
test_data = TensorDataset(torch.as_tensor(test_points).float(),
                          torch.as_tensor(test_directions).view(-1, 1).float())
train_loader = DataLoader(train_data, batch_size=64, shuffle=True)
test_loader = DataLoader(test_data, batch_size=64)

class SquareModel(nn.Module):
    def __init__(self, n_features, hidden_dim, n_outputs):
        super(SquareModel, self).__init__()
        self.hidden_dim = hidden_dim
        self.n_features = n_features
        self.n_outputs = n_outputs
        self.hidden = None
        # Simple RNN
        self.basic_rnn = nn.RNN(self.n_features, self.hidden_dim, batch_first=True)
        # self.basic_rnn = nn.GRU(self.n_features, self.hidden_dim, batch_first=True) # Simply change the RNN to GRU will make the testing errors go to 0.
        # Classifier to produce as many logits as outputs
        self.classifier = nn.Linear(self.hidden_dim, self.n_outputs)
                
    def forward(self, X):
        # X is batch first (N, L, F)
        # output is (N, L, H)
        # final hidden state is (1, N, H)
        batch_first_output, self.hidden = self.basic_rnn(X)
        
        # only last item in sequence (N, 1, H)
        last_output = batch_first_output[:, -1]
        # classifier will output (N, 1, n_outputs)
        out = self.classifier(last_output)
        # final output is (N, n_outputs)
        out = F.sigmoid(out)
        return out


model = SquareModel(n_features=2, hidden_dim=2, n_outputs=1)
loss_fn = nn.BCEWithLogitsLoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)

simple_rnn = ModelTrainer(model, loss_fn, optimizer)
simple_rnn.set_loaders(train_loader, test_loader)
simple_rnn.train(100)
simple_rnn.plot_losses()

ModelTrainer.loader_apply(test_loader,simple_rnn.correct)
# tensor([[509, 512],
#        [488, 512]])
# (509+488)/(512+512) = 97.4% correction