import numpy as np
import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F

from torch.utils.data import DataLoader, Dataset, random_split, TensorDataset
from torch.nn.utils import rnn as rnn_utils

seed = 0
rv = np.random.RandomState(seed)

def generate_sequences(n=128):
    """
    Generates sequences of points forming squares, either clockwise or counter-clockwise.

    Each sequence consists of points that represent the corners of a square. The sequence can
    either go around the square in a clockwise or counter-clockwise direction. This is determined
    randomly for each sequence. The sequences are also slightly randomized by adding a small
    noise to each point.

    Args:
        n (int): The number of sequences to generate. Default is 128.

    Returns:
        tuple: A tuple containing two elements:
               - A list of arrays, where each array represents a sequence of points (corners of a square).
               - An array indicating the direction of each sequence (0 for counter-clockwise, 1 for clockwise).

    Example:
        >>> sequences, directions = generate_sequences(n=5)
        >>> print(sequences[0])  # Prints first sequence of points
        >>> print(directions[0])  # Prints direction of the first sequence (0 or 1)
    """
    basic_corners = np.array([[-1, -1], [-1, 1], [1, 1], [1, -1]])
    bases = rv.randint(4, size=n)  # Starting corner indices for each sequence.
    directions = np.random.randint(2, size=n)  # Direction (0 for CCW, 1 for CW) for each sequence.

    # Generating the point sequences.
    points = [basic_corners[[(b + i) % 4 for i in range(4)]][::d*2-1] + np.random.randn(4, 2) * 0.1
              for b, d in zip(bases, directions)]

    return points, directions

train_points, train_directions = generate_sequences(n=256)
test_points, test_directions = generate_sequences(n=1024)

train_data = TensorDataset(torch.as_tensor(train_points).float(), 
                           torch.as_tensor(train_directions).view(-1, 1).float())
test_data = TensorDataset(torch.as_tensor(test_points).float(),
                          torch.as_tensor(test_directions).view(-1, 1).float())

train_loader = DataLoader(train_data, batch_size=16, shuffle=True)
test_loader = DataLoader(test_data, batch_size=16)

class SquareModel(nn.Module):
    def __init__(self, n_features, hidden_dim, n_outputs):
        super(SquareModel, self).__init__()
        self.hidden_dim = hidden_dim
        self.n_features = n_features
        self.n_outputs = n_outputs
        self.hidden = None
        # Simple RNN
        self.basic_rnn = nn.RNN(self.n_features, self.hidden_dim, batch_first=True)
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
        return out.view(-1, self.n_outputs)

torch.manual_seed(123)
model = SquareModel(n_features=2, hidden_dim=2, n_outputs=1)
loss = nn.BCEWithLogitsLoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)