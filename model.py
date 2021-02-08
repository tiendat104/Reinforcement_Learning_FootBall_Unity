import torch
import torch.nn as nn
import torch.nn.functional as F
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
class QNetwork(nn.Module):
    def __init__(self, state_size, action_size, seed, fc1_units=128, fc2_units=256):
        super(QNetwork, self).__init__()
        self.seed = torch.manual_seed(seed)
        self.relu = nn.LeakyReLU(0.2,inplace = True)
        self.pool = nn.MaxPool2d(kernel_size=2,stride=2)
        fc1 = 128
        fc2 = 256
        fc3 = 256
        fc4 = 128

        self.conv1a = nn.Conv2d(3,32, kernel_size=5,stride=1,padding=1)   
        self.conv1b = nn.Conv2d(32,64,kernel_size=5,stride=1, padding = 1)

        self.conv2a = nn.Conv2d(64,64, kernel_size=3,stride=1,padding=1)   
        self.conv2b = nn.Conv2d(64,128,kernel_size=3,stride=1, padding = 1)
        self.conv3 = nn.Conv2d(128,128,kernel_size=2,stride=2, padding = 0)





        self.fc1 = nn.Linear(state_size, fc1)
        self.bn1 = nn.BatchNorm1d(fc1)
        self.fc2 = nn.Linear(fc1, fc2)
        self.bn2 = nn.BatchNorm1d(fc2)
        self.fc3 = nn.Linear(fc2, fc3)
        self.bn3 = nn.BatchNorm1d(fc3)
        self.fc4 = nn.Linear(fc3,fc4)
        self.bn4 = nn.BatchNorm1d(fc4)
        self.res = nn.Linear(fc4, action_size)
        self.dropout = nn.Dropout(0.2) 


    def set_random(self):
        torch.nn.init.xavier_uniform_(self.fc1.weight)
        self.fc1.bias.data.fill_(0.01)
        torch.nn.init.xavier_uniform_(self.fc2.weight)
        self.fc2.bias.data.fill_(0.01)
        torch.nn.init.xavier_uniform_(self.fc3.weight)
        self.fc3.bias.data.fill_(0.01)

    def set_params(self, params):
        self.load_state_dict(params)

    def forward(self, state):
        """Build a network that maps state -> action values."""


        x = self.relu(self.conv1a(state))

        x = self.relu(self.conv1b(x))

        x = self.pool(x)

        x = self.relu(self.conv2a(x))

        x = self.relu(self.conv2b(x))

        x = self.relu(self.conv3(x))

        x = x.view(x.size(0), -1)
        #print(x.size())

        x = self.relu(self.bn3(self.fc3(x)))
        x = self.relu(self.bn4(self.fc4(x)))
        
        return self.res(x)




























































































