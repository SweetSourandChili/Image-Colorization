import torch
import os
import matplotlib.pyplot as plt
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision.transforms as transforms
import hw3utils

batch_size = 16
max_num_epoch = 10
hps = {'lr': 0.001}

# ---- options ----
DEVICE_ID = 'cpu'  # set to 'cpu' for cpu, 'cuda' / 'cuda:0' or similar for gpu.
LOG_DIR = 'checkpoints'
VISUALIZE = False  # set True to visualize input, prediction and the output from the last batch
LOAD_CHKPT = False

torch.multiprocessing.set_start_method('spawn', force=True)
if torch.backends.mps.is_available():
    DEVICE_ID = "mps"

# ---- utility functions -----


def get_loaders(batch_size, device):
    data_root = 'ceng483-f22-hw3-dataset'
    train_set = hw3utils.HW3ImageFolder(root=os.path.join(data_root, 'train'), device=device)
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=0)
    val_set = hw3utils.HW3ImageFolder(root=os.path.join(data_root, 'val'), device=device)
    val_loader = torch.utils.data.DataLoader(val_set, batch_size=batch_size, shuffle=False, num_workers=0)
    # Note: you may later add test_loader to here.
    return train_loader, val_loader


# ---- ConvNet -----
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 8, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(8, 8, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(8, 8, kernel_size=3, padding=1)

        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.fc1 = nn.Linear(10 * 10 * 8, 4096)
        self.fc2 = nn.Linear(4096, 4096)
        self.fc3 = nn.Linear(4096, 3 * 80 * 80)

    def forward(self, grayscale_image):
        # apply your network's layers in the following lines:
        x = self.pool(F.relu(self.conv1(grayscale_image)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        x = x.view(-1, 10 * 10 * 8)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        x = x.view(-1, 3, 80, 80)
        return x


# ---- training code -----
device = torch.device(DEVICE_ID)
print('device: ' + str(device))
net = Net().to(device=device)
criterion = nn.MSELoss()
optimizer = optim.SGD(net.parameters(), lr=hps['lr'])
train_loader, val_loader = get_loaders(batch_size, device)

if LOAD_CHKPT:
    print('loading the model from the checkpoint')
    model.load_state_dict(os.path.join(LOG_DIR, 'checkpoint.pt'))

print('training begins')
for epoch in range(max_num_epoch):
    running_loss = 0.0  # training loss of the network
    for iteri, data in enumerate(train_loader, 0):
        inputs, targets = data  # inputs: low-resolution images, targets: high-resolution images.

        optimizer.zero_grad()  # zero the parameter gradients

        # do forward, backward, SGD step
        preds = net(inputs)
        loss = criterion(preds, targets)
        loss.backward()
        optimizer.step()

        # print loss
        running_loss += loss.item()
        print_n = 100  # feel free to change this constant
        if iteri % print_n == (print_n - 1):  # print every print_n mini-batches
            print('[%d, %5d] network-loss: %.3f' %
                  (epoch + 1, iteri + 1, running_loss / 100))
            running_loss = 0.0
            # note: you most probably want to track the progress on the validation set as well (needs to be implemented)

        if (iteri == 0) and VISUALIZE:
            hw3utils.visualize_batch(inputs, preds, targets)

    print('Saving the model, end of epoch %d' % (epoch + 1))
    if not os.path.exists(LOG_DIR):
        os.makedirs(LOG_DIR)
    torch.save(net.state_dict(), os.path.join(LOG_DIR, 'checkpoint.pt'))
    hw3utils.visualize_batch(inputs, preds, targets, os.path.join(LOG_DIR, 'example.png'))

print('Finished Training')
