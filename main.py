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
max_num_epoch = 100
hps = {'lr': 0.001}
num_kernels = 8

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
        self.conv1 = nn.Conv2d(1, num_kernels, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(num_kernels, num_kernels, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(num_kernels, num_kernels, kernel_size=3, padding=1)
        self.conv4 = nn.Conv2d(num_kernels, 3, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(self, grayscale_image):
        # apply your network's layers in the following lines:
        x = F.relu(self.conv1(grayscale_image))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = self.conv4(x)
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

last_val_loss = float("inf")
print('training begins')
for epoch in range(max_num_epoch):
    running_loss = 0.0  # training loss of the network
    for iteri, data in enumerate(train_loader, 0):
        inputs, targets = data  # inputs: low-resolution images, targets: high-resolution images.

        net.train()
        optimizer.zero_grad()  # zero the parameter gradients

        # do forward, backward, SGD step
        preds = net(inputs)
        loss = criterion(preds, targets)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()

        if (iteri == 0) and VISUALIZE:
            hw3utils.visualize_batch(inputs, preds, targets)

    print('Saving the model, end of epoch %d' % (epoch + 1))
    if not os.path.exists(LOG_DIR):
        os.makedirs(LOG_DIR)
    torch.save(net.state_dict(), os.path.join(LOG_DIR, 'checkpoint.pt'))
    hw3utils.visualize_batch(inputs, preds, targets, os.path.join(LOG_DIR, 'example.png'))

    # validation loss
    with torch.no_grad():
        val_loss = 0.0
        for ind, (val_inputs, val_targets) in enumerate(val_loader):
            val_preds = net.forward(val_inputs)
            val_loss_iter = criterion(val_preds, val_targets)
            val_loss += val_loss_iter.item()
    print('Epoch = {} | Train Loss = {:.2f}\tVal Loss = {:.2f}'.format(epoch+1, running_loss, val_loss))
    # stop epochs if not improving
    print(last_val_loss-val_loss)
    if (last_val_loss - val_loss) < 0.1:
        print("The number of epochs of the model is: %d" % (epoch + 1))
        break
    last_val_loss = val_loss
    val_loss = 0.0
    running_loss = 0.0

print('Finished Training')
