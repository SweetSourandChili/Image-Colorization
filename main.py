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
num_kernels = (2, 4, 8)
learning_rates = (0.0001, 0.001, 0.1)
conv_layers = (1, 2, 4)

# ---- options ----
DEVICE_ID = 'cpu'  # set to 'cpu' for cpu, 'cuda' / 'cuda:0' or similar for gpu.
LOG_DIR = 'checkpoints'
VISUALIZE = False  # set True to visualize input, prediction and the output from the last batch
LOAD_CHKPT = False
SAVE_MODEL = False

torch.multiprocessing.set_start_method('spawn', force=True)
if torch.backends.mps.is_available():
    DEVICE_ID = "mps"


# ---- utility functions -----


def write_output(filename, string_to_write):
    f = open(os.path.join("outputs", filename), "w")
    f.write(string_to_write)
    f.close()


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
    def __init__(self, conv_layers, num_kernel):
        super(Net, self).__init__()
        if conv_layers <= 1:
            num_kernel = 1
        self.conv_layers = conv_layers
        self.conv1 = nn.Conv2d(1, num_kernel, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(num_kernel, num_kernel, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(num_kernel, num_kernel, kernel_size=3, padding=1)
        self.convs = (self.conv1, self.conv2, self.conv3)
        self.conv4 = nn.Conv2d(num_kernel, 3, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(self, x):
        # apply your network's layers in the following lines:
        for i in range(self.conv_layers - 1):
            x = F.relu(self.convs[i](x))
        x = torch.tanh(self.conv4(x))
        x = x.view(-1, 3, 80, 80)
        return x


if LOAD_CHKPT:
    print('loading the model from the checkpoint')
    model.load_state_dict(os.path.join(LOG_DIR, 'checkpoint.pt'))


def train_model(conv_layer, num_kernel, lr):
    string_to_write = ""
    # ---- training code -----
    device = torch.device(DEVICE_ID)
    print('device: ' + str(device))
    net = Net(conv_layer, num_kernel).to(device=device)
    criterion = nn.MSELoss()
    margin_loss = nn.MarginRankingLoss()
    optimizer = optim.SGD(net.parameters(), lr=lr)
    train_loader, val_loader = get_loaders(batch_size, device)

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

        print(epoch+1, end="-")
        if SAVE_MODEL:
            print('Saving the model, end of epoch %d' % (epoch + 1))
            if not os.path.exists(LOG_DIR):
                os.makedirs(LOG_DIR)
            torch.save(net.state_dict(), os.path.join(LOG_DIR, 'checkpoint.pt'))

        hw3utils.visualize_batch(inputs, preds, targets, os.path.join(LOG_DIR, 'example.png'))

        # validation loss
        with torch.no_grad():
            net.eval()
            val_loss = 0.0
            for ind, (val_inputs, val_targets) in enumerate(val_loader):
                val_preds = net.forward(val_inputs)
                val_loss_iter = margin_loss(12, val_preds, val_targets)
                val_loss += val_loss_iter.item()
        string_to_write += 'Epoch = {} | Train Loss = {:.2f}\tVal Loss = {:.2f}\n'.format(epoch + 1, running_loss, val_loss)
        # stop epochs if not improving
        if (last_val_loss - val_loss) / last_val_loss < 0:
            print("The number of epochs of the model is: %d" % (epoch + 1))
            break
        last_val_loss = val_loss

    write_output("Conv_layer:{}\tNum_kernel:{}\tlr:{}".format(conv_layer, num_kernel, lr), string_to_write)
    print('Finished Training')
    return net, epoch, last_val_loss


best_conf = {"conv_layer": 1, "num_kernel": 8, "learning_rate": 0.1, "best": ""}
"""
least_loss = float("inf")
for conv_layer in conv_layers:
    for num_kernel in num_kernels:
        for learning_rate in learning_rates:
            print("Configuration:")
            configuration = "Convolution Layers: {}\tNumber of Kernels: {}\t Learning Rate: {}".format(conv_layer,
                                                                                                       num_kernel,
                                                                                                       learning_rate)
            print(configuration)
            model, epoch, last_val = train_model(conv_layer, num_kernel, learning_rate)
            print("Epochs stopped at: {}\t Validation Loss: {}".format(epoch, last_val))
            if least_loss > last_val:
                least_loss = last_val
                best_conf["conv_layer"] = conv_layer
                best_conf["num_kernel"] = num_kernel
                best_conf["learning_rate"] = learning_rate
                best_conf["best"] = configuration
"""
print("The best configuration:\n" + best_conf["best"])
print("Train model with the best configuration...")
model, _, _ = train_model(best_conf["conv_layer"], best_conf["num_kernel"], best_conf["learning_rate"])
