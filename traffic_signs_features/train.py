import torch 
import torch.nn as nn
import time
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from pathlib import Path

# TODO:
# - add per class accuracy
# - continuation 
# - better prints when training
# - saving checkpoints

class TrainingExperiment:
    def __init__(self, net, criterion, train_data, val_data=None):
        '''
        args:
            - criterion: loss fcn
            - train_data: training dataset dataloader
            - val_data: validation set dataloader:
        '''
        self.net = net
        self.train_data = train_data
        self.val_data = val_data
        self.criterion = criterion
        # stats
        self.epoch_loss = []
        self.epoch_acc = []
        self.epoch_count = 0

    def plot_loss(self):
        plt.plot(np.concatenate(self.epoch_loss)) # concatenate different runs over the experiment
        plt.xlabel('epoch')
        plt.title('loss')
        plt.legend(['train', 'validation'])
        plt.show()

    def plot_acc(self):
        plt.plot(np.concatenate(self.epoch_acc)) # concatenate different runs over the experiment
        plt.xlabel('epoch')
        plt.title('accuracy')
        plt.legend(['train', 'validation'])
        plt.show()
    
    def export_model(self, out_dir:Path):
        out_dir = Path(out_dir)
        out_dir.mkdir(exist_ok=True, parents=True)
        # export for inference only
        # torch.save(self.net, out_dir/f"{self.net.__class__.__name__}_epoch_{self.epoch_count}.pth")
        
        # # export for future usage here
        torch.save({
                    'epoch': self.epoch_count,
                    'model_state_dict': self.net.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'loss': self.epoch_loss[-1][-1],
                    }, 
                    out_dir/f"{self.net.__class__.__name__}_epoch_{self.epoch_count}.pth")

    def train(self, optimizer, num_epoch, scheduler=None):
        '''
        train
        args: 
            - optimizer
            - num_epoch: int
        '''
        self.optimizer = optimizer
        dataloaders = {'train': self.train_data, 'val':self.val_data}
        
        loss_hist = np.ones((num_epoch,2))*np.inf
        acc_hist = np.zeros((num_epoch, 2))

        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.net.to(device)
        print(f"Starting training of {self.net}, using device {device}")

        for epoch in range(num_epoch):
            time_start = time.time()
            print(f"epoch: {self.epoch_count}")
            # iterate between train and val phase
            for phi, phase in enumerate(dataloaders):
                if dataloaders[phase] is None:
                    break 
                if phase == 'train':
                    self.net.train()
                else:
                    self.net.eval()
                
                epoch_loss = 0.0
                correct_cls = 0.0
                # tqdm around enumerate?
                for i, data_batch in tqdm(enumerate(dataloaders[phase]), total=len(dataloaders[phase])):
                    inputs, labels = data_batch
                    inputs, labels = inputs.to(device), labels.to(device)
                    # TODO: not getting batches form dataloaders. dataloaders not proper
                    optimizer.zero_grad()
                    out = self.net(inputs) # (batch_size, class_num)
                    loss = self.criterion(out, labels)
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()
                                            
                    pred = torch.argmax(out, 1) # predicted class for each sample in batch
                    correct_cls += torch.sum(pred == labels).item()
                    epoch_loss += loss.item()
                
                loss_hist[epoch, phi] = epoch_loss  
                epoch_acc = correct_cls / len(dataloaders[phase].dataset)   
                acc_hist[epoch, phi] = epoch_acc
                print(f"{phase.upper()} loss: {epoch_loss} | acc: {epoch_acc}")
            
            if scheduler:
                scheduler.step()
            self.epoch_count += 1
        
        self.epoch_loss.append(loss_hist)
        self.epoch_acc.append(acc_hist)
        print(f"training finished after {epoch+1} iterations")

from pathlib import Path
import torch.optim as optim

from .models import *
from .dataset import build_dataset_out_of_dir_structure, get_data_loaders, visualize_sample, get_smaller_dataloader

if __name__=='__main__':
    img_dir = 'total_data_CNN03'
    resize_to = (128,128)

    imgs, labls, labls_2_id = build_dataset_out_of_dir_structure(Path(img_dir))

    # dataloaders
    # train_data, test_data, val_data = get_data_loaders(imgs, labls, labls_2_id, 
    #                                                    batch_size=1, resize_to=resize_to)

    # quick overfit to a small subset of available dataset
    tiny_data = get_smaller_dataloader(800, imgs, labls, labls_2_id, batch_size=4, resize_to=resize_to)
    tiny_data_val = get_smaller_dataloader(400, imgs, labls, labls_2_id, batch_size=4, resize_to=resize_to)

    # net = Model1Vgg19(len(labls_2_id), img_size=resize_to)
    # net = ModelTinyHruz(num_out_classes=len(labls_2_id), img_size=resize_to)
    net = ResnetTiny(num_out_classes=len(labls_2_id))

    loss = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=0.002)

    tr = TrainingExperiment(net, loss, tiny_data, val_data=tiny_data_val)
    tr.train(optimizer, 3)
    tr.plot_loss()    
    tr.plot_acc()