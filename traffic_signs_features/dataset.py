### PYTORCH DATASET AND DATALOADER HERE
import typing as T
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision.io import read_image, ImageReadMode
import torchvision.transforms as transf
import matplotlib.pyplot as plt
import json

# TODO: standardize --normalize with std and mean of the whole dataset

CLASSES_FILE = 'info.json' # inside the dataset dir

class Dataset(Dataset):
    def __init__(self, images:tuple, labels:tuple, id_to_label:tuple, transform=None):
        self.images = images
        self.labels = labels
        self.transform = transform
        self.id_to_label = id_to_label

    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        '''returns image as a tensor and corresponding label'''
        # images expected already loaded in memory
        img = self.images[idx]
        if self.transform:
            img = self.transform(img)
        # if self.target_transform:
        #     label = self.target_transform(label)
        label_id = self.id_to_label.index(self.labels[idx])
        return img, torch.tensor(label_id)
    
    def get_label(self, label_id):
        '''get category string label given it's id'''
        return self.id_to_label[label_id.item()]


def get_data_loaders(imgs, labels, id_to_label, transform=None, train_size=0.8, 
                     test_size=0.1, batch_size=16, resize_to=(224,224)) -> T.Tuple[DataLoader, DataLoader, DataLoader]:
    '''Get (train, test, val) dataloaders'''
    # normalization to 0-1
    transform = transf.Compose([
        transf.Normalize((0.0,0.0,0.0), (255,255,255)),
        # transf.RandomHorizontalFlip(p=0.5),
        # transf.RandomCrop(size=(224,224)) # TODO: not sure about random crop or resize
        transf.Resize(resize_to)
        ])
    
    dataset = Dataset(imgs, labels, id_to_label, transform=transform)

    train_len = int(np.round(dataset.__len__() * train_size))
    test_len = int(np.round(dataset.__len__() * test_size))
    val_len = dataset.__len__() - train_len - test_len
    print(f"data split; train:{train_len}, test:{test_len}, val:{val_len}")


    return tuple([DataLoader(data, batch_size=batch_size) 
                  for data in random_split(dataset, [train_len, test_len, val_len])])


def get_smaller_dataloader(size, imgs, labels, id_to_label, transform=None, 
                           batch_size=16, resize_to=(224,224)):
    '''returns dataloader with subset of size size of the dataset'''
    # normalization to 0-1
    transform = transf.Compose([
        transf.Normalize((0.0,0.0,0.0), (255,255,255)),
        # transf.RandomHorizontalFlip(p=0.5),
        # transf.RandomCrop(size=(224,224)) # TODO: not sure about random crop or resize
        transf.Resize(resize_to)
    ])
    
    
    dataset = Dataset(imgs, labels, id_to_label, transform=transform)
    small_dataset, _ = random_split(dataset, [size, dataset.__len__()-size])
    
    return DataLoader(small_dataset, batch_size=batch_size)

def build_dataset_out_of_dir_structure(parent_dir:Path, expected_labels:list=None):
    '''
    iterates over directories named as labels returns list of images and label

    returns: (images, labels, labels_to_id, label_counts)
    '''
    IMG_SUFFIXES = ["jpg","png"]
    
    images = []
    labels = [] # lables[i] : label for image images[i]
    labels_to_id = [] # category id -> string label mapping
    label_counts = {}
    for dir in parent_dir.iterdir():
        if expected_labels and dir.name not in expected_labels:
            # assert dir.name in expected_labels
            print(f"{dir.name} not found in expected labels")
        else:
            print(f"{dir.name} loaded")
        if dir.name not in labels_to_id:
            label_counts[dir.name] = 0
        for img_p in dir.glob('*'):
            if img_p.name.split(".")[-1] not in IMG_SUFFIXES:
                print("unexpected image suffix:", img_p.name.split(".")[-1])
            try:
                img = read_image(str(img_p), mode=ImageReadMode.RGB)
                if img.shape[0] != 3:
                    raise Exception(f"error 3 channels img expected,instead got: {img.shape}")
                else:
                    images.append(img.float())
                    labels.append(dir.name)
                    label_counts[dir.name] += 1
            except Exception as ex:
                print(f"loading {img_p} failed:\n{ex}")
    print(f"Dataset of {len(images)} images loaded")

    with open(parent_dir/CLASSES_FILE, 'r') as f:
        classes_dict = json.load(f)

    return tuple(images), tuple(labels), tuple(classes_dict['id_to_label']), label_counts

def visualize_sample(dataloader):
    ''''''
    figure = plt.figure(figsize=(8, 8))
    cols, rows = 3, 3
    dt = iter(dataloader.dataset)
    for i in range(1, cols * rows + 1):
        img, label = next(dt)
        img *= 255 # unnormalize
        img = img.permute(1,2,0).int()
        figure.add_subplot(rows, cols, i)
        # first dataset is the subset second is the reference to the main one...
        plt.title(dataloader.dataset.dataset.get_label(label))
        plt.axis("off")
        plt.imshow(img.cpu().numpy())
    plt.show()


if __name__=='__main__':
    SIGNS_CATEGORY_NAMES = (
        'A10', 'A11', 'A12', 'A14', 'A15', 'A16', 'A17', 'A18', 'A19', 'A1a', 'A1b', 'A22', 'A24', 'A28', 'A29', 'A2a', 'A2b', 'A30', 'A31a', 'A31b', 'A31c', 'A32a', 'A32b', 'A4', 'A5a', 'A6a', 'A6b', 'A7a', 'A8', 'A9', 'B1', 'B11', 'B12', 'B13', 'B14', 'B15', 'B16', 'B17', 'B19', 'B2', 'B20a', 'B20b', 'B21a', 'B21b', 'B24a', 'B24b', 'B26', 'B28', 'B29', 'B32', 'B4', 'B5', 'B6', 'C1', 'C10a', 'C10b', 'C13a', 'C14a', 'C2a', 'C2b', 'C2c', 'C2d', 'C2e', 'C2f', 'C3a', 'C3b', 'C4a', 'C4b', 'C4c', 'C7a', 'C9a', 'C9b', 'E1', 'E11', 'E11c', 'E12', 'E13', 'E2a', 'E2b', 'E2c', 'E2d', 'E3a', 'E3b', 'E4', 'E5', 'E6', 'E7a', 'E7b', 'E8a', 'E8b', 'E8c', 'E8d', 'E8e', 'E9', 'I2', 'IJ1', 'IJ10', 'IJ11a', 'IJ11b', 'IJ14c', 'IJ15', 'IJ2', 'IJ3', 'IJ4a', 'IJ4b', 'IJ4c', 'IJ4d', 'IJ4e', 'IJ5', 'IJ6', 'IJ7', 'IJ8', 'IJ9', 'IP10a', 'IP10b', 'IP11a', 'IP11b', 'IP11c', 'IP11e', 'IP11g', 'IP12', 'IP13a', 'IP13b', 'IP13c', 'IP13d', 'IP14a', 'IP15a', 'IP15b', 'IP16', 'IP17', 'IP18a', 'IP18b', 'IP19', 'IP2', 'IP21', 'IP21a', 'IP22', 'IP25a', 'IP25b', 'IP26a', 'IP26b', 'IP27a', 'IP3', 'IP31a', 'IP4a', 'IP4b', 'IP5', 'IP6', 'IP7', 'IP8a', 'IP8b', 'IS10b', 'IS11a', 'IS11b', 'IS11c', 'IS12a', 'IS12b', 'IS12c', 'IS13', 'IS14', 'IS15a', 'IS15b', 'IS16b', 'IS16c', 'IS16d', 'IS17', 'IS18a', 'IS18b', 'IS19a', 'IS19b', 'IS19c', 'IS19d', 'IS1a', 'IS1b', 'IS1c', 'IS1d', 'IS20', 'IS21a', 'IS21b', 'IS21c', 'IS22a', 'IS22c', 'IS22d', 'IS22e', 'IS22f', 'IS23', 'IS24a', 'IS24b', 'IS24c', 'IS2a', 'IS2b', 'IS2c', 'IS2d', 'IS3a', 'IS3b', 'IS3c', 'IS3d', 'IS4a', 'IS4b', 'IS4c', 'IS4d', 'IS5', 'IS6a', 'IS6b', 'IS6c', 'IS6e', 'IS6f', 'IS6g', 'IS7a', 'IS8a', 'IS8b', 'IS9a', 'IS9b', 'IS9c', 'IS9d', 'O2', 'P1', 'P2', 'P3', 'P4', 'P6', 'P7', 'P8', 'UNKNOWN', 'X1', 'X2', 'X3', 'XXX', 'Z2', 'Z3', 'Z4a', 'Z4b', 'Z4c', 'Z4d', 'Z4e', 'Z7', 'Z9'
    ) 

    
    img_dir = 'total_data_CNN03'
    
    imgs, labls, labls_2_id = build_dataset_out_of_dir_structure(Path(img_dir), expected_labels=SIGNS_CATEGORY_NAMES)

    # dataloaders
    train_data, test_data, val_data = get_data_loaders(imgs, labls, labls_2_id)

    visualize_sample(train_data)
    
    print()