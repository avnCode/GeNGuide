import numpy as np
from torchvision import datasets, transforms
from utils.toolkit import split_images_labels
import random
from copy import deepcopy
import json
from PIL import Image
import torch
import yaml
from PIL import Image, ImageFile
import warnings
from tqdm import tqdm

torch.manual_seed(2024)
np.random.seed(2024)
import os
class iData(object):
    train_trsf = []
    test_trsf = []
    common_trsf = []
    class_order = None


class iCIFAR10(iData):
    name = 'cifar10'
    num_classes = 10
    use_path = False
    with open('./exps/genguide_cifar.json', 'r') as file:
        config = json.load(file)
    
    train_trsf = [
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ColorJitter(brightness=63/255)
    ]
    test_trsf = []
    common_trsf = [
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.4914, 0.4822, 0.4465), std=(0.2023, 0.1994, 0.2010)),
    ]

    class_order = np.arange(10).tolist()

    def download_data(self):
        train_dataset = datasets.cifar.CIFAR10('./data', train=True, download=True)
        test_dataset = datasets.cifar.CIFAR10('./data', train=False, download=True)
        self.train_data, self.train_targets = train_dataset.data, np.array(train_dataset.targets)
        self.test_data, self.test_targets = test_dataset.data, np.array(test_dataset.targets)
        self.org_targets = deepcopy(self.train_targets)
        self.train_idx = np.arange(len(self.train_targets))

class iCIFAR10_224(iCIFAR10):
    train_trsf = [
        transforms.RandomResizedCrop(224, interpolation=3),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=63/255)
    ]
    test_trsf = [
        transforms.Resize(256, interpolation=3),
        transforms.CenterCrop(224),
    ]


# class iWEBVISION(iData):
#     name = 'webvision'
#     num_classes = 14
#     use_path = False
#     config = yaml.load(open("./exps/webvision_spr.yaml"), Loader=yaml.FullLoader)
#     print("*************************************************", config)
#     ImageFile.LOAD_TRUNCATED_IMAGES = True
#     train_trsf = [
#         transforms.Resize((config['x_h'], config['x_w'])),
#         transforms.RandomHorizontalFlip(),
#         transforms.ColorJitter(brightness=63/255)
#     ]
#     test_trsf = [
#         transforms.Resize(256),
#         transforms.CenterCrop(224),
#     ]
#     common_trsf = [
#         transforms.ToTensor(),
#         transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
#     ]

#     idx_to_realname = []
#     with open(os.path.join(config['data_root'], name, 'info', 'synsets.txt'), 'r') as f:
#         for i, cls in enumerate(f.readlines()):
#             idx_to_realname.append(cls.strip())

#     def download_data(self):
#         train_infos_pth = os.path.join(self.config['data_root'], self.name, 'info', 'train_filelist_google.txt')
#         val_infos_pth = os.path.join(self.config['data_root'], self.name, 'info', 'val_filelist.txt')
#         LABEL_LIST = [412, 480, 506, 395, 421, 121, 498, 762, 48, 896, 32, 414, 147, 436]
        
#         train_data_list = []
#         train_targets_list = []
#         val_data_list = []
#         val_targets_list = []
#         transform = transforms.Compose(self.train_trsf)
#         with open(train_infos_pth, 'r') as f:  
#             for info in tqdm(f.readlines(), desc="Downloading Train Data"):  
#                 name, label = info.split(' ')  
#                 label = int(label)  
#                 if label not in LABEL_LIST:  
#                     continue  
#                 with Image.open(os.path.join(self.config['data_root'], self.name, name)) as img:  
#                     tmp = transform(img)  
#                     train_data_list.append(np.array(tmp))  
#                     train_targets_list.append(LABEL_LIST.index(label))

#         with open(val_infos_pth, 'r') as f:
#             for info in f.readlines():
#                 name, label = info.split(' ')
#                 name = os.path.join('val_images', name)
#                 label = int(label)
#                 if label not in LABEL_LIST:
#                     continue
#                 with Image.open(os.path.join(self.config['data_root'], self.name, name)) as img:
#                     tmp = transform(img)
#                     val_data_list.append(np.array(tmp))
#                     val_targets_list.append(LABEL_LIST.index(label))

#         self.train_data = np.array(train_data_list)
#         self.train_targets = np.array(train_targets_list)
#         self.test_data = np.array(val_data_list)
#         self.test_targets = np.array(val_targets_list)
#         self.org_targets = deepcopy(self.train_targets)
#         self.train_idx = np.arange(len(self.train_targets))

# class iWEBVISION_224(iWEBVISION):
#     train_trsf = [
#         transforms.RandomResizedCrop(224, interpolation=3),
#         transforms.RandomHorizontalFlip(),
#         transforms.ColorJitter(brightness=63/255)
#     ]
#     test_trsf = [
#         transforms.Resize(256, interpolation=3),
#         transforms.CenterCrop(224),
#     ]


class iCIFAR100(iData):
    use_path = False
    train_trsf = [
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=63/255)
    ]
    test_trsf = []
    common_trsf = [
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.5071, 0.4867, 0.4408), std=(0.2675, 0.2565, 0.2761)),
    ]

    class_order = np.arange(100).tolist()

    def download_data(self):
        train_dataset = datasets.cifar.CIFAR100('./data', train=True, download=True)
        test_dataset = datasets.cifar.CIFAR100('./data', train=False, download=True)
        self.train_data, self.train_targets = train_dataset.data, np.array(train_dataset.targets)
        self.test_data, self.test_targets = test_dataset.data, np.array(test_dataset.targets)
        self.org_targets = deepcopy(self.train_targets)
        self.train_idx = np.arange(len(self.train_targets))

class iCIFAR100_224(iCIFAR100):
    train_trsf = [
        transforms.RandomResizedCrop(224, interpolation=3),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=63/255)
    ]
    test_trsf = [
        transforms.Resize(256, interpolation=3),
        transforms.CenterCrop(224),
    ]

class iImageNet1000(iData):
    use_path = True
    train_trsf = [
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=63/255)
    ]
    test_trsf = [
        transforms.Resize(256),
        transforms.CenterCrop(224),
    ]
    common_trsf = [
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]

    class_order = np.arange(1000).tolist()

    def download_data(self):
        assert 0,"You should specify the folder of your dataset"
        train_dir = '[DATA-PATH]/train/'
        test_dir = '[DATA-PATH]/val/'

        train_dset = datasets.ImageFolder(train_dir)
        test_dset = datasets.ImageFolder(test_dir)

        self.train_data, self.train_targets = split_images_labels(train_dset.imgs)
        self.test_data, self.test_targets = split_images_labels(test_dset.imgs)

class iImageNet100(iData):
    use_path = True
    train_trsf = [
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
    ]
    test_trsf = [
        transforms.Resize(256),
        transforms.CenterCrop(224),
    ]
    common_trsf = [
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]

    class_order = np.arange(1000).tolist()

    def download_data(self):
        train_dir = 'data/imagenet100/train/'
        test_dir = 'data/imagenet100/val/'

        train_dset = datasets.ImageFolder(train_dir)
        test_dset = datasets.ImageFolder(test_dir)

        self.train_data, self.train_targets = split_images_labels(train_dset.imgs)
        self.test_data, self.test_targets = split_images_labels(test_dset.imgs)


class iImageNetR(iData):
    use_path = True
    train_trsf = [
        transforms.RandomResizedCrop(224, interpolation=3),
        transforms.RandomHorizontalFlip(),
    ]
    test_trsf = [
        transforms.Resize(256, interpolation=3),
        transforms.CenterCrop(224),
    ]
    common_trsf = [
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]

    class_order = np.arange(1000).tolist()

    def download_data(self):
        train_dir = 'data/imagenet-r/train/'
        test_dir = 'data/imagenet-r/val/'

        train_dset = datasets.ImageFolder(train_dir)
        test_dset = datasets.ImageFolder(test_dir)

        self.train_data, self.train_targets = split_images_labels(train_dset.imgs)
        self.test_data, self.test_targets = split_images_labels(test_dset.imgs)

class iCUB200_224(iData):
    use_path = True
    train_trsf = [
        transforms.Resize((300, 300), interpolation=3),
        transforms.RandomCrop((224, 224)),
        transforms.RandomHorizontalFlip(),
    ]
    test_trsf = [
        transforms.Resize(256, interpolation=3),
        transforms.CenterCrop(224),
    ]
    common_trsf = [
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
    class_order = np.arange(1000).tolist()

    def download_data(self):
        train_dir = 'data/cub_200/train/'
        test_dir = 'data/cub_200/val/'
        train_dset = datasets.ImageFolder(train_dir)
        test_dset = datasets.ImageFolder(test_dir)
        self.train_data, self.train_targets = split_images_labels(train_dset.imgs)
        self.test_data, self.test_targets = split_images_labels(test_dset.imgs)

class iCARS196_224(iData):
    use_path = True
    train_trsf = [
        transforms.Resize((300, 300), interpolation=3),
        transforms.RandomCrop((224, 224)),
        transforms.RandomHorizontalFlip(),
    ]
    test_trsf = [
        transforms.Resize(256, interpolation=3),
        transforms.CenterCrop(224),
    ]
    common_trsf = [
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
    class_order = np.arange(1000).tolist()

    def download_data(self):
        train_dir = 'data/cars196/train/'
        test_dir = 'data/cars196/val/'

        train_dset = datasets.ImageFolder(train_dir)
        test_dset = datasets.ImageFolder(test_dir)
        self.train_data, self.train_targets = split_images_labels(train_dset.imgs)
        self.test_data, self.test_targets = split_images_labels(test_dset.imgs)


class iResisc45_224(iData): 
    use_path = True
    train_trsf = [
        transforms.Resize((300, 300), interpolation=3),
        transforms.RandomCrop((224, 224)),
        transforms.RandomHorizontalFlip(),
    ]
    test_trsf = [
        transforms.Resize(256, interpolation=3),
        transforms.CenterCrop(224),
    ]
    common_trsf = [
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
    class_order = np.arange(1000).tolist()

    def download_data(self):
        train_dir = 'data/resisc45/train/'
        test_dir = 'data/resisc45/val/'

        train_dset = datasets.ImageFolder(train_dir)
        test_dset = datasets.ImageFolder(test_dir)
        self.train_data, self.train_targets = split_images_labels(train_dset.imgs)
        self.test_data, self.test_targets = split_images_labels(test_dset.imgs)



class iSketch345_224(iData):
    use_path = True
    train_trsf = [
        transforms.Resize((300, 300), interpolation=3),
        transforms.RandomCrop((224, 224)),
        transforms.RandomHorizontalFlip(),
    ]
    test_trsf = [
        transforms.Resize(256, interpolation=3),
        transforms.CenterCrop(224),
    ]
    common_trsf = [
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
    class_order = np.arange(1000).tolist()

    def download_data(self):
        train_dir = 'data/sketch345/train/'
        test_dir = 'data/sketch345/val/'

        train_dset = datasets.ImageFolder(train_dir)
        test_dset = datasets.ImageFolder(test_dir)
        self.train_data, self.train_targets = split_images_labels(train_dset.imgs)
        self.test_data, self.test_targets = split_images_labels(test_dset.imgs)