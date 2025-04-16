import logging
import numpy as np
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
from utils.data import iCIFAR10, iCIFAR100, iImageNet100, iImageNet1000, iCIFAR10_224, iCIFAR100_224, iImageNetR, iCUB200_224, iResisc45_224, iCARS196_224, iSketch345_224
from copy import deepcopy
import random
import json
import torch
import yaml
torch.manual_seed(2025)
np.random.seed(2025)

class DataManager(object):
    def __init__(self, dataset_name, shuffle, seed, init_cls, increment):
        self.dataset_name = dataset_name
        self._setup_data(dataset_name, shuffle, seed)
        assert init_cls <= len(self._class_order), 'No enough classes.'
        self._increments = [init_cls]
        while sum(self._increments) + increment < len(self._class_order):
            self._increments.append(increment)
        offset = len(self._class_order) - sum(self._increments)
        if offset > 0:
            self._increments.append(offset)

    @property
    def nb_tasks(self):
        return len(self._increments)

    def get_task_size(self, task):
        return self._increments[task]

    def get_dataset(self, indices, source, mode, appendent=None, ret_data=False, with_raw=False, with_noise=False):
        if source == 'train':
            x, y, z, org, train_idx = self._train_data, self._train_targets, self._ranking, self._org_targets, self.train_idx
        elif source == 'test':
            x, y, z, org, train_idx = self._test_data, self._test_targets, self._ranking, self._org_targets, self.train_idx
        else:
            raise ValueError('Unknown data source {}.'.format(source))

        if mode == 'train':
            trsf = transforms.Compose([*self._train_trsf, *self._common_trsf])
        elif mode == 'flip':
            trsf = transforms.Compose([*self._test_trsf, transforms.RandomHorizontalFlip(p=1.), *self._common_trsf])
        elif mode == 'test':
            trsf = transforms.Compose([*self._test_trsf, *self._common_trsf])
        else:
            raise ValueError('Unknown mode {}.'.format(mode))

        data, targets, ranking, orginal_target, train_ids = [], [], [], [], []
        for idx in indices:
            class_data, class_targets, class_ranking, class_orgLabel, class_train_ids = self._select(x, y, z, org, train_idx, low_range=idx, high_range=idx+1)         #just selecting x,y where label index matches with idx
            data.append(class_data)
            targets.append(class_targets)
            ranking.append(class_ranking)
            orginal_target.append(class_orgLabel)
            train_ids.append(class_train_ids)

        if appendent is not None and len(appendent) != 0:
            appendent_data, appendent_targets = appendent
            data.append(appendent_data)
            targets.append(appendent_targets)

        data, targets, ranking, orginal_target, train_ids = np.concatenate(data), np.concatenate(targets), np.concatenate(ranking), np.concatenate(orginal_target), np.concatenate(train_ids)

        if ret_data:
            return data, targets, ranking, orginal_target, train_ids, DummyDataset(data, targets, ranking, train_ids,trsf, self.use_path, with_raw, with_noise)
        else:
            return DummyDataset(data, targets, ranking, train_ids, trsf, self.use_path, with_raw, with_noise)

    def get_dataset_with_split(self, indices, source, mode, appendent=None, val_samples_per_class=0):
        if source == 'train':
            x, y, z = self._train_data, self._train_targets, self._ranking
        elif source == 'test':
            x, y, z = self._test_data, self._test_targets, self._ranking
        else:
            raise ValueError('Unknown data source {}.'.format(source))

        if mode == 'train':
            trsf = transforms.Compose([*self._train_trsf, *self._common_trsf])
        elif mode == 'test':
            trsf = transforms.Compose([*self._test_trsf, *self._common_trsf])
        else:
            raise ValueError('Unknown mode {}.'.format(mode))

        train_data, train_targets = [], []
        val_data, val_targets = [], []
        np.random.seed(1997)

        for idx in indices:
            class_data, class_targets = self._select(x, y, low_range=idx, high_range=idx+1)
            val_indx = np.random.choice(len(class_data), val_samples_per_class, replace=False)
            train_indx = list(set(np.arange(len(class_data))) - set(val_indx))
            val_data.append(class_data[val_indx])
            val_targets.append(class_targets[val_indx])
            train_data.append(class_data[train_indx])
            train_targets.append(class_targets[train_indx])

        if appendent is not None:
            appendent_data, appendent_targets = appendent
            for idx in range(0, int(np.max(appendent_targets))+1):
                append_data, append_targets = self._select(appendent_data, appendent_targets,
                                                           low_range=idx, high_range=idx+1)
                val_indx = np.random.choice(len(append_data), val_samples_per_class, replace=False)
                train_indx = list(set(np.arange(len(append_data))) - set(val_indx))
                val_data.append(append_data[val_indx])
                val_targets.append(append_targets[val_indx])
                train_data.append(append_data[train_indx])
                train_targets.append(append_targets[train_indx])

        train_data, train_targets = np.concatenate(train_data), np.concatenate(train_targets)
        val_data, val_targets = np.concatenate(val_data), np.concatenate(val_targets)

        return DummyDataset(train_data, train_targets, trsf, self.use_path), \
            DummyDataset(val_data, val_targets, trsf, self.use_path)

    def udt(self, input):
        self._ranking = np.ones_like(self._train_targets, dtype=np.float64)*input


    def _setup_data(self, dataset_name, shuffle, seed):
        idata = _get_idata(dataset_name)
        idata.download_data()
        with open('./exps/genguide_cifar.json', 'r') as file:
            config = json.load(file)
        # Data
        self._train_data, self._train_targets = idata.train_data, idata.train_targets
        self._test_data, self._test_targets = idata.test_data, idata.test_targets
        self._org_targets, self.train_idx = idata.org_targets, idata.train_idx
        self.use_path = idata.use_path
        self._ranking = np.ones_like(self._train_targets, dtype=np.float64)

        # Transforms
        self._train_trsf = idata.train_trsf
        self._test_trsf = idata.test_trsf
        self._common_trsf = idata.common_trsf
        self.num_classes = len(np.unique(self._test_targets))
        self.class_to_idx = {'apple': 0,
                            'aquarium_fish': 1,
                            'baby': 2,
                            'bear': 3,
                            'beaver': 4,
                            'bed': 5,
                            'bee': 6,
                            'beetle': 7,
                            'bicycle': 8,
                            'bottle': 9,
                            'bowl': 10,
                            'boy': 11,
                            'bridge': 12,
                            'bus': 13,
                            'butterfly': 14,
                            'camel': 15,
                            'can': 16,
                            'castle': 17,
                            'caterpillar': 18,
                            'cattle': 19,
                            'chair': 20,
                            'chimpanzee': 21,
                            'clock': 22,
                            'cloud': 23,
                            'cockroach': 24,
                            'couch': 25,
                            'crab': 26,
                            'crocodile': 27,
                            'cup': 28,
                            'dinosaur': 29,
                            'dolphin': 30,
                            'elephant': 31,
                            'flatfish': 32,
                            'forest': 33,
                            'fox': 34,
                            'girl': 35,
                            'hamster': 36,
                            'house': 37,
                            'kangaroo': 38,
                            'keyboard': 39,
                            'lamp': 40,
                            'lawn_mower': 41,
                            'leopard': 42,
                            'lion': 43,
                            'lizard': 44,
                            'lobster': 45,
                            'man': 46,
                            'maple_tree': 47,
                            'motorcycle': 48,
                            'mountain': 49,
                            'mouse': 50,
                            'mushroom': 51,
                            'oak_tree': 52,
                            'orange': 53,
                            'orchid': 54,
                            'otter': 55,
                            'palm_tree': 56,
                            'pear': 57,
                            'pickup_truck': 58,
                            'pine_tree': 59,
                            'plain': 60,
                            'plate': 61,
                            'poppy': 62,
                            'porcupine': 63,
                            'possum': 64,
                            'rabbit': 65,
                            'raccoon': 66,
                            'ray': 67,
                            'road': 68,
                            'rocket': 69,
                            'rose': 70,
                            'sea': 71,
                            'seal': 72,
                            'shark': 73,
                            'shrew': 74,
                            'skunk': 75,
                            'skyscraper': 76,
                            'snail': 77,
                            'snake': 78,
                            'spider': 79,
                            'squirrel': 80,
                            'streetcar': 81,
                            'sunflower': 82,
                            'sweet_pepper': 83,
                            'table': 84,
                            'tank': 85,
                            'telephone': 86,
                            'television': 87,
                            'tiger': 88,
                            'tractor': 89,
                            'train': 90,
                            'trout': 91,
                            'tulip': 92,
                            'turtle': 93,
                            'wardrobe': 94,
                            'whale': 95,
                            'willow_tree': 96,
                            'wolf': 97,
                            'woman': 98,
                            'worm': 99}

        self.super_classes = [["beaver", "dolphin", "otter", "seal", "whale"],
                            ["aquarium_fish", "flatfish", "ray", "shark", "trout"],
                            ["orchid", "poppy", "rose", "sunflower", "tulip"],
                            ["bottle", "bowl", "can", "cup", "plate"],
                            ["apple", "mushroom", "orange", "pear", "sweet_pepper"],
                            ["clock", "keyboard", "lamp", "telephone", "television"],
                            ["bed", "chair", "couch", "table", "wardrobe"],
                            ["bee", "beetle", "butterfly", "caterpillar", "cockroach"],
                            ["bear", "leopard", "lion", "tiger", "wolf"],
                            ["bridge", "castle", "house", "road", "skyscraper"],
                            ["cloud", "forest", "mountain", "plain", "sea"],
                            ["camel", "cattle", "chimpanzee", "elephant", "kangaroo"],
                            ["fox", "porcupine", "possum", "raccoon", "skunk"],
                            ["crab", "lobster", "snail", "spider", "worm"],
                            ["baby", "boy", "girl", "man", "woman"],
                            ["crocodile", "dinosaur", "lizard", "snake", "turtle"],
                            ["hamster", "mouse", "rabbit", "shrew", "squirrel"],
                            ["maple_tree", "oak_tree", "palm_tree", "pine_tree", "willow_tree"],
                            ["bicycle", "bus", "motorcycle", "pickup_truck", "train"],
                            ["lawn_mower", "rocket", "streetcar", "tank", "tractor"]]
        
        # Order
        if shuffle:
            np.random.seed(1997)
        # Order
        order = [i for i in range(len(np.unique(self._train_targets)))]
        if shuffle:
            np.random.seed(seed)
            order = np.random.permutation(len(order)).tolist()
        else:
            order = idata.class_order
        self._class_order = order

    def asym_cifar10(self):
        self._class_order = [9,1,2,0,5,3,8,6,7,4]
        print("Asymmetric Cifar10 class order")
        
    
    def sym_cifar10(self):
        self._class_order = [7,6,2,3,0,1,5,9,8,4]
        print("Symmetric Cifar10 class order")

    def webvision(self):
        file_path = './episodes/webvision-split_epc1_a.yaml'
        with open(file_path, 'r') as file:
            data = yaml.safe_load(file)
        order = []
        for subset in data:
            for s in subset['subsets']:
                order.append(s[1])
        self._class_order = order
        print(order)
        print("Webvision class order initialised")
    
    def rnd_cifar100(self):
        file_path = './episodes/cifar100rnd-split_epc1_a.yaml'
        with open(file_path, 'r') as file:
            data = yaml.safe_load(file)
        order = []
        for subset in data:
            for s in subset['subsets']:
                order.append(s[1])
        self._class_order = order
        print("Random Cifar100 class order initialised")

    def superclass_cifar100(self):
        file_path = './episodes/cifar100sup-split_epc1_a.yaml'
        with open(file_path, 'r') as file:
            data = yaml.safe_load(file)
        order = []
        for subset in data:
            for s in subset['subsets']:
                order.append(s[1])
        self._class_order = order
        print("Superclass Cifar100 class order initialised")

    def map_class(self):
        self._train_targets = _map_new_class_index(self._train_targets, self._class_order)
        self._test_targets = _map_new_class_index(self._test_targets, self._class_order) 
        self._org_targets = _map_new_class_index(self._org_targets, self._class_order)
        print("Mapping Done with New Class Order")


    def _select(self, x, y, z, org, train_idx, low_range, high_range):
        idxes = np.where(np.logical_and(y >= low_range, y < high_range))[0]
        return x[idxes], y[idxes], z[idxes], org[idxes], train_idx[idxes]


    def add_noise_cifar10(self, noise_asym, noise_rate):
        np.random.seed(1997)
        if noise_asym:
            source_class = [9, 2, 3, 5, 4]
            target_class = [1, 0, 5, 3, 7]
            self.add_asymmetric_noise(source_class, target_class, noise_rate)
            print(f"{noise_rate} Asymmetric Noise Added")                           

        else:
            self.add_symmetric_noise(list(range(self.num_classes)), noise_rate)
            print(f"{noise_rate} Symmetric Noise Added")                           


    def add_noise_cifar100(self, noise_super, noise_rate):
        if noise_super:
            for super_cls in self.super_classes:
                cls_idx = [self.class_to_idx[c] for c in super_cls]   
                self.add_symmetric_noise(cls_idx, noise_rate)
                print(f"{noise_rate} SuperClass Noise Added")                           
        else:
            self.add_symmetric_noise(list(range(self.num_classes)), noise_rate)
            print(f"{noise_rate} Random Noise Added")  


    def add_symmetric_noise(self, source_class, noise_rate):
        np.random.seed(1997)
        for y in source_class:
            random_target = [t for t in source_class if t != y]
            tindx = [i for i, x in enumerate(self._org_targets) if x == y]
            for i in tindx[:round(len(tindx) * noise_rate)]:
                self._train_targets[i] = random.choice(random_target)

    def add_asymmetric_noise(self, source_class, target_class, noise_rate):
        np.random.seed(1997)
        for s, t in zip(source_class, target_class):
            cls_idx = np.where(np.array(self._org_targets) == s)[0]
            n_noisy = int(noise_rate * cls_idx.shape[0])
            noisy_sample_index = np.random.choice(list(cls_idx), n_noisy, replace=False)
            for idx in noisy_sample_index:
                self._train_targets[idx] = t

    def _update_rank(self, ranklist, idx):
        self._ranking[idx] = ranklist.cpu()

    def _get_weights(self, idx):
        return self._ranking[idx]

    def _get_org_targets(self, idx):
        return self._org_targets[idx]

class DummyDataset(Dataset):
    def __init__(self, images, labels, rnk, train_ids, trsf, use_path=False, with_raw=False, with_noise=False):
        assert len(images) == len(labels), 'Data size error!'
        np.random.seed(1997)
        self.images = images
        self.labels = labels
        self.ranking = rnk
        self.train_ids = train_ids
        self.trsf = trsf
        self.use_path = use_path
        self.with_raw = with_raw
        
        if use_path and with_raw:
            self.raw_trsf = transforms.Compose([transforms.Resize((500, 500)), transforms.ToTensor()])
        else:
            self.raw_trsf = transforms.Compose([transforms.ToTensor()])
        if with_noise:
            class_list = np.unique(self.labels)
            self.ori_labels = deepcopy(labels)
            for cls in class_list:
                random_target = class_list.tolist()
                random_target.remove(cls)
                tindx = [i for i, x in enumerate(self.ori_labels) if x == cls]
                for i in tindx[:round(len(tindx)*0.2)]:
                    self.labels[i] = random.choice(random_target)
            

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        np.random.seed(1997)
        if self.use_path:
            load_image = pil_loader(self.images[idx])
            image = self.trsf(load_image)
        else:
            load_image = Image.fromarray(self.images[idx])
            image = self.trsf(load_image)
        label = self.labels[idx]
        rank = self.ranking[idx]
        train_id = self.train_ids[idx]
        if self.with_raw:
            return idx, image, label, self.raw_trsf(load_image) 
        return idx, image, label, rank, train_id


def _map_new_class_index(y, order):
    return np.array(list(map(lambda x: order.index(x), y)))


def _get_idata(dataset_name):
    name = dataset_name.lower()
    if name == 'cifar10':
        return iCIFAR10()
    elif name == 'cifar100':
        return iCIFAR100()
    elif name == 'cifar10_224':
        return iCIFAR10_224()
    elif name == 'cifar100_224':
        return iCIFAR100_224()
    elif name == 'webvision':
        return iWEBVISION()
    elif name == 'webvision_224':
        return iWEBVISION_224()
    elif name == 'imagenet1000':
        return iImageNet1000()
    elif name == "imagenet100":
        return iImageNet100()
    elif name == "imagenet-r":
        return iImageNetR()
    elif name == 'cub200_224':
        return iCUB200_224()
    elif name == 'resisc45':
        return iResisc45_224()
    elif name == 'cars196_224':
        return iCARS196_224()
    elif name == 'sketch345_224':
        return iSketch345_224()
    else:
        raise NotImplementedError('Unknown dataset {}.'.format(dataset_name))


def pil_loader(path):
    '''
    Ref:
    https://pytorch.org/docs/stable/_modules/torchvision/datasets/folder.html#ImageFolder
    '''
    # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, 'rb') as f:
        img = Image.open(f)
        return img.convert('RGB')


def accimage_loader(path):
    '''
    Ref:
    https://pytorch.org/docs/stable/_modules/torchvision/datasets/folder.html#ImageFolder
    accimage is an accelerated Image loader and preprocessor leveraging Intel IPP.
    accimage is available on conda-forge.
    '''
    import accimage
    try:
        return accimage.Image(path)
    except IOError:
        # Potentially a decoding problem, fall back to PIL.Image
        return pil_loader(path)


def default_loader(path):
    '''
    Ref:
    https://pytorch.org/docs/stable/_modules/torchvision/datasets/folder.html#ImageFolder
    '''
    from torchvision import get_image_backend
    if get_image_backend() == 'accimage':
        return accimage_loader(path)
    else:
        return pil_loader(path)
