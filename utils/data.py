import numpy as np
from torchvision import datasets, transforms
import time
from utils.autoaugment import CIFAR10Policy, ImageNetPolicy


class iData(object):
    train_trsf = []
    test_trsf = []
    common_trsf = []
    class_order = None


def build_transform(is_train, args):
    input_size = 224
    resize_im = input_size > 32
    if is_train:
        scale = (0.05, 1.0)
        ratio = (3. / 4., 4. / 3.)

        transform = [
            transforms.RandomResizedCrop(input_size, scale=scale, ratio=ratio),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ColorJitter(brightness=63 / 255),
            # CIFAR10Policy(),
            ImageNetPolicy(),
            transforms.ToTensor(),
        ]
        return transform

    t = []
    if resize_im:
        size = int((256 / 224) * input_size)
        t.append(
            transforms.Resize(size, interpolation=3),  # to maintain same ratio w.r.t. 224 images
        )
        t.append(transforms.CenterCrop(input_size))
    t.append(transforms.ToTensor())

    # return transforms.Compose(t)
    return t


class iCIFAR224(iData):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.use_path = False

        self.train_trsf = build_transform(True, args)
        self.test_trsf = build_transform(False, args)
        self.common_trsf = [
            # transforms.ToTensor(),
        ]

        self.class_order = np.arange(100).tolist()

    def download_data(self):
        train_dataset = datasets.cifar.CIFAR100("./data", train=True, download=True)
        test_dataset = datasets.cifar.CIFAR100("./data", train=False, download=True)
        self.train_data, self.train_targets = train_dataset.data, np.array(
            train_dataset.targets
        )
        self.test_data, self.test_targets = test_dataset.data, np.array(
            test_dataset.targets
        )

class CUB(iData):
    def __init__(self, args):
        self.use_path = False
        self.args = args
        self.train_trsf = build_transform(True, None)
        self.test_trsf = build_transform(False, None)
        self.common_trsf = []

        self.class_order = np.arange(200).tolist()

    def download_data(self):
        # assert 0, "You should specify the folder of your dataset"
        train_dir = "./data/cub/train/"
        test_dir = "./data/cub/test/"
        train_dset = datasets.ImageFolder(train_dir)
        test_dset = datasets.ImageFolder(test_dir)

        self.train_data, self.train_targets = split_images_labels_imagenet(train_dset.imgs)
        self.test_data, self.test_targets = split_images_labels_imagenet(test_dset.imgs)


class iImageNetA(iData):
    def __init__(self, args):
        self.use_path = False
        self.args = args
        self.train_trsf = build_transform(True, None)
        self.test_trsf = build_transform(False, None)
        self.common_trsf = []

        self.class_order = np.arange(200).tolist()

    def download_data(self):
        # assert 0, "You should specify the folder of your dataset"
        train_dir = "./data/imagenet-a/train/"
        test_dir = "./data/imagenet-a/test/"
        train_dset = datasets.ImageFolder(train_dir)
        test_dset = datasets.ImageFolder(test_dir)

        self.train_data, self.train_targets = split_images_labels_imagenet(train_dset.imgs)
        self.test_data, self.test_targets = split_images_labels_imagenet(test_dset.imgs)


class omnibenchmark(iData):
    def __init__(self, args):
        self.use_path = False
        self.args = args
        self.train_trsf = build_transform(True, None)
        self.test_trsf = build_transform(False, None)
        self.common_trsf = []

        self.class_order = np.arange(300).tolist()

    def download_data(self):
        # assert 0, "You should specify the folder of your dataset"
        train_dir = "./data/omnibenchmark/train/"
        test_dir = "./data/omnibenchmark/test/"
        train_dset = datasets.ImageFolder(train_dir)
        test_dset = datasets.ImageFolder(test_dir)

        self.train_data, self.train_targets = split_images_labels_imagenet(train_dset.imgs)
        self.test_data, self.test_targets = split_images_labels_imagenet(test_dset.imgs)


class iFood101(iData):
    def __init__(self, args):
        self.use_path = False
        self.args = args
        self.train_trsf = build_transform(True, None)
        self.test_trsf = build_transform(False, None)
        self.common_trsf = []

        self.class_order = np.arange(101).tolist()

    def download_data(self):
        # assert 0, "You should specify the folder of your dataset"
        train_dir = "./data/food101/train/"
        test_dir = "./data/food101/test/"
        train_dset = datasets.ImageFolder(train_dir)
        test_dset = datasets.ImageFolder(test_dir)

        self.train_data, self.train_targets = split_images_labels_imagenet(train_dset.imgs)
        self.test_data, self.test_targets = split_images_labels_imagenet(test_dset.imgs)
        
        
class CARs(iData):
    def __init__(self, args):
        self.use_path = False
        self.args = args
        self.train_trsf = build_transform(True, None)
        self.test_trsf = build_transform(False, None)
        self.common_trsf = []

        self.class_order = np.arange(196).tolist()

    def download_data(self):
        # assert 0, "You should specify the folder of your dataset"
        train_dir = "./data/CARS/train/"
        test_dir = "./data/CARS/test/"
        train_dset = datasets.ImageFolder(train_dir)
        test_dset = datasets.ImageFolder(test_dir)

        self.train_data, self.train_targets = split_images_labels_imagenet(train_dset.imgs)
        self.test_data, self.test_targets = split_images_labels_imagenet(test_dset.imgs)


from PIL import Image
from tqdm import tqdm
import time
from multiprocessing import Pool
from multiprocessing.dummy import Pool as ThreadPool
from functools import partial

mode = '1'

def get_pil_img(nimg, size):
    with open(nimg[0], "rb") as f:
        img = Image.open(f).convert("RGB")
        if size is not None:
            img = np.array(img.resize((size, size)))
        else:
            img = np.array(img)
    label = nimg[1]
    return img, label


def split_images_labels_imagenet(imgs, size=None):
    # split trainset.imgs in ImageFolder
    if mode == '0':
        t0 = time.time()
        images = []
        labels = []
        for item in tqdm(imgs):
            with open(item[0], "rb") as f:
                img = Image.open(f).convert("RGB")
                if size is not None:
                    img = np.array(img.resize((size, size)))
                else:
                    img = np.array(img)
            images.append(img)
            labels.append(item[1])
        t1 = time.time()
        print("for loop takes %.2f s" % (t1 - t0))
    if mode == '1':
        t2 = time.time()
        pfunc = partial(get_pil_img, size=size)
        pool = Pool(processes=12)
        results = pool.map(pfunc, imgs)
        pool.close()
        pool.join()
        images, labels = zip(*results)
        t3 = time.time()
        print("Pool takes %.2f s" % (t3 - t2))
    if mode == '2':
        t4 = time.time()
        pfunc = partial(get_pil_img, size=size)
        pool = ThreadPool(processes=4)
        results = pool.map(pfunc, imgs)
        pool.close()
        pool.join()
        images, labels = zip(*results)
        t5 = time.time()
        print("ThreadPool takes %.2f s" % (t5 - t4))
        
    n_labels = np.array(labels)
    print("Finish")
    return images, n_labels
