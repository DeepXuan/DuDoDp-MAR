import math
import random

from PIL import Image
import blobfile as bf
from mpi4py import MPI
import numpy as np
from torch.utils.data import DataLoader, Dataset, Sampler
import imageio.v2 as imageio
import cv2
from geometry.misc import get_config
from geometry.build_gemotry import initialization, imaging_geo
from torch_radon.radon import FanBeam
import torch
import os

class SinoImageSampler(Sampler):
    def __init__(self, dataset, batch_size):
        super(Sampler, self).__init__()
        self.dataset = dataset
        self.batch_size = batch_size	
        self.leng = len(dataset)
        self.indices = range(len(dataset))	
        self.count = int(len(dataset) / self.batch_size)

    def __iter__(self):
        list_count = list(range((self.leng-1) // 4096 + 1))
        for i in range(len(list_count)):
            n = random.sample(list_count, 1)[0]
            list_count.remove(n)
            start = n*4096
            end = (n+1)*4096 if (n+1)*4096 < self.leng else self.leng
            list_minicount = list(range(end-start))
            for j in range((end-start-1) // self.batch_size + 1):
                indices = random.sample(list_minicount, min(self.batch_size, len(list_minicount)))
                for k in range(min(self.batch_size, len(list_count))):
                    list_minicount.remove(indices[k])
                indices = [x+start for x in indices]
                # print(indices)
                yield indices
                
    def __len__(self):
        return self.count
    
class SinoSampler(Sampler):
    def __init__(self, dataset, batch_size):
        super(Sampler, self).__init__()
        self.dataset = dataset
        self.batch_size = batch_size	
        self.leng = len(dataset)
        self.indices = range(len(dataset))	
        self.count = int(len(dataset) / self.batch_size)

    def __iter__(self):
        list_count = list(range((self.leng-1) // (4096*640) + 1))
        for i in range(len(list_count)):
            n = random.sample(list_count, 1)[0]
            list_count.remove(n)
            start = n*(4096*640)
            end = (n+1)*(4096*640) if (n+1)*(4096*640) < self.leng else self.leng
            list_minicount = list(range(end-start))
            for j in range((end-start-1) // self.batch_size + 1):
                indices = random.sample(list_minicount, min(self.batch_size, len(list_minicount)))
                for k in range(min(self.batch_size, len(list_count))):
                    list_minicount.remove(indices[k])
                indices = [x+start for x in indices]
                # print(indices)
                yield indices
                
    def __len__(self):
        return self.count

def load_data(
    *,
    dataset_name,
    data_dir,
    batch_size,
    image_size,
    class_cond=False,
    deterministic=False,
    random_crop=False,
    random_flip=True,
):
    """
    For a dataset, create a generator over (images, kwargs) pairs.

    Each images is an NCHW float tensor, and the kwargs dict contains zero or
    more keys, each of which map to a batched Tensor of their own.
    The kwargs dict can be used for class labels, in which case the key is "y"
    and the values are integer tensors of class labels.

    :param data_dir: a dataset directory.
    :param batch_size: the batch size of each returned pair.
    :param image_size: the size to which images are resized.
    :param class_cond: if True, include a "y" key in returned dicts for class
                       label. If classes are not available and this is true, an
                       exception will be raised.
    :param deterministic: if True, yield results in a deterministic order.
    :param random_crop: if True, randomly crop the images for augmentation.
    :param random_flip: if True, randomly flip the images for augmentation.
    """
    if not data_dir:
        raise ValueError("unspecified data directory")
    all_files = _list_image_files_recursively(data_dir)
    print(len(all_files))
    with open('/media/siat04/硬盘/codes/PatchDiffusion-Pytorch-main/test_640geo_dir.txt', 'r') as f:
        lines = f.readlines()
        for l in lines:
            # print(data_dir+l[:-7]+'.png')
            if data_dir+l[:-7]+'.png' in all_files:
                all_files.remove(data_dir+l[:-7]+'.png')
    print(len(all_files))
    classes = None
    if class_cond:
        # Assume classes are the first part of the filename,
        # before an underscore.
        classes = [bf.basename(path).split("_")[0] for path in all_files]
    
    if dataset_name == 'DeepLesion':
        dataset = DeepLesionDataset(
            image_size,
            all_files,
            classes=classes,
            shard=MPI.COMM_WORLD.Get_rank(),
            num_shards=MPI.COMM_WORLD.Get_size(),
            random_crop=random_crop,
            random_flip=random_flip,
        )
    elif dataset_name == 'DeepLesionSino':
        dataset = DeepLesionSinoDataset(
            image_size,
            data_dir,
            classes=classes,
            shard=MPI.COMM_WORLD.Get_rank(),
            num_shards=MPI.COMM_WORLD.Get_size(),
            random_crop=random_crop,
            random_flip=random_flip,
        )
    elif dataset_name == 'DeepLesionSinoImage':
        dataset = DeepLesionSinoImageDataset(
            image_size,
            data_dir,
            random_crop=random_crop,
            random_flip=random_flip,
        )
    else: 
        dataset = ImageDataset(
        image_size,
        all_files,
        classes=classes,
        shard=MPI.COMM_WORLD.Get_rank(),
        num_shards=MPI.COMM_WORLD.Get_size(),
        random_crop=random_crop,
        random_flip=random_flip,
    )
    if dataset_name=="DeepLesionSinoImage":
        sampler = SinoImageSampler(dataset, batch_size)
        loader = DataLoader(dataset, sampler=sampler, num_workers=1)
    elif dataset_name=="DeepLesionSino":
        sampler = SinoSampler(dataset, batch_size)
        loader = DataLoader(dataset, sampler=sampler, num_workers=1)
    elif deterministic:
        loader = DataLoader(
            dataset, batch_size=batch_size, shuffle=False, num_workers=1, drop_last=True
        )
    else:
        loader = DataLoader(
            dataset, batch_size=batch_size, shuffle=True, num_workers=1, drop_last=True
        )
    while True:
        yield from loader


def _list_image_files_recursively(data_dir):
    results = []
    for entry in sorted(bf.listdir(data_dir)):
        full_path = bf.join(data_dir, entry)
        ext = entry.split(".")[-1]
        if "." in entry and ext.lower() in ["jpg", "jpeg", "png", "gif"]:
            results.append(full_path)
        elif bf.isdir(full_path):
            results.extend(_list_image_files_recursively(full_path))
    return results


class ImageDataset(Dataset):
    def __init__(
        self,
        resolution,
        image_paths,
        classes=None,
        shard=0,
        num_shards=1,
        random_crop=False,
        random_flip=True,
    ):
        super().__init__()
        self.resolution = resolution
        self.local_images = image_paths[shard:][::num_shards]
        self.local_classes = None if classes is None else classes[shard:][::num_shards]
        self.random_crop = random_crop
        self.random_flip = random_flip

    def __len__(self):
        return len(self.local_images)

    def __getitem__(self, idx):
        path = self.local_images[idx]
        with bf.BlobFile(path, "rb") as f:
            pil_image = Image.open(f)
            pil_image.load()
        pil_image = pil_image.convert("RGB")

        if self.random_crop:
            arr = random_crop_arr(pil_image, self.resolution)
        else:
            arr = center_crop_arr(pil_image, self.resolution)

        if self.random_flip and random.random() < 0.5:
            arr = arr[:, ::-1]

        arr = arr.astype(np.float32) / 127.5 - 1

        out_dict = {}
        if self.local_classes is not None:
            out_dict["y"] = np.array(self.local_classes[idx], dtype=np.int64)
        return np.transpose(arr, [2, 0, 1]), out_dict
    

class DeepLesionDataset(Dataset):
    def __init__(
        self,
        resolution,
        image_paths,
        classes=None,
        shard=0,
        num_shards=1,
        random_crop=False,
        random_flip=True,
    ):
        super().__init__()
        self.resolution = resolution
        self.local_images = image_paths[shard:][::num_shards]
        self.local_classes = None if classes is None else classes[shard:][::num_shards]
        self.random_crop = random_crop
        self.random_flip = random_flip

    def __len__(self):
        return len(self.local_images)

    def __getitem__(self, idx):
        path = self.local_images[idx]
     
        img = imageio.imread(path) 
        img = img.astype(np.float16) - 32768.
 
        img = np.clip(img, -1000, 2500)
        img = (img + 1000.) / 5208. * 255.

        img = cv2.resize(img, (self.resolution, self.resolution), interpolation=cv2.INTER_LINEAR)

        img = np.expand_dims(img, axis=-1)

        # if self.random_crop:
        #     arr = random_crop_arr(img, self.resolution)
        # else:
        #     arr = center_crop_arr(img, self.resolution)

        # if self.random_flip and random.random() < 0.5:
        #     img = img[::-1, :]

        img = img.astype(np.float32) / 127.5 - 1

        out_dict = {}
        if self.local_classes is not None:
            out_dict["y"] = np.array(self.local_classes[idx], dtype=np.int64)
        return np.transpose(img, [2, 0, 1]), out_dict
    
class DeepLesionSinoImageDataset(Dataset):
    def __init__(
        self,
        resolution,
        image_paths,
        random_crop=False,
        random_flip=True,
    ):
        super().__init__()
        self.resolution = resolution
        self.random_crop = random_crop
        self.random_flip = random_flip
        self.path = image_paths
        self.npy_files = sorted(os.listdir(self.path))

    def __len__(self):
        return 925696

    def __getitem__(self, idx):
        imgs = []
        n = idx[0] // 4096
        npy_path = self.npy_files[n]
        sino_npy = np.load(os.path.join(self.path, npy_path))
        for i in idx:
            img = sino_npy[i % 4096, ...]
            img = img.astype(np.float16) / 65535. * 255.
            # img = cv2.resize(img, (640, 640), interpolation=cv2.INTER_LINEAR)
            img = np.expand_dims(img, axis=0)

        # if self.random_crop:
        #     arr = random_crop_arr(img, self.resolution)
        # else:
        #     arr = center_crop_arr(img, self.resolution)

        # if self.random_flip and random.random() < 0.5:
        #     img = img[::-1, :]

            img = img.astype(np.float32) / 127.5 - 1
            imgs.append(img)
        img = np.stack(imgs, axis=0)
        out_dict = {}
        # if self.local_classes is not None:
        #     out_dict["y"] = np.array(self.local_classes[idx], dtype=np.int64)
        return img, out_dict
    
class DeepLesionSinoImageDataset(Dataset):
    def __init__(
        self,
        resolution,
        image_paths,
        random_crop=False,
        random_flip=True,
    ):
        super().__init__()
        self.resolution = resolution
        self.random_crop = random_crop
        self.random_flip = random_flip
        self.path = image_paths
        self.npy_files = sorted(os.listdir(self.path))

    def __len__(self):
        return 925696

    def __getitem__(self, idx):
        imgs = []
        n = idx[0] // 4096
        npy_path = self.npy_files[n]
        sino_npy = np.load(os.path.join(self.path, npy_path))
        for i in idx:
            img = sino_npy[i % 4096, ...]
            img = img.astype(np.float16) / 65535. * 255.
            # img = cv2.resize(img, (640, 640), interpolation=cv2.INTER_LINEAR)
            img = np.expand_dims(img, axis=0)

        # if self.random_crop:
        #     arr = random_crop_arr(img, self.resolution)
        # else:
        #     arr = center_crop_arr(img, self.resolution)

        # if self.random_flip and random.random() < 0.5:
        #     img = img[::-1, :]

            img = img.astype(np.float32) / 127.5 - 1
            imgs.append(img)
        img = np.stack(imgs, axis=0)
        out_dict = {}
        # if self.local_classes is not None:
        #     out_dict["y"] = np.array(self.local_classes[idx], dtype=np.int64)
        return img, out_dict
    
class DeepLesionSinoDataset(Dataset):
    def __init__(
        self,
        resolution,
        image_paths,
        classes=None,
        shard=0,
        num_shards=1,
        random_crop=False,
        random_flip=True,
    ):
        super().__init__()
        self.resolution = resolution
        self.random_crop = random_crop
        self.random_flip = random_flip
        self.path = image_paths
        self.local_classes = None if classes is None else classes[shard:][::num_shards],
        self.npy_files = sorted(os.listdir(self.path))

    def __len__(self):
        return 925696 * 640

    def __getitem__(self, idx):
        imgs = []
        views = []
        n = idx[0] // (4096*640)
        npy_path = self.npy_files[n]
        sino_npy = np.load(os.path.join(self.path, npy_path))
        for i in idx:
            view = i % (4096*640) % 640
            img = sino_npy[i % (4096*640) // 640, view, :]
            img = img.astype(np.float16) / 65535. * 255.
            img = np.expand_dims(img, axis=0)
            view = np.array((np.sin(view/640*2*np.pi), np.cos(view/640*2*np.pi)))

        # if self.random_crop:
        #     arr = random_crop_arr(img, self.resolution)
        # else:
        #     arr = center_crop_arr(img, self.resolution)

        # if self.random_flip and random.random() < 0.5:
        #     img = img[::-1, :]

            img = img.astype(np.float32) / 127.5 - 1
            imgs.append(img)
            views.append(view)
        img = np.stack(imgs, axis=0)
        view = np.stack(views, axis=0)
        out_dict = {}
        
        out_dict["view"] = view.astype(np.float32)
        return img, out_dict

def center_crop_arr(pil_image, image_size):
    # We are not on a new enough PIL to support the `reducing_gap`
    # argument, which uses BOX downsampling at powers of two first.
    # Thus, we do it by hand to improve downsample quality.
    while min(*pil_image.size) >= 2 * image_size:
        pil_image = pil_image.resize(
            tuple(x // 2 for x in pil_image.size), resample=Image.BOX
        )

    scale = image_size / min(*pil_image.size)
    pil_image = pil_image.resize(
        tuple(round(x * scale) for x in pil_image.size), resample=Image.BICUBIC
    )

    arr = np.array(pil_image)
    crop_y = (arr.shape[0] - image_size) // 2
    crop_x = (arr.shape[1] - image_size) // 2
    return arr[crop_y : crop_y + image_size, crop_x : crop_x + image_size]


def random_crop_arr(pil_image, image_size, min_crop_frac=0.8, max_crop_frac=1.0):
    min_smaller_dim_size = math.ceil(image_size / max_crop_frac)
    max_smaller_dim_size = math.ceil(image_size / min_crop_frac)
    smaller_dim_size = random.randrange(min_smaller_dim_size, max_smaller_dim_size + 1)

    # We are not on a new enough PIL to support the `reducing_gap`
    # argument, which uses BOX downsampling at powers of two first.
    # Thus, we do it by hand to improve downsample quality.
    while min(*pil_image.size) >= 2 * smaller_dim_size:
        pil_image = pil_image.resize(
            tuple(x // 2 for x in pil_image.size), resample=Image.BOX
        )

    scale = smaller_dim_size / min(*pil_image.size)
    pil_image = pil_image.resize(
        tuple(round(x * scale) for x in pil_image.size), resample=Image.BICUBIC
    )

    arr = np.array(pil_image)
    crop_y = random.randrange(arr.shape[0] - image_size + 1)
    crop_x = random.randrange(arr.shape[1] - image_size + 1)
    return arr[crop_y : crop_y + image_size, crop_x : crop_x + image_size]
