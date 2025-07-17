import torch
import numpy as np
import os
from torch.utils.data import Dataset
from torchvision.datasets import ImageFolder
from PIL import Image
from tqdm import tqdm
import torch.distributed as dist
from multiprocessing import Manager
from concurrent.futures import ProcessPoolExecutor
import io

class CustomDataset(Dataset):
    def __init__(self, feature_dir, label_dir):
        self.feature_dir = feature_dir
        self.label_dir = label_dir
        self.flip = 'flip' in self.feature_dir

        aug_feature_dir = feature_dir.replace('ten_crop/', 'ten_crop_105/')
        aug_label_dir = label_dir.replace('ten_crop/', 'ten_crop_105/')
        if os.path.exists(aug_feature_dir) and os.path.exists(aug_label_dir):
            self.aug_feature_dir = aug_feature_dir
            self.aug_label_dir = aug_label_dir
        else:
            self.aug_feature_dir = None
            self.aug_label_dir = None

        # self.feature_files = sorted(os.listdir(feature_dir))
        # self.label_files = sorted(os.listdir(label_dir))
        # TODO: make it configurable
        self.feature_files = [f"{i}.npy" for i in range(1281167)]
        self.label_files = [f"{i}.npy" for i in range(1281167)]

    def __len__(self):
        assert len(self.feature_files) == len(self.label_files), \
            "Number of feature files and label files should be same"
        return len(self.feature_files)

    def __getitem__(self, idx):
        if self.aug_feature_dir is not None and torch.rand(1) < 0.5:
            feature_dir = self.aug_feature_dir
            label_dir = self.aug_label_dir
        else:
            feature_dir = self.feature_dir
            label_dir = self.label_dir
                   
        feature_file = self.feature_files[idx]
        label_file = self.label_files[idx]

        features = np.load(os.path.join(feature_dir, feature_file))
        if self.flip:
            aug_idx = torch.randint(low=0, high=features.shape[1], size=(1,)).item()
            features = features[:, aug_idx]
        labels = np.load(os.path.join(label_dir, label_file))
        return torch.from_numpy(features), torch.from_numpy(labels)
    
def get_ddp_info():
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        rank = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
    else:
        rank, world_size, local_rank = 0, 1, 0
    return rank, world_size, local_rank

def _load_image_bytes(path_label):
    path, label = path_label
    with open(path, 'rb') as f:
        img_bytes = f.read()
    return (img_bytes, label)

def _parallel_load(samples, num_workers=64):
    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        results = list(tqdm(executor.map(_load_image_bytes, samples), total=len(samples)))
    return results

class SharedImageFolder(Dataset):
    def __init__(self, root, transform=None, num_workers=64):
        self.transform = transform
        self.rank, self.world_size, self.local_rank = get_ddp_info()

        self.manager = Manager()
        self.shared_list = self.manager.list()

        if self.rank == 0:
            print(f"[Rank 0] Preloading data from {root}")
            dataset = ImageFolder(root)
            self.classes = dataset.classes
            self.class_to_idx = dataset.class_to_idx
            loaded = _parallel_load(dataset.samples, num_workers=num_workers)
            self.shared_list.extend(loaded)
            print(f"[Rank 0] Preload complete: {len(self.shared_list)} samples")
        if self.world_size > 1:
            dist.barrier()  # wait for preload

        # read metadata from shared list
        while len(self.shared_list) == 0:
            pass  # wait for rank 0 to fill list
        self.samples = self.shared_list

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        img_bytes, label = self.samples[index]
        image = Image.open(io.BytesIO(img_bytes)).convert('RGB')
        if self.transform:
            image = self.transform(image)
        return image, label


def build_imagenet(args, transform):
    return ImageFolder(args.data_path, transform=transform)
    # return SharedImageFolder(args.data_path, transform=transform)

def build_imagenet_code(args):
    feature_dir = f"{args.code_path}/imagenet{args.image_size}_codes"
    label_dir = f"{args.code_path}/imagenet{args.image_size}_labels"
    assert os.path.exists(feature_dir) and os.path.exists(label_dir), \
        f"please first run: bash scripts/autoregressive/extract_codes_c2i.sh ..."
    return CustomDataset(feature_dir, label_dir)