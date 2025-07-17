from ffcv.writer import DatasetWriter
from ffcv.fields import RGBImageField, IntField
from torchvision.datasets import ImageFolder
from torchvision import transforms
from torch.utils.data import DataLoader
from tqdm import tqdm

import os

# ==== Step 1: 定义数据路径和输出路径 ====
DATASET_PATH = '/path/to/imagenet/train'   # 原始ImageFolder格式的ImageNet路径
OUTPUT_PATH = '/path/to/imagenet_train.beton'  # 输出的FFCV文件

# ==== Step 2: 加载ImageFolder数据 ====
transform = transforms.Compose([
    transforms.PILToTensor(),  # 保留原图像大小，不进行resize
])

dataset = ImageFolder(DATASET_PATH, transform=transform)

# ==== Step 3: 设置字段（RGB图像、标签） ====
writer = DatasetWriter(
    OUTPUT_PATH,
    {
        'image': RGBImageField(write_mode='smart', max_resolution=None),  # 保留原图大小
        'label': IntField()
    }
)

# ==== Step 4: 写入beton文件 ====
writer.from_indexed_dataset(dataset)
