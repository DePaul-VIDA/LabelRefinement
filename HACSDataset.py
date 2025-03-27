import numpy
import torch
import torchvision
import cv2
from filedata import FileData
from fileloader import load_people
from torch.utils.data import Dataset, DataLoader
from mmflow.apis import inference_model, init_model
import numpy as np
from random import randint

import json
from abc import abstractmethod

IMAGE_WIDTH = 640

BACKGROUND = 0

class Sample:
    def __init__(self, video_link):
        self.video_link = video_link
        self.start_time
        self.end_time
        self.label
        self.start_frame
        self.end_frame

    def get_tIoU(self, start_frame, end_frame):
        union = float(max(end_frame, self.end_frame) - min(self.start_frame, start_frame))
        intersection = float(min(end_frame, self.end_frame) - max(self.start_frame, start_frame))
        return intersection / union



class OnlineDatasetLoader():

    def __init__(self, file_name):
        self.file_name = file_name
        self.training_samples, self.testing_samples = self.load_file()
        self.n_samples = len(self.training_samples) + len(self.testing_samples)

    def parse_samples(self, data):
        #return training, testing, validation samples
        pass

    def load_file(self):
        f = open(self.file_name)
        data = json.load(f)
        f.close()
        return self.parse_samples(data)

class OnlineDataset(Dataset):
    def __len__(self):
        self.n_samples

    def __getitem__(self, index):
        pass

class HACSDatasetLoader(OnlineDatasetLoader):
    def __init__(self, file_name):
        super.__init__(file_name)

    def parse_samples(self, data):
        return [], [], []


class HACSDatasetSignals(OnlineDataset):
     def __init__(self, vids, device, window_size, unde_sample):
         super.__init__()
         self.vids = vids

     def __getitem__(self, index):
        return []

     def __len__(self):
         return self.n_samples


class HACSDatasetFrames(OnlineDataset):
     def __init__(self, vids, device, window_size, unde_sample):
         super.__init__()
         self.vids = vids

     def __getitem__(self, index):
        return []

     def __len__(self):
         return self.n_samples
