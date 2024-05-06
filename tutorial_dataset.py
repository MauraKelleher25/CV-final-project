import json
import cv2
import numpy as np

from torch.utils.data import Dataset


class MyDataset(Dataset):
    def __init__(self):
        # self.data = []
        # with open('./training/fill50k/prompt.json', 'rt') as f:
        #     for line in f:
        #         self.data.append(json.loads(line))
        self.data = []
        self.target_size = (576, 540)
        with open('training_data.json', 'rt') as f:
            self.data = json.load(f)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]

        source_filename = item['source']
        target_filename = item['target']
        prompt = item['prompt']

        source = cv2.imread('/home/yajvan/CV-final-project/ISIC2017/' + source_filename)
        target = cv2.imread('/home/yajvan/CV-final-project/ISIC2017/' + target_filename)

        print(source.shape) 
        print(target.shape)

        # Resize images to the target size
        source = cv2.resize(source, self.target_size, interpolation=cv2.INTER_AREA)
        target = cv2.resize(target, self.target_size, interpolation=cv2.INTER_AREA)

        # Check and convert images to RGB if not already
        if source.shape[2] == 1:
            print('statement entered')
            source = cv2.cvtColor(source, cv2.COLOR_GRAY2RGB)
    
        # Convert BGR to RGB
        source = cv2.cvtColor(source, cv2.COLOR_BGR2RGB)
        target = cv2.cvtColor(target, cv2.COLOR_BGR2RGB)
    
        # Normalize source images to [0, 1].
        source = source.astype(np.float32) / 255.0
    
        # Normalize target images to [-1, 1].
        target = (target.astype(np.float32) / 127.5) - 1.0
    
        return dict(jpg=target, txt=prompt, hint=source)

