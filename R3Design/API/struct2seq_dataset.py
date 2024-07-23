import os
import numpy as np
from tqdm import tqdm
import _pickle as cPickle

import torch.utils.data as data
from .utils import cached_property


class Struct2SeqDataset(data.Dataset):
    def __init__(self, path='./',  mode='train'):
        self.path = path
        self.mode = mode
        self.data = self.cache_data[mode]
    
    @cached_property
    def cache_data(self):
        alphabet_set = set(['A', 'U', 'C', 'G'])
        
        with open('/tancheng/experiments/RDesign/train_RFAM.txt', 'r') as file:
    # 读取文件的内容
            data = file.read()
    # 将文本拆分为行
        lines = data.split('\n')
        second_column_data = []

        for line in lines:

            columns = line.split()

            if len(columns) >= 2:
                second_column_data.append(columns[1])
        second_column_data = list(set(second_column_data))
        
        
        if os.path.exists(self.path):
            data_dict = {'train': [], 'val': [], 'test': []}
            for split in ['train', 'val', 'test']:
                data = cPickle.load(open(os.path.join(self.path, split + '_data.pt'), 'rb'))
                for entry in tqdm(data):
                    
                    ## Below is the selection
                    ingore_entry = False
                    for item in second_column_data:
                        if item in entry["name"]:
                            ingore_entry = True
                            break
                    if ingore_entry:
                        continue
                    
                    
                    for key, val in entry['coords'].items():
                        entry['coords'][key] = np.asarray(val)
                    bad_chars = set([s for s in entry['seq']]).difference(alphabet_set)
                    if len(bad_chars) == 0:
                        data_dict[split].append(entry)
            return data_dict
        else:
            raise "no such file:{} !!!".format(self.path)

    def change_mode(self, mode):
        self.data = self.cache_data[mode]
    
    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return self.data[index]