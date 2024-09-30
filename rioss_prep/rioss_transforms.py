import torch
from torchvision.transforms import Normalize

class ToBinaryTransformd:
    def __init__(self, keys, label_num=1):
        self.keys = keys
        self.label_num = label_num


    def __call__(self, sample):
        for key in self.keys:
            sample[key] = torch.where(
                sample[key] == self.label_num, 
                torch.tensor(1, dtype=sample[key].dtype), 
                torch.tensor(0, dtype=sample[key].dtype)
            )
        return sample
    
class EnsureChanellFirstd:
    def __init__(self, keys):
        self.keys = keys
    
    def __call__(self, sample):
        for key in self.keys:
            sample[key] = sample[key].reshape(-1, sample[key].shape[-2], sample[key].shape[-1])
        return sample
    
class Downsampled:
    def __init__(self, keys):
        self.keys = keys
    
    def __call__(self, sample):
        for key in self.keys:
            sample[key] = sample[key].reshape(-1, *sample[key].shape)
        return sample
    

class PrintSampled:
    def __init__(self, keys):
        self.keys = keys
    
    def __call__(self, sample):
        for key in self.keys:
            print(sample[key])
        return sample


    

   

