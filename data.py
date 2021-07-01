from torch.utils.data import Dataset, DataLoader
import os
import librosa
import numpy as np
import pytorch_lightning as pl

    
class DataPre(pl.LightningDataModule):
    def __init__(self, *args):
        pass
    
    

def leerDatosAudio(path):
    with open(path, 'rb') as f:
        wave = np.load(f) 
        nombres = np.load(f)
        
    print(wave.shape)
    print(len(nombres))
    return wave,nombres