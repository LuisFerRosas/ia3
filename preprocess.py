import hyperparams as hp

from torch.utils.data import Dataset, DataLoader
import os
import librosa
import numpy as np

import collections
from scipy import signal
import torch as t
import math
from nuevoPreproseced import sequence_partitura,cargarVocabulario


class DatasetsCompleto(Dataset):
    """LJSpeech dataset."""

    def __init__(self, pathAudio, pathPartituras,pathVocabulario,pathAP):
        """
        Args:
            pathAudio (string): Path del directorio de audios.
            root_dir (string): Directory with all the wavs.

        """
        self.pathAudioProcesado=pathAP
        self.nombreAudios = os.listdir(pathAudio)
        self.pathPartituras = pathPartituras
        self.vocabPartitura =cargarVocabulario(pathVocabulario=pathVocabulario)
        
        
    def load_wav(self, filename):
        return librosa.load(filename, sr=hp.sample_rate)

    def __len__(self):
        return len(self.nombreAudios)

    def __getitem__(self, idx):
        wav_name = self.nombreAudios[idx]
        

        text = np.asarray(sequence_partitura(self.pathPartituras+'/'+wav_name[:-3]+'xml',self.vocabPartitura), dtype=np.int32)
        mel = np.load(self.pathAudioProcesado+ wav_name[:-4] + '.pt.npy')
        mel_input = np.concatenate([np.zeros([1,hp.num_mels], np.float32), mel[:-1,:]], axis=0)
        text_length = len(text)
        pos_text = np.arange(1, text_length + 1)
        pos_mel = np.arange(1, mel.shape[0] + 1)

        sample = {'text': text, 'mel': mel, 'text_length':text_length, 'mel_input':mel_input, 'pos_mel':pos_mel, 'pos_text':pos_text}

        return sample
    

    
def collate_fn_transformer(batch):

    # Puts each data field into a tensor with outer dimension batch size
    if isinstance(batch[0], collections.Mapping):

        text = [d['text'] for d in batch]
        mel = [d['mel'] for d in batch]
        mel_input = [d['mel_input'] for d in batch]
        text_length = [d['text_length'] for d in batch]
        pos_mel = [d['pos_mel'] for d in batch]
        pos_text= [d['pos_text'] for d in batch]
        
        text = [i for i,_ in sorted(zip(text, text_length), key=lambda x: x[1], reverse=True)]
        mel = [i for i, _ in sorted(zip(mel, text_length), key=lambda x: x[1], reverse=True)]
        mel_input = [i for i, _ in sorted(zip(mel_input, text_length), key=lambda x: x[1], reverse=True)]
        pos_text = [i for i, _ in sorted(zip(pos_text, text_length), key=lambda x: x[1], reverse=True)]
        pos_mel = [i for i, _ in sorted(zip(pos_mel, text_length), key=lambda x: x[1], reverse=True)]
        text_length = sorted(text_length, reverse=True)
        # PAD sequences with largest length of the batch
        text = _prepare_data(text).astype(np.int32)
        mel = _pad_mel(mel)
        mel_input = _pad_mel(mel_input)
        pos_mel = _prepare_data(pos_mel).astype(np.int32)
        pos_text = _prepare_data(pos_text).astype(np.int32)


        return t.LongTensor(text), t.FloatTensor(mel_input), t.LongTensor(pos_text), t.LongTensor(pos_mel), t.LongTensor(text_length)

    raise TypeError(("batch must contain tensors, numbers, dicts or lists; found {}"
                     .format(type(batch[0]))))
    
def collate_fn_postnet(batch):

    # Puts each data field into a tensor with outer dimension batch size
    if isinstance(batch[0], collections.Mapping):

        mel = [d['mel'] for d in batch]
        mag = [d['mag'] for d in batch]
        
        # PAD sequences with largest length of the batch
        mel = _pad_mel(mel)
        mag = _pad_mel(mag)

        return t.FloatTensor(mel), t.FloatTensor(mag)

    raise TypeError(("batch must contain tensors, numbers, dicts or lists; found {}"
                     .format(type(batch[0]))))

def _pad_data(x, length):
    _pad = 0
    return np.pad(x, (0, length - x.shape[0]), mode='constant', constant_values=_pad)

def _prepare_data(inputs):
    max_len = max((len(x) for x in inputs))
    return np.stack([_pad_data(x, max_len) for x in inputs])

def get_param_size(model):
    params = 0
    for p in model.parameters():
        tmp = 1
        for x in p.size():
            tmp *= x
        params += tmp
    return params

def get_dataset():
    return DatasetsCompleto(hp.path_audio,hp.path_partituras,hp.path_vocabularioPartitura,hp.path_audio_procesado)



def _pad_mel(inputs):
    _pad = 0
    def _pad_one(x, max_len):
        mel_len = x.shape[0]
        return np.pad(x, [[0,max_len - mel_len],[0,0]], mode='constant', constant_values=_pad)
    max_len = max((x.shape[0] for x in inputs))
    return np.stack([_pad_one(x, max_len) for x in inputs])

