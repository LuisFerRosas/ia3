import numpy as np
import librosa
import os
from pydub import AudioSegment
from pydub.playback import play
from scipy.io.wavfile import write

def resampleAudio(pathInit,pathFinal):
    audio_in_file = pathInit
    audio_out_file = pathFinal

    # create 1 sec of silence audio segment
    one_sec_segment = AudioSegment.silent(duration=15000)  #duration in milliseconds

    #read wav file to an audio segment
    song = AudioSegment.from_wav(audio_in_file)

    #Add above two audio segments    
    final_song =song+ one_sec_segment 

    #Either save modified audio
    final_song.export(audio_out_file, format="wav")

    #Or Play modified audio
    # play(final_song)


def segundosFile(path):
    files= os.listdir(path=path)
    
    for x in files:
      y,sr=librosa.load(os.path.join(path,x))
      seg =librosa.get_duration(y=y,sr=sr)
      if seg>=20:
        print("archivo : "+x+" seg : "+str(seg))
      
    
def ejecutarAlargarAudio(path,path2):
    files= os.listdir(path=path)
    for file in files:
        resampleAudio(os.path.join(path,file),os.path.join(path2,file))
         
def recortarAudio(path,path2):
    files= os.listdir(path=path)
    for file in files:
        y,sr=librosa.load(os.path.join(path,file),44100,duration=20)
        # res= librosa.resample(y=y,orig_sr=44100,target_sr=1300)
        write(os.path.join(path2,file),44100,y)
        print(file)
    
def feature_mfcc(
    waveform, 
    sample_rate,
    n_mfcc = 40,
    fft = 1024,
    winlen = 512,
    window='hamming',
    #hop=256, # increases # of time steps; was not helpful
    mels=128
    ):

    # Compute the MFCCs for all STFT frames 
    # 40 mel filterbanks (n_mfcc) = 40 coefficients
    mfc_coefficients=librosa.feature.mfcc(
        y=waveform, 
        sr=sample_rate, 
        n_mfcc=n_mfcc,
        n_fft=fft, 
        win_length=winlen, 
        window=window, 
        #hop_length=hop, 
        n_mels=mels, 
        fmax=sample_rate/2
        ) 

    return mfc_coefficients
import matplotlib.pyplot as plt
import librosa.display
from sklearn.preprocessing import StandardScaler

def scalerDatos(path):
    waveform=[]
    nombres=[]
    scaler = StandardScaler()
    files= os.listdir(path=path)
    count=0
    for file in files:
        y,sr=librosa.load(os.path.join(path,file),44100)
        mfc=feature_mfcc(y,sr)
        mfc=scaler.fit_transform(mfc)
        mfc= np.expand_dims(mfc,axis = 0)
        waveform.append(mfc)
        nombres.append(file)
        count+=1
        print('\r'+f'Processed {count }/{len(files)} ',end='')
        
    return waveform,nombres
        
def cargarArchivosNPY(path):
    with open(path, 'rb') as f:
        wave = np.load(f) 
        nombres = np.load(f)
        
    print(wave.shape)
    print(len(nombres))
    return wave,nombres


if __name__ == '__main__':
    # segundosFile('datos2/validateAudio/')
    # ejecutarAlargarAudio('datos2/audio','datos2/audio2')
    # recortarAudio('datos2/audio2','datos2/audio3')
    # wave,nombre= scalerDatos('datos2/audio3')
    # with open('archivosnumpy/train_audio_nombre.npy', 'wb') as f:
    #     np.save(f, wave)
    #     np.save(f,nombre)
   
    # with open('archivosnumpy/train_audio_nombre.npy', 'rb') as f:
    #     wave = np.load(f) 
    #     nombres = np.load(f)
        
    # print(wave.shape)
    # print(len(nombres))
    #////////////////////////////////////////////////////////
    #VALIDATE
    # ejecutarAlargarAudio('datos2/validateAudio','datos2/validateAudio2')
    # recortarAudio('datos2/validateAudio2','datos2/validateAudio3')
    # wave,nombre= scalerDatos('datos2/validateAudio3')
    # with open('archivosnumpy/valid_audio_nombre.npy', 'wb') as f:
    #     np.save(f, wave)
    #     np.save(f,nombre)
   
    with open('archivosnumpy/valid_audio_nombre.npy', 'rb') as f:
        wave = np.load(f) 
        nombres = np.load(f)
        
    print(wave.shape)
    print(len(nombres))