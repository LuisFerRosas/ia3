import os
import music21 as m21
import json
import librosa
import numpy as np
from scipy.signal.signaltools import resample
from scipy.io.wavfile import write
import sys
DATASETPATH="dataset audio"

# for root, dirs, files in os.walk("dataset audio/",topdown=False ):
#     for name in files:
#         print("archivos") 
#         print(os.path.join(root, name))
#     for name in dirs:
#         print("directorios")   
#         print(os.path.join(root, name))
        
        
def load_file(filepath):
    y2, sr = librosa.load(filepath,sr=44100)
    return y2,sr
    

def spectogram_stft(waveform , hop_length,n_fft):
    caracteristica=librosa.core.stft(y=waveform,hop_length=hop_length,n_fft=n_fft)
    return abs(caracteristica)


def spectogram_mel(wavefor,sr):
    caracteristica= librosa.feature.melspectrogram(y=wavefor,sr=sr,)
    return abs(caracteristica)


def load_songs(dataset_path):
    
    songs =[]
    dato={};

    # go through all the files in dataset and load them with music21
    for path, subdirs, files in os.walk(dataset_path,topdown=False):
        for file in files:

            # consider only kern files
            if file[-3:] == "wav":
                #print(file)
                y,sr = load_file(os.path.join(path,file))
                resampled=librosa.resample(y,sr,1200)
                write('resample.wav',1200,resampled)
                print(y.shape)
                # wave= spectogram_stft(y,hop_length=1024,n_fft=4096)
                wave=spectogram_mel(wavefor=resampled,sr=1200)
                print(wave.shape)
                # songs.append(wave)
                wave2= np.ravel(np.array(wave))# aplanado
                
                # faltante=3000000- len(wave2)
                # ceros=np.zeros(faltante)
                # wave3=np.append(wave2,ceros)
                
                
                dato[file]=wave2.tolist();
                
                #print(len(wave3))
                #songs.append(wave4)
                
    return dato

#///////////////////////////////////////////
# cargado y guardado de audios
# songs= load_songs(DATASETPATH)

# np.save("sonidos.npy",songs)
# for nombre in songs:
#     print(nombre);
#     print(len(songs[nombre]))
#///////////////////////////////////////////
# newDIc=np.load("sonidos.npy",allow_pickle=True).item()
# print(newDIc)



def load_partituras(dataset_path):
    """Loads all kern pieces in dataset using music21.

    :param dataset_path (str): Path to dataset
    :return songs (list of m21 streams): List containing all pieces
    """
    songs = []
    nombres=[]
    # go through all the files in dataset and load them with music21
    for path, subdirs, files in os.walk(dataset_path,topdown=False):
        for file in files:

            # consider only kern files
            if file[-3:] == "xml":
                print(file)
                nombres.append(file)
                try:
                    song = m21.converter.parse(os.path.join(path, file))
                except OSError as err:
                    print("OS error: {0}".format(err))
                except ValueError:
                    print("Could not convert data to an integer.")
                except:
                    print("Unexpected error:", sys.exc_info()[0])
                    
                
                songs.append(song)
    return songs,nombres



def encode_song(song):
    encoded_song = []
   
    liga="";

    for event in song.flat.notesAndRests:

        # handle notes
        if isinstance(event, m21.note.Note):
            symbol = event.pitch.midi # 60
            con = event.tie
            if(con==None):
                # print(con)
                liga=con
            else:
            #    print(con.type) 
               liga=con.type
            
            
            # print(symbol)
        # handle rests
        elif isinstance(event, m21.note.Rest):
            symbol = "r"

        # convert the note/rest into time series notation
        duracionNota = event.duration.quarterLength 
        
        encoded_song.append(symbol)
        encoded_song.append(duracionNota)
        encoded_song.append(liga)
        encoded_song.append("/")
      

    # cast encoded song to str
    encoded_song = " ".join(map(str, encoded_song))

    return encoded_song+" "


def generarVocabulario(partes,pathSave):
    todo=""
    for unaparte in partes:
        decodePartitura= encode_song(unaparte)
        todo=todo+decodePartitura
    partido=todo.split(" / ")
    partido.pop()
    partido=list(set(partido))
    partido.insert(0,"<eos>")
    partido.insert(0,"<sos>")
    partido.insert(0,"<pad>")
    # print(partido)
    mappings = {}
    # create mappings
    for i, symbol in enumerate(partido):
        mappings[symbol] = i

    # save voabulary to a json file
    with open(pathSave+"vocabulario2.json", "w") as fp:
        json.dump(mappings, fp, indent=4)
    
    
def generarDic(partes,nombrefile):
    completo={}
    dato=[]
    with open("vocabulario2.json", "r") as fp:
        mappings = json.load(fp)
        
    for i, unaparte in enumerate(partes):
        decodePartitura= encode_song(unaparte)
        partido=decodePartitura.split(" / ")
        partido.pop()
        dato.append(mappings["<sos>"])
        for symbol in partido:
            dato.append(mappings[symbol])
        dato.append(mappings["<eos>"])
        completo[nombrefile[i]]=dato
        dato=[]
    # np.save("partituras.npy",completo)
    # print(completo)
    
        
def sequence_partitura(pathPartitura,vocabulario):
    dato=[]
    song = m21.converter.parse(pathPartitura)
    decodePartitura=encode_song(song)
    partido=decodePartitura.split(" / ")
    partido.pop()
    dato.append(vocabulario["<sos>"])
    for symbol in partido:
        dato.append(vocabulario[symbol])
    dato.append(vocabulario["<eos>"])
    
    return dato
    
def cargarVocabulario(pathVocabulario):
    with open(pathVocabulario, "r") as fp:
        mappings = json.load(fp)
    return mappings
    
    
def guardarVocabulario(pathPartituras,pathSave):
    parte, nombres= load_partituras(pathPartituras)
    generarVocabulario(parte,pathSave)

# parte, nombres= load_partituras("datos/partituras")
# generarVocabulario(parte)
# generarDic(parte,nombres)
#print(parte)
if __name__ == '__main__':
    # with open("vocabulario2.json", "r") as fp:
    #     mappings = json.load(fp)
    # partedeco = sequence_partitura("datos/partituras/1.xml",mappings)
    # print(partedeco)
    guardarVocabulario(pathPartituras="datos/partituras",pathSave="datos/")
    