import os
import hyperparams as hp
import numpy as np
from utils import get_spectrograms
def generarDatos(dataset_path):
    
    for path, subdirs, files in os.walk(dataset_path,topdown=False):
        for file in files:

            # consider only kern files
            if file[-3:] == "wav":
                mel, mag =get_spectrograms(os.path.join(dataset_path,file) )
                np.save("datos/audioProcesado/"+file[:-4] + '.pt', mel)
                # np.save("datos/audioProcesado/"+file[:-4] + '.mag', mag)




if __name__ == '__main__':
    generarDatos("datos/audio")