import os
import numpy as np
from scipy import signal
import soundfile as sf

# Directories of .wav files ending with '/'
anechoic_dir = '/Volumes/Material/Eze/Drive/Tesis-Ezequiel/00-Audios/anechoic/'
rir_dir = '/Volumes/Material/Eze/Drive/Tesis-Ezequiel/00-Audios/rir/'
# Search for .wav files in the given directories and build the filenames list
anechoic_filenames = [_ for _ in os.listdir(anechoic_dir) if _.endswith(".wav")]
rir_filenames = [_ for _ in os.listdir(rir_dir) if _.endswith(".wav")]

output_dir = '/Users/eze/Desktop/untitledfolder/'

sr = int(16e3)

for i in range(len(anechoic_filenames)):
    anechoic_fn = anechoic_filenames[i]
    # reads wav file and converts to float64
    anechoic, _ = sf.read(anechoic_dir + anechoic_fn)
#    anechoicL = len(anechoic)

    for j in range(len(rir_filenames)):
        rir_fn = rir_filenames[j]
        # reads wav file and converts to float64
        rir, _ = sf.read(rir_dir + rir_fn)
        #rirL = len(rir)
        # zero pad to have length rir+anechoic-1
        #rir = np.append(rir, anechoicL-1)
        # zero pad to have length rir+anechoic-1
        #anechoic = np.append(anechoic, rirL-1)
        conv = signal.convolve(anechoic, rir, method='auto')

        output_fn = anechoic_fn.rstrip('.wav') + '-' + rir_fn
        max_level = max(abs(conv))
        conv = conv / max_level
        sf.write(output_dir + output_fn, conv, sr)
