from sklearn.preprocessing import MinMaxScaler, StandardScaler, Normalizer
import librosa
import scipy.io.wavfile as sci_wav  # Open wav files

def rev_crop(reverberant, anechoic_len):
    """ARREGLAR?
    Crop reverberant audios, in such a way that its length is equal to:
    [ length of its corresponding anechoic  + (number_neighbour_frames * win_length) ]. 
    This is performed in order to ensure the same length of anechoic and reverberant STFT
    which will conform the feature pairs to feed the NN
    Note that IF the filenames of the anechoic and reverberant are not related, the function
    will not work. The files used here have the following syntax to ensure an easy association of
    the anechoic file that gave birth to the reverberant version:
    anechoic:       ABCD_##.wav
    reverberant:    ABCD_##-rir-#.#-r#.wav 
    The first 7 characters always correlate thus allowing to map the reverberant with its corresponding
    anechoic file.
    reverberant: audio vector.
    anechoic_len: Length of the anechoic audio file that corresponds to the reverberant audio file
    """
    reverberant_cropped = reverberant[:, 0 : anechoic_len + neighbour_frames ]
    return reverberant_cropped

def rescale(x, flatten, min, max):
    """Scales data to the range given by [min, max]. If x is a matix it scales across the 
    first axis (i.e. for 2D arrays running vertically downwards across rows)
    http://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.MinMaxScaler.html 
    x: numpy array.
    flatten: boolean. If true, it flattens x to 1D, then reshapes back.
    min: float. Minimum number to scale data to. 
    max: float. Maximum number to scale data to. 
    """ 

    if flatten:
        if x.ndim > 1:
            r, c = np.shape(x)
            x = x.flatten()
            #Neccesary because flatten removes dimension and StandardScaler() requires 2dim
            x = x.reshape(-1,1)
            scaler = MinMaxScaler(feature_range=(min, max))
            xrescaled = scaler.fit_transform(x)
            #xrescaled = np.reshape(xrescaled, (r, c))
            xrescaled = xrescaled.reshape(r, c)
            return xrescaled
        else:
            #Necessary because StandardScaler() requires 2dim
            x = x.reshape(-1,1)
            r, c = np.shape(x)
            scaler = MinMaxScaler(feature_range=(min, max))
            xrescaled = scaler.fit_transform(x)
            xrescaled = xrescaled.reshape(r, )
            return xrescaled

    else:
        if x.ndim > 1:
            scaler = MinMaxScaler(feature_range=(min, max))
            xrescaled = scaler.fit_transform(x)
            return xrescaled
        else:
            #Necessary because StandardScaler() requires 2dim
            x = x.reshape(-1,1)
            r, c = np.shape(x)
            scaler = MinMaxScaler(feature_range=(min, max))
            xrescaled = scaler.fit_transform(x)
            xrescaled = xrescaled.reshape(r, )
            return xrescaled

def standardize(x, flatten):
    """Standardize features by removing the mean and scaling to unit variance. If x is a 
    matrix it scales across the first axis (i.e. for 2D arrays running vertically 
    downwards across rows).
    Assumes Gaussian distribution. Data ends with a mean of 0 and a standard deviation of 1.
    http://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.StandardScaler.html
    directory.
    x: numpy array
    flatten: boolean. If true, it flattens x to 1D, then reshapes back.
    """ 
    if flatten:
        if x.ndim > 1:
            r, c = np.shape(x)
            x = x.flatten()
            #Necessary because flatten removes dimension and StandardScaler() requires 2dim
            x = x.reshape(-1,1)
            scaler = StandardScaler().fit(x)
            xstandard = scaler.transform(x)
            xstandard = xstandard.reshape(r, c)
            return xstandard
        else:
            #Necessary because StandardScaler() requires 2dim
            x = x.reshape(-1,1)
            r, c = np.shape(x)
            scaler = StandardScaler().fit(x)
            xstandard = scaler.transform(x)
            xstandard = xstandard.reshape(r, )
            return xstandard

    else:
        if x.ndim > 1:
            scaler = StandardScaler().fit(x)
            xstandard = scaler.transform(x)
            return xstandard
        else:
            #Neccesary because StandardScaler() requires 2dim
            x = x.reshape(-1,1)
            r, c = np.shape(x)
            scaler = StandardScaler().fit(x)
            xstandard = scaler.transform(x)
            xstandard = xstandard.reshape(r, )
            return xstandard
            
def STFTanalysis(x, type_return):
    """Calculates the Short-Time Fourier transform 
    x: file dir
    type_return: String. 'complex', 'mag' or 'phase'. 
    """

    ystft = librosa.stft(x, n_fft=n_fft, hop_length=hop_length, win_length=win_length, window='hamming', center=True, pad_mode='reflect')
    if type_return == 'complex':
        return ystft 
    elif type_return == 'mag':
        ystft, _ = librosa.magphase(ystft)
        return ystft
    elif type_return == 'phase':
        _, ystft = librosa.magphase(ystft)
        #Phase angle in radians
        ystft = np.angle(ystft)
        return ystft

def powerdB(x):
    """Convert a power spectrogram (amplitude squared) to decibel (dB) units
    x: spectrogram
    """    
    #x_dB = 10 * log10(x**2) = 20 log10(x)
    return librosa.power_to_db(x**2)    

def export(export_format, reverberant_fn, fmatrix):

    if export_format == 'JSON':
        output =  output_dir + reverberant_fn.rstrip('.wav') + '.json'
        with open(output, 'w') as f:
            json.dump(fmatrix, f, ensure_ascii=False)

    elif export_format == 'CSV':
        output =  output_dir + reverberant_fn.rstrip('.wav') + '.csv'
        df = pd.DataFrame(fmatrix)
        df.to_csv(output, columns=None, header=False, index=False, mode='w', encoding='utf-8')

def featureMatrix(rev, anec):
    '''
    Building of the feature vectors. The iteration is across the columns. The result is a matrix of
    size [columns, feature_length] = [columns,1799] for the case of 5 neighbor frames,
    which comes from: anechoic_col_n + reverberant_col_n + reverberant_col_n+1 + ... + reverberant_col_n+5
    = 257 + 257*6 for the case of STFT size of 512. 
    The matrix is arranged in a way such that the first 1542 values of each row are reverberant (input). 
    The last 257 anechoic (target).
    rev: Reverberant STFT matrix
    anec: Anechoic STFT matrix
    '''
    r, c = np.shape(anec)
    fmatrix = np.empty(shape=(c,feature_length), dtype=float)

    for k in range(c):
        fvector = np.append(reverberant_STFT_dB[:, k : k+1+neighbour_frames], anechoic_STFT_dB[:,k])
        #Row vector
        fvector = fvector.reshape(1,feature_length)
        fmatrix[k,:] = fvector
    return fmatrix 

def read_wav_files(filepath,sr):
    '''Returns a list of audio waves
    Params:
        filepath
        sr= Sampling rate of wavfiles
    Returns:
        audio signals in float32
    '''
    audiofile,_ = librosa.load(filepath,sr=sr)
    return audiofile