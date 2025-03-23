import numpy as np
import heapq
import scipy.signal as signal
from scipy import ndimage

#Based on https://github.com/rrlyman/phase-reconstruction

def dxdw(x):
    ''' return the derivative of x with respect to frequency'''
    xp = np.pad(x,1,mode='edge')
    dw = (xp[1:-1,2:]-xp[1:-1,:-2])/2

    return dw
    
def dxdt(x):
    ''' return the derivative of x with respect to time'''   
    xp = np.pad(x,1,mode='edge')      
    dt = (xp[2:,1:-1]-xp[:-2,1:-1])/(2)

    return dt

def calculate_synthesis_window(win_length=2048, hop_length=512, n_fft=2048,window=None):
    gsynth = np.zeros_like(window)
    for l in range(int(n_fft)):
        denom = 0                
        for n in range(-win_length//hop_length, win_length//hop_length+1):
            dl = l-n*hop_length
            if dl >=0 and dl < win_length:
                denom += window[dl]**2
            gsynth[l] = window[l]/denom

    return gsynth

def get_default_window(n_fft):
    lambda_ = (-n_fft**2/(8*np.log(0.01)))**.5
    lambdasqr = lambda_**2
    gamma = 2*np.pi*lambdasqr
    window=np.array(signal.windows.gaussian(2*n_fft+1, lambda_*2, sym=False))[1:2*n_fft+1:2]

    return window, gamma, lambdasqr

def pghi(x, win_length=2048, hop_length=512, n_fft=None, window=None, synthesis_window=None,gamma=None,lambdasqr=None, tol=1e-6):
    """
    Performs phase gradient heap integration to estimate the phase of a spectrogram given the magnitude
    x:           magnitude spectrogram
    win_length:  size (in samples) of the analysis window
    hop_length:  number of samples between two consecutive windows
    n_fft:       (default = win_length), number of samples to calculate the FFT. If it is larger than win_length, window will get padded
    window:      (default is a gaussian window, it is recommended to use it). Array of size win_length.
    synthesis_window: window to use for the synthesis step. By default it will be calculated from window
    gamma:       constant related to the window (if not given, is calculated from lambdasqr)
    lambdasqr:   constant related to the window (if not given, the default value corresponds to the gaussian window)
    tol:         small signal relative magnitude filtering size
    """
    if n_fft is None: n_fft = win_length
    if gamma is not None: lambdasqr = gamma/(2*np.pi)
    if window is None:
        window, gamma, lambdasqr = get_default_window(n_fft)
    if synthesis_window is None:
        gsynth = calculate_synthesis_window(win_length, hop_length, n_fft, window)
    
    m2 = int(win_length/2) + 1
    n_frames = x.shape[0]
    wbin = 2*np.pi/win_length
    logs = np.log(x+1e-50)
    fmul = gamma/(hop_length * win_length)

    tgrad = dxdw(logs)/fmul + (2*np.pi*hop_length/win_length)*np.arange(m2)
    fgrad = -fmul*dxdt(logs) + np.pi

    dphaseNE =  tgrad  + fgrad
    dphaseSE =  tgrad - fgrad
    phase = np.random.random_sample(x.shape)*2*np.pi

    mask = x > (tol*np.max(x))
    active_padded = np.pad(mask,1,mode='constant',constant_values=False)

    while np.any(active_padded):
        labeled, nr_objects = ndimage.label(active_padded[1:n_frames+1,1:m2+1])
        startpoints = ndimage.maximum_position(x*active_padded[1:n_frames+1,1:m2+1], labeled, range(nr_objects+1))[1:]
        for startpoint in startpoints:
            h=[]
            n0,m0 = startpoint
            phase[n0,m0] = 0
            heapq.heappush(h, (-x[n0,m0],n0,m0)) 
            active_padded[n0+1,m0+1]=False
            while len(h) > 0:
                s=heapq.heappop(h)            
                n,m = s[1],s[2]
                if active_padded[(n+1), m+2]: # North             
                    active_padded[(n+1), m+2]=False # padded is 1 indexed                               
                    phase[n, m+1]=  phase[n,  m] +(fgrad[n,  m] + fgrad[n, m+1])/2 
                    heapq.heappush(h, (-x[n, m+1],n,m+1))                                       
                if active_padded[(n+1), m]: # South 
                    active_padded[(n+1), m]=False # padded is 1 indexed                             
                    phase[n, m-1]=  phase[n,  m] - (fgrad[n,  m] + fgrad[n, m-1])/2 
                    heapq.heappush(h, (-x[n, m-1],n,m-1))    
                                                          
                if active_padded[(n+2), m+1]:  # East 
                    active_padded[(n+2), m+1]=False # padded is 1 indexed                          
                    phase[(n+1), m]=  phase[n,  m] + (tgrad[n,  m] + tgrad[(n+1), m])/2 
                    heapq.heappush(h, (-x[(n+1), m], n+1,m))   
                                                                      
                if active_padded[n, m+1]: # West            
                    active_padded[n, m+1]=False # padded is 1 indexed                             
                    phase[(n-1), m]=  phase[n,  m] - (tgrad[n,  m] + tgrad[(n-1), m])/2 
                    heapq.heappush(h, (-x[(n-1), m],n-1,m))
                                                                 
    return phase

def stft(x,win_length=2048,hop_length=512,window=None):
    """
    Performs the short fourier transform of x in a way that is compatible with PGHI
    x:           time domain signal
    win_length:  size (in samples) of the analysis window
    hop_length:  number of samples between two consecutive windows
    window:      array of size win_length with the window to apply
    """
    L = x.shape[0] - win_length
    if window is None:
        window, _, _ = get_default_window(win_length)
    return np.stack([np.fft.rfft(window*x[ix:ix + win_length]) for ix in range(0, L, hop_length)])

def istft(X, win_length=2048,hop_length=512,window=None,synthesis_window=None):
    """
    Returns a spectrogram to the time domain by using the Overlap and Add method
    X:                spectrogram
    win_length:       size (in samples) of the analysis window
    hop_length:       number of samples between two consecutive windows
    window:           array of size win_length with the window to apply
    synthesis_window: window to use for the synthesis step. By default it will be calculated from window
    """
    N = X.shape[0]
    vr=np.fft.irfft(X)
    sig = np.zeros((N*hop_length+win_length))

    if window is None:
        window, _, _ = get_default_window(win_length)
    if synthesis_window is None:
        synthesis_window = calculate_synthesis_window(win_length, hop_length, win_length, window)

    for n in range(N):
        vs = vr[n]*synthesis_window
        sig[n*hop_length: n*hop_length+win_length] += vs
    
    return sig

def griffin_lim(X, win_length=2048,hop_length=512,window=None,synthesis_window=None, n_iters=100):
    """
    An implementation of the Griffin Lim algorithm
    X:                spectrogram (can be complex and have an initial phase)
    win_length:       size (in samples) of the analysis window
    hop_length:       number of samples between two consecutive windows
    window:           array of size win_length with the window to apply
    synthesis_window: window to use for the synthesis step. By default it will be calculated from window
    n_iters:          number of iterations to perform
    """
    if window is None:
        window, _, _ = get_default_window(win_length)
    if synthesis_window is None:
        synthesis_window = calculate_synthesis_window(win_length, hop_length, win_length, window)
    
    mag_X = np.abs(X)
    for i in range(n_iters):
        x = istft(X, win_length, hop_length, window, synthesis_window)
        X = stft(x,win_length,hop_length,window)
        X = mag_X*np.exp(1.0j*np.angle(X))
    return x