import numpy as np
import librosa
from essentia import Pool, array
import essentia.standard as ess
import scipy.sparse as sparse
from scipy.signal import spectrogram

class FeatureExtractor:  
    def __init__(self, features='cens') -> None:
        self.features = features
        
        if features == 'cens':
            self.extract = self.generate_cens
        elif features == 'hpcp':
            self.extract = self.generate_hpcp
        elif features == 'mfcc':
            self.extract = self.generate_mfcc
    
    def generate_mfcc(self, filename):
        hopSize=512
        (XAudio, Fs) = getAudioLibrosa(filename)
        XMFCC = self.getMFCCsLibrosa(XAudio, Fs, 4*hopSize, hopSize, lifterexp = 0.6, NMFCC = 20)
        XMFCC = (XMFCC - np.mean(XMFCC)) / np.std(XMFCC)
        return XMFCC

    def generate_hpcp(self, filename):
        hopSize=512
        (XAudio, Fs) = getAudioLibrosa(filename)
        # XHPCP = self.getHPCPEssentia(XAudio, Fs, 2048, hopSize, NChromaBins = 12)
        XHPCP = getHPCP(XAudio, 22050, 2048, 512, 12, squareMags=False)
        XHPCP = (XHPCP - np.mean(XHPCP)) / np.std(XHPCP)
        return XHPCP
    
    def generate_cens(self, filename):
        (XAudio, Fs) = getAudioLibrosa(filename)
        XCENS = self.getCENSLibrosa(XAudio)
        XCENS = (XCENS - np.mean(XCENS)) / np.std(XCENS)
        return XCENS
       
     
def getAudioLibrosa(filename):
    """
    Use librosa to load audio
    :param filename: Path to audio file
    :return (XAudio, Fs): Audio in samples, sample rate
    """
    XAudio, Fs = librosa.load(filename)
    XAudio = librosa.core.to_mono(XAudio)
    return (XAudio, Fs)

# compute frame-wise hpcp with default params
def getHPCPEssentia(XAudio, Fs, winSize, hopSize, squareRoot=False, NChromaBins=36, NHarmonics = 0):
    """
    Wrap around the essentia library to compute HPCP features
    :param XAudio: A flat array of raw audio samples
    :param Fs: Sample rate
    :param winSize: Window size of each STFT window
    :param hopSize: Hop size between STFT windows
    :param squareRoot: Do square root compression?
    :param NChromaBins: How many chroma bins (default 36)
    :returns H: An (NChromaBins x NWindows) matrix of all \
        chroma windows
    """
    spectrum = ess.Spectrum()
    window = ess.Windowing(size=winSize, type='hann')
    spectralPeaks = ess.SpectralPeaks()
    hpcp = ess.HPCP(size=NChromaBins, harmonics=NHarmonics)
    H = []
    for frame in ess.FrameGenerator(array(XAudio), frameSize=winSize, hopSize=hopSize, startFromZero=True):
        S = spectrum(window(frame))
        freqs, mags = spectralPeaks(S)
        H.append(hpcp(freqs, mags))
    H = np.array(H)
    H = H.T
    return H

def getMFCCsLibrosa(XAudio, Fs, winSize, hopSize = 512, NBands = 40, fmax = 8000, NMFCC = 20, lifterexp = 0):
    """
    Get MFCC features using librosa functions
    :param XAudio: A flat array of audio samples
    :param Fs: Sample rate
    :param winSize: Window size to use for STFT
    :param hopSize: Hop size to use for STFT (default 512)
    :param NBands: Number of mel bands to use
    :param fmax: Maximum frequency
    :param NMFCC: Number of MFCC coefficients to return
    :param lifterexp: Lifter exponential
    :return X: An (NMFCC x NWindows) array of MFCC samples
    """
    X = librosa.feature.mfcc(XAudio, Fs, n_mfcc=NMFCC, hop_length=hopSize, n_mels=NBands, fmax=fmax)
    #Do liftering
    coeffs = np.arange(NMFCC)**lifterexp
    coeffs[0] = 1
    X = coeffs[:, None]*X
    X = np.array(X, dtype = np.float32)
    return X

def getCENSLibrosa(XAudio):
    X = librosa.feature.chroma_cens(XAudio, hop_length=512)
    X = np.array(X, dtype = np.float32)
    return X

HPCP_PRECISION = 1e-5
STFT_MIN = 1e-10

import scipy.sparse as sparse
from scipy.signal import spectrogram

def get1DPeaks(X, doParabolic=True, MaxPeaks = -1):
    """
    Find peaks in intermediate locations using parabolic interpolation
    :param X: A 1D array in which to find interpolated peaks
    :param doParabolic: Whether to use parabolic interpolation to get refined \
        peak estimates (default True)
    :param MaxPeaks: The maximum number of peaks to consider\
        (default -1, consider all peaks)
    :return (bins, freqs): p is signed interval to the left/right of the max
        at which the true peak resides, and b is the peak value
    """
    idx = np.arange(1, X.size-1)
    idx = idx[(X[idx-1] < X[idx])*(X[idx+1] < X[idx])]
    vals = X[idx]
    if doParabolic:
        #Reference:
        # https://ccrma.stanford.edu/~jos/parshl/Peak_Detection_Steps_3.html
        alpha = X[idx-1]
        beta = X[idx]
        gamma = X[idx+1]
        p = 0.5*(alpha - gamma)/(alpha-2*beta+gamma)
        idx = np.array(idx, dtype = np.float64) + p
        vals = beta - 0.25*(alpha - gamma)*p
    else:
        idx = np.array(idx, dtype = np.float64)
    if MaxPeaks > 0:
        if len(vals) > MaxPeaks:
            idxx = np.argsort(-vals)
            vals = vals[idxx[0:MaxPeaks]]
            idx = idx[idxx[0:MaxPeaks]]
    return (idx, vals)        

def unitMaxNorm(x):
    m = np.max(x)
    if m < HPCP_PRECISION:
        m = 1.0
    return x/m

def getHPCP(XAudio, Fs, winSize, hopSize, NChromaBins = 36, minFreq = 40, maxFreq = 5000, 
            bandSplitFreq = 500, refFreq = 440, NHarmonics = 0, windowSize = 1,
            MaxPeaks = 100, doParabolic = True, dodB = False, squareMags = True):
    """
    My implementation of HPCP
    :param XAudio: The raw audio
    :param Fs: The sample rate
    :param winSize: The window size of each HPCP window in samples
    :param hopSize: The hop size between windows
    :param NChromaBins: The number of semitones for each HPCP window (default 36)
    :param minFreq: Minimum frequency to consider (default 40hz)
    :param maxFreq: Maximum frequency to consider (default 5000hz)
    :param bandSplitFreq: The frequency separating low and high bands (default 500hz)
    :param refFreq: Reference frequency (440hz default)
    :param NHarmonics: The number of harmonics to contribute to each semitone (default 0)
    :param windowSize: Size in semitones of window used for weighting
    :param MaxPeaks: The maximum number of peaks to include per window
    :param doParabolic: Do parabolic interpolation when finding peaks
    :param dodB: Whether to use dB instead of linear magnitudes (default False)
    :param squareMags: Whether to square the linear magnitudes of the contributions
        from the spectrogram
    """
    #Squared cosine weight type

    NWin = int(np.floor((len(XAudio)-winSize)/float(hopSize))) + 1
    binFrac,_,S = spectrogram(XAudio[0:winSize], nperseg=winSize, window='blackman')
    #Setup center frequencies of HPCP
    NBins = int(NChromaBins*np.ceil(np.log2(float(maxFreq)/minFreq)))
    freqs = np.zeros(NBins)
    binIdx = -1*np.ones(NBins)
    for i in range(NChromaBins):
        f = refFreq*2.0**(float(i)/NChromaBins)
        while f > minFreq*2:
            f /= 2.0
        k = i
        while f <= maxFreq:
            freqs[k] = f
            binIdx[k] = i
            k += NChromaBins
            f *= 2.0
    freqs = freqs[binIdx >= 0]
    binIdx = binIdx[binIdx >= 0]
    idx = np.argsort(freqs)
    freqs = freqs[idx]
    binIdx = binIdx[idx]
    freqsNorm = freqs/Fs #Normalize to be fraction of sampling frequency
    maxFreqIdx = int(np.ceil(winSize*float(maxFreq)/Fs))
    minFreqIdx = int(np.floor(winSize*float(minFreq)/Fs))

    #Do STFT window by window
    H = []
    for i in range(NWin):
        #Compute spectrogram and pull out relevant portions
        _,_,S = spectrogram(XAudio[i*hopSize:i*hopSize+winSize], nperseg=winSize, window='blackman')
        S = S.flatten()
        S = S[0:maxFreqIdx]
        
        if dodB:
            #Convert to dB
            S = np.maximum(S, STFT_MIN)
            S = np.log(S)
        
        #Do parabolic interpolation on each peak
        (pidxs, pvals) = get1DPeaks(S, doParabolic=doParabolic, MaxPeaks=MaxPeaks)
        pidxs /= float(winSize) #Normalize to be fraction of sampling frequency
        
        #Figure out number of semitones from each unrolled bin
        ratios = pidxs[:, None]/freqsNorm[None, :]
        ratios[ratios == 0] = 1
        delta = np.abs(np.log2(ratios))*NChromaBins
        
        #Weight by squared cosine window
        weights = (np.cos((delta/windowSize)*np.pi/2)**2)*(delta <= windowSize)
        pvals = pvals[:, None]*weights
        if squareMags:
            pvals = pvals**2
        hpcpUnrolled = np.sum(pvals, 0)
        
        #Make hpcp low and hpcp high
        hpcplow = hpcpUnrolled[freqs <= minFreq]
        binIdxLow = binIdx[freqs <= minFreq]
        hpcplow = sparse.coo_matrix((hpcplow, (np.zeros(binIdxLow.size), binIdxLow)), 
            shape=(1, NChromaBins)).todense()
        hpcphigh = hpcpUnrolled[freqs > minFreq]
        binIdxHigh = binIdx[freqs > minFreq]
        hpcphigh = sparse.coo_matrix((hpcphigh, (np.zeros(binIdxHigh.size), binIdxHigh)), 
            shape=(1, NChromaBins)).todense()
        
        #unitMax normalization of low and high individually, then sum
        hpcp = unitMaxNorm(hpcplow) + unitMaxNorm(hpcphigh)
        hpcp = unitMaxNorm(hpcp)
        hpcp = np.array(hpcp).flatten()
        H.append(hpcp.tolist())
    H = np.array(H)
    H = H.T
    return H

