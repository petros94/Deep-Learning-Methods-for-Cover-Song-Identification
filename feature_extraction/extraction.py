import numpy as np
import librosa
import torchaudio
from essentia import Pool, array
import essentia.standard as ess

from hpcp import TonalDescriptorsExtractor, down_post


class FeatureExtractor:
    def __init__(self, features='cens') -> None:
        self.features = features
        
        if features == 'cens':
            self.extract = self.generate_cens
        elif features == 'hpcp':
            self.extract = self.generate_hpcp
        elif features == 'mfcc':
            self.extract = self.generate_mfcc
        elif features == 'wav':
            self.extract = self.generate_wavs
        elif features == 'hpcp-shs100k':
            self.extract = self.generate_hpcp_shs100k
    
    def generate_mfcc(self, filename):
        hopSize=512
        (XAudio, Fs) = getAudioLibrosa(filename)
        XMFCC = getMFCCsLibrosa(XAudio, Fs, 4*hopSize, hopSize, lifterexp = 0.6, NMFCC = 20)
        XMFCC = (XMFCC - np.mean(XMFCC)) / np.std(XMFCC)
        return XMFCC

    def generate_hpcp(self, filename):
        hopSize=512
        (XAudio, Fs) = getAudioLibrosa(filename)
        XHPCP = getHPCPEssentia(XAudio, Fs, 2048, hopSize, NChromaBins = 12)
        XHPCP = (XHPCP - np.mean(XHPCP)) / np.std(XHPCP)
        return XHPCP

    def generate_hpcp_shs100k(self, filename):
        # sty: 1: hpcp_hpcp, 2: hpcp_npy, 4: 2dfm_npy
        pool = essentia.Pool()
        loader = essentia.streaming.MonoLoader(filename=filename)
        tonalExtractor = TonalDescriptorsExtractor()
        loader.audio >> tonalExtractor.signal
        for desc, output in tonalExtractor.outputs.items():
            output >> (pool, desc)
        essentia.run(loader)

        down_post(pool, 20)
        return pool['down_sample_hpcp']
    
    def generate_cens(self, filename):
        (XAudio, Fs) = getAudioLibrosa(filename)
        XCENS = getCENSLibrosa(XAudio)
        XCENS = (XCENS - np.mean(XCENS)) / np.std(XCENS)
        return XCENS

    def generate_wav(self, filename):
        XWAV = speech_file_to_array_fn(filename)
        return XWAV
       
     
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

def speech_file_to_array_fn(path):
    speech_array, _sampling_rate = torchaudio.load(path)
    resampler = torchaudio.transforms.Resample(_sampling_rate, 16000)
    speech = resampler(speech_array).squeeze().numpy()
    if speech.shape[0] == 2:
        speech = speech[1:, :]
    return speech