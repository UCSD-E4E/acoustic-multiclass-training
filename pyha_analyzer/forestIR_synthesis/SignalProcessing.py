__author__ = "Shoken KANEKO"

import numpy as np
import os
from scipy.signal import stft as spstft
from scipy.signal import istft as spistft
from scipy.fft import irfft
from forestIR_synthesis.Constants import soundVel

na = np.newaxis
join = os.path.join
floatX = np.float64
pi = np.pi

def stft(sig_TxnCh, fs, winSize, stepSize=None):
    if stepSize is None:
        stepSize = winSize//2
    noverlap = winSize - stepSize
    assert winSize > stepSize
    freqs,times,spec_FxnumChxnumSteps = \
        spstft(sig_TxnCh, fs=fs, window='hann', nperseg=winSize, noverlap=noverlap,
               nfft=None, detrend=False, return_onesided=False, boundary='zeros', padded=True, axis=0)
    if sig_TxnCh.ndim==2:
        spec = np.array(spec_FxnumChxnumSteps).transpose([2,0,1])  # [numSteps x F x numCh] <- [F x numCh x numSteps]
    elif sig_TxnCh.ndim==1:
        spec = spec_FxnumChxnumSteps.T  # [numSteps x F]  <-  [F x numSteps]
    else:
        raise NotImplementedError
    return freqs, times, spec

def istft(spec_TxFxnCh, fs, winSize, stepSize):
    if stepSize is None:
        stepSize = winSize//2
    noverlap = winSize - stepSize
    assert winSize > stepSize
    _, sig_TxnCh = spistft(spec_TxFxnCh, fs=fs, window='hann', nperseg=winSize, noverlap=noverlap,
                           nfft=None, input_onesided=False, boundary=True, time_axis=0, freq_axis=1)
    return sig_TxnCh

def getZeroPadded(sig, retSigLen):
    if sig.ndim==1:
        T = len(sig)
        assert retSigLen>T
        ret = np.concatenate([sig, np.zeros([retSigLen - T],dtype=sig.dtype)])
    elif sig.ndim==2:
        T,C = sig.shape
        assert retSigLen>T
        ret = np.concatenate([sig, np.zeros([retSigLen - T,C],dtype=sig.dtype)],axis=0)
    else:
        raise NotImplementedError
    return ret

def dist(p1,p2):
    # p1, p2: [3]
    return np.sum((p1-p2)**2,axis=-1)**0.5

def dist_hori(p1,p2):
    # p1, p2: [3]
    return np.sum((p1-p2)[:2]**2,axis=-1)**0.5

def dists(ps1,ps2):
    # ps1: [nPnts1 x 3]
    # ps2: [nPnts2 x 3]
    return np.sum((ps1[:,na,:]-ps2[na,:,:])**2,axis=-1)**0.5 # [nPnts1 x nPnts2]

def delayInSec(p1,p2):
    d = dist(p1, p2)
    return d / soundVel

def delaysInSec(ps1,ps2):
    d = dists(ps1, ps2)
    return d / soundVel  # [nPnts1 x nPnts2]

def delayInSamples(p1,p2,fs):
    d = dist(p1,p2)
    return d / soundVel * fs

def convolve(sig1, sig2, domain="time"):
    # sig1: [T1] or [T1 x C1]
    # sig2: [T2] or [T2 x C2]
    if sig1.ndim == 1:
        sig1_ = sig1.reshape([len(sig1),1])
    else:
        sig1_ = sig1
    if sig2.ndim == 1:
        sig2_ = sig2.reshape([len(sig2),1])
    else:
        sig2_ = sig2

    if domain=="time":
        ret = []
        for c1 in range(sig1_.shape[1]):
            ret.append([np.convolve(sig1_[:,c1],sig2_[:,c2]) for c2 in range(sig2_.shape[1])])
        # ret: [c1][c2][T3]
        return np.array(ret).transpose([2,0,1])  # [T3 x c1 x c2]
    elif domain=="freq":
        from scipy.fft import rfft
        n = len(sig1_) + len(sig2_) - 1
        sig1_f = rfft(sig1_,n=n,axis=0)  # [Fh x c1]
        sig2_f = rfft(sig2_,n=n,axis=0)  # [Fh x c2]
        conved_f = sig1_f[:,:,na] * sig2_f[:,na,:]  # [Fh x c1 x c2]
        conved = irfft(conved_f,axis=0)  # [T x c1 x c2]
        return conved.real
    else:
        raise NotImplementedError

def getDelayedSig(sig, fs, delayInSec=None, distance=None, nIRwithoutDelay = 128):
    # sig: [T]
    assert sig.ndim==1
    # set distance to None if only time delay but no amplitude reduction should be applied
    # set delayInSec to None if delay should be computed from distance
    if delayInSec is None or delayInSec==0.0:
        assert distance is not None
        delayInSec = distance / soundVel
    if distance is None or distance==0.0:
        distance = 1.0
    assert distance != 0
    delayInSamples = delayInSec * fs
    delayIRLen = int(delayInSamples + nIRwithoutDelay//2)
    if delayInSamples > nIRwithoutDelay//2:
        delayToAdd = int(delayInSamples - nIRwithoutDelay // 2)
    else:
        delayToAdd = 0
    ts = np.arange(start=delayToAdd,stop=delayIRLen)
    x = pi * (ts - delayInSamples)
    delayIR = np.sin(x)/(x * distance)
    if delayIRLen > delayInSamples and delayInSamples - int(delayInSamples) == 0:
        delayIR[int(delayInSamples) - delayToAdd] = 1.0 / distance
    delayed = np.convolve(sig, delayIR)
    return delayed, delayToAdd

def getDelayedSig_batch(sigs, fs, delaysInSec=None, nIRwithoutDelay = 128):
    # sigs: [T x C]
    # delaysInSec, distances: [C]
    assert sigs.ndim == 2
    T,C = sigs.shape
    delaysInSamples = delaysInSec * fs  # [C] ... this is a floating point number
    delaysInSamples_int = delaysInSamples.astype(int)
    delaysToAdd = delaysInSamples_int - int(nIRwithoutDelay // 2)  # [C]
    effectiveIRLen = nIRwithoutDelay
    delaysToAdd[delaysInSamples <= nIRwithoutDelay//2] = 0
    ts = np.arange(effectiveIRLen)  # [T]
    ts = np.tile(ts.reshape([effectiveIRLen,1]),reps=(1,C))  # [T x C]
    ts = ts + delaysToAdd  # [T x C]
    x = pi * (ts - delaysInSamples)  # [T x C]
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        delayIRs = np.sin(x)/x  # [T x C]
    singulars = (x==0) * (delaysInSamples == delaysInSamples_int)
    delayIRs[singulars] = 1.0
    delayed = np.array([np.convolve(sigs[:,i], delayIRs[:,i]) for i in range(C)]).T  # [T x C]
    return delayed, delaysToAdd  # [T x C], [C]

def getDelayedSig_batch_freqDomain(sigs, fs, delaysInSec=None, nIRwithoutDelay=128, fftLen=128*3):
    # sigs: [Fh x C]
    # delaysInSec, distances: [C]
    assert sigs.ndim == 2
    _,C = sigs.shape
    delaysInSamples = delaysInSec * fs  # [C] ... this is a floating point number
    delaysInSamples_int = delaysInSamples.astype(np.int32)
    delaysToAdd = delaysInSamples_int - int(nIRwithoutDelay // 2)  # [C]
    effectiveIRLen = nIRwithoutDelay
    delaysToAdd[delaysInSamples <= nIRwithoutDelay//2] = 0
    ts = np.arange(effectiveIRLen)  # [T]
    ts = np.tile(ts.reshape([effectiveIRLen,1]),reps=(1,C))  # [T x C]
    ts = ts + delaysToAdd  # [T x C]
    x = pi * (ts - delaysInSamples)  # [T x C]
    delayIRs = np.sin(x)/x  # [T x C]
    singulars = (x==0) * (delaysInSamples == delaysInSamples_int)
    delayIRs[singulars] = 1.0
    from scipy.fft import rfft, irfft
    delayIRs_freqDomain = rfft(delayIRs,n=fftLen,axis=0)  # [Fh x C]
    delayed_freqDomain = sigs * delayIRs_freqDomain
    delayed = irfft(delayed_freqDomain, axis=0)  # [T x C]
    return delayed, delaysToAdd  # [T x C], [C]

def getFreqsForGivenFilterLength(nTaps, fs):
    nFreqBins = nTaps
    df = fs / nFreqBins
    freqs = np.arange(nFreqBins) * df
    freqs_half = freqs[:nFreqBins // 2 + 1]
    return freqs, freqs_half

def getHPF(nTaps, fs, fcut):
    import scipy.signal as spsig
    assert nTaps%2==1
    fir = spsig.firwin(nTaps, cutoff=fcut, fs=fs, pass_zero=False)
    return fir

def getBPF(nTaps, fs, bandFreq_low, bandFreq_high):
    import scipy.signal as spsig
    assert nTaps%2==1
    fir = spsig.firwin(nTaps,cutoff=[bandFreq_low,bandFreq_high],fs=fs, pass_zero=False)
    return fir

def get_dB_from_amplitude(spec_TxF, eps=1e-8):
    return 20*np.log(np.abs(spec_TxF)+eps)/np.log(10)

def getAmplitudeFromdB(dB):
    amp = np.exp(dB*np.log(10)/20)
    return amp

def getIRFromSpectrum(spec_FxC):
    from scipy.fft import ifft
    ir = ifft(spec_FxC, axis=0).real
    return ir  # [T x C]

def getIRFromSpectrum_irfft(spec_hFxC):
    ir = irfft(spec_hFxC, axis=0).real
    return ir  # [T x C]

def getLongerSignalByRepeating(sig, desiredLen):
    # sig: [T] or [T x C]
    T = len(sig)
    nRep = int(desiredLen/T)+1
    if sig.ndim==2:
        rep = np.tile(sig, reps=[nRep, 1])
    else:
        rep = np.tile(sig, reps=nRep)
    assert len(rep)>=desiredLen
    return rep[:desiredLen]

def addSignal(sig1, sig2, delayToAdd, channel):
    # sig1: [T1 x C]
    # sig2: [T2]
    # delayToAdd: position in sig1 to start adding sig2
    assert int(delayToAdd)==delayToAdd
    T1 = len(sig1)
    T2 = len(sig2)
    tailInSig1 = min(T1, delayToAdd + T2)
    tailInSig2 = min(T2, T1 - delayToAdd)
    assert channel is not None
    if delayToAdd<T1:
        sig1[delayToAdd:tailInSig1, channel] += sig2[0:tailInSig2]

def getSoundPressure_in_Pascal_from_dBSPL(dBSPL):
    return getAmplitudeFromdB(dBSPL)*0.00002

def getSoundPressureLevel_in_dBSPL_from_Pascal(pa):
    return 20*np.log10(pa/0.00002)

def getSignal_scaled_to_desired_SPL_Aweighted(sig, fs, spl_in_dB):
    rms_A = getAweightedRMS(sig, fs)
    rms_A_desired = getSoundPressure_in_Pascal_from_dBSPL(spl_in_dB)
    return sig * rms_A_desired/rms_A

def getSNR(sig,noise,fs,doAweighting=0):
    if doAweighting==0:
        sigRMS = getRMS(sig)
        noiseRMS = getRMS(noise)
    else:
        sigRMS = getAweightedRMS(sig,fs)
        noiseRMS = getAweightedRMS(noise,fs)
    snr = 20*np.log10(sigRMS/noiseRMS)
    return snr

def getRMS(sig_t):
    return np.mean(sig_t**2)**0.5

def getEnergy(sig_t):
    return np.sum(sig_t**2)

def getLinearGainCoefsOfAweighting(freqs):
    import librosa
    weights = librosa.core.A_weighting(freqs)  # in deciBell
    weights_lin = getAmplitudeFromdB(weights)
    return weights_lin

def getAweightedRMS(sig_t, fs, energy=0):
    assert sig_t.ndim==1
    import scipy as sp
    sig_f = sp.fft.rfft(sig_t)
    nfft = len(sig_t)
    freqs = sp.fft.rfftfreq(nfft, 1/fs)
    weights_lin = getLinearGainCoefsOfAweighting(freqs)
    assert len(freqs)==nfft//2+1
    sig_f_weighted = sig_f * weights_lin  # [F]
    sig_t_weighted = sp.fft.irfft(sig_f_weighted)
    if energy==1:
        return getEnergy(sig_t_weighted)
    rms = getRMS(sig_t_weighted)
    return rms

def getImpulse(sigLen):
    # create src signal
    srcSig = np.zeros(sigLen, dtype=np.float64)
    srcSig[0] = 1
    return srcSig

if __name__=="__main__":
    pass