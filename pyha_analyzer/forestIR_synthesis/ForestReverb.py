__author__ = "Shoken KANEKO"

import numpy as np
import os
from scipy.fft import rfft, irfft
from scipy.special import jn, yn
import time

na = np.newaxis
join = os.path.join
norm = np.linalg.norm

from forestIR_synthesis.Constants import pi, soundVel
import forestIR_synthesis.Wav
import forestIR_synthesis.SignalProcessing as sigp
from forestIR_synthesis.SignalProcessing import addSignal, getImpulse
from forestIR_synthesis.SignalProcessing import getFreqsForGivenFilterLength
from forestIR_synthesis.SignalProcessing import getIRFromSpectrum_irfft
from forestIR_synthesis.SignalProcessing import dists, getDelayedSig, floatX

def computeAirAbsorptionCoef(
        freqs,
        temperature_in_celsius=20.0,
        pressure_in_kiloPascal=101.325,
        relativeHumidity_in_percent=50.0):
    """
    freqs: array-like with size [F].
        frequencies to compute the air absorption coefficient alpha,
        where the sound pressure amplitude A(x) = A0 * exp(-alpha * x)
    implementation based on JavaScript code from here:
        http://resource.npl.co.uk/acoustics/techguides/absorption/ (accessed: 6/10/2020)
    """
    temperature_in_kelvin = 273.15 + temperature_in_celsius
    temp_ref = 293.15
    temp_rel = temperature_in_kelvin/temp_ref
    T_01 = 273.15 + 0.01  # Triple point isotherm temperature in [K]
    P_ref = 101.325  # Reference atmospheric pressure in [kPa]
    P_rel = pressure_in_kiloPascal / P_ref
    P_sat_over_P_ref = 10**((-6.8346 * (T_01 / temperature_in_kelvin)**1.261) + 4.6151)
    H = relativeHumidity_in_percent * (P_sat_over_P_ref / P_rel)
    Fro = P_rel * (24 + 40400 * H * (0.02 + H) / (0.391 + H))
    Frn = P_rel / np.sqrt(temp_rel) * (9 + 280 * H * np.power(np.e, (-4.17 * (np.power(temp_rel, (-1 / 3)) - 1))))
    Xc = 0.0000000000184 / P_rel * np.sqrt(temp_rel)
    f2 = freqs**2
    Xo = 0.01275 * np.power(np.e, (-2239.1 / temperature_in_kelvin)) * np.power((Fro + (f2 / Fro)), -1)
    Xn = 0.1068  * np.power(np.e, (-3352   / temperature_in_kelvin)) * np.power((Frn + (f2 / Frn)), -1)
    alpha = 20 * np.log10(np.e) * f2 * (Xc + np.power(temp_rel, (-5 / 2)) * (Xo + Xn))
    return alpha  # [F]

def getFilterBankForAirAbsorption(ntaps, fs, dists, fftLen):
    freqs, freqs_half = sigp.getFreqsForGivenFilterLength(ntaps, fs)
    alphas = computeAirAbsorptionCoef(freqs_half)  # [hF]
    attenuations_dB = alphas[:, na] * dists[na, :]  # [hF x M] ... in dB
    attenuations = sigp.getAmplitudeFromdB(-attenuations_dB)
    filters = irfft(attenuations, axis=0)  # [F=ntaps x M]
    filters = np.roll(filters, shift=ntaps // 2, axis=0)
    specs = rfft(filters,n=fftLen,axis=0)
    return filters, specs  # [ntaps x nDists]

def computeGammaTable(N,a,freqs):
    # a: tree radius
    # N: maximum order of Bessel functions
    # freqs: [F]
    F = len(freqs)
    ks = 2*pi*freqs/soundVel  # [F]
    kas = ks*a  # [F]
    ns = np.arange(N+2)  # [N+2]
    jns = [jn(ns,ka) for ka in kas]  # [F x N+2]
    yns = [yn(ns,ka) for ka in kas]  # [F x N+2]
    jns = np.array(jns)
    yns = np.array(yns)
    tanGamma_n = np.zeros([F,N+1],dtype=floatX)
    for n in range(1,N+1):
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            tanGamma_n[:,n] = (jns[:,n-1] - jns[:,n+1])/(yns[:,n-1] - yns[:,n+1])
    tanGamma_n[:,0] = jns[:,1]/yns[:,1]
    tanGamma_n[0, :] = 0
    gamma_n = np.arctan(tanGamma_n)  # [F x N+1]
    sinGamma_n = np.sin(gamma_n)
    cosGamma_n = np.cos(gamma_n)
    expiGamma_n = cosGamma_n + 1j * sinGamma_n
    sinexpiGamma_n = sinGamma_n * expiGamma_n  # [F x N+1]
    sinexpiGamma_n[:,1:] *= 2  # multiplying En
    sinexpiGamma_n[0,:] = 0
    return sinexpiGamma_n  # [F x N+1], (freq)^-0.5 * En * sinGamma_n * expiGamma_n
    #  the rest:
    #    multiply cosnphi(freq, angle),
    #    sum over all n,
    #    mult (freqs**-0.5)
    #    mult ((2/pi)**0.5 * np.exp(1j*pi*0.25)),
    #    apply delay and multiply 1/dist**0.5

def computeAngleDependentCylinderScatteringFilter(N, a, freqs, angles_in_radian):
    # angles_in_radian: [A]
    gammaTable = computeGammaTable(N,a,freqs)  # [F x N+1]
    ns = np.arange(N+1)  # [N+1]
    cosnphis = np.cos(ns[na,:]*angles_in_radian[:,na])  # [A x N+1]
    sum = np.sum(gammaTable[:,na,:]*cosnphis[na,:,:], axis=-1)  # [F x A]
    const = ((2/pi)**0.5 * np.exp(1j*pi*0.25))
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        sum = sum * ((freqs**-0.5)*const)[:,na]  # [F x A]
    sum[0,:] = 1e-8
    return sum
    #  the rest:
    #    apply delay and multiply 1/dist**0.5

def computeAngleDependentSphereRadiationFilter(maxOrder, a, freqs, angles_in_radian):
    # angles_in_radian: [A]
    import scipy.special as sp
    N = maxOrder
    ns = np.arange(maxOrder+1)  # [N+1]
    kas = 2*np.pi*freqs/soundVel * a  # [F]
    jnps = np.array([sp.spherical_jn(ns,ka) for ka in kas])  # [F x N+1]
    ynps = np.array([sp.spherical_yn(ns,ka) for ka in kas])  # [F x N+1]
    hnps = jnps + 1j * ynps  # [F x N+1]
    coss = np.cos(angles_in_radian)  # [A]
    Pncoss = np.array([sp.lpn(N, z=cos)[0] for cos in coss])  # [A x N+1]
    numerator = ((2*ns+1)*(-1j)**(ns-1)) * Pncoss  # [A x N+1]
    numerator_LF = (((2*ns+1)*(-1j)**(ns-1)) * Pncoss)[na,:,:] * (kas[:,na]**ns[na,:])[:,na,:] # [F x A x N+1]
    denominator = (kas**2)[:,na] * hnps  # [F x N+1]
    denominator_LF = 1j*(ns+1)*sp.factorial2(2*ns-1)  #/(kas[:,na]**ns[na,:])  # [F x N+1]
    summand = numerator[na,:,:] / denominator[:,na,:]  # [F x A x N+1]
    summand_LF = numerator_LF / denominator_LF  # [F x A x N+1]
    for idx,ka in enumerate(kas):
        for n in ns:
            if ka<=0.01*n:
                summand[idx,:,n] = summand_LF[idx,:,n]
    sum = np.sum(summand, axis=-1)  # [F x A]
    return sum  # [F x A]

def getFilterBankForAngleDependentCylinderScattering(
        treeRad=0.25, fs=24000, steps=128, nAngleBins=180, N=50, fftLen=128*3):
    a = treeRad
    freqs, freqs_half = getFreqsForGivenFilterLength(steps, fs)
    angles_in_radian = np.arange(0,nAngleBins+1) * 180/nAngleBins * pi/180
    Ns = [N]  # 50 is just enough
    F = len(freqs)
    A = len(angles_in_radian)
    filter_NxhFxA = np.zeros((len(Ns),F//2+1,A),dtype=np.complex128)
    for n,N in enumerate(Ns):
        filter_hFxA = computeAngleDependentCylinderScatteringFilter(N, a, freqs_half, angles_in_radian)
        filter_NxhFxA[n,:,:] = filter_hFxA

    filter_FxA = np.zeros([F,A],dtype=np.complex128)
    filter_FxA[:F//2+1,:] = filter_hFxA[:,:]
    filter_FxA[F//2+1:,:] = filter_hFxA[1:F//2,:][::-1,:].conj()
    irs = getIRFromSpectrum_irfft(filter_hFxA)  # [T x C]
    irs = np.roll(irs, shift=steps//2, axis=0)
    specs = rfft(irs,n=fftLen,axis=0)  # [Fh(=2T//2+1=T+1) x nAngles]
    return irs, specs  # [T x nAngles], [T+1 x nAngles]

def getFilterBankForSourceDirectivity(fs, ntaps, nAngleBins=180, birdHeadRad=0.025, maxOrder=80, fftLen=128, doPlot=0):
    a = birdHeadRad
    freqs, freqs_half = getFreqsForGivenFilterLength(ntaps, fs)
    angles_in_radian = np.arange(0, nAngleBins + 1) * 180 / nAngleBins * pi / 180
    Ns = [maxOrder]
    Fh = len(freqs_half)
    F = len(freqs)
    A = len(angles_in_radian)
    filter_NxFhxA = np.zeros((len(Ns), Fh, A), dtype=np.complex128)
    for n, N in enumerate(Ns):
        filter_FhxA = computeAngleDependentSphereRadiationFilter(N, a, freqs_half, angles_in_radian)
        filter_NxFhxA[n, :, :] = filter_FhxA
    filter_FxA = np.zeros([F, A], dtype=np.complex128)
    filter_FxA[:F // 2 + 1, :] = filter_FhxA[:, :]
    filter_FxA[F // 2 + 1:, :] = filter_FhxA[1:F // 2, :][::-1, :].conj()
    irs = getIRFromSpectrum_irfft(filter_FhxA)  # [T x C]
    irs = np.roll(irs, shift=ntaps // 2, axis=0)
    rms_0 = sigp.getAweightedRMS(irs[:,0], fs, energy=1)
    irs = irs / rms_0**0.5
    specs = rfft(irs, n=fftLen, axis=0)  # [Fh(=2T//2+1=T+1) x nAngles]
    return irs, specs  # [T x nAngles], [T+1 x nAngles]

def getTreePoss(forestRange_x, forestRange_y, nTrees, algo="uniform", seed=1234):

    if algo.lower()=="uniform":
        np.random.seed(seed)
        treePoss = np.random.uniform(size=[nTrees,2])
    else:
        raise NotImplementedError

    assert treePoss.shape[0]==nTrees
    treePoss[:, 0] *= (forestRange_x[1] - forestRange_x[0])
    treePoss[:, 1] *= (forestRange_y[1] - forestRange_y[0])
    treePoss += np.array([forestRange_x[0], forestRange_y[0]])
    treePoss = np.concatenate([treePoss, np.zeros([nTrees,1])], axis=-1)  # [nTrees x 3]
    return treePoss  # [nTrees x 3]

def getRotMat2D(theta):
    c = np.cos(theta)
    s = np.sin(theta)
    return np.array([[c,-s],[s,c]])

def getRotMat3D_hori(theta):
    c = np.cos(theta)
    s = np.sin(theta)
    return np.array([[c,-s,0],[s,c,0],[0,0,1]])

def getCosOf3Pnts(p1,p2,p3):
    v1 = (p2 - p1)[:2]
    v2 = (p3 - p2)[:2]
    cos = v1.dot(v2)/(norm(v1)*norm(v2))
    return cos

def getCosOf2Vecs(v1,v2):
    cos = v1.dot(v2)/(norm(v1)*norm(v2))
    return cos

def getCossOf3Pnts(p1,p2s,p3):
    # p2s: [N x 3]
    v1 = (p2s[:,:2] - p1[:2])  # [N x 2]
    v2 = (p3[:2] - p2s[:,:2])
    coss = np.sum(v1*v2,axis=-1)/(np.sum(v1**2,axis=-1)**0.5 * np.sum(v2**2,axis=-1)**0.5)
    return coss  # [N]

def getCossOf2Vecs_batch(v1,v2s):
    # v2s: [N x 3]
    coss = np.sum(v1*v2s,axis=-1)/(np.sum(v1**2,axis=-1)**0.5 * np.sum(v2s**2,axis=-1)**0.5)
    return coss  # [N]

def getAndStoreMultipleScatteringFilter(filterBank_treeScattering_TxA, angleBins, dicFilter):
    if len(angleBins)==1:
        if str(angleBins) in dicFilter:
            return dicFilter[str(angleBins)], dicFilter
        else:
            dicFilter[str(angleBins)] = filterBank_treeScattering_TxA[:,angleBins[0]]
            return dicFilter[str(angleBins)], dicFilter
    if str(angleBins[:-1]) in dicFilter:
        filter_new = np.convolve(filterBank_treeScattering_TxA[:,angleBins[-1]],dicFilter[str(angleBins[:-1])])
        dicFilter[str(angleBins)] = filter_new
        return filter_new, dicFilter
    else:
        filter_new, dic_Filter = getAndStoreMultipleScatteringFilter(filterBank_treeScattering_TxA, angleBins[:-1], dicFilter)
        filter_new = np.convolve(filterBank_treeScattering_TxA[:,angleBins[-1]],filter_new)
        dicFilter[str(angleBins)] = filter_new
        return filter_new, dicFilter


def getAirFilterIdx(dist, D):
    return min(int(dist // 10), D)

def getAirFilterIdx_batch(dists, Dary):
    return np.minimum(dists // 10, Dary).astype(int)

def simulateForestIR(
        nTrees, posSrc, micPoss, fs, sigLen_in_samples=None, forestRange_x=None, forestRange_y=None, reflectionCoef=2.0,
        treeRadiationExponent=0.7, applyCylindricalScatteringFilter=1, applyAirAbsorption=1, maxDist=20000.0,
        maxReflOrder=1, seed=1234, floorReflectionCoef=0.8, ntaps_treeScattering=128, ntaps_airAbsorption=128,
        ntaps_delay=128, sourceDirectivity=0, ntaps_sourceDirectivity=128, sourceDirectivityVec=None,
        samplingAlgo="uniform"
    ):
    # scalable forest reverb generator.
    assert maxReflOrder<=1,"Higher order scattering is not supported."
    assert applyAirAbsorption==1
    assert applyCylindricalScatteringFilter==1
    print("simulating forest IR...")
    if forestRange_x is None:
        forestRange_x = [0.0,1000.0]
    if forestRange_y is None:
        forestRange_y = [0.0,1000.0]
    if nTrees>0:
        treePoss = getTreePoss(forestRange_x, forestRange_y, nTrees, algo=samplingAlgo, seed=seed)
    else:
        treePoss = None
    if nTrees==0:
        ntaps_treeScattering = 0
    dists_src_to_mics_direct = dists(posSrc.reshape([1,3]), micPoss)[0]  # [nMics]
    posSrc_floorMirror = posSrc.copy()
    posSrc_floorMirror[2] *= -1
    dists_src_to_mics_floor = dists(posSrc_floorMirror.reshape([1,3]), micPoss)[0]  # [nMics]
    if nTrees>0:
        dists_src_to_trees = dists(posSrc.reshape([1,3])[:,:2], treePoss[:,:2])[0]  # [nTrees]
        dists_trees_to_mic = dists(micPoss[:,:2], treePoss[:,:2])  # [nMics x nTrees]
        dists_src_to_mics_reflect = dists_src_to_trees[na,:] + dists_trees_to_mic  # [nMics x nTrees]
        distDiffs_src_to_mics_reflect = dists_src_to_mics_reflect - dists_src_to_mics_direct[:,na]  # [nMics x nTrees]
        distsMult_src_to_mics_reflect = dists_src_to_trees[na,:] * dists_trees_to_mic**treeRadiationExponent  # [nMics x nTrees]
        dists_src_to_mics_reflect_ = dists_src_to_mics_reflect
        distsMult_src_to_mics_reflect_ = distsMult_src_to_mics_reflect

    domain = "freq"
    if domain=="freq":
        fftLen = ntaps_treeScattering + ntaps_airAbsorption + ntaps_delay + sourceDirectivity * ntaps_sourceDirectivity
    else:
        fftLen = 128
    Bt = 100  # batch size in batched path processing
    assert nTrees%Bt==0, "The number of trees must be "+str(Bt)+"n (n: integer)."
    if applyCylindricalScatteringFilter==1 and nTrees>0:
        filterBank_treeScattering_TxA, filterBank_treeScattering_freqDomain_FhxA = \
            getFilterBankForAngleDependentCylinderScattering(fs=fs, steps=ntaps_treeScattering, fftLen=fftLen)
        filterBank_treeScattering_TxA *= reflectionCoef
        filterBank_treeScattering_freqDomain_FhxA *= reflectionCoef
        filterBank_treeScattering_freqDomain_FhxA_ = filterBank_treeScattering_freqDomain_FhxA

    distBins = np.arange(0,maxDist+10,10)
    if applyAirAbsorption==1:
        filterBank_airAbsorption_TxD, filterBank_airAbsorption_freqDomain_FhxD = \
            getFilterBankForAirAbsorption(ntaps_airAbsorption, fs, dists=distBins, fftLen=fftLen)
        D = filterBank_airAbsorption_TxD.shape[1]
        Dary = np.ones([Bt]) * D
        filterBank_airAbsorption_freqDomain_FhxD_ = filterBank_airAbsorption_freqDomain_FhxD
        Dary_ = Dary

    if sourceDirectivity==1:
        filterBank_directivity_TxA, filterBank_directivity_FhxA = getFilterBankForSourceDirectivity(
            fs, ntaps=ntaps_sourceDirectivity, fftLen=fftLen)

    nMic = len(micPoss)
    if sigLen_in_samples is None:
        diagonalLen = ((forestRange_y[1]-forestRange_y[0])**2 + (forestRange_x[1]-forestRange_x[0])**2)**0.5
        sigLen_in_samples = int(diagonalLen / soundVel * fs)
    signal = np.zeros([sigLen_in_samples, nMic], dtype=floatX)
    impulse = getImpulse(1)
    # treeIndices = np.arange(nTrees, dtype=int)

    T1 = len(signal)
    posSrc_ = posSrc
    treePoss_ = treePoss
    micPoss_ = micPoss

    for m in range(nMic):
        print("computing ir from src to mic",m,"...")
        tic = time.time()
        if nTrees>0:
            distDiffs_src_to_mic_reflect = distDiffs_src_to_mics_reflect[m,:]  # [nTrees]

        # compute direct path
        direct = impulse
        # apply air absorption
        if applyAirAbsorption==1:
            airFilterIdx = getAirFilterIdx(dists_src_to_mics_direct[m],D)
            airFilter = filterBank_airAbsorption_TxD[:,airFilterIdx]
            direct = np.convolve(direct,airFilter)
        if sourceDirectivity==1:
            cos_dir = getCosOf2Vecs(sourceDirectivityVec,micPoss[m]-posSrc)
            cos_dir = min(1,cos_dir)
            cos_dir = max(-1,cos_dir)
            angle_dir = np.arccos(cos_dir) * 180/pi
            angleBin_dir = (angle_dir+0.5).astype(int)
            dirFilter = filterBank_directivity_TxA[:,angleBin_dir]
            direct = np.convolve(direct,dirFilter)

        # apply delay and distance attenuation
        delayedSig_direct_m, delayToAdd = getDelayedSig(direct, fs, delayInSec=0.0, distance=dists_src_to_mics_direct[m])
        addSignal(signal, delayedSig_direct_m, delayToAdd=delayToAdd, channel=m)

        # compute floor reflection
        floor = impulse
        # apply air absorption
        if applyAirAbsorption==1:
            airFilterIdx = getAirFilterIdx(dists_src_to_mics_direct[m],D)
            airFilter = filterBank_airAbsorption_TxD[:,airFilterIdx]
            floor = airFilter
        if sourceDirectivity==1:
            micPos_refl = micPoss[m].copy()
            micPos_refl[2] *= -1
            cos_dir = getCosOf2Vecs(sourceDirectivityVec,micPos_refl-posSrc)
            cos_dir = min(1,cos_dir)
            cos_dir = max(-1,cos_dir)
            angle_dir = np.arccos(cos_dir) * 180/pi
            angleBin_dir = (angle_dir+0.5).astype(int)
            dirFilter = filterBank_directivity_TxA[:,angleBin_dir]
            floor = np.convolve(floor, dirFilter)

        # apply delay and distance attenuation
        delayedSig_floor_m, delayToAdd = getDelayedSig(floor, fs, delayInSec=0.0, distance=dists_src_to_mics_floor[m])
        addSignal(signal, delayedSig_floor_m * floorReflectionCoef, delayToAdd=delayToAdd, channel=m)

        # compute 1st order scattering by trees
        if nTrees>0:
            coss = getCossOf3Pnts(posSrc_, treePoss_, micPoss_[m])
            coss[coss<-1] = -1
            coss[coss>1] = 1
            angles = np.arccos(coss) * 180.0/pi
            angleBins = (angles+0.5).astype(int)
            # if np.max()
            assert applyAirAbsorption==1
            assert applyCylindricalScatteringFilter==1

            if sourceDirectivity==1:
                treePoss_[:,2] = micPoss[m,2]
                coss_dir = getCossOf2Vecs_batch(sourceDirectivityVec, treePoss_ - posSrc)
                coss_dir[coss_dir<-1] = -1
                coss_dir[coss_dir>1] = 1
                angles_dir = np.arccos(coss_dir) * 180 / pi
                angleBins_dir = (angles_dir + 0.5).astype(int)

            if domain=="time":
                for t in range(nTrees):
                    # apply tree scattering and air absorption
                    scatteringFilter = filterBank_treeScattering_TxA[:,angleBins[t]]  # [T1]
                    airFilter = filterBank_airAbsorption_TxD[:, getAirFilterIdx(dists_src_to_mics_reflect[m,t],D)]  # [T2]
                    reflected_m_t = np.convolve(scatteringFilter, airFilter)  # [T1+T2-1]
                    # apply delay and distance attenuation
                    delayedSig_reflect_m_t, delayToAdd = getDelayedSig(
                        reflected_m_t, fs, delayInSec=dists_src_to_mics_reflect[m,t]/soundVel, distance=0.0,
                        nIRwithoutDelay=ntaps_delay)
                    delayedSig_reflect_m_t /= distsMult_src_to_mics_reflect[m,t]
                    # add signal to result
                    addSignal(signal, delayedSig_reflect_m_t, delayToAdd=delayToAdd, channel=m)
            elif domain=="freq":
                for tHead in range(0,nTrees,Bt):
                    # apply tree scattering and air absorption
                    scatteringFilter_freqDomain = filterBank_treeScattering_freqDomain_FhxA_[:,angleBins[tHead:tHead+Bt]]  # [Fh x Bt]
                    airFilter_freqDomain = filterBank_airAbsorption_freqDomain_FhxD_[:, getAirFilterIdx_batch(
                        dists_src_to_mics_reflect_[m,tHead:tHead+Bt],Dary_)]  # [Fh x Bt]
                    reflected_m_Bt_freqDomain = scatteringFilter_freqDomain * airFilter_freqDomain  # [Fh x Bt]
                    if sourceDirectivity==1:
                        reflected_m_Bt_freqDomain = reflected_m_Bt_freqDomain * filterBank_directivity_FhxA[:,angleBins_dir[tHead:tHead+Bt]]
                    # apply delay
                    delayedSigs_reflect_m_Bt, delaysToAdd = \
                        sigp.getDelayedSig_batch_freqDomain(
                            reflected_m_Bt_freqDomain,
                            fs, dists_src_to_mics_reflect_[m,tHead:tHead+Bt]/soundVel,
                            nIRwithoutDelay=ntaps_delay, fftLen=fftLen)  # [T1+T2+T3 x Bt], [Bt]
                    # apply amplitude attenuation
                    delayedSigs_reflect_m_Bt /= distsMult_src_to_mics_reflect_[m, tHead:tHead+Bt]

                    T2 = fftLen
                    tailsInSig1 = np.minimum(T1, delaysToAdd + T2)
                    tailsInSig2 = np.minimum(T2, T1 - delaysToAdd)

                    t__ = np.where(delaysToAdd<T1)[0]
                    for t in t__:
                        signal[delaysToAdd[t]:tailsInSig1[t], m] += \
                            delayedSigs_reflect_m_Bt[:tailsInSig2[t],t]

            else:
                raise NotImplementedError

        toc = time.time()
        print("  time:",toc-tic)
    return signal  # [T x mMic]


def getMatDistance(mat1, mat2):
    diff = mat1 - mat2
    dist = np.sum(diff * diff.conj()).real ** 0.5
    return dist

def computeDistancesOfMatrices(dic1,dic2):
    # dic: sorted dictionary in which values are ndarrays of same shape
    len1 = len(dic1)
    len2 = len(dic2)
    resultMat = np.zeros([len1,len2],dtype=np.float64)
    i = 0
    for k, v in dic1.items():
        j = 0
        for l, w in dic2.items():
            resultMat[i,j] = getMatDistance(v,w)
            j += 1
        i += 1
    return resultMat

def generateSampleForestIR():
    fs = 24000
    ir = simulateForestIR(
        nTrees=100000,
        posSrc=np.array([500,500,1.5]),
        micPoss=np.array([510,500,1.5]).reshape([1,3]),
        fs=fs,
        sigLen_in_samples=fs*5)
    Wav.writeWav(ir, "sampleForestIR.wav", fs, 1)

if __name__ == "__main__":

    generateSampleForestIR()