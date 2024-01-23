__author__ = "Shoken KANEKO"

import numpy as np
import os
import soundfile as sf

na = np.newaxis
join = os.path.join

def writeWav(ary, path, fs, normalize=0):
    # ary: [T x nChannels] or [T]
    if normalize==1:
        maxAmp = np.max(np.abs(ary))
        ary_ = ary / maxAmp
    else:
        maxAmp = 1.0
        ary_ = ary
    if ary_.ndim==1:
        ary_ = ary_.reshape([len(ary_),1])
    _,ext = os.path.splitext(path)
    sf.write(path,ary_,fs,subtype="FLOAT",format=ext[1:].upper())
    gainFactor = 1.0/maxAmp
    return gainFactor

if __name__ == "__main__":
    pass