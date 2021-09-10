import numpy as np

PI = np.math.pi

def ewt_beta(x):
    if x < 0:
        bm = 0
    elif x > 1:
        bm = 1
    else:
        bm = (x**4) * (35.-84.*x+70.*(x**2)-20.*(x**3))    
    return bm


def ewt_meyer_scaling(w1, gamma, n):
    mi = np.math.floor(n/2)
    w = np.fft.fftshift(np.linspace(0, 2*PI - 2*PI/n, n))
    w[:mi] -= 2*PI

    aw = np.abs(w)
    yms = np.zeros(n)

    an = 1/(2*gamma*w1)
    pbn = (1+gamma)*w1
    mbn = (1-gamma)*w1

    for i in range(n):
        if aw[i] <= mbn:
            yms[i] = 1
        elif aw[i] >= mbn and aw[i] <= pbn:
            yms[i] = np.math.cos(PI*ewt_beta(an*(aw[i]-mbn))/2)

    yms = np.fft.ifftshift(yms)

    return yms


def ewt_meyer_wavelet(wn, wm, gamma, n):
    mi = np.math.floor(n/2)
    w = np.fft.fftshift(np.arange(0, 2*PI, 2*PI/n))
    w[:mi] -= 2*PI

    aw = np.abs(w)
    ymw = np.zeros(n)

    an = 1/(2*gamma*wn)
    am = 1/(2*gamma*wm)
    pbn = (1+gamma)*wn
    pbm = (1+gamma)*wm
    mbn = (1-gamma)*wn
    mbm = (1-gamma)*wm

    for i in range(n):
        if (aw[i] >= pbn) and (aw[i] <= mbm):
            ymw[i] = 1
        elif (aw[i] >= mbm) and (aw[i] <= pbm):
            ymw[i] = np.math.cos(PI*ewt_beta(am*(aw[i]-mbm))/2)
        elif (aw[i] >= mbn) and (aw[i] <= pbn):
            ymw[i] = np.math.sin(PI*ewt_beta(an*(aw[i]-mbn))/2)
    
    ymw = np.fft.ifftshift(ymw)
    return ymw


def ewt_meyer_filter_bank(boundaries, n):
    len_boundaries = boundaries.shape[0]

    gamma = 1
    for i in range(len_boundaries-1):
        r = (boundaries[i+1] - boundaries[i]) / (boundaries[i+1] + boundaries[i])
        if r < gamma:
            gamma = r

    r = (PI - boundaries[len_boundaries-1]) / (PI + boundaries[len_boundaries-1])
    if r < gamma:
        gamma = r

    gamma *= (1 - 1/n)

    mfb = np.zeros((n, len_boundaries+1))
    
    mfb[:, 0] = ewt_meyer_scaling(boundaries[0], gamma, n)

    for i in range(len_boundaries-1):
        mfb[:, i+1] = ewt_meyer_wavelet(boundaries[i], boundaries[i+1], gamma, n)
    
    mfb[:, len_boundaries] = ewt_meyer_wavelet(boundaries[len_boundaries-1], PI, gamma, n)
    
    return mfb

def mode_eval_each_channel(zz, mfb):
    ft = np.fft.fft(zz)

    ewt = np.zeros((mfb.shape[1], mfb.shape[0]), dtype='complex_') # ewt.shape = 5*300

    for i in range(mfb.shape[1]):
        ewt[i] = np.real(np.fft.ifft(np.conjugate(mfb[:, i])))*ft

    return ewt

import scipy.io as sio
import numpy as np
import matplotlib.pyplot as plt

data = sio.loadmat('data.mat')['data']

x = data

fs = 100

for i in range(3):
    x[:,i] = (x[:,i] - np.mean(x[:,i])) / np.std(x[:,i])

combined = np.sum(x, axis = 1)/np.math.sqrt(3)

ff = np.fft.fft(combined)

frequencies = np.array([5, 10, 20, 30])

boundaries = np.array([(frequency * (2*PI)) / fs for frequency in frequencies])
print(boundaries)
mfb = ewt_meyer_filter_bank(boundaries, len(ff))
print(mfb.shape)

xxx = np.linspace(0, 1, mfb.shape[0])*fs

legend = []
for i in range(mfb.shape[1]):
    plt.plot(xxx, mfb[:, i])
    legend.append(i)

plt.xlim(0, round(173.61/2))
plt.ylim(0,2)
plt.title('Projection based multivariate EWT filter bank')
plt.legend(legend)
plt.show()

modal = np.zeros((mfb.shape[1], mfb.shape[0], 3), dtype='complex_')

for i in range(3):
    modal[:, :, i] = mode_eval_each_channel(x[:, i], mfb)

print(modal.shape)