# -*- coding: utf-8 -*-
"""
Created on Thu Apr 28 16:06:50 2022

@author: koenv
"""

#%% Loading Packages


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import scipy.signal      

from scipy.fftpack import fft    
from scipy.stats import chi2 



#%% FFT tool



# I have left this part in for now, but this will be a way shorter function that is more related to the previous sections.


def wave_spectrum(data,nfft,Fs):
    ''' Compute variance spectral density spectrum of the time-series and its 
    90% confidence intervals. 
    The time-series is first divided into blocks of length nfft before being 
    Fourier-transformed.

    INPUT
      data    timeseries 
      nfft    block length
       Fs     sampling frequency (Hz)
    
    OUTPUT
      E       variance spectral density. If data is in meters, E is in m^2/Hz
      f       frequency axis (Hz)
      confLow and confUpper     Lower and upper 90% confidence interval; 
                                (Multiplication factors for E)  '''
    
    # 1. PRELIMINARY CALCULATIONS
    # ---------------------------
    n = len(data)                # length of the time-series
    nfft = int(nfft - (nfft%2))  # the length of the window should be an even number

    data = scipy.signal.detrend(data)      # detrend the time-series
    
    nBlocks = int(n/nfft)        # number of blocks (use of int to make it an integer)

    data_new = data[0:nBlocks*nfft] # (we work only with the blocks which are complete)

    # we organize the initial time-series into blocks of length nfft 
    dataBlock = np.reshape(data_new,(nBlocks,nfft))  # each column of dataBlock is one block
    
    
    print(dataBlock)
    
    
    # 2. CALCULATION VARIANCE DENSITY SPECTRUM
    # ----------------------------------------

    # definition frequency axis
    
    delta_f = Fs/nfft      # frequency resolution of the spectrum df = 1/[Duration of one block]
    f = np.arange(0, Fs/2 + delta_f, delta_f)   # frequency axis (Fs/2 = Fnyquist = max frequency)
    fId = np.arange(0,len(f))

    # Calculate the variance for each block and for each frequency
    
    fft_data = fft(dataBlock, n = nfft, axis = 1)      # Fourier transform of the data 
    fft_data = fft_data[:,fId]            # Only one side needed
    A = 2.0/nfft*np.real(fft_data)          # A(i,b) and B(i,b) contain the Fourier coefficients Ai and Bi for block b
    B = 2.0/nfft*np.imag(fft_data)          #  -- see LH's book, page 325 for definition of Ai and Bi
                                          # /!\ assumes that mean(eta)=0 

    amplitudes = (A**2 + B**2)**(1/2)
    phases = np.arctan(-B/A)

    variance = (amplitudes)**2/2.                  # E(i,b) = ai^2/2 = variance at frequency fi for block b. 

    # We finally average the variance over the blocks, and divide by df to get the spectra
    
    amplitudes = np.mean(amplitudes, axis = 0)/delta_f
    phases = np.mean(phases, axis = 0)/delta_f
    variance = np.mean(variance, axis = 0)/delta_f
    
#     # 3. CONFIDENCE INTERVALS
#     # -----------------------
#     edf = round(nBlocks*2)   # Degrees of freedom 
#     alpha = 0.1              # calculation of the 90% confidence interval

#     confLow = edf/chi2.ppf(1-alpha/2,edf)    # see explanations on confidence intervals given in lecture 3 
#     confUpper  = edf/chi2.ppf(alpha/2,edf)
    
    return f, amplitudes, phases, variance