import numpy as np
import math
from midinotes import midinotes
import matplotlib.pyplot as plt

# This transform is implemented as specified in "An efficient algorithm for 
# the calculation of a constant Q transform" (Brown 1992)

# This function generates frequencies between the minimum and maximum frequency 
# on a semitone scale by default (2^(1/12) spacing).  
def gen_freqs(min_freq, max_freq, tone='semi'):
    freqs = []
    curr_freq = min_freq
    tone_spacing = 0 
    k = 0

    if tone == 'quarter':
        tone_spacing = 1.0/24.0
    else:
        print('gen_freqs: Default of semitone spacing being used to generate frequencies')
        tone_spacing = 1.0/12.0

    while curr_freq < max_freq:
        curr_freq = pow(math.pow(2, tone_spacing), k) * min_freq
        freqs.append(curr_freq)
        k += 1 
    
    print(freqs)
    return freqs

def hamming_window(N, a0):
    window = np.zeros(N)    
    for n in range(N):
        window[n] = a0 - (1 - a0) * math.cos((2*np.pi*n)/(N-1))

    return window

# This function generates all of the windows for given sampling rate and midinote range
# Manual control of FFT length is not recommended: The minimum length is bounded by the formula 
# sampling rate * Q / mininum_frequency.  The default fft_length allows a minimum frequency (in most cases)
# of 44100 * 17 / 1024 ~= 732

# In the future, I may want the frequency range to be adjustable on the fly. We can make this possible in real time
# by computing kernels for ALL of the midinotes when a Constant-Q object is initialized.
def gen_kernels(midi_low, midi_high, sampling_rate, a0=25/46, Q=17, fft_length=1024, MINVAL=0.0005):
    if type(midi_low) != int or type(midi_high) != int:
        raise Exception('midi_low and midi_high must be integers in the range 0-167')
    elif midi_low >= midi_high:
        raise Exception('midi_low cannot be greater than or equal to midi_high: {} is not less than {}'.format(midi_low, midi_high)) 
    elif midi_low < 0 or midi_high < 0 or midi_low > 167 or midi_high > 167:
        raise Exception('Acceptable midinotes are in the range 0-167. You entered range: {}-{}'.format(midi_low, midi_high))

    freqs = midinotes[midi_low:midi_high+1]

    # Generate window lengths
    N = [None] * len(freqs)  
    for k_cq in range(len(N)):
        N[k_cq] = int(sampling_rate * Q / freqs[k_cq])

    N_max = N[0]
    t_kernels = [None] * len(N)
    s_kernels = [None] * len(N)
    sum_bounds = [None] * len(N)

    # These loops generate temporal kernels and take FFT's of them to generate spectral kernels
    for k_cq in range(len(N)):

        # This loop generates the temporal kernels by multiplying a hamming window
        # by an exponential of the k_cq component frequency
        t_kernels[k_cq] = [0] * N_max 
        for n in range(int(N_max/2 - N[k_cq]/2), int(N_max/2 + N[k_cq]/2)):
            t_kernels[k_cq][n] = (a0 - (1 - a0) * math.cos((2*np.pi*(n-(N_max/2 - N[k_cq]/2))
            /N[k_cq]))) * np.exp(2*np.pi*freqs[k_cq]*(n-N_max/2)*1j/sampling_rate) / N[k_cq] 
          
        t_kernels[k_cq] = np.real(t_kernels[k_cq])
        s_kernels[k_cq] = [0] * fft_length

        s_kernels[k_cq] = np.conj(np.fft.fft(t_kernels[k_cq]))
        s_kernels[k_cq] = s_kernels[k_cq][0:int(len(s_kernels[k_cq])/2)]

        non_zero = []
        for i in range(len(s_kernels[k_cq])):
            if np.absolute(s_kernels[k_cq][i]) <= MINVAL:
                s_kernels[k_cq][i] = 0
            else:
                non_zero.append(i) 
        # TODO: Adjust MINVAL
        sum_bounds[k_cq] = (min(non_zero), max(non_zero))

    return s_kernels, sum_bounds, N
