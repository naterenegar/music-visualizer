import numpy as np
import math
from midinotes import midinotes, quarter_tones
import matplotlib.pyplot as plt

# This transform is implemented as specified in "An efficient algorithm for 
# the calculation of a constant Q transform" (Brown 1992)
def hamming_window(N, shift, a0):
    window = np.ones(int(N)) * a0 - (1 - a0) * np.cos(2*np.pi*(np.arange(N)-shift)/N)
    return window

def kernel_get(N, a0):
    window = hamming_window(N, a0)



# In the future, I may want the frequency range to be adjustable on the fly. We can make this possible in real time
# by computing kernels for ALL of the midinotes when a Constant-Q object is initialized.
def gen_kernels(midi_low, midi_high, sampling_rate, a0=25/46, Q=34, fft_length=1024, MINVAL=0.0005):
    if type(midi_low) != int or type(midi_high) != int:
        raise Exception('midi_low and midi_high must be integers in the range 0-167')
    elif midi_low >= midi_high:
        raise Exception('midi_low cannot be greater than or equal to midi_high: {} is not less than {}'.format(midi_low, midi_high)) 
    elif midi_low < 0 or midi_high < 0 or midi_low > 167 or midi_high > 167:
        raise Exception('Acceptable midinotes are in the range 0-167. You entered range: {}-{}'.format(midi_low, midi_high))

    if Q != 17 and Q != 34:
        raise Exception('Q must be either 17 for semitones or 34 for quarter tones')

    if Q == 17:
        freqs = midinotes[midi_low:midi_high+1]
    elif Q == 34:
        freqs = quarter_tones[midi_low*2:midi_high*2+1]

    # Generate window lengths
    N = np.zeros(len(freqs))
    for k_cq in range(len(N)):
        N[k_cq] = int(round(sampling_rate * Q / freqs[k_cq]))

    N_max = int(N[0])
    N_max = fft_length
    t_kernels = np.zeros((len(N), N_max), dtype='complex_')
    s_kernels = np.zeros((len(N), int(fft_length / 2)), dtype='complex_')
    sum_bounds = [0] * len(N)

    # These loops generate temporal kernels and take FFT's of them to generate spectral kernels
    for k_cq in range(len(N)):

        # This loop center aligns the kernels
        lower = int(N_max/2 - N[k_cq]/2)
        upper = int(N_max/2 + N[k_cq]/2) 
        idxs = np.arange(lower, upper)
        t_kernels[k_cq][lower:upper] = np.multiply(hamming_window(N[k_cq], lower, a0), np.exp(2*np.pi*freqs[k_cq]*(idxs - int(N_max/2))*1j/sampling_rate)) / N[k_cq]
        s_kernels[k_cq] = np.conj(np.fft.fft(np.real(t_kernels[k_cq])))[0:int(len(t_kernels[k_cq])/2)]

        non_zero = []
        for i in range(len(s_kernels[k_cq])):
            if np.absolute(s_kernels[k_cq][i]) <= MINVAL:
                s_kernels[k_cq][i] = 0
            else:
                non_zero.append(i) 

        # TODO: Adjust MINVAL
        sum_bounds[k_cq] = (min(non_zero), max(non_zero)+1)

    return s_kernels, sum_bounds, N
