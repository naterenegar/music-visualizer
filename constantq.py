import numpy as np
import math
from midinotes import midinotes

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


# This function generates all of the windows for given q and frequency range
def gen_kernels(min_freq, max_freq, sampling_rate, a0=25/46, Q=17, fft_length=2048, MINVAL=0.001):
    freqs = gen_freqs(min_freq, max_freq) 

    # Bound min and max frequencies
    if max_freq > 8000:
        max_freq = 8000
        print("gen_kernels: max_freq was greater than 8000 Hz. Setting to 8000 Hz")
    if min_freq < 8:
        print("gen_kernels: min_freq was less than 8 Hz. Setting to 8 Hz")


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

        # This piece of code samples the temporal kernels to the desired length of the fft
        # I believe this is causing the transform to behave incorrectly
        if N[k_cq] > 1024:
            delta_n = int(N_max/fft_length)
            for k in range(fft_length):
                s_kernels[k_cq][k] = t_kernels[k_cq][k*delta_n]
        else:
                s_kernels[k_cq] = np.fft.fft(t_kernels[k_cq])
        s_kernels[k_cq] = np.fft.fft(s_kernels[k_cq])
        s_kernels[k_cq] = s_kernels[k_cq][0:int(len(s_kernels[k_cq])/2)]

#        plt.plot(abs(s_kernels[k_cq]))
#        plt.show()

        non_zero = []
        for i in range(len(s_kernels[k_cq])):
            if np.absolute(s_kernels[k_cq][i]) <= MINVAL:
                s_kernels[k_cq][i] = 0
            else:
                non_zero.append(i) 
        sum_bounds[k_cq] = (min(non_zero), max(non_zero))

    return s_kernels, sum_bounds, N
