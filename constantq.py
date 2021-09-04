import numpy as np
import math
from math import ceil, floor
from midinotes import midinotes, quarter_tones
import matplotlib.pyplot as plt
from scipy.signal import filtfilt, blackmanharris


# This implementation adapted from the MATLAB implementation of the algorithm
# described in "CONSTANT-Q TRANSFORM TOOLBOX FOR MUSIC PROCESSING"
# https://code.soundsoftware.ac.uk/projects/constant-q-toolbox

class cqt(object):

    # Coefficients generated in MATLAB using: [B A] = butter(6, 0.5, 'low')
    butter6_A = np.array([1.00000000, -0.00000000, 0.77769596, -0.00000000,
        0.11419943, -0.00000000, 0.00175093])
    butter6_B = np.array([0.02958822, 0.17752934, 0.44382335, 0.59176447,
        0.44382335, 0.17752934, 0.02958822])

    def __init__(self, fmin, fmax, bins, fs, q=1, hop_factor=0.25, thresh=0.0005, kernel=None, B=None, A=None):
        if q <= 0  or q > 1:
            print("Warning: \n    Invalid q of " + str(q) + ". Setting to default of 1")
            q = 1

        if hop_factor <= 0 or hop_factor > 1:
            print("Warning: \n    Invalid hop factor of " + str(hop_factor) + ". Setting to default of 0.25")
            hop_factor = 0.25

        if kernel:
            self.kernel = kernel

        if B:
            self.filtB = B
        else:
            self.filtB = self.butter6_B

        if A:
            self.filtA = A
        else:
            self.filtA = self.butter6_A

        self.thresh = thresh
        self.hop_factor = hop_factor
        self.q = q
        self.fs = fs
        self.bins = int(bins)
        self.num_octaves = ceil(math.log(fmax/fmin, 2))
        self.fmax = fmax
        self.fmin = (fmax / pow(2, self.num_octaves)) * pow(2, 1/bins) # set fmin to first bin in the lowest octave

    def next_bin(self, freq):
        return freq * pow(2, 1 / self.bins)

    def prev_bin(self, freq):
        return freq / pow(2, 1 / self.bins)

    def gen_kernels(self):
        oct_low = self.next_bin(self.fmax / 2)

        assert(oct_low < self.fmax)

        Q = self.q / (pow(2, 1 / self.bins) - 1)

        Nk_min = round(Q * self.fs / self.fmax) 
        Nk_max = round(Q * self.fs / oct_low)

        assert(Nk_min < Nk_max)

        half_max = ceil(Nk_max / 2)
        atom_shift_samples = ceil(Nk_min * self.hop_factor)
   
        assert(atom_shift_samples < Nk_min)

        # We want center of the biggest atom to be at an integer of the shift stride
        first_center_idx = atom_shift_samples * ceil(half_max / atom_shift_samples) 
        assert(first_center_idx >= half_max)
    
        # Use the FFT length that just fits the biggest atom
        fft_len = pow(2, ceil(math.log(first_center_idx + half_max, 2)))
        num_atoms_per_fft = floor((fft_len - first_center_idx - half_max) / atom_shift_samples) + 1

        self.num_interleave_cqts = num_atoms_per_fft
        self.atom_hop_sec = atom_shift_samples / self.fs
        
        last_center_idx = first_center_idx + atom_shift_samples * (num_atoms_per_fft - 1)
        fft_shift = atom_shift_samples * num_atoms_per_fft
        fft_overlap = (fft_len - fft_shift) * 100 / fft_len

        kernel = np.zeros((fft_len, self.bins * num_atoms_per_fft), dtype=np.complex)
        

        for k in range(self.bins):
            fk = oct_low * pow(2, k/self.bins) 
            Nk = Q * self.fs / fk
            win = blackmanharris(round(Nk))
        
            exp_seq = np.exp(np.arange(0, round(Nk)) * 2j * np.pi * fk / self.fs)
            t_kernel = (1 / Nk) * np.multiply(win, exp_seq)
            
            base = k * num_atoms_per_fft
            for i in range(num_atoms_per_fft):
                start = first_center_idx - ceil(Nk / 2) + (i * atom_shift_samples)
                end = start + round(Nk)
                assert(end < fft_len)
                kernel[start:end,base+i] = np.copy(t_kernel)
                kernel[:,base+i] = np.fft.fft(kernel[:,base+i])
                kernel[np.where(np.abs(kernel[:,base+i]) < self.thresh),base+i] = 0
                kernel[:,base+i] = kernel[:,base+i] / fft_len

        # TODO: Any further normalization 
        self.kernel = kernel


# This transform is implemented as specified in "An efficient algorithm for 
# the calculation of a constant Q transform" (Brown 1992)
def hamming_window(N, shift, a0):
    #window = (a0 - (1 - a0) * np.cos(2*np.pi*(np.arange(int(N))-shift)/N)) / N
    window = (a0 - (1 - a0) * np.cos(2*np.pi*(np.arange(int(N)))/N)) / N
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
        idxs = np.arange(N[k_cq])
        t_kernels[k_cq][lower:upper] = np.multiply(hamming_window(N[k_cq], lower, a0), 
                                                    np.exp(2*np.pi*freqs[k_cq]*(idxs - int(N_max/2))*1j/sampling_rate))
        s_kernels[k_cq] = np.conj(np.fft.fft(np.real(t_kernels[k_cq])))[0:int(len(t_kernels[k_cq])/2)]

#        plt.plot(hamming_window(N[k_cq], lower, a0))
#        plt.show()
#        plt.plot(t_kernels[k_cq])
#        plt.show()
#        plt.plot(np.abs(s_kernels[k_cq]))
#        plt.show()

        non_zero = []
        for i in range(len(s_kernels[k_cq])):
            if np.absolute(s_kernels[k_cq][i]) <= MINVAL:
                s_kernels[k_cq][i] = 0
            else:
                non_zero.append(i) 

        # TODO: Adjust MINVAL
        sum_bounds[k_cq] = (min(non_zero), max(non_zero)+1)

    return s_kernels, sum_bounds, N
