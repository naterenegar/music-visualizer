import pyaudio, wave, struct, math
import numpy as np
import matplotlib.pyplot as plt
import librosa.core as lc
from pyqtgraph.Qt import QtGui, QtCore
import pyqtgraph as pg

#------------------ Useful functions ------------------#

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
        curr_freq = math.pow(math.pow(2, tone_spacing), k) * min_freq
        freqs.append(curr_freq)
        k += 1 

    return freqs

def hamming_window(N, a0):
    window = np.zeros(N)    
    for n in range(N):
        window[n] = a0 - (1 - a0) * math.cos((2*np.pi*n)/(N-1))

    return window

def print_bins(cqt, notes):
    print(chr(27) + "[2J")
    for k in range(len(cqt)):
            print(notes[k] + ' ' + '|' * int(cqt[k]/0.001)) 

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
        t_kernels[k_cq] = [0] * N_max 
        for n in range(int(N_max/2 - N[k_cq]/2), int(N_max/2 + N[k_cq]/2)):
            t_kernels[k_cq][n] = (a0 - (1 - a0) * math.cos((2*np.pi*(n-(N_max/2 - N[k_cq]/2))/N[k_cq]))) * np.exp(2*np.pi*freqs[k_cq]*(n-N_max/2)*1j/sampling_rate) / N[k_cq] 

        t_kernels[k_cq] = np.real(t_kernels[k_cq])
        s_kernels[k_cq] = [0] * fft_length
        delta_n = int(N_max/fft_length)
        for k in range(fft_length):
            s_kernels[k_cq][k] = t_kernels[k_cq][k*delta_n]
        s_kernels[k_cq] = np.fft.fft(s_kernels[k_cq])
        s_kernels[k_cq] = s_kernels[k_cq][0:int(len(s_kernels[k_cq])/2)]

        non_zero = []
        for i in range(len(s_kernels[k_cq])):
            if np.absolute(s_kernels[k_cq][i]) <= MINVAL:
                s_kernels[k_cq][i] = 0
            else:
                non_zero.append(i) 
        sum_bounds[k_cq] = (min(non_zero), max(non_zero) )

    return s_kernels, sum_bounds, N


#------------------ Main Code ------------------#

# Set up for pyqtgraph
app = QtGui.QApplication([])
win = pg.GraphicsWindow(title="Constant-Q Transform of Audio")
win.resize(1000,600)
win.setWindowTitle('Audio Visualizer')

pg.setConfigOptions(antialias=True)

cqtplot = win.addPlot(title='Constant-Q Transform')
curve = cqtplot.plot(pen='y')
cqtplot.setRange(yRange=(0, 0.1))
cqtplot.enableAutoRange('y', False)

# Opening the audio file
wf = wave.open("./audio/plaza.wav", "rb")

# Making a pyaudio object
p = pyaudio.PyAudio()   

# Open a stream with the object
channels = wf.getnchannels()
stream = p.open(format=p.get_format_from_width(wf.getsampwidth()),
                channels=wf.getnchannels(),
                rate=wf.getframerate(),
                output=True)

# Read in the initial data as bytes and generate the window function
CHUNK = 1024 
#BINS = 8 
a0 = 25.0/46.0
hamming = hamming_window(CHUNK, a0)
data = wf.readframes(CHUNK)
data_converted = np.zeros(CHUNK) 
kernels, bounds, N = gen_kernels(8.175, 123.5, wf.getframerate(), fft_length=CHUNK)
cqt = [0] * len(kernels)
notes = ['C1', 'Db', 'D', 'Eb', 'E', 'F', 'Gb', 'G', 'Ab', 'A', 'Bb', 'B',
'C2', 'Db', 'D', 'Eb', 'E', 'F', 'Gb', 'G', 'Ab', 'A', 'Bb', 'B',
'C3', 'Db', 'D', 'Eb', 'E', 'F', 'Gb', 'G', 'Ab', 'A', 'Bb', 'B',
'C4', 'Db', 'D', 'Eb', 'E', 'F', 'Gb', 'G', 'Ab', 'A', 'Bb', 'B', 'C5']
prev_bins = [1] * len(kernels)


def update():
    global CHUNK, data, data_converted, kernels, bounds, N, cqt, notes, prev_bins, p, wf, curve

    #while len(data) == CHUNK*4:
    stream.write(data)  
    n = 0 

    for i in struct.iter_unpack('%ih' % (channels), data):
        data_converted[n] = i[0]
        n += 1

    data_converted = [float(val) / pow(2, 15) for val in data_converted]
    data_converted = np.multiply(data_converted, hamming)
    freq = np.fft.fft(data_converted)
    freq = freq[0:int(len(freq)/2)]

    for k_cq in range(len(cqt)):
        cqt[k_cq] = 0
        for k in range(bounds[k_cq][0], bounds[k_cq][1]+1):
            cqt[k_cq] += freq[k] * kernels[k_cq][k]   
        cqt[k_cq] = abs(cqt[k_cq])

    curve.setData(cqt)

    #    print(chr(27) + "[2J")
    #    for k in range(len(cqt)):
    #        if cqt[k] >= prev_bins[k]:
    #            print(notes[k] + ' ' + '|' * int(cqt[k]/0.001)) 
    #            prev_bins[k] = cqt[k]
    #        elif prev_bins[k] > 0:
    #            prev_bins[k] -= 0.1
    #            print(notes[k] + ' ' + '|' * int((prev_bins[k])/0.001)) 
    #        else:
    #            print(notes[k]) 
       
    data = wf.readframes(CHUNK)
timer = QtCore.QTimer()
timer.timeout.connect(update)
timer.start(1000*int(CHUNK/wf.getframerate()))

QtGui.QApplication.instance().exec_()
stream.stop_stream()
stream.close()

p.terminate()
