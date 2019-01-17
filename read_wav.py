import pyaudio, wave, struct, math
import numpy as np
import matplotlib.pyplot as plt
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



#------------------ Main Code ------------------#

# Set up for pyqtgraph
app = QtGui.QApplication([])
win = pg.GraphicsWindow(title="Constant-Q Transform of Audio")
win.resize(1000,600)
win.setWindowTitle('Audio Visualizer')

pg.setConfigOptions(antialias=True)

cqtplot = win.addPlot(title='Constant-Q Transform')
curve = cqtplot.plot(pen='g', fillLevel=0, fillBrush='g', stepMode=True)
cqtplot.setRange(yRange=(0, 0.1))
cqtplot.enableAutoRange('y', False)

# Opening the audio file
wf = wave.open("./audio/spirited_away.wav", "rb")

# Making a pyaudio object
p = pyaudio.PyAudio()   

# Open a stream with the object
channels = wf.getnchannels()
stream = p.open(format=p.get_format_from_width(wf.getsampwidth()),
                channels=wf.getnchannels(),
                rate=wf.getframerate(),
                output=True)

# Read in the initial chunk of data, and make an array for the float representatin of it
CHUNK = 1024 
bytes_data = wf.readframes(CHUNK)
float_data = np.zeros(CHUNK) 
kernels, bounds, N = gen_kernels(65.4063913251, 523, wf.getframerate(), fft_length=CHUNK)
cqt = [0] * len(kernels)
prev_bins = [1] * len(kernels)
x_vals = [0] * (len(kernels) + 1)
for i in range(len(x_vals)):
    x_vals[i] = i

hamming = hamming_window(CHUNK, 25/46)


def update():
    global CHUNK, bytes_data, float_data, kernels, bounds, N, cqt, notes, prev_bins, p, wf, curve

    stream.write(bytes_data)  

    # This loop converts the bytes data to float data we can easily work with
    n = 0 
    for i in struct.iter_unpack('%ih' % (channels), bytes_data):
        float_data[n] = i[0]
        n += 1
    
    # Here we normalize the float data
    float_data = [float(val) / pow(2, 15) for val in float_data]
    float_data *= hamming
    data_fft = np.fft.fft(float_data)
    data_fft = data_fft[0:int(len(data_fft)/2)]

    for k_cq in range(len(cqt)):
        cqt[k_cq] = 0
        for k in range(bounds[k_cq][0], bounds[k_cq][1]+1):
            cqt[k_cq] += data_fft[k] * kernels[k_cq][k]   
        cqt[k_cq] = np.abs(cqt[k_cq])

        
    curve.setData(y=cqt, x=x_vals)
    bytes_data = wf.readframes(CHUNK)

timer = QtCore.QTimer()
timer.timeout.connect(update)
timer.start(1000*int(CHUNK/wf.getframerate()))

QtGui.QApplication.instance().exec_()
stream.stop_stream()
stream.close()

p.terminate()
