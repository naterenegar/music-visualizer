import pyaudio, wave, struct, math
import numpy as np
import matplotlib.pyplot as plt
from pyqtgraph.Qt import QtGui, QtCore
import pyqtgraph as pg
import constantq as cq
import time

# Set up for pyqtgraph
app = QtGui.QApplication([])
win = pg.GraphicsWindow(title="Constant-Q Transform of Audio")
win.resize(1000,600)
win.setWindowTitle('Music Visualizer')

pg.setConfigOptions(antialias=True)

cqtplot = win.addPlot(title='Constant-Q Transform')
curve = cqtplot.plot(shadowPen='g', fillLevel=0, fillBrush=.5, stepMode=True)
cqtplot.setRange(yRange=(0, 100))
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

# Read in the initial chunk of data, and make an array for the float representatin of it
CHUNK = 1024 
bytes_data = wf.readframes(CHUNK)
float_data = np.zeros(CHUNK) 

# Set up the cqt kernels with appropriate FFT lengths 
kernels16, bounds16, N16 = cq.gen_kernels(30, 41, wf.getframerate(), fft_length=16384)
kernels8, bounds8, N8 = cq.gen_kernels(42, 53, wf.getframerate(), fft_length=8192)
kernels4, bounds4, N4 = cq.gen_kernels(54, 64, wf.getframerate(), fft_length=4096)
kernels2, bounds2, N2 = cq.gen_kernels(65, 77, wf.getframerate(), fft_length=2048)
kernels, bounds, N = cq.gen_kernels(78, 127, wf.getframerate(), fft_length=1024)
cqt = [0] * 98
prev_bins = [0] * 98 
vals = [i for i in range(99)]
hamming = cq.hamming_window(CHUNK, 25/46)

first_transform = True # This will mark the first set of data we transform.  This is used for the slow falling effect

data_cumulator = [0] * 16384

# empty lists for different length fft's
data_fft = [0] * 1024
two_fft = [0] * 2048
four_fft = [0] * 4096
eight_fft = [0] * 8192
sixteen_fft = [0] * 16384


def update():
    global CHUNK, bytes_data, float_data, kernels, bounds, N, cqt, notes, prev_bins, p, wf, curve, first_transform, counter, data_cumulator
    global data_fft, two_fft, four_fft, eight_fft, sixteen_fft
    start = time.time()

    # This loop converts the bytes data to float data we can easily work with
    n = 0 
    for i in struct.iter_unpack('%ih' % (channels), bytes_data):
        float_data[n] = i[0]
        n += 1
  
    data_loaded = time.time()  

    # Here we normalize the float data to a 0-1 range then append the data to the cumulator
    float_data = [float(val) / pow(2, 15) for val in float_data]

    data_cumulator[0:15360] = data_cumulator[1024:16384].copy()
    data_cumulator[15360:16384] = float_data

    sixteen_fft = np.fft.fft(data_cumulator)  
    sixteen_fft = sixteen_fft[0:len(sixteen_fft/2)] 
    eight_fft = np.fft.fft(data_cumulator[-8192:16384])
    eight_fft = eight_fft[0:len(eight_fft/2)]
    four_fft = np.fft.fft(data_cumulator[-4096:16384])
    four_fft = four_fft[0:len(four_fft/2)]
    two_fft = np.fft.fft(data_cumulator[-2048:16384])
    two_fft = two_fft[0:len(two_fft/2)]
    data_fft = np.fft.fft(data_cumulator[-1024:16384])
    data_fft = data_fft[0:len(data_fft/2)]

    fft_done = time.time()

    for k_cq in range(98):
        cqt[k_cq] = 0

        if 30 <= k_cq+30 <= 41:
            for k in range(bounds16[k_cq][0], bounds16[k_cq][1]+1):
                cqt[k_cq] += sixteen_fft[k] * kernels16[k_cq][k]
            cqt[k_cq] = np.abs(cqt[k_cq]) 
        elif 42 <= k_cq+30 <= 53:
            for k in range(bounds8[k_cq-12][0], bounds8[k_cq-12][1]+1):
                cqt[k_cq] += eight_fft[k] * kernels8[k_cq-12][k]
            cqt[k_cq] = np.abs(cqt[k_cq])
        elif 54 <= k_cq+30 <= 64:
            for k in range(bounds4[k_cq-24][0], bounds4[k_cq-24][1]+1):
                cqt[k_cq] += four_fft[k] * kernels4[k_cq-24][k]
            cqt[k_cq] = np.abs(cqt[k_cq])
        elif 65 <= k_cq+30 <= 77:
            for k in range(bounds2[k_cq-35][0], bounds2[k_cq-35][1]+1):
                cqt[k_cq] += two_fft[k] * kernels2[k_cq-35][k]
            cqt[k_cq] = np.abs(cqt[k_cq])
        else:
            for k in range(bounds[k_cq-48][0], bounds[k_cq-48][1]+1):
                cqt[k_cq] += data_fft[k] * kernels[k_cq-48][k]
            cqt[k_cq] = np.abs(cqt[k_cq])

        if cqt[k_cq] < prev_bins[k_cq]:
            prev_bins[k_cq] = 0.90 * prev_bins[k_cq]
        else:
            prev_bins[k_cq] = cqt[k_cq]

    cqt_done = time.time()

    if first_transform:
        prev_bins = cqt.copy()
        first_transform = False 

    stream.write(bytes_data)
    curve.setData(y=prev_bins, x=vals)
    data_displayed = time.time()
    bytes_data = wf.readframes(CHUNK)

#    print("Load time: " + str(data_loaded-start))
#    print("FFT Time: " + str(fft_done - data_loaded))
#    print("CQT Time: " + str(cqt_done - fft_done))
#    print("Display Time: " + str(data_displayed - cqt_done))
#    print("Total Time: " + str(data_displayed - start))

timer = QtCore.QTimer()
timer.timeout.connect(update)
timer.start(1000*int(CHUNK/wf.getframerate()))

QtGui.QApplication.instance().exec_()
stream.stop_stream()
stream.close()

p.terminate()
