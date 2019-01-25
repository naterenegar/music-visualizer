import pyaudio, wave, struct, math
import numpy as np
import matplotlib.pyplot as plt
from pyqtgraph.Qt import QtGui, QtCore
import pyqtgraph as pg
import constantq as cq

# Set up for pyqtgraph
app = QtGui.QApplication([])
win = pg.GraphicsWindow(title="Constant-Q Transform of Audio")
win.resize(1000,600)
win.setWindowTitle('Music Visualizer')

pg.setConfigOptions(antialias=True)

cqtplot = win.addPlot(title='Constant-Q Transform')
curve = cqtplot.plot(shadowPen='g', fillLevel=0, fillBrush=.5, stepMode=True)
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

# Read in the initial chunk of data, and make an array for the float representatin of it
CHUNK = 4096 
bytes_data = wf.readframes(CHUNK)
float_data = np.zeros(CHUNK) 
kernels, bounds, N = cq.gen_kernels(54, 107, wf.getframerate(), fft_length=CHUNK)
cqt = [0] * len(kernels)
prev_bins = [0] * len(kernels) # 2000 is an arbitrary number, perhaps use sys.maxint in the future?
x_vals = [i for i in range((len(kernels) + 1))]
fft_x_vals = [i for i in range(CHUNK+1)]
hamming = cq.hamming_window(CHUNK, 25/46)

first_transform = True # This will mark the first set of data we transform.  This is used for the slow falling effect

def update():
    global CHUNK, bytes_data, float_data, kernels, bounds, N, cqt, notes, prev_bins, p, wf, curve, first_transform

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
            cqt[k_cq] += data_fft[k] * np.conj(kernels[k_cq][k])
        cqt[k_cq] = np.abs(cqt[k_cq])

        if cqt[k_cq] < prev_bins[k_cq]:
            prev_bins[k_cq] = 0.90 * prev_bins[k_cq]
        else:
            prev_bins[k_cq] = cqt[k_cq]

    if first_transform:
        prev_bins = cqt.copy()
        first_transform = False 

    stream.write(bytes_data)
    curve.setData(y=prev_bins, x=x_vals)
    bytes_data = wf.readframes(CHUNK)

timer = QtCore.QTimer()
timer.timeout.connect(update)
timer.start(1000*int(CHUNK/wf.getframerate()))

QtGui.QApplication.instance().exec_()
stream.stop_stream()
stream.close()

p.terminate()
