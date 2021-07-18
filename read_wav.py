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

num_circles = 30 

cqtplot = win.addPlot(title='Constant-Q Transform', row=0, col=0)
circleplot = win.addPlot(title='Circular Plot', row=0, col=1)
circleplot.setRange(xRange=(-1, 1), yRange=(-1,1))
circle_curve = circleplot.plot(symbol='o',symbolPen='g',symbolSize=200/num_circles,connect=np.zeros(num_circles))
curve = cqtplot.plot([0,1], [0], shadowPen='g', fillLevel=0, fillBrush=.5, stepMode='center')
cqtplot.setRange(yRange=(0, 100))
cqtplot.enableAutoRange('y', False)

# Opening the audio file
wf = wave.open("./audio/note76.wav", "rb")
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
data_bytes = wf.readframes(CHUNK)
float_data = np.zeros(CHUNK) 

# Set up the cqt kernels with appropriate FFT lengths 
# We do multiple FFTs so that the energy doesn't linger for too long in the
# high frequencies
kernels16, bounds16, N16 = cq.gen_kernels(42, 53, wf.getframerate(), fft_length=16384) # 16384 / 44100 = 372 ms
kernels8, bounds8, N8 = cq.gen_kernels(54, 65, wf.getframerate(), fft_length=8192)    # 8192  / 44100 = 186 ms
kernels4, bounds4, N4 = cq.gen_kernels(66, 77, wf.getframerate(), fft_length=4096)   # 4096  / 44100 = 93  ms
kernels2, bounds2, N2 = cq.gen_kernels(78, 89, wf.getframerate(), fft_length=2048)   # 2048  / 44100 = 46  ms
kernels, bounds, N = cq.gen_kernels(90, 101, wf.getframerate(), fft_length=1024)      # 1024  / 44100 = 23  ms

n_bins_list = [len(kernels16), len(kernels8), len(kernels4), len(kernels2), len(kernels)]
n_bins_acc = [0] * len(n_bins_list)
n_bins_acc[0] = n_bins_list[0]

for i in range(1, len(n_bins_list)):
    n_bins_acc[i] = n_bins_acc[i-1] + n_bins_list[i]

n_bins = n_bins_acc[4]
cqt = np.zeros(n_bins)
prev_bins = np.zeros(n_bins)
vals = [i for i in range(n_bins + 1)]

first_transform = True # This will mark the first set of data we transform.  This is used for the slow falling effect

data_cumulator = np.zeros(16384)

# empty lists for different length fft's
fft1 = np.zeros(512, dtype='complex_')
fft2 = np.zeros(1024, dtype='complex_')
fft4 = np.zeros(2048, dtype='complex_')
fft8 = np.zeros(4096, dtype='complex_')
fft16 = np.zeros(8192, dtype='complex_')

counter = np.zeros(num_circles) 
exp_circ = np.arange(num_circles, dtype='complex_') / num_circles 

# We should be running the transform Fs / window_stride times per second, so
# 44100 / 1024 ~= 43
# If we change the stride to 1050 then we run an even 42 times per second
def update():
    global CHUNK, data_bytes, float_data, kernels, bounds, N, cqt, notes, prev_bins, p, wf, curve, first_transform, counter, data_cumulator
    global fft1, fft2, fft4, fft8, fft16
    global exp_circ, num_circles, counter
    start = time.time()

    # This loop converts the bytes data to float data we can easily work with
    n = 0 
    for i in struct.iter_unpack('%ih' % (channels), data_bytes):
        float_data[n] = i[0]
        n += 1
  
    data_loaded = time.time()  

    # Here we normalize the float data to a 0-1 range then append the data to the cumulator
    float_data = [float(val) / pow(2, 15) for val in float_data]

    # Shift over the window
    data_cumulator[0:15360] = np.copy(data_cumulator[1024:16384])
    data_cumulator[15360:16384] = float_data

    # Calculate the new FFTs, with their last sample on the end of the window
    fft1 = np.fft.fft(data_cumulator[-1024:16384])[0:512]
    fft2 = np.fft.fft(data_cumulator[-2048:16384])[0:1024]
    fft4 = np.fft.fft(data_cumulator[-4096:16384])[0:2048]
    fft8 = np.fft.fft(data_cumulator[-8192:16384])[0:4096]
    fft16 = np.fft.fft(data_cumulator)[0:8192]
    fft_done = time.time()

    for k_cq in range(n_bins):
        cqt[k_cq] = 0
        if 0 <= k_cq < n_bins_acc[0]:
            lower = bounds16[k_cq][0]
            upper = bounds16[k_cq][1]
            cqt[k_cq] = np.abs(np.sum(np.multiply(fft16[lower:upper], kernels16[k_cq][lower:upper])))

        elif n_bins_acc[0] <= k_cq < n_bins_acc[1]:
            k_cq_tmp = k_cq - n_bins_acc[0]
            lower = bounds8[k_cq_tmp][0]
            upper = bounds8[k_cq_tmp][1]
            cqt[k_cq] = np.abs(np.sum(np.multiply(fft8[lower:upper], kernels8[k_cq_tmp][lower:upper])))

        elif n_bins_acc[1] <= k_cq < n_bins_acc[2]:
            k_cq_tmp = k_cq - n_bins_acc[1]
            lower = bounds4[k_cq_tmp][0]
            upper = bounds4[k_cq_tmp][1]
            cqt[k_cq] = np.abs(np.sum(np.multiply(fft4[lower:upper], kernels4[k_cq_tmp][lower:upper])))

        elif n_bins_acc[2] <= k_cq < n_bins_acc[3]:
            k_cq_tmp = k_cq - n_bins_acc[2]
            lower = bounds2[k_cq_tmp][0]
            upper = bounds2[k_cq_tmp][1]
            cqt[k_cq] = np.abs(np.sum(np.multiply(fft2[lower:upper], kernels2[k_cq_tmp][lower:upper])))

        else:
            k_cq_tmp = k_cq - n_bins_acc[3]
            lower = bounds[k_cq_tmp][0]
            upper = bounds[k_cq_tmp][1]
            cqt[k_cq] = np.abs(np.sum(np.multiply(fft1[lower:upper], kernels[k_cq_tmp][lower:upper])))

        # This is really just a first order difference equation y[n] = 0.9 * y[n-1]
        if cqt[k_cq] < 0.95 * prev_bins[k_cq]:
            prev_bins[k_cq] = 0.95 * prev_bins[k_cq]
        else:
            prev_bins[k_cq] = cqt[k_cq]

    cqt_done = time.time()

    if first_transform:
        prev_bins = cqt.copy()
        first_transform = False 

    # Now we want to take the bins and collaspe average them so that we can
    # turn them into circles  
    bins_per_circle = int(n_bins / num_circles)
    for i in range(num_circles-1):
        counter[i] += np.sum(cqt[i*bins_per_circle:bins_per_circle*(i+1)])
    counter[num_circles-1] += np.sum(cqt[i*bins_per_circle:])
   
    new_points = np.multiply(exp_circ, np.exp(counter * 1j / 5000))

    circle_curve.setData(x=np.real(new_points), y=np.imag(new_points))
    stream.write(data_bytes)
    curve.setData(y=prev_bins, x=vals)
    data_displayed = time.time()
    data_bytes = wf.readframes(CHUNK)

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
