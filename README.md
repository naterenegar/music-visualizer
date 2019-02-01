# Spectrum Analyzer 
## A Real Time Constant-Q Transform Implementation
This project utilizes the algorithm described in [this paper](http://academics.wellesley.edu/Physics/brown/pubs/effalgV92P2698-P2701.pdf) by Brown and Puckette to visualize music in a more accurate way than a conventional FFT.  

## How does it work?
The Constant-Q transform is ideal for musical applications because it transforms to log-space frequency.  This gives better resolution at lower frequencies, allowing us to accurately distinguish between notes.  Each kernel (defined in the mentioned paper) can be modeled as a vector, and the magnitude of a particular note can be noted as complex magnitude of the dot between the FFT of the data and the kernel. 

A sliding window of `16384` samples moves `1024` samples at a time, and the CQT is taken of the window each time it advances.  To decrease the minimum frequency by an octave, the window must double in size.

## Issues / Future Features
- Only `*.wav` files are supported currently, should find a way to support `*.mp3`
- Add functionality to change songs, pause, play, skip (i.e. turn this into a music player that shows the spectrum)
- Add on the fly changes to the CQT (e.g. change the frequency bounds)

