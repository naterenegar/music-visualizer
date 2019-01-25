# Spectrum Analyzer 
## A Real Time Constant-Q Transform Implementation
This project utilizes the algorithm described in [this paper](http://academics.wellesley.edu/Physics/brown/pubs/effalgV92P2698-P2701.pdf) by Brown and Puckette to visualize music in a more accurate way than a conventional FFT.  

## How does it work?
The Constant-Q transform is ideal for musical applications because it transforms to log-space frequency.  This gives better resolution at lower frequencies, allowing us to accurately distinguish between notes.  Each kernel (defined in the mentioned paper) can be modeled as a vector, and the magnitude of a particular note can be noted as complex magnitude of the dot between the FFT of the data and the kernel.

## Issues
The current framerate is pretty low.  With `sampling_rate = 44100` and `Q = 17` (required for quarter tone spacing), the minimum frequency for `fft_length = 1024` is `sampling_rate * Q / fft_length ~= 732 Hz`. We can decrease the minimum frequency by increasing `fft_length`, however this impacts the refresh rate, with a transform happening every `fft_length / sampling_rate` seconds. I'm looking for a way to do a real time Constant-Q transform for low frequencies with a reasonable refresh rate.