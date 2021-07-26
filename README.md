# Music Visualizer 
## A Real Time Constant-Q Transform Implementation
One algorithm for music visualiztion is the Constant-Q Transform. The CQT can identify individual notes. This project utilizes the specific algorithm described in [this paper](http://academics.wellesley.edu/Physics/brown/pubs/effalgV92P2698-P2701.pdf) by Brown and Puckette to extract frequency content in a more appropriate way than the standard FFT.  

## How does it work?
The goal of the Constant-Q transform is to maintain a constant ratio, Q, between the frequency of a DFT bin and the frequency to the next DFT bin (f/delta).  With the standard DFT, this ratio is proportional to frequency because the bin spacing is constant. A constant spacing ratio makes geometrically spaced bins, matching the geometric spacing of notes. 

The main insight of the paper mentioned above is that Parseval's Identity can be applied to the definition of the CQT, which means the FFT can be used to speed up computation. 

A sliding window of `16384` samples moves `1024` samples at a time, and the CQT is taken of the window each time it advances.  To decrease the minimum frequency by an octave, the window must double in size. This is necessary to maintain the constant frequency resolution.

## Issues / Future Features
- Only `*.wav` files are supported currently, should find a way to support `*.mp3`
- Add functionality to change songs, pause, play, skip (i.e. turn this into a music player that shows the spectrum)
- Add on the fly changes to the CQT (e.g. change the frequency bounds)
- Rewrite the transform code in C++ for easy portability to a microcontroller (I want to make this into hardware at some point!)

