# This script generates the midinote frequencies in Hz
mut = [0.0] * 128 
mut[69] = 440.0
for i in range(70, 128):
    mut[i] = pow(2, 1/12) * mut[i-1]
for i in range(68, -1, -1):
    mut[i] = mut[i+1] / pow(2, 1/12)

midinotes = tuple(mut)


