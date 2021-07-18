# This script generates the midinote frequencies in Hz
# TODO: If we migrate to C++, these can be calculated at compile time with
# constexpr. Will this end up in ROM for a microcontroller?

mut = [0.0] * 128 
mut[69] = 440.0
for i in range(70, 128):
    mut[i] = pow(2, 1/12) * mut[i-1]
for i in range(68, -1, -1):
    mut[i] = mut[i+1] / pow(2, 1/12)

midinotes = tuple(mut)

quarter_tones = [0.0] * 256
quarter_tones[138] = 440.0
for i in range(139, 256):
    quarter_tones[i] = pow(2, 1/24) * quarter_tones[i-1]
for i in range(137, -1, -1):
    quarter_tones[i] = quarter_tones[i+1] / pow(2, 1/24)

quarter_tones = tuple(quarter_tones)


