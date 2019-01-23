function generate_wav_file(freqs, duration, filename)
Fs = 44100;
audio = zeros(1, Fs*duration);
t = 1/Fs:1/Fs:duration;
for i = 1:length(freqs)
    audio = audio + cos(2*pi*freqs(i).*t)/length(freqs);
end

audiowrite(filename, audio, Fs)
end

