fmin = 174.6;
freqs = [];
f = fmin;
k_cq = 1;
fs = 44100;
Q = 17;
a0 = 25/46;
while f < 8000
    freqs = [freqs, f];
    f = (2^(1/12)^k_cq)*fmin;
    k_cq = k_cq + 1;
end
    
N = zeros(size(freqs));
for k_cq = 1:length(N)
    N(k_cq) = round((fs * Q)/freqs(k_cq));
end
for k_cq = 1:length(freqs)
    window = zeros(1, N(1));
    for n = (round(N(1)/2 - N(k_cq)/2)+1):(round(N(1)/2 + N(k_cq)/2)+1)
        window(n) = (a0 - (1 - a0) * cos(2*pi*(n-(N(1)/2 - N(k_cq)/2)+1)/N(k_cq))) * exp(2*pi*freqs(k_cq)*(n-N(1)/2)*1j/fs) / N(k_cq);
    end
    figure;
    subplot(2, 1, 1);
    plot(real(window))
    fft_len = 2048;
    win_fft = zeros(1, fft_len);
    sample_n = floor(N(1)/fft_len);
    for i = 1:fft_len
        win_fft(i) = real(window(i*sample_n));
    end  
    
    subplot(2, 1, 2);
    plot(abs(fft(win_fft)))
end
        
        
        
