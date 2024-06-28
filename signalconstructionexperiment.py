import numpy as np
import matplotlib.pyplot as plt

def compute_fourier_coeffs(fft_freq, fft_result, target_frequencies):
    # Initialize coefficients
    # A_0 = fft_result[0].real / len(fft_result)  # DC component
    C_n = []
    S_n = []

    for target_freq in target_frequencies:
        # Find the closest FFT coefficient to the target frequency
        idx = (np.abs(fft_freq - target_freq)).argmin()
        
        # Compute the coefficients
        C_n.append(2 * fft_result[idx].real / len(fft_result))
        S_n.append(-2 * fft_result[idx].imag / len(fft_result))
    
    return C_n, S_n

def reconstruct_signal(A_0, C_n, S_n, target_frequencies, time_vector):
    # Initialize the reconstructed signal with float64 type
    reconstructed_signal = np.zeros_like(time_vector, dtype=np.float64)
    
    # Add the DC component
    reconstructed_signal += A_0
    
     # Add each frequency component point by point
    for i, t in enumerate(time_vector):
        for n, (C, S, f) in enumerate(zip(C_n, S_n, target_frequencies), 1):
            reconstructed_signal[i] += C * np.cos(2 * np.pi * f * t) + S * np.sin(2 * np.pi * f * t)
    
    return reconstructed_signal



# Example usage
fft_results = {
    'tesu': (np.array([2.0, 15.0]), np.array([5.341932763340897+141.98924167308073j, 0.4611564998896712+15.486827661637781j])),
    'taicu': (np.array([2.0, 15.0]), np.array([-967.2329398157063+707.9364057164689j,-132.1153316377344+740.7576529692277j])),
    'tisu': (np.array([2.0, 15.0]), np.array([7.18918875439495+145.2885836897493j,3.098025041150109+15.191452847355375j])),
}
target_frequencies = [2, 15]
sampling_interval=1/86400
#using A_0 as mean of data
A_0={
    'tesu': (22.018211967372796),
    'taicu': (21.33066783664287),
    'tisu': (22.741420843065665),
}

# Define the range of time in seconds
start_time = 0
end_time = 86400  # 86400 seconds = 24 hours

# Create the time vector with a step size of 1 second
time_vector = np.arange(start_time, end_time + 1, 1)


for product_flag, (fft_freq, fft_result) in fft_results.items():
    C_n, S_n = compute_fourier_coeffs(fft_freq, fft_result, target_frequencies)
    print(f'Fourier Coefficients for {product_flag}:')
    print(f'A_0: {A_0}')
    for i, (C, S) in enumerate(zip(C_n, S_n)):
        print(f'C_{i+1}: {C}, S_{i+1}: {S}')
    print('')

    # Reconstruct the signal
    reconstructed_signal = reconstruct_signal(A_0[product_flag], C_n, S_n, target_frequencies, time_vector)

    # Plot the reconstructed signal
    plt.figure(figsize=(14, 7))
    plt.plot(time_vector, reconstructed_signal, label=f'Reconstructed Signal for {product_flag}')
    plt.xlabel('Time (days)')
    plt.ylabel('Amplitude')
    plt.title(f'Reconstructed Signal for {product_flag}')
    plt.legend()
    plt.grid(True)
    plt.show()