#Header
#Note all Coefficients adn Inputs needed can be given by running fft_tempanalysis_onefile.py

#Imports
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

#This function Transforms the Complex Coefficients into Simple Sinusoidal Fourier Series Constant
def compute_fourier_coeffs(fft_freq, fft_result, target_frequencies,L):
    # Initialize coefficients
    A_n = []
    B_n = []

    for target_freq in target_frequencies:
        # Find the closest FFT coefficient to the target frequency
        idx = (np.abs(fft_freq - target_freq)).argmin()
        
        # Compute the coefficients
        A_n.append(2 * fft_result[idx].real / L)
        B_n.append(-2 * fft_result[idx].imag / L)
    
    return A_n, B_n

#This Signal Reconstructs the Signal from the Principle Sinusoidal Coefficients
def reconstruct_signal(A_0, A_n, B_n, target_frequencies, time_vector, L):
    # Initialize the reconstructed signal with float64 type
    reconstructed_signal = np.zeros_like(time_vector, dtype=np.float64)
    
    # Add the DC component
    reconstructed_signal += A_0
    
     # Add each frequency component point by point
    for i, t in enumerate(time_vector):
        for n, (C, S, f) in enumerate(zip(A_n, B_n, target_frequencies), 1):
            reconstructed_signal[i] += C * np.cos(2 * np.pi * f * t/L) + S * np.sin(2 * np.pi * f * t/L)
    
    return reconstructed_signal

# This is where you enter the Constants provided by fft_tempanalysis_onefile
fft_results = {
    'tesu': (np.array([2,15]), np.array([ 5.341932763340897+141.98924167308073j,0.4611564998896712+15.486827661637781j])),
    'taicu': (np.array([1, 2, 6, 8, 10, 15, 30]), np.array([-777.0011344928828+753.1330223387438j, -967.2329398157063+707.9364057164689j, 283.38875790826006-9.710721789929035j,-209.3126651868165+31.003807286664014j, 111.22818430281612+270.19284713207486j, -132.1153316377344+740.7576529692277j, -165.1638369745679+170.23471930346335j])),
    'tisu': (np.array([2,15]), np.array([7.18918875439495+145.2885836897493j,3.098025041150109+15.191452847355375j])),
}

#Specify Target Frequencies
target_frequencies = [1,2,6,8,10,15,30]

#Specify Sampling Interval
sampling_interval=1/86400

#Enter A_0 as mean of data
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
L = end_time - start_time  # Interval length

#This For Loop executes the functions, and plots the Reconstructed Signal
for product_flag, (fft_freq, fft_result) in fft_results.items():

    #Call the function to transform the complex coefficients into simple coefficients
    A_n, B_n = compute_fourier_coeffs(fft_freq, fft_result, target_frequencies, L)

    #Print out Results
    print(f'Fourier Coefficients for {product_flag}:')
    print(f'A_0: {A_0[product_flag]}')
    for i, (C, S) in enumerate(zip(A_n, B_n)):
        print(f'A_{i+1}: {C}, B_{i+1}: {S}')
    print('')

    #Call the Function to Reconstruct the Signal
    reconstructed_signal = reconstruct_signal(A_0[product_flag], A_n, B_n, target_frequencies, time_vector, L)

    # Plot the reconstructed signal
    # plt.figure(figsize=(14, 7))
    # plt.plot(time_vector, reconstructed_signal, 'or', label=f'Reconstructed Signal for {product_flag}')
    # plt.xlabel('Time (days)')
    # plt.ylabel('Amplitude')
    # plt.title(f'Reconstructed Signal for {product_flag}')
    # plt.legend()
    # plt.grid(True)
    # plt.show()

#Let's Construct a Synthetic time Series 

#Choose 2/day and 1/rev as the fundamental frequencies
synthetic_freqs=[2/86400, 15/86400]

#Choose some constants
a_n= [-0.2, -0.1]

#Choose some more constacts
b_n=[0.2,0.1] 

# Create synthetic time series
synthetic_signal = np.zeros_like(time_vector, dtype=np.float64)

#Choose a DC for the Signal
A_0=22.0

#Add the DC component into the Signal
synthetic_signal+=A_0
for freq, amp_sin, amp_cos in zip(synthetic_freqs, a_n, b_n):
    synthetic_signal += amp_sin * np.sin(2 * np.pi * freq * time_vector)+amp_cos*np.cos(2*np.pi*freq*time_vector) 
# Plot the synthetic signal
plt.figure(figsize=(14, 7))
plt.plot(time_vector, synthetic_signal, 'or', label='Synthetic Signal')
plt.xlabel('Time (seconds)')
plt.ylabel('Amplitude')
plt.title('Synthetic Signal')
plt.legend()
plt.grid(True)
plt.show()

# Perform FFT on synthetic signal
fft_result_synthetic = np.fft.fft(synthetic_signal)
fft_freq_synthetic = np.fft.fftfreq(len(time_vector), d=1)  # Sampling interval of 1 second

# Filter only positive frequencies
positive_freq_idxs = np.where(fft_freq_synthetic > 0)
fft_freq_synthetic = fft_freq_synthetic[positive_freq_idxs]
fft_result_synthetic = fft_result_synthetic[positive_freq_idxs]

# Plot the FFT result (magnitude spectrum)
plt.figure(figsize=(14, 7))
plt.plot(fft_freq_synthetic*86400, np.abs(fft_result_synthetic), 'or')
plt.xscale('log')
plt.yscale('log')
plt.xlabel('Frequency (Cycles/Day)')
plt.ylabel('Amplitude')
plt.title('FFT of Synthetic Signal')
plt.grid(True)
plt.show()

# Create a DataFrame for the FFT results
fft_df = pd.DataFrame({
    'Frequency (Cycles/Day)': fft_freq_synthetic*86400,
    'Amplitude': np.abs(fft_result_synthetic)
})

# Save the DataFrame to a text file (CSV format)
fft_df.to_csv('fft_synthetic_results.csv', index=False)

# Optionally, save the DataFrame to a text file (TSV format)
# fft_df.to_csv('fft_results.tsv', index=False, sep='\t')



#APPENDIX WITH ALL RELEVANT CONSTANTS FOR TESU DATASET:
    # Sampling Frequency in Cycles/Day for tesu is : 18001.61307347744
    # DC Component (A_0) for tesu: 22.018211967372793
    # Product Flag: tesu
    # Target Frequency: 1.1574074074074073e-05 Hz
    # Closest Frequency: 1.1573825306368635e-05 Hz
    # Amplitude: 226.7293376364118
    # Phase: 1.9050367794867125
    # FFT Coefficient: (-74.37895616910404+214.18208007263473j)

    # Product Flag: tesu
    # Target Frequency: 2.3148148148148147e-05 Hz
    # Closest Frequency: 2.314765061273727e-05 Hz
    # Amplitude: 142.08969349162726
    # Phase: 1.533191966946512
    # FFT Coefficient: (5.341932763340897+141.98924167308073j)

    # Product Flag: tesu
    # Target Frequency: 0.00017361111111111112 Hz
    # Closest Frequency: 0.00017360737959552954 Hz
    # Amplitude: 15.493692146762811
    # Phase: 1.5410277853285008
    # FFT Coefficient: (0.4611564998896712+15.486827661637781j)

    # Product Flag: tesu
    # Target Frequency: 0.00034722222222222224 Hz
    # Closest Frequency: 0.00034721475919105907 Hz
    # Amplitude: 4.9088739190240345
    # Phase: 1.5393247634806346
    # FFT Coefficient: (0.15446443494159667+4.906443099763061j)

    # Product Flag: tesu
    # Target Frequency: 0.0005208333333333333 Hz
    # Closest Frequency: 0.0005208221387865886 Hz
    # Amplitude: 3.139719906218296
    # Phase: 1.5498637788006246
    # FFT Coefficient: (0.06571753812070902+3.139032063341624j)

#APPENDIX WITH ALL RELEVANT CONSTANTS FOR TAICU DATASET:
    # Sampling Frequency in Cycles/Day for taicu is : 18001.613073581626
    # DC Component (A_0) for taicu: 21.330667836642878
    # Product Flag: taicu
    # Target Frequency: 1.1574074074074073e-05 Hz
    # Closest Frequency: 1.1573825306435618e-05 Hz
    # Amplitude: 1082.0998624620177
    # Phase: 2.371791930763293
    # FFT Coefficient: (-777.0011344928828+753.1330223387438j)

    # Product Flag: taicu
    # Target Frequency: 2.3148148148148147e-05 Hz
    # Closest Frequency: 2.3147650612871236e-05 Hz
    # Amplitude: 1198.6298487870583
    # Phase: 2.509764003171289
    # FFT Coefficient: (-967.2329398157063+707.9364057164689j)

    # Product Flag: taicu
    # Target Frequency: 6.944444444444444e-05 Hz
    # Closest Frequency: 6.944295183861371e-05 Hz
    # Amplitude: 283.55508499490503
    # Phase: -0.0342530303531434
    # FFT Coefficient: (283.38875790826006-9.710721789929035j)

    # Product Flag: taicu
    # Target Frequency: 9.259259259259259e-05 Hz
    # Closest Frequency: 9.259060245148495e-05 Hz
    # Amplitude: 211.59637963319918
    # Phase: 2.9945399003316573
    # FFT Coefficient: (-209.3126651868165+31.003807286664014j)

    # Product Flag: taicu
    # Target Frequency: 0.00011574074074074075 Hz
    # Closest Frequency: 0.00011573825306435618 Hz
    # Amplitude: 292.19151874179715
    # Phase: 1.1802769241377875
    # FFT Coefficient: (111.22818430281612+270.19284713207486j)

    # Product Flag: taicu
    # Target Frequency: 0.00017361111111111112 Hz
    # Closest Frequency: 0.00017360737959653427 Hz
    # Amplitude: 752.4469159257862
    # Phase: 1.7472921642940522
    # FFT Coefficient: (-132.1153316377344+740.7576529692277j)

    # Product Flag: taicu
    # Target Frequency: 0.00034722222222222224 Hz
    # Closest Frequency: 0.00034721475919306853 Hz
    # Amplitude: 237.18969771153763
    # Phase: 2.341076667191042
    # FFT Coefficient: (-165.1638369745679+170.23471930346335j)

    # Product Flag: taicu
    # Target Frequency: 0.0005208333333333333 Hz
    # Closest Frequency: 0.0005208221387896029 Hz
    # Amplitude: 7.972026950718086
    # Phase: 1.5926898973943675
    # FFT Coefficient: (-0.17452219184524376+7.970116417438898j)

#APPENDIX WITH ALL RELEVANT CONSTANTS FOR TISU DATASET:
    # Sampling Frequency in Cycles/Day for tisu is : 18001.61307225105
    # DC Component (A_0) for tisu: 22.741420843065665
    # Product Flag: tisu
    # Target Frequency: 1.1574074074074073e-05 Hz
    # Closest Frequency: 1.1574468260155204e-05 Hz
    # Amplitude: 235.71404121813973
    # Phase: 1.8914507836883303
    # FFT Coefficient: (-74.29417284242142+223.69954204031643j)

    # Product Flag: tisu
    # Target Frequency: 2.3148148148148147e-05 Hz
    # Closest Frequency: 2.3148936520310408e-05 Hz
    # Amplitude: 145.46634313654687
    # Phase: 1.521354521765035
    # FFT Coefficient: (7.18918875439495+145.2885836897493j)

    # Product Flag: tisu
    # Target Frequency: 0.00017361111111111112 Hz
    # Closest Frequency: 0.00017361702390232805 Hz
    # Amplitude: 15.504128442741141
    # Phase: 1.3696227662557579
    # FFT Coefficient: (3.098025041150109+15.191452847355375j)

    # Product Flag: tisu
    # Target Frequency: 0.00034722222222222224 Hz
    # Closest Frequency: 0.0003472340478046561 Hz
    # Amplitude: 4.21041267879033
    # Phase: 1.5929463275134568
    # FFT Coefficient: (-0.09325301808551012+4.209379859354144j)

    # Product Flag: tisu
    # Target Frequency: 0.0005208333333333333 Hz
    # Closest Frequency: 0.0005208510717069842 Hz
    # Amplitude: 3.313037648044643
    # Phase: 1.4714100222555904
    # FFT Coefficient: (0.32872876707291154+3.2966886196697294j)

































