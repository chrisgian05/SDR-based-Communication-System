import numpy as np
import matplotlib.pyplot as plt

# =========================
# PARAMETERS
# =========================
NUM_BITS = 2000
SAMPLES_PER_SYMBOL = 10

snr_db = 10            
beta = 0.25            # roll-off factor RRC
span = 6               # φίλτρο σε σύμβολα

# =========================
# 1. TRANSMITTER
# =========================
bits = np.random.randint(0, 2, NUM_BITS)

symbols = 2 * bits -1

# Upsampling (ΔΕΝ είναι oversampling)
upsampled_symbols = np.zeros(len(symbols) * SAMPLES_PER_SYMBOL)
upsampled_symbols[::SAMPLES_PER_SYMBOL] = symbols

# =========================
# 2. RRC FILTER DESIGN
# =========================
num_taps = span * SAMPLES_PER_SYMBOL + 1
t = np.linspace(-span/2, span/2, num_taps)
rrc_filter = np.zeros(num_taps)

for i in range(num_taps):
    ti = t[i]
    if abs(ti) < 1e-12:
        rrc_filter[i] = 1.0 - beta + (4*beta)/np.pi
    elif np.isclose(abs(4*beta*ti), 1.0):
        rrc_filter[i] = (beta/np.sqrt(2)) * (
            (1 + 2/np.pi) * np.sin(np.pi/(4*beta)) +
            (1 - 2/np.pi) * np.cos(np.pi/(4*beta))
        )
    else:
        numerator = np.sin(np.pi*ti*(1-beta)) + 4*beta*ti*np.cos(np.pi*ti*(1+beta))
        denominator = np.pi*ti*(1-(4*beta*ti)**2)
        rrc_filter[i] = numerator / denominator

# Normalize energy
rrc_filter *= np.sqrt(SAMPLES_PER_SYMBOL / np.sum(rrc_filter**2))

# =========================
# 3. PULSE SHAPING
# =========================
tx_signal = np.convolve(upsampled_symbols, rrc_filter, mode ='full')

# =========================
# 4. CHANNEL (AWGN)
# =========================
snr_linear = 10**(snr_db / 10)

sig_power = np.mean(tx_signal**2)

noise_variance = (sig_power * SAMPLES_PER_SYMBOL) / (2 * snr_linear)

noise = np.sqrt(noise_variance) * np.random.randn(len(tx_signal))
rx_signal = tx_signal + noise

# =========================
# 5. MATCHED FILTER (RECEIVER)
# =========================
rx_filtered = np.convolve(rx_signal, rrc_filter, mode='full')

# =========================
# 6. EYE DIAGRAM
# =========================
eye_span = 2 * SAMPLES_PER_SYMBOL

plt.figure()
for i in range(0, len(rx_filtered) - eye_span, SAMPLES_PER_SYMBOL):
    plt.plot(rx_filtered[i:i+eye_span], color='blue', alpha=0.2)

plt.title("Eye Diagram")
plt.xlabel("Sample index")
plt.ylabel("Amplitude")
plt.grid(True)
plt.show()

# =========================
# 7. SYMBOL SAMPLING
# =========================
group_delay = (num_taps - 1) // 2
total_delay = 2 * group_delay

sample_offset = total_delay 

rx_samples = rx_filtered[sample_offset::SAMPLES_PER_SYMBOL]

rx_samples = rx_samples[:NUM_BITS]

# =========================
# 8. DECISION & BER
# =========================
detected_symbols = np.where(rx_samples >=0, 1, -1)

detected_bits = np.where(detected_symbols == 1, 1, 0)

min_len = min(len(bits), len(detected_bits))

bit_errors = np.sum(bits[:min_len] != detected_bits[:min_len])

ber = bit_errors / min_len

print(ber)