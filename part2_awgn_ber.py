import numpy as np
import matplotlib.pyplot as plt

# =========================
# PARAMETERS
# =========================
NUM_BITS = 10000
SAMPLES_PER_SYMBOL = 10

snr_db_range = np.arange(-2, 11, 1)  # SNR σε dB
ber_values = []

# =========================
# 1. TRANSMITTER
# =========================
bits = np.random.randint(0, 2, NUM_BITS)

symbols = 2 * bits - 1

tx_signal = np.repeat(symbols, SAMPLES_PER_SYMBOL)

# =========================
# LOOP OVER SNR VALUES
# =========================
for snr_db in snr_db_range:

    # =========================
    # 2. CHANNEL (AWGN)
    # =========================
    signal_power = np.mean(tx_signal**2)

    snr_linear = 10**(snr_db / 10)

    noise_variance = (signal_power * SAMPLES_PER_SYMBOL) / (2 * snr_linear)

    noise = np.sqrt(noise_variance) * np.random.randn(len(tx_signal))

    rx_signal = tx_signal + noise

    # =========================
    # 3. RECEIVER
    # =========================
    # Downsampling (symbol-rate sampling)
    rx_samples = rx_signal[::SAMPLES_PER_SYMBOL]

    detected_symbols = np.where(rx_samples >=0, 1, -1)

    detected_bits = np.where(detected_symbols == 1, 1, 0)

    # =========================
    # 4. BER CALCULATION
    # =========================
    bit_errors = np.sum(bits != detected_bits)

    ber = bit_errors / len(bits)

    ber_values.append(ber)

# =========================
# 5. PLOT BER vs SNR
# =========================
plt.figure()
plt.semilogy(snr_db_range, ber_values, 'o-')
plt.xlabel("SNR (dB)")
plt.ylabel("Bit Error Rate (BER)")
plt.title("BER vs SNR for BPSK over AWGN")
plt.grid(True)
plt.show()

# =========================
# 6. PLOT TIME-DOMAIN SIGNAL
# =========================
plt.figure()
plt.plot(tx_signal[:200], label="Tx signal")
plt.plot(rx_signal[:200], label="Rx signal (with noise)", alpha=0.7)
plt.legend()
plt.grid(True)
plt.show()

# =========================
# 7. CONSTELLATION DIAGRAM
# =========================
plt.figure()
plt.scatter(rx_samples, np.zeros_like(rx_samples), alpha=0.5)
plt.title(f"Constellation at SNR = {snr_db} dB")
plt.xlabel("In-phase")
plt.grid(True)
plt.show()