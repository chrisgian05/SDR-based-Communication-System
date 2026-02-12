import numpy as np
import matplotlib.pyplot as plt

# =========================
# PARAMETERS
# =========================
NUM_BITS = 20          
SAMPLES_PER_SYMBOL = 10  

# =========================
# 1. GENERATE RANDOM BITS
# =========================
bits = np.random.randint(0, 2, NUM_BITS)

print("Bits:")
print(bits)

# =========================
# 2. BPSK MAPPING
# =========================
symbols = 2 * bits - 1

print("BPSK symbols:")
print(symbols)

# =========================
# 3. OVERSAMPLING
# =========================
tx_signal = np.repeat(symbols, SAMPLES_PER_SYMBOL)

# =========================
# 4. TIME AXIS
# =========================
t = np.arange(len(tx_signal))

# =========================
# 5. PLOT TIME-DOMAIN SIGNAL
# =========================
plt.figure()
plt.plot(t, tx_signal)
plt.title("BPSK Signal in Time Domain")
plt.xlabel("Sample index")
plt.ylabel("Amplitude")
plt.grid(True)
plt.show()

# =========================
# 6. CONSTELLATION DIAGRAM
# =========================
plt.figure()
I = symbols
Q = np.zeros(len(symbols))
plt.scatter(I, Q)
plt.title("BPSK Constellation Diagram")
plt.xlabel("In-phase (I)")
plt.ylabel("Quadrature (Q)")
plt.axhline(0)
plt.axvline(0)
plt.grid(True)
plt.show()