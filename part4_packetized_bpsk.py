import numpy as np

# =========================
# PARAMETERS
# =========================
NUM_BITS = 2000
SAMPLES_PER_SYMBOL = 10

snr_db = 10
beta = 0.25
span = 6

# =========================
# 1. DEFINE PACKET FORMAT
# =========================
PREAMBLE = [1,0,1,0,1,0,1,0]*2   # 16 bits
PAYLOAD_SIZE = 8                 # 8-bit command
PARITY = 1                       # 1-bit parity

# =========================
# 2. CREATE PACKETS
# =========================
num_packets = NUM_BITS // PAYLOAD_SIZE
payloads = np.random.randint(0, 2, (num_packets, PAYLOAD_SIZE))

def compute_parity(bits):
    return np.array([np.sum(bits) % 2])

payloads_with_parity = [np.concatenate((p, compute_parity(p))) for p in payloads]

packets = [np.concatenate((PREAMBLE, pwp)) for pwp in payloads_with_parity]

bitstream = np.concatenate(packets)

# =========================
# 3. TRANSMITTER (BPSK + Upsampling + Pulse shaping)
# =========================
symbols = 2 * bitstream - 1

upsampled_symbols = np.zeros(len(symbols) * SAMPLES_PER_SYMBOL)
upsampled_symbols[::SAMPLES_PER_SYMBOL] = symbols

# ---- RRC FILTER ----
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

rrc_filter *= np.sqrt(SAMPLES_PER_SYMBOL / np.sum(rrc_filter**2))

tx_signal = np.convolve(upsampled_symbols, rrc_filter, mode='full')

# =========================
# 4. CHANNEL (AWGN)
# =========================
snr_linear = 10**(snr_db / 10)

sig_power = np.mean(tx_signal**2)

noise_variance = (sig_power * SAMPLES_PER_SYMBOL) / (2 * snr_linear)

noise = np.sqrt(noise_variance) * np.random.randn(len(tx_signal))
rx_signal = tx_signal + noise

# =========================
# 5. RECEIVER (Matched Filter + Sampling)
# =========================
rx_filtered = np.convolve(rx_signal, rrc_filter, mode='full')

total_delay = num_taps - 1
sample_offset = int(total_delay) 
rx_samples = rx_filtered[sample_offset::SAMPLES_PER_SYMBOL]
expected_total_bits = len(bitstream)
rx_samples = rx_samples[:expected_total_bits]

# =========================
# 6. DECISION
# =========================
detected_symbols = np.where(rx_samples >=0, 1, -1)
detected_bits = np.where(detected_symbols == 1, 1, 0)

# =========================
# 7. PACKET DETECTION 
# =========================
def find_preamble_robust(detected_bits, preamble_bits, threshold_errors=1):
    preamble_len = len(preamble_bits)
    n_bits = len(detected_bits)
    preamble_indices = []
    
    for i in range(n_bits - preamble_len + 1):
        segment = detected_bits[i : i + preamble_len]
        num_errors = np.sum(segment != preamble_bits)
        
        if num_errors <= threshold_errors:
            preamble_indices.append(i)
            
    cleaned = []
    if preamble_indices:
        cleaned.append(preamble_indices[0])
        for idx in preamble_indices[1:]:
            if idx > cleaned[-1] + preamble_len:
                cleaned.append(idx)
    return cleaned

preamble_indices = find_preamble_robust(detected_bits, PREAMBLE, threshold_errors=1)

# =========================
# 8 & 9. REVISED EVALUATION
# =========================
# Αντί για sliding window, σπάμε το bitstream σε σταθερά blocks των 25 bits
packet_len = len(PREAMBLE) + PAYLOAD_SIZE + PARITY
num_expected_packets = len(bitstream) // packet_len

# reshape για να έχουμε [num_packets x 25]
rx_packet_matrix = detected_bits[:num_expected_packets * packet_len].reshape(-1, packet_len)

correct_packets = 0
for i in range(num_expected_packets):
    # Split το πακέτο
    rx_preamble = rx_packet_matrix[i, :16]
    rx_payload = rx_packet_matrix[i, 16:24]
    rx_parity = rx_packet_matrix[i, 24]
    
    # 1. Έλεγχος αν το preamble είναι σωστό (συγχρονισμός)
    if np.array_equal(rx_preamble, PREAMBLE):
        # 2. Έλεγχος αν το payload είναι ολόιδιο με το αρχικό
        if np.array_equal(rx_payload, payloads[i]):
            correct_packets += 1

print(f"--- Final Check ---")
print(f"Total Sent: {num_packets}")
print(f"Correct: {correct_packets}")
print(f"PER: {(num_packets - correct_packets)/num_packets:.2%}")