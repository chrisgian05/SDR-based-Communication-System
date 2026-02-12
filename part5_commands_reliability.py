import numpy as np

# =========================
# PARAMETERS
# =========================
SAMPLES_PER_SYMBOL = 10
snr_db = 10
beta = 0.25
span = 6

# =========================
# 1. DEFINE PACKET FORMAT
# =========================
PREAMBLE = np.array([1,0,1,0,1,0,1,0]*2)  
PAYLOAD_SIZE = 8
PARITY_SIZE = 1
PACKET_LEN = 16 + 8 + 1

# =========================
# 2. DEFINE COMMANDS (8-bit)
# =========================
COMMANDS = {
    "FORWARD":  np.array([0,0,0,0,0,0,0,1]),
    "BACKWARD": np.array([0,0,0,0,0,0,1,0]),
    "LEFT":     np.array([0,0,0,0,0,0,1,1]),
    "RIGHT":    np.array([0,0,0,0,1,0,0,0]),
    "STOP":     np.array([0,0,0,0,1,0,0,1])
}

# =========================
# 3. CREATE COMMAND SEQUENCE
# =========================
command_sequence = ["FORWARD", "LEFT", "FORWARD", "RIGHT", "STOP"]

payloads = [COMMANDS[cmd] for cmd in command_sequence]

def compute_parity(bits):
    return np.array([np.sum(bits) % 2])

packets = []
for payload in payloads:
    packet = np.concatenate((PREAMBLE,
                             payload,
                             compute_parity(payload)))
    packets.append(packet)

bitstream = np.concatenate(packets)

# =========================
# 4. TRANSMITTER (BPSK)
# =========================
symbols = 2 * bitstream - 1

upsampled_symbols = np.zeros(len(symbols) * SAMPLES_PER_SYMBOL)
upsampled_symbols[::SAMPLES_PER_SYMBOL] = symbols

# ---- RRC FILTER ----
num_taps = span*SAMPLES_PER_SYMBOL + 1
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
# 5. CHANNEL (AWGN)
# =========================
snr_linear = 10**(snr_db/10)
sig_power = np.mean(tx_signal**2)
noise_variance = (sig_power * SAMPLES_PER_SYMBOL) / (2 * snr_linear)

noise = np.sqrt(noise_variance) * np.random.randn(len(tx_signal))
rx_signal = tx_signal + noise

# =========================
# 6. RECEIVER
# =========================
rx_filtered = np.convolve(rx_signal, rrc_filter, mode='full')

total_delay = num_taps - 1
sample_offset = int(total_delay)
rx_samples = rx_filtered[sample_offset::SAMPLES_PER_SYMBOL]
expected_total_bits = len(bitstream)
rx_samples = rx_samples[:expected_total_bits]

detected_symbols = np.where(rx_samples >=0, 1, -1)
detected_bits = np.where(detected_symbols == 1, 1, 0)

# =========================
# 7. ROBUST PREAMBLE DETECTION
# =========================
def find_packets_structured(bits, preamble, packet_len, threshold=1):
    preamble_len = len(preamble)
    indices = []
    i = 0
    while i <= len(bits) - packet_len:
        # Έλεγχος αν στο τρέχον σημείο υπάρχει το preamble
        segment = bits[i : i + preamble_len]
        errors = np.sum(segment != preamble)
        
        if errors <= threshold:
            indices.append(i)
            # Αφού βρήκαμε πακέτο, πηδάμε ακριβώς στο επόμενο πιθανό σημείο
            i += packet_len 
        else:
            # Αν δεν βρήκαμε, προχωράμε κατά 1 bit για να ψάξουμε παρακάτω
            i += 1
    return indices

preamble_positions = find_packets_structured(detected_bits, PREAMBLE, PACKET_LEN)

# =========================
# 8. PACKET EXTRACTION
# =========================
correct_packets = 0
total_detected = 0

def decode_command(payload):
    for name, bits in COMMANDS.items():
        if np.array_equal(bits, payload):
            return name
    return "UNKNOWN"

for start in preamble_positions:

    end = start + PACKET_LEN
    if end > len(detected_bits):
        continue  # incomplete packet

    total_detected += 1
    packet = detected_bits[start:end]

    rx_preamble = packet[0:16]
    rx_payload = packet[16:24]
    rx_parity = packet[24]

    # parity check
    if compute_parity(rx_payload)[0] != rx_parity:
        print(f"Packet at bit {start:3d}: [FAILED] Parity Error")
        continue

    cmd = decode_command(rx_payload)
    if cmd != "UNKNOWN":
        print(f"Packet at bit {start:3d}: [OK] Command: {cmd}")
        correct_packets += 1
    else:
        print(f"Packet at bit {start:3d}: [FAILED] Unknown Command Payload")

# =========================
# 9. PERFORMANCE
# =========================
sent_count = len(command_sequence)
PER = (sent_count - correct_packets) / sent_count

print("\nSent packets:", len(command_sequence))
print("Detected packets:", total_detected)
print("Correct packets:", correct_packets)
print("Packet Error Rate:", PER)