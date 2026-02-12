import numpy as np

# =========================
# PARAMETERS
# =========================
SAMPLES_PER_SYMBOL = 10
snr_db = 10
beta = 0.25
span = 6
MAX_RETRANS = 10

# =========================
# PACKET FORMAT
# =========================
PREAMBLE = np.array([1,0,1,0,1,0,1,0]*2)
PAYLOAD_SIZE = 8
PARITY_SIZE = 1
PACKET_LEN = len(PREAMBLE) + PAYLOAD_SIZE + PARITY_SIZE

# =========================
# COMMANDS
# =========================
COMMANDS = {
    "FORWARD":  np.array([0,0,0,0,0,0,0,1]),
    "BACKWARD": np.array([0,0,0,0,0,0,1,0]),
    "LEFT":     np.array([0,0,0,0,0,0,1,1]),
    "RIGHT":    np.array([0,0,0,0,1,0,0,0]),
    "STOP":     np.array([0,0,0,0,1,0,0,1])
}

ACK_BITS  = np.array([1,1,1,1,1,1,1,1])
NACK_BITS = np.array([0,0,0,0,0,0,0,0])

command_sequence = ["FORWARD", "LEFT", "RIGHT", "STOP"]

# =========================
# COMMON FUNCTIONS
# =========================
def compute_parity(bits):
    return np.array([np.sum(bits) % 2])

def create_packet(payload):
    return np.concatenate((PREAMBLE,
                           payload,
                           compute_parity(payload)))

# ----- MODEM (TX + RX) -----
def modem_transmit_receive(bitstream):

    # BPSK
    symbols = 2 * bitstream - 1
    upsampled = np.zeros(len(symbols) * SAMPLES_PER_SYMBOL)
    upsampled[::SAMPLES_PER_SYMBOL] = symbols

    # RRC
    num_taps = span*SAMPLES_PER_SYMBOL + 1
    t = np.linspace(-span/2, span/2, num_taps)
    rrc_filter = np.zeros(num_taps)

    for i in range(num_taps):
        ti = t[i]
        if abs(ti) < 1e-12:
            rrc_filter[i] = 1.0 - beta + (4*beta)/np.pi
        elif np.isclose(abs(4*beta*ti), 1.0):
            rrc_filter[i] = (beta/np.sqrt(2)) * (
                (1 + 2/np.pi)*np.sin(np.pi/(4*beta)) +
                (1 - 2/np.pi)*np.cos(np.pi/(4*beta))
            )
        else:
            numerator = np.sin(np.pi*ti*(1-beta)) + \
                        4*beta*ti*np.cos(np.pi*ti*(1+beta))
            denominator = np.pi*ti*(1-(4*beta*ti)**2)
            rrc_filter[i] = numerator/denominator

    rrc_filter *= np.sqrt(SAMPLES_PER_SYMBOL / np.sum(rrc_filter**2))

    tx_signal = np.convolve(upsampled, rrc_filter, mode='full')

    # AWGN
    snr_linear = 10**(snr_db/10)
    sig_power = np.mean(tx_signal**2)
    noise_var = (sig_power * SAMPLES_PER_SYMBOL) / (2 * snr_linear)
    noise = np.sqrt(noise_var) * np.random.randn(len(tx_signal))

    rx_signal = tx_signal + noise

    # RX
    rx_filtered = np.convolve(rx_signal, rrc_filter, mode='full')
    total_delay = num_taps - 1
    sample_offset = int(total_delay)
    rx_samples = rx_filtered[sample_offset::SAMPLES_PER_SYMBOL]
    rx_samples = rx_samples[:len(bitstream)]

    detected_symbols = np.where(rx_samples >=0, 1, -1)
    detected_bits = np.where(detected_symbols == 1, 1, 0)

    return detected_bits

# ----- ROBUST DETECTION -----
def find_packets_structured(bits, preamble, packet_len, threshold=1):
    preamble_len = len(preamble)
    indices = []
    i = 0
    while i <= len(bits) - packet_len:
        segment = bits[i:i+preamble_len]
        errors = np.sum(segment != preamble)
        if errors <= threshold:
            indices.append(i)
            i += packet_len
        else:
            i += 1
    return indices

def check_packet(bits):
    positions = find_packets_structured(bits, PREAMBLE, PACKET_LEN)
    if not positions:
        return False, None

    start = positions[0]
    packet = bits[start:start+PACKET_LEN]

    payload = packet[len(PREAMBLE) : len(PREAMBLE) + PAYLOAD_SIZE]
    parity  = packet[len(PREAMBLE) + PAYLOAD_SIZE]

    if compute_parity(payload)[0] != parity:
        return False, None

    return True, payload

# =========================
# STOP AND WAIT ARQ
# =========================
total_retransmissions = 0
successful_commands = 0

for cmd in command_sequence:

    payload = COMMANDS[cmd]
    success = False
    attempts = 0

    while not success and attempts < MAX_RETRANS:

        attempts += 1

        # ---- SEND COMMAND ----
        tx_packet = create_packet(payload)
        rx_bits = modem_transmit_receive(tx_packet)

        # ---- ROBOT SIDE ----
        valid, rx_payload = check_packet(rx_bits)

        if valid:
            ack_packet = create_packet(ACK_BITS)
        else:
            ack_packet = create_packet(NACK_BITS)

        # ---- ACK BACK ----
        ack_rx = modem_transmit_receive(ack_packet)
        ack_valid, ack_payload = check_packet(ack_rx)

        if ack_valid and np.array_equal(ack_payload, ACK_BITS):
            success = True
            successful_commands += 1
        else:
            total_retransmissions += 1

    if success:
        print(f"{cmd} delivered in {attempts} attempt(s)")
    else:
        print(f"{cmd} FAILED after {MAX_RETRANS} attempts")

# =========================
# PERFORMANCE
# =========================
avg_retrans = total_retransmissions / successful_commands if successful_commands > 0 else total_retransmissions
print("\nTotal Commands:", len(command_sequence))
print("Successful:", successful_commands)
print("Total Retransmissions:", total_retransmissions)
print("Average Retransmissions per Command:", avg_retrans)