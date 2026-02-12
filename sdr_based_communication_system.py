import numpy as np
import matplotlib.pyplot as plt

# =============================================================================
# CONFIGURATION
# =============================================================================
SAMPLES_PER_SYMBOL = 10
SNR_DB = 10  
beta = 0.25
span = 6

WINDOW_SIZE = 3
COMMAND_SEQUENCE = ["FORWARD", "LEFT", "FORWARD", "RIGHT", "STOP"]

# Packet Structure
PREAMBLE = np.array([1, 0, 1, 0, 1, 0, 1, 0] * 2)
PAYLOAD_SIZE = 8
CRC_SIZE = 8
PACKET_LEN = len(PREAMBLE) + PAYLOAD_SIZE + CRC_SIZE

# =============================================================================
# UTILITIES & ERROR DETECTION
# =============================================================================
def compute_crc8(bits):
    poly = np.array([1, 0, 0, 0, 0, 0, 1, 1, 1]) # x^8 + x^2 + x + 1
    msg = np.append(bits, np.zeros(8, dtype=int))
    for i in range(len(bits)):
        if msg[i] == 1:
            msg[i:i+9] = np.logical_xor(msg[i:i+9], poly)
    return msg[-8:].astype(int)

# =============================================================================
# PHYSICAL LAYER 
# =============================================================================
class PhysicalLayer:
    def __init__(self):
        self.total_bits_sent = 0
        self.total_bit_errors = 0

    def transmit_receive(self, bitstream, snr_db):
        # 1. Modulation (BPSK)
        symbols = 2 * bitstream - 1
        upsampled = np.zeros(len(symbols) * SAMPLES_PER_SYMBOL)
        upsampled[::SAMPLES_PER_SYMBOL] = symbols

        # 2. RRC Filter
        num_taps = span*SAMPLES_PER_SYMBOL + 1
        t = np.linspace(-span/2, span/2, num_taps)
        rrc = np.zeros(num_taps)

        for i in range(num_taps):
            ti = t[i]
            if abs(ti) < 1e-12:
                rrc[i] = 1.0 - beta + (4*beta)/np.pi
            elif np.isclose(abs(4*beta*ti), 1.0):
                rrc[i] = (beta/np.sqrt(2)) * (
                    (1 + 2/np.pi)*np.sin(np.pi/(4*beta)) +
                    (1 - 2/np.pi)*np.cos(np.pi/(4*beta))
                )
            else:
                numerator = np.sin(np.pi*ti*(1-beta)) + \
                            4*beta*ti*np.cos(np.pi*ti*(1+beta))
                denominator = np.pi*ti*(1-(4*beta*ti)**2)
                rrc[i] = numerator/denominator

        rrc *= np.sqrt(SAMPLES_PER_SYMBOL / np.sum(rrc**2))

        # 3. Channel (AWGN)
        tx_sig = np.convolve(upsampled, rrc, mode='full')
        snr_linear = 10**(snr_db/10)
        noise_var = (np.mean(tx_sig**2) * SAMPLES_PER_SYMBOL) / (2 * snr_linear)
        noise = np.sqrt(noise_var) * np.random.randn(len(tx_sig))
        rx_sig = tx_sig + noise

        # 4. Receiver Filter & Sampling
        rx_filt = np.convolve(rx_sig, rrc, mode='full')
        offset = num_taps - 1
        sample_offset = int(offset)
        rx_samples = rx_filt[sample_offset::SAMPLES_PER_SYMBOL]
        rx_samples = rx_samples[:len(bitstream)]
        detected_symbols = np.where(rx_samples >=0, 1, -1)
        detected_bits = np.where(detected_symbols == 1, 1, 0)

        # --- BER LOGGING ---
        errors = np.sum(bitstream != detected_bits)
        self.total_bits_sent += len(bitstream)
        self.total_bit_errors += errors

        return detected_bits

    def get_current_ber(self):
        if self.total_bits_sent == 0: return 0
        return self.total_bit_errors / self.total_bits_sent

# =============================================================================
# DATA LINK LAYER
# =============================================================================
class DataLinkLayer:
    def __init__(self, threshold=1):
        self.threshold = threshold

    def create_packet(self, payload):
        crc = compute_crc8(payload)
        return np.concatenate((PREAMBLE, payload, crc))

    def verify_packet(self, bits):
        for i in range(len(bits) - PACKET_LEN + 1):
            if np.sum(bits[i:i+len(PREAMBLE)] != PREAMBLE) <= self.threshold:
                packet = bits[i : i+PACKET_LEN]
                payload = packet[len(PREAMBLE) : len(PREAMBLE)+PAYLOAD_SIZE]
                received_crc = packet[len(PREAMBLE)+PAYLOAD_SIZE:]
                
                if np.array_equal(compute_crc8(payload), received_crc):
                    return True, payload
        return False, None

# =============================================================================
# MAIN SYSTEM
# =============================================================================
def run_robot_control():
    phy = PhysicalLayer()
    dll = DataLinkLayer(threshold=2) # Πιο robust threshold για χαμηλό SNR
    
    COMMANDS = {
        "FORWARD":  np.array([0,0,0,0,0,0,0,1]),
        "BACKWARD": np.array([0,0,0,0,0,0,1,0]),
        "LEFT":     np.array([0,0,0,0,0,0,1,1]),
        "RIGHT":    np.array([0,0,0,0,1,0,0,0]),
        "STOP":     np.array([0,0,0,0,1,0,0,1])
    }
    
    payloads = [COMMANDS[cmd] for cmd in COMMAND_SEQUENCE]
    results = [False] * len(payloads)
    total_tx = 0
    
    print(f"--- ROBOT CONTROL SYSTEM (Sliding Window ARQ) ---")
    print(f"Target SNR: {SNR_DB}dB | Window Size: {WINDOW_SIZE}")
    print("-" * 50)

    base = 0
    while base < len(payloads):
        for i in range(base, min(base + WINDOW_SIZE, len(payloads))):
            if not results[i]:
                total_tx += 1
                tx_bits = dll.create_packet(payloads[i])
                rx_bits = phy.transmit_receive(tx_bits, SNR_DB)
                
                success, _ = dll.verify_packet(rx_bits)
                
                if success:
                    results[i] = True
                    print(f"[OK]   {COMMAND_SEQUENCE[i]} (ID: {i})")
                else:
                    print(f"[FAIL] {COMMAND_SEQUENCE[i]} (ID: {i}) -> CRC Error")

        while base < len(payloads) and results[base]:
            base += 1

    # Final Results
    print("-" * 50)
    print("FINAL PERFORMANCE METRICS:")
    print(f"Link BER (Bit Error Rate):    {phy.get_current_ber():.5f}")
    print(f"Packet Throughput Efficiency: {len(payloads)/total_tx:.2%}")
    print(f"Total Retransmissions:        {total_tx - len(payloads)}")

if __name__ == "__main__":
    run_robot_control()

