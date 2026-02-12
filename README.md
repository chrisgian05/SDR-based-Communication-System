# Wireless Robot Control Simulator (BPSK & Sliding Window ARQ)

An end-to-end communication system simulator written in Python, designed to transmit robotic control commands over a noisy wireless channel. This project simulates both the **Physical Layer (Layer 1)** and the **Data Link Layer (Layer 2)** of the OSI model.

## Key Features

- **Physical Layer (PHY):**
  - **Modulation:** Binary Phase Shift Keying (BPSK).
  - **Pulse Shaping:** Root Raised Cosine (RRC) filters for bandwidth efficiency and ISI reduction.
  - **Channel Model:** Additive White Gaussian Noise (AWGN).
  - **Synchronization:** Robust preamble detection with configurable error thresholds.

- **Data Link Layer (DLL):**
  - **Error Detection:** Cyclic Redundancy Check (CRC-8) using polynomial division.
  - **Flow Control:** Sliding Window ARQ (Selective Repeat) for efficient command delivery.
  - **Reliability:** Automated retransmission of corrupted frames.

- **Analytics:**
  - Real-time Bit Error Rate (BER) monitoring.
  - Packet Delivery Efficiency tracking.

## System Architecture

The project is structured into modular classes:
1. `PhysicalLayer`: Handles modulation, filtering, and the noise environment.
2. `DataLinkLayer`: Handles framing, CRC calculation, and packet validation.
3. `RobotControl`: Orchestrates the command sequence delivery.

## Performance Metrics

The simulator evaluates performance based on the Signal-to-Noise Ratio (SNR). At low SNR (e.g., 5dB), the **Selective Repeat** mechanism ensures 100% command delivery by retransmitting only the corrupted frames, significantly outperforming simple Stop-and-Wait protocols.

## How to Run

1. Ensure you have `numpy` installed:
   ```bash
   pip install numpy
