# Cortex: Neuromorphic Flight Stabilization System

> **NOTE:** This repository contains proprietary research and development code. It is intended for academic review and portfolio purposes only. Unauthorized copying, distribution, modification, or commercial use of this software is strictly prohibited.

## Project Abstract
Cortex is a bio-inspired flight control system designed to mitigate pilot-induced oscillation and distinguish between intentional maneuvers and mechanical failure. Unlike traditional PID controllers which react linearly to error, Cortex employs a **Spiking Neural Network (SNN)** architecture based on **Predictive Coding** principles.

The system utilizes a Leaky Integrate-and-Fire (LIF) neuron model to process real-time telemetry. It establishes a dynamic threshold for "Danger" (Excitation) and suppresses this signal using an Efference Copy of the pilot's control inputs (Inhibition). This allows the system to autonomously intervene only when the physical state of the drone deviates significantly from the expected outcome of the pilot's input.

## Mathematical Model
The core decision logic is governed by a dynamical system modeling a biological neuron.

### 1. Synaptic Integration
The input current $I(t)$ is calculated as the differential between the environmental danger signal (Excitatory) and the pilot's intent (Inhibitory).

$$I(t) = \sum (w_i \cdot x_i) - (k \cdot u_{pilot})$$

Where:
* $x_i$: Normalized sensor inputs (Roll, Pitch, Gyroscopic Vibration).
* $w_i$: Synaptic weights optimized via evolutionary algorithms.
* $u_{pilot}$: Magnitude of control stick input.
* $k$: Inhibition gain factor.

### 2. Leaky Integrate-and-Fire (LIF)
The membrane potential $V(t)$ represents the system's "Panic Level." It accumulates charge from the input current while continuously leaking charge over time.

$$V(t) = V(t-1) \cdot \tau_{decay} + I(t)$$

If $V(t) > V_{threshold}$, the neuron spikes, triggering the emergency recovery protocol.

## System Architecture

The controller operates in a continuous 32ms control loop within the Webots simulation environment.

### Sensory Processing (Excitation)
* **Inertial Measurement:** Reads raw accelerometer and gyroscope data.
* **Normalization:** Applies pre-calculated normalization factors (`snn_normalization.npy`) derived from flight data analysis to scale inputs between 0.0 and 1.0.
* **Weighting:** Applies learned synaptic weights (`snn_weights.npy`) to quantify the severity of the current physical state.

### Cognitive Control (Inhibition)
* **Efference Copy:** Intercepts the raw RC signals sent to the motors.
* **Prediction:** Estimates the expected drone orientation based on the control input.
* **Cancellation:** Subtracts the expected movement from the actual sensor readings. If the pilot *intends* to roll 30 degrees, the inhibition signal cancels the danger signal, preventing a false positive.

### Black Box Telemetry
A dedicated monitoring module visualizes the internal state of the SNN in real-time, plotting the membrane potential against the firing threshold. This serves as a diagnostic tool to verify the "Excitatory-Inhibitory" balance during flight testing.

