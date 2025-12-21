import os
import sys
import numpy as np
import pandas as pd

WEBOTS_HOME = r"D:\Webots"
if not os.path.exists(WEBOTS_HOME):
    print(f"CRITICAL: Webots not found at {WEBOTS_HOME}")
    sys.exit(1)

os.environ["WEBOTS_HOME"] = WEBOTS_HOME
libs_path = os.path.join(WEBOTS_HOME, 'lib', 'controller')
os.environ["PATH"] = libs_path + os.pathsep + os.environ["PATH"]
sys.path.append(os.path.join(WEBOTS_HOME, 'lib', 'controller', 'python'))

try:
    os.add_dll_directory(libs_path)
except AttributeError:
    pass

from controller import Robot, Keyboard

WEIGHTS_FILE = "snn_weights.npy"
if not os.path.exists(WEIGHTS_FILE):
    print("ERROR: SNN Weights not found. Run train_snn_lif.py first.")
    sys.exit(1)

synapses = np.load(WEIGHTS_FILE)
print(f"Loaded SNN Synapses: {synapses}")

class SNN_Drone:
    def __init__(self):
        self.robot = Robot()
        self.timestep = int(self.robot.getBasicTimeStep())
        
        self.imu = self.robot.getDevice("inertial unit")
        self.imu.enable(self.timestep)
        self.gps = self.robot.getDevice("gps")
        self.gps.enable(self.timestep)
        self.gyro = self.robot.getDevice("gyro")
        self.gyro.enable(self.timestep)
        self.kb = self.robot.getKeyboard()
        self.kb.enable(self.timestep)
        
        self.motors = []
        names = ["front left propeller", "front right propeller", "rear left propeller", "rear right propeller"]
        for name in names:
            m = self.robot.getDevice(name)
            m.setPosition(float('inf'))
            m.setVelocity(0)
            self.motors.append(m)

        self.target_altitude = 0.0
        self.emergency_mode = False 
        
        self.membrane_potential = 0.0
        self.spike_buffer = 0
        self.LIF_DECAY = 0.8

    def clamp(self, value, low, high):
        return max(low, min(high, value))

    def run(self):
        k_vert_p = 3.0
        k_roll_p = 8.0
        k_pitch_p = 8.0
        
        print("\n--- NEUROMORPHIC SNN ONLINE ---")
        print("Model: Leaky Integrate-and-Fire (LIF)")
        print("Output: Membrane Voltage (V_mem)")

        while self.robot.step(self.timestep) != -1:
            sim_time = self.robot.getTime()
            
            rpy = self.imu.getRollPitchYaw()
            roll, pitch = rpy[0], rpy[1]
            alt = self.gps.getValues()[2]
            gyro = self.gyro.getValues()

            if alt > 0.5:
                i_roll = abs(roll) / 3.14
                i_pitch = abs(pitch) / 1.5
                i_vib = abs(gyro[2]) / 10.0
                
                input_current = np.array([i_roll, i_pitch, i_vib])
                incoming_current = np.dot(input_current, synapses)
                
                self.membrane_potential = (self.membrane_potential * self.LIF_DECAY) + incoming_current
                
                if self.membrane_potential > 1.0:
                    self.spike_buffer += 1
                    self.membrane_potential = 0.0
                else:
                    self.spike_buffer = max(0, self.spike_buffer - 1)
                
                if self.spike_buffer >= 3 and not self.emergency_mode:
                    print(f"\n[{sim_time:.2f}s] *** NEURON SPIKE DETECTED *** EMERGENCY LANDING.")
                    self.emergency_mode = True

            fault_trigger = False
            r_man = 0.0
            p_man = 0.0
            
            key = self.kb.getKey()
            while key != -1:
                if not self.emergency_mode:
                    if key == self.kb.SHIFT + self.kb.UP: self.target_altitude += 0.02
                    elif key == self.kb.SHIFT + self.kb.DOWN: self.target_altitude -= 0.02
                    elif key == self.kb.UP: p_man = -0.3
                    elif key == self.kb.DOWN: p_man = 0.3
                    elif key == self.kb.RIGHT: r_man = -0.3
                    elif key == self.kb.LEFT: r_man = 0.3
                
                if key == ord(' '): fault_trigger = True
                if key == ord('R'): 
                    self.emergency_mode = False
                    self.membrane_potential = 0.0
                    self.spike_buffer = 0
                    print("NEURON RESET.")
                key = self.kb.getKey()

            if self.target_altitude < 0.0: self.target_altitude = 0.0
            if self.emergency_mode: self.target_altitude = 0.0
            
            alt_err = self.clamp(self.target_altitude - alt + 0.6, -1, 1)
            v_in = 68.5 + (k_vert_p * pow(alt_err, 3.0))
            
            safe_roll = self.clamp(roll, -0.35, 0.35)
            safe_pitch = self.clamp(pitch, -0.35, 0.35)
            
            r_in = (k_roll_p * safe_roll) + gyro[0] + r_man
            p_in = (k_pitch_p * safe_pitch) + gyro[1] + p_man

            m = [0.0]*4
            m[0] = v_in - r_in + p_in
            m[1] = v_in + r_in + p_in
            m[2] = v_in - r_in - p_in
            m[3] = v_in + r_in - p_in

            if self.emergency_mode and alt < 0.2:
                m = [0.0]*4
                
            if fault_trigger: m[0] -= 100 

            self.motors[0].setVelocity(m[0])
            self.motors[1].setVelocity(-m[1])
            self.motors[2].setVelocity(-m[2])
            self.motors[3].setVelocity(m[3])

            if int(sim_time*10)%2==0:
                state = "AI_LAND" if self.emergency_mode else "MANUAL"
                volts = "|" * int(self.membrane_potential * 10)
                print(f"\r[{state}] Alt:{alt:.2f}m | V_mem: {self.membrane_potential:.2f} [{volts:<10}] | Spikes: {self.spike_buffer}   ", end="")

if __name__ == "__main__":
    SNN_Drone().run()
