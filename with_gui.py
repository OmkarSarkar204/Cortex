import os
import sys
import numpy as np
import tkinter as tk
from collections import deque

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

if not os.path.exists("snn_weights.npy"):
    print("ERROR: Run the training script first!")
    sys.exit(1)

synaptic_weights = np.load("snn_weights.npy")
norm_factors = np.load("snn_normalization.npy")

print(f"Loaded Weights: {synaptic_weights}")

class BlackBoxGUI:
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("CORTEX BLACK BOX")
        self.root.geometry("400x300")
        self.root.configure(bg="black")
        self.root.attributes('-topmost', True)

        self.lbl_status = tk.Label(self.root, text="SYSTEM INITIALIZING", font=("Consolas", 20, "bold"), bg="black", fg="grey")
        self.lbl_status.pack(pady=10)

        self.metrics_frame = tk.Frame(self.root, bg="black")
        self.metrics_frame.pack(pady=5)
        
        self.lbl_volt = tk.Label(self.metrics_frame, text="VOLTAGE: 0.00", font=("Consolas", 12), bg="black", fg="#00ff00")
        self.lbl_volt.pack(side="left", padx=10)
        
        self.lbl_thresh = tk.Label(self.metrics_frame, text="THRESH: 1.50", font=("Consolas", 12), bg="black", fg="white")
        self.lbl_thresh.pack(side="left", padx=10)

        self.canvas_height = 150
        self.canvas_width = 380
        self.canvas = tk.Canvas(self.root, width=self.canvas_width, height=self.canvas_height, bg="#111111", highlightthickness=1, highlightbackground="grey")
        self.canvas.pack(pady=10)

        self.thresh_y = self.map_value(1.5)
        self.canvas.create_line(0, self.thresh_y, self.canvas_width, self.thresh_y, fill="red", dash=(4, 4))
        self.canvas.create_text(30, self.thresh_y - 10, text="CRASH LIMIT", fill="red", font=("Arial", 8))

        self.data_len = 100
        self.voltage_data = deque([0.0]*self.data_len, maxlen=self.data_len)

    def map_value(self, v):
        v_display = max(0.0, v)
        y = self.canvas_height - (v_display * (self.canvas_height / 2.0))
        return y

    def update(self, voltage, emergency_mode):
        if emergency_mode:
            self.lbl_status.config(text="EMERGENCY LANDING", fg="red")
            self.lbl_volt.config(fg="red")
        else:
            self.lbl_status.config(text="CORTEX SAFE", fg="#00ff00")
            self.lbl_volt.config(fg="#00ff00")

        self.lbl_volt.config(text=f"VOLTAGE: {voltage:.2f}")

        self.voltage_data.append(voltage)
        self.canvas.delete("wave")

        points = []
        x_gap = self.canvas_width / (self.data_len - 1)
        
        for i, val in enumerate(self.voltage_data):
            x = i * x_gap
            y = self.map_value(val)
            points.append(x)
            points.append(y)
        
        if len(points) >= 4:
            self.canvas.create_line(points, fill="#00ff00", width=2, tags="wave", smooth=True)

        self.root.update()

class IntelligentDrone:
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
        
        self.v_mem = 0.0
        self.spike_buffer = 0

        self.gui = BlackBoxGUI()

    def clamp(self, value, low, high):
        return max(low, min(high, value))

    def run(self):
        k_vert_p = 3.0
        k_roll_p = 8.0
        k_pitch_p = 8.0
        
        print("\n--- CORTEX SNN: PREDICTIVE MODE ---")
        print("Logic: Learned Danger (Excitation) vs. Pilot Intent (Inhibition)")

        while self.robot.step(self.timestep) != -1:
            sim_time = self.robot.getTime()
            
            rpy = self.imu.getRollPitchYaw()
            roll, pitch = rpy[0], rpy[1]
            alt = self.gps.getValues()[2]
            gyro = self.gyro.getValues()
            vib = gyro[2]

            r_man = 0.0
            p_man = 0.0
            fault_trigger = False
            
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
                    self.v_mem = 0
                    print("RESET.")
                key = self.kb.getKey()

            excitation = 0.0
            inhibition = 0.0

            if alt > 0.5:
                n_roll = abs(roll) / norm_factors[0]
                n_pitch = abs(pitch) / norm_factors[1]
                n_vib = abs(vib) / norm_factors[2]
                
                inputs = np.array([n_roll, n_pitch, n_vib])
                excitation = np.dot(inputs, synaptic_weights)
                
                stick_mag = abs(r_man) + abs(p_man)
                inhibition = stick_mag * 3.0 
                
                input_current = excitation - inhibition
                
                self.v_mem = (self.v_mem * 0.8) + input_current
                
                if self.v_mem > 1.5:
                    self.spike_buffer += 1
                    self.v_mem = 0
                else:
                    self.spike_buffer = max(0, self.spike_buffer - 0.2)
                    
                if self.spike_buffer > 4 and not self.emergency_mode:
                    print(f"\n[{sim_time:.2f}s] UNINTENDED MOTION DETECTED! (Exc:{excitation:.2f} > Inh:{inhibition:.2f})")
                    self.emergency_mode = True

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

            if self.emergency_mode and alt < 0.2: m = [0.0]*4
            if fault_trigger: m[0] -= 100 

            self.motors[0].setVelocity(m[0])
            self.motors[1].setVelocity(-m[1])
            self.motors[2].setVelocity(-m[2])
            self.motors[3].setVelocity(m[3])

            if int(sim_time*10)%2==0:
                self.gui.update(self.v_mem, self.emergency_mode)

if __name__ == "__main__":
    IntelligentDrone().run()
