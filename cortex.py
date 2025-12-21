import os
import sys
import math
import pickle
import pandas as pd
import numpy as np

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

BRAIN_FILE = "cortex_brain.pkl"
if not os.path.exists(BRAIN_FILE):
    print("ERROR: Brain file missing.")
    sys.exit(1)

with open(BRAIN_FILE, 'rb') as f:
    cortex = pickle.load(f)

class SuperStableDrone:
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
        self.fault_buffer = 0
        
        self.FAULT_LIMIT = 2
        self.MAX_TILT = 0.35

    def clamp(self, value, low, high):
        return max(low, min(high, value))

    def run(self):
        k_vert_p = 3.0
        k_roll_p = 8.0
        k_pitch_p = 8.0
        
        print("\n--- CORTEX SUPER-STABLE (UNLIMITED ALTITUDE) ---")
        print("System: Tilt Locked to 20° max.")
        print("System: Reflex Speed set to FAST (2 frames).")
        print("Controls: Arrow Keys (Fly) | Shift+Arrows (Climb) | Space (Crash Test)")

        while self.robot.step(self.timestep) != -1:
            sim_time = self.robot.getTime()
            
            rpy = self.imu.getRollPitchYaw()
            roll, pitch = rpy[0], rpy[1]
            alt = self.gps.getValues()[2]
            gyro = self.gyro.getValues()

            if alt > 0.5:
                data_point = pd.DataFrame([[roll, pitch, gyro[2]]], columns=["roll", "pitch", "vibration_z"])
                pred = cortex.predict(data_point)[0]
                
                if pred == "FAULT": 
                    self.fault_buffer += 1
                else: 
                    self.fault_buffer = 0
                
                if self.fault_buffer >= self.FAULT_LIMIT and not self.emergency_mode:
                    print(f"\n[{sim_time:.2f}s] !!! CRASH IMMINENT !!! KILLING MOTORS.")
                    self.emergency_mode = True

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
                    self.fault_buffer = 0
                    print("SYSTEM RESET.")
                key = self.kb.getKey()

            if self.target_altitude < 0.0: self.target_altitude = 0.0
            
            if self.emergency_mode:
                self.target_altitude = 0.0
            
            alt_err = self.clamp(self.target_altitude - alt + 0.6, -1, 1)
            v_in = 68.5 + (k_vert_p * pow(alt_err, 3.0))
            
            safe_roll = self.clamp(roll, -self.MAX_TILT, self.MAX_TILT)
            safe_pitch = self.clamp(pitch, -self.MAX_TILT, self.MAX_TILT)
            
            r_in = (k_roll_p * safe_roll) + gyro[0] + r_man
            p_in = (k_pitch_p * safe_pitch) + gyro[1] + p_man

            m = [0.0]*4
            m[0] = v_in - r_in + p_in
            m[1] = v_in + r_in + p_in
            m[2] = v_in - r_in - p_in
            m[3] = v_in + r_in - p_in

            if self.emergency_mode and alt < 0.2:
                m = [0.0, 0.0, 0.0, 0.0]
                
            if fault_trigger:
                m[0] -= 100 

            self.motors[0].setVelocity(m[0])
            self.motors[1].setVelocity(-m[1])
            self.motors[2].setVelocity(-m[2])
            self.motors[3].setVelocity(m[3])

            if int(sim_time*10)%5==0:
                state = "AUTOPILOT" if self.emergency_mode else "MANUAL"
                print(f"\r[{state}] Alt: {alt:.2f}m | TILT: {roll:.2f} | DANGER: {self.fault_buffer}/2   ", end="")

if __name__ == "__main__":
    SuperStableDrone().run()
