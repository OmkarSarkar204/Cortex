import os
import sys
import math
import csv

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

desktop = os.path.join(os.path.expanduser("~"), "Desktop")
LOG_FILENAME = os.path.join(desktop, "cortex_data_v2.csv")

class DataCollectorV2:
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

        self.log_file = open(LOG_FILENAME, 'w', newline='')
        self.writer = csv.writer(self.log_file)
        self.writer.writerow(["time", "roll", "pitch", "altitude", "vibration_z", "label"])
        
        print(f"--- COLLECTOR V2 READY ---")
        print(f"Saving to: {LOG_FILENAME}")
        self.target_altitude = 0.0

    def clamp(self, value, low, high):
        return max(low, min(high, value))

    def run(self):
        k_vertical_thrust = 68.5
        k_vertical_p = 3.0
        k_roll_p = 50.0
        k_pitch_p = 30.0
        
        print("\nFLIGHT PLAN:")
        print("1. [Shift+Up] Takeoff to 1.5m")
        print("2. [Arrows] FLY AGGRESSIVELY (Lean forward/back/left/right) -> TEACH 'SAFE'")
        print("3. [SPACE] CRASH (Mid-air vibration) -> TEACH 'FAULT'")

        while self.robot.step(self.timestep) != -1:
            sim_time = self.robot.getTime()
            
            rpy = self.imu.getRollPitchYaw()
            roll, pitch = rpy[0], rpy[1]
            altitude = self.gps.getValues()[2]
            gyro = self.gyro.getValues()
            vib_z = gyro[2]

            label = "SAFE"
            fault = False
            roll_man = 0.0
            pitch_man = 0.0
            
            key = self.kb.getKey()
            while key != -1:
                if key == self.kb.SHIFT + self.kb.UP: self.target_altitude += 0.05
                elif key == self.kb.SHIFT + self.kb.DOWN: self.target_altitude -= 0.05
                elif key == self.kb.UP: pitch_man = -2.0
                elif key == self.kb.DOWN: pitch_man = 2.0
                elif key == self.kb.RIGHT: roll_man = -2.0
                elif key == self.kb.LEFT: roll_man = 2.0
                
                if key == ord(' '): 
                    fault = True
                    label = "FAULT"
                
                key = self.kb.getKey()

            if self.target_altitude > 2.0: self.target_altitude = 2.0
            if self.target_altitude < 0.0: self.target_altitude = 0.0

            alt_err = self.clamp(self.target_altitude - altitude + 0.6, -1, 1)
            v_in = 68.5 + (3.0 * pow(alt_err, 3.0))
            
            r_in = (50.0 * self.clamp(roll, -1, 1)) + gyro[0] + roll_man
            p_in = (30.0 * self.clamp(pitch, -1, 1)) + gyro[1] + pitch_man

            m_fl = v_in - r_in + p_in
            m_fr = v_in + r_in + p_in
            m_rl = v_in - r_in - p_in
            m_rr = v_in + r_in - p_in

            if fault:
                noise = math.sin(sim_time * 50) * 30
                m_fl = m_fl - 50 + noise

            self.motors[0].setVelocity(m_fl)
            self.motors[1].setVelocity(-m_fr)
            self.motors[2].setVelocity(-m_rl)
            self.motors[3].setVelocity(m_rr)

            self.writer.writerow([sim_time, roll, pitch, altitude, vib_z, label])
            self.log_file.flush()
            
            if int(sim_time * 10) % 5 == 0:
                print(f"\rRec: {label} | Pitch: {pitch:.2f} | Roll: {roll:.2f}   ", end="")

if __name__ == "__main__":
    controller = DataCollectorV2()
    controller.run()
