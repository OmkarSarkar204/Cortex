"""MAVLink flight controller communication adapter."""

from pymavlink import mavutil


class FCAdapter:
    """Handles MAVLink communication with a flight controller."""

    def __init__(self, connection_string="COM3", baud=115200):
        """Initialize MAVLink connection parameters."""
        self.connection_string = connection_string
        self.baud = baud
        self.master = None

    def connect(self):
        """Establish MAVLink connection and wait for heartbeat."""
        self.master = mavutil.mavlink_connection(
            self.connection_string,
            baud=self.baud
        )
        print("Waiting for heartbeat...")
        self.master.wait_heartbeat()
        print("Connected to FC")

    def get_attitude(self):
        """Return roll, pitch, yaw from ATTITUDE message."""
        msg = self.master.recv_match(type="ATTITUDE", blocking=False)
        if msg:
            return msg.roll, msg.pitch, msg.yaw
        return None

    def get_imu(self):
        """Return gyro values from HIGHRES_IMU message."""
        msg = self.master.recv_match(type="HIGHRES_IMU", blocking=False)
        if msg:
            return msg.xgyro, msg.ygyro, msg.zgyro
        return None

    def send_land(self):
        """Send MAV_CMD_NAV_LAND command to flight controller."""
        self.master.mav.command_long_send(
            self.master.target_system,
            self.master.target_component,
            mavutil.mavlink.MAV_CMD_NAV_LAND,
            0,
            0, 0, 0, 0, 0, 0, 0
        )
        print("Land command sent")

    def send_motor_override(self, values):
        """Send RC channel override values."""
        self.master.mav.rc_channels_override_send(
            self.master.target_system,
            self.master.target_component,
            *values
        )
