# -*- coding: utf-8 -*-
import random
import serial
import threading
import time
import cv2
import numpy as np
from struct import unpack
import RPi.GPIO as GPIO


class RoboBrain:
    def __init__(
        self,
        use_ultrasonic_sensor: bool,
        use_regular_camera: bool,
        use_TOF_camera: bool,
        stationary_testing: bool = False,
        control_manualy: bool = False,
        TOF_data_header: bytes = b"\x00\xFF",
        regular_camera_index: int = 0,
    ) -> None:
        """
        Initialize a class instance with camera and sensor setup options.
        Parameters:
            - use_ultrasonic_sensor (bool): Determines if robot should use data from the ultrasonic sensor
            - use_regular_camera (bool): Specifies wether or not the robot should use information from a camera. If not true the camera horizontal and verticle servos will also not be used.
            - use_TOF_camera (bool): Determines wether or not the robot should use information from a TOF camera or depth camera.
            - stationary_testing (bool): Determines if the system is in stationary testing mode.
            - TOF_data_header (bytes): Specifies the header to use when finding the needed image data from the TOF camera.

        Returns:
            - None: This constructor does not return any value.

        Example:
            - __init__(use_ultrasonic_sensor=True, use_regular_camera = True, use_TOF_camera = True, stationary_testing=True, TOF_data_header=b"\\x00\\xFF") -> Initializes the class in stationary mode using depth cam and the header b"\\x00\\xFF" for the specific TOF Cam used (in this case the header works with the Sipeed MaixSense-A010).
        """
        self.running: bool = (
            False  # variable that controlls wether or not the robot should be currently thinking
        )
        self.stationary_testing = stationary_testing
        self.control_manualy = control_manualy
        self.com = None
        self.TOF_cam_connected: bool = False  # is the TOF camera currently connected
        self.TOF_data_header: bytes = TOF_data_header  #
        self.use_ultrasonic_sensor: bool = use_ultrasonic_sensor
        self.use_regular_camera: bool = use_regular_camera
        self.use_TOF_camera: bool = use_TOF_camera
        self.face_detected: bool = False  # has a face been detected currently

        if self.use_regular_camera:
            self.reg_cam: cv2.VideoCapture = cv2.VideoCapture(
                regular_camera_index
            )  # regular color camera

        if self.use_TOF_camera:
            self.open_TOF_cam()

        self.setup_GPIO()

    def setup_GPIO(self):
        """
        Configure the GPIO pins and PWM for motor and sensor controls.
        Parameters:
            - self (object): The instance of the class containing attributes like stationary_testing, use_regular_camera, and use_ultrasonic_sensor.
        Returns:
            - None: This function does not return a value; it configures hardware settings.
        Example:
            - setup_GPIO(self) -> None
        """
        GPIO.setwarnings(False)
        GPIO.setmode(GPIO.BCM)

        if not self.stationary_testing:
            GPIO.setup(L_MOTOR_ENABLE_PIN, GPIO.OUT, initial=GPIO.HIGH)
            GPIO.setup(LEFT_FW_PIN, GPIO.OUT, initial=GPIO.LOW)
            GPIO.setup(LEFT_BK_PIN, GPIO.OUT, initial=GPIO.LOW)
            GPIO.setup(R_MOTOR_ENABLE_PIN, GPIO.OUT, initial=GPIO.HIGH)
            GPIO.setup(RIGHT_FW_PIN, GPIO.OUT, initial=GPIO.LOW)
            GPIO.setup(RIGHT_BK_PIN, GPIO.OUT, initial=GPIO.LOW)

            # Set the PWM pin with a frequency of 2000hz
            self.pwm_L_motor_enable = GPIO.PWM(L_MOTOR_ENABLE_PIN, 2000)
            self.pwm_R_motor_enable = GPIO.PWM(R_MOTOR_ENABLE_PIN, 2000)

            self.pwm_L_motor_enable.start(0)
            self.pwm_R_motor_enable.start(0)

        if self.use_TOF_camera or self.use_ultrasonic_sensor:
            GPIO.setup(SENSOR_SERVO_PIN, GPIO.OUT)

            self.TOF_or_ultrasonic_servo = GPIO.PWM(SENSOR_SERVO_PIN, 50)
            self.TOF_or_ultrasonic_servo.start(0)
        else:
            self.TOF_or_ultrasonic_servo = None

        if self.use_regular_camera:
            GPIO.setup(HCS_PIN, GPIO.OUT)
            GPIO.setup(VCS_PIN, GPIO.OUT)

            self.camera_horizontal_servo = GPIO.PWM(HCS_PIN, 50)
            self.camera_vertical_servo = GPIO.PWM(VCS_PIN, 50)

            self.camera_vertical_servo.start(0)
            self.camera_horizontal_servo.start(0)
        else:
            self.camera_horizontal_servo = None
            self.camera_vertical_servo = None

        if self.use_ultrasonic_sensor:
            GPIO.setup(ULTRASONIC_IN_PIN, GPIO.IN)
            GPIO.setup(ULTRASONIC_OUT_PIN, GPIO.OUT)

        self.servo_pos = {
            self.TOF_or_ultrasonic_servo: 900,
            self.camera_vertical_servo: 900,
            self.camera_horizontal_servo: 900,
        }

    def cleanup_GPIO(self):
        """
        Cleanup GPIO settings by stopping active PWM signals and releasing GPIO resources.
        Parameters:
            - self (object): An instance of a class where this method is defined. It should include attributes to determine the configuration, such as `stationary_testing`, `use_TOF_camera`, `use_ultrasonic_sensor`, `use_regular_camera`, and PWM attributes.
        Returns:
            - None: This function does not return any value. It performs cleanup operations for GPIO settings.
        Example:
            - cleanup_GPIO(self) -> None
        """
        try:
            if not self.stationary_testing:
                self.pwm_L_motor_enable.stop()
                self.pwm_R_motor_enable.stop()

            if self.use_TOF_camera or self.use_ultrasonic_sensor:
                self.TOF_or_ultrasonic_servo.stop()

            if self.use_regular_camera:
                self.camera_vertical_servo.stop()
                self.camera_horizontal_servo.stop()

            GPIO.cleanup()
        except Exception as e:
            print(f"Error cleaning up GPIO recources: {e}")

    def open_TOF_cam(self):
        """
        Open and initialize the TOF (Time-of-Flight) camera connection.

        Parameters:
            - self (object): Instance of the class containing the method, expected to have 'com' as an attribute.

        Returns:
            - None: This function does not return any value.

        Example:
            - open_TOF_cam() -> None
        """
        try:
            self.com = serial.Serial()
            self.com.port = "/dev/ttyUSB0"
            self.com.baudrate = 115200
            self.com.timeout = 1  # Set a timeout for reading
            self.com.open()
            self.TOF_cam_connected = True
            try:
                # Send initialization commands
                self.com.write("AT+DISP=1\r".encode("utf-8"))  # Turn on display
                time.sleep(0.1)  # Wait for the command to take effect
                self.com.write(
                    "AT+DISP=3\r".encode("utf-8")
                )  # Set to USB and LCD display
                time.sleep(0.1)  # Wait for the command to take effect
                self.com.write(
                    "AT+BINN=4\r".encode("utf-8")
                )  # Set binning to four for simplicity (we don't need 100x100 pixels to see what the closest object is in most cases)

            except Exception as e:
                print(f"Error sending commands to TOF camera: {e}")
        except Exception as e:
            print(f"Error opening TOF cam serial connection: {e}")

    def close_TOF_cam(self):
        """
        Close the connection to the TOF camera if it is currently connected.
        Parameters:
            - self (object): The instance of the class containing the TOF camera connection properties.
        Returns:
            - None: The function does not return any value. It performs an operation to close the TOF camera connection.
        Example:
            - close_TOF_cam() -> None (Assuming 'self' is an instance with TOF_cam_connected set to True)
        """
        try:
            if self.TOF_cam_connected:
                self.com.close()
                self.TOF_cam_connected = False
        except Exception as e:
            print(f"Error closing TOF cam serial connection: {e}")

    def close_reg_cam(self):
        """
        Closes the regular camera if it is in use.
        Parameters:
            - self: Instance of the class containing methods and properties including the camera.
        Returns:
            - None: This function does not return a value.
        Example:
            - close_reg_cam() -> None
        """
        if self.use_regular_camera:
            try:
                self.reg_cam.release()
            except Exception as e:
                print(f"Error releasing regular camera: {e}")

    def recieve_TOF_frame(self):
        """
        Receives and processes a Time-of-Flight (TOF) frame from a serial connection, returning an image if successful.
        Parameters:
            - self (object): Instance of the class containing serial connection properties and status.
        Returns:
            - np.ndarray or None: A 2D image array constructed from frame data if a complete frame is received and valid, otherwise returns None.
        Example:
            - Calling recieve_TOFframe() on a TOF_cam_connected instance will attempt to read and decode frame data to return an image if successful, otherwise it loops or stops on disconnection.
        """
        raw_data = b""

        def decode_data(raw_data):
            idx = raw_data.find(self.TOF_data_header)
            if idx < 0:
                return False, None

            raw_data = raw_data[idx:]
            if len(raw_data) < 20:
                return False, None

            data_len = unpack("H", raw_data[2:4])[0]
            frame_len = len(self.TOF_data_header) + 2 + data_len + 2

            if len(raw_data) < frame_len:
                return False, None

            frame = raw_data[:frame_len]
            raw_data = raw_data[frame_len:]

            checksum = frame[-2]
            if checksum != sum(frame[:-2]) % 256:
                print("Checksum failed")
                return False, None

            res_rows = unpack("B", frame[14:15])[0]
            res_cols = unpack("B", frame[15:16])[0]
            frame_data = [
                unpack("B", frame[20 + i : 21 + i])[0] for i in range(data_len - 16)
            ]

            img = np.array(frame_data, dtype=np.uint8).reshape((res_rows, res_cols))
            return True, img

        while self.TOF_cam_connected:
            try:
                if self.com.in_waiting:
                    data = self.com.read(self.com.in_waiting)
                    raw_data += data
                    have_frame, img = decode_data(raw_data)
                    if have_frame:
                        print("image recieved")
                        return img

            except serial.SerialException as e:
                print(f"Disconnected from port: {e}")
                self.TOF_cam_connected = False
                self.com.close()
                break

    def proccess_TOF_frame(self, depth_map: np.ndarray):
        """
        Finds the closest point in the depth map and returns its distance in centimeters.

        Parameters:
            - depth_map: 2D NumPy array representing the grayscale depth image,
                         where pixel intensity corresponds to distance.

        Returns:
            - float: Closest distance in centimeters.
        """

        # Default formula when UNIT = 0 if not multiply pixle values by unit value
        distances = (depth_map / 5.1) ** 2
        distances = distances.min() - 45  # minus 45 because of weird mappings
        return distances

    def retrieve_proccess_TOF_frame(self) -> float:
        """
        Retrieve and process a Time-of-Flight (TOF) frame to calculate distance.

        Parameters:
            - self: Represents the instance of the class. No additional parameters are required.

        Returns:
            - float: The calculated distance in centimeters if a valid distance is obtained from the TOF frame.

        Example:
            - retrieve_proccess_TOF_frame() -> 150.0
        """
        while True:
            depth_map = self.recieve_TOF_frame()
            distance = self.proccess_TOF_frame(depth_map)
            if distance > 0:
                print(f"Calculated distance of {distance}cm from TOF frame.")
                return distance
            else:
                print(
                    "Error calculating distance from TOF frame, retrieving new frame."
                )

    def ultrasonic_GPIO_getData(self):
        """
        Measure the distance using an ultrasonic sensor connected to GPIO pins.
        Parameters:
            - None
        Returns:
            - float: The distance measured by the ultrasonic sensor in centimeters, or -1 if the measurement times out.
        Example:
            - ultrasonic_GPIO_getData() -> 10.5
        """
        GPIO.output(ULTRASONIC_OUT_PIN, GPIO.LOW)
        time.sleep(0.000002)
        GPIO.output(ULTRASONIC_OUT_PIN, GPIO.HIGH)
        time.sleep(0.000015)
        GPIO.output(ULTRASONIC_OUT_PIN, GPIO.LOW)

        t3 = time.time()

        while not GPIO.input(ULTRASONIC_IN_PIN):
            t4 = time.time()
            if (t4 - t3) > 0.03:
                return -1

        t1 = time.time()
        while GPIO.input(ULTRASONIC_IN_PIN):
            t5 = time.time()
            if (t5 - t1) > 0.03:
                return -1

        t2 = time.time()
        time.sleep(0.01)
        #    print "distance is %d " % (((t2 - t1)* 340 / 2) * 100)
        return ((t2 - t1) * 340 / 2) * 100

    def get_and_validate_ultrasonic_data(self):
        """Validate and process ultrasonic sensor data to obtain a reliable distance measurement.
        This function reads raw data from an ultrasonic sensor, filters out invalid readings, and calculates the average of the middle three valid measurements to determine an accurate distance value.
        Parameters:
            None
        Returns:
            - float: The averaged distance measured by the ultrasonic sensor after filtering through valid readings in cm.
        Example:
            - interpret_ultrasonic_data() -> 124.5
        """
        num = 0
        ultrasonic = []
        while num < 5:
            distance = self.ultrasonic_GPIO_getData()
            while int(distance) == -1:
                distance = self.ultrasonic_GPIO_getData()
                print("Tdistance is %f" % (distance))
            while int(distance) >= 500 or int(distance) == 0:
                distance = self.ultrasonic_GPIO_getData()
                print("Edistance is %f" % (distance))
            ultrasonic.append(distance)
            num = num + 1
            time.sleep(0.01)
        print(ultrasonic)
        distance = (ultrasonic[1] + ultrasonic[2] + ultrasonic[3]) / 3
        print("distance is %f" % (distance))
        return distance

    def servo_appointed_detection(
        self, servo: GPIO.PWM, pos: int
    ):  # The specified servo rotates to the specified angle
        """Rotate the specified servo to the desired angle.
        Parameters:
            - servo (GPIO.PWM): The servo motor to be controlled.
            - pos (int): The desired position angle for the servo, constrained between 0 and 180 degrees.
        Returns:
            - None: This function does not return any value.
        Example:
            - servo_appointed_detection(servo_motor, 90)
              This rotates 'servo_motor' to a 90-degree angle.
        """
        if pos < 0:
            pos = 0
        if servo == self.camera_vertical_servo:
            if pos < 60:
                pos = 60
        elif pos > 180:
            pos = 180
        if self.servo_pos[servo] != 900:
            waittime = abs(self.servo_pos[servo] - pos) / 180
        else:
            waittime = 0.5
        self.servo_pos[servo] = pos
        servo.ChangeDutyCycle(2.5 + 10 * pos / 180)
        time.sleep(waittime)  # Allow time for the servo to reach the desired angle
        servo.ChangeDutyCycle(0)  # Stop the PWM signal

    def move_fowards(self, left_motor_speed: int, right_motor_speed: int):
        """
        Moves the motors of a robot or device forward at the specified speeds.

        Parameters:
            - left_motor_speed (int): The speed percentage (0-100) for the left motor.
            - right_motor_speed (int): The speed percentage (0-100) for the right motor.

        Returns:
            - None: This function does not return any value.

        Example:
            - move_forwards(50, 50) -> None
        """
        if left_motor_speed < 0 or left_motor_speed > 100:
            raise ValueError("Left motor speed must be between 0 and 100")
        if right_motor_speed < 0 or right_motor_speed > 100:
            raise ValueError("Right motor speed must be between 0 and 100")

        GPIO.output(LEFT_FW_PIN, GPIO.HIGH)  # Left motor rotates forward
        GPIO.output(LEFT_BK_PIN, GPIO.LOW)  # Left motor does not rotate backwards
        GPIO.output(RIGHT_FW_PIN, GPIO.HIGH)  # Right motor rotates forward
        GPIO.output(RIGHT_BK_PIN, GPIO.LOW)  # Right motor does not rotate backwards
        self.pwm_L_motor_enable.ChangeDutyCycle(
            left_motor_speed
        )  # change the left side motors to the left_motor_speed variable
        self.pwm_R_motor_enable.ChangeDutyCycle(
            right_motor_speed
        )  # change the right side motors to the right_motor_speed variable

    def move_backwards(self, left_motor_speed: int, right_motor_speed: int):
        """
        Moves the robot backwards by setting the motor directions and speeds.
        Parameters:
            - left_motor_speed (int): Speed for the left motor; typically between 0 and 100.
            - right_motor_speed (int): Speed for the right motor; typically between 0 and 100.
        Returns:
            - None: This function does not return a value.
        Example:
            - move_backwards(50, 50) -> None

        This function causes both the left and right motors to spin backwards causing the robot to move backwards.
        """
        GPIO.output(LEFT_FW_PIN, GPIO.LOW)  # Left motor does not rotate forward
        GPIO.output(LEFT_BK_PIN, GPIO.HIGH)  # Left motor rotates backwards
        GPIO.output(RIGHT_FW_PIN, GPIO.LOW)  # Right motor does not forward
        GPIO.output(RIGHT_BK_PIN, GPIO.HIGH)  # Right motor rotates backwards
        self.pwm_L_motor_enable.ChangeDutyCycle(
            left_motor_speed
        )  # change the left side motors to the left_motor_speed variable
        self.pwm_R_motor_enable.ChangeDutyCycle(
            right_motor_speed
        )  # change the right side motors to the right_motor_speed variable

    def turn_left(
        self, left_motor_speed: int, right_motor_speed: int
    ):  # Only the right motor goes fowards causing the robot to turn left but not stay in place as it does so
        """
        Moves the robot left by controlling motor speeds.

        Parameters:
            - left_motor_speed (int): Speed of the left motor, set using PWM for gradual speed control.
            - right_motor_speed (int): Speed of the right motor, set using PWM for gradual speed control.

        Returns:
            - None: This function does not return a value.

        Example:
            - turn_left(0, 50) -> None

        This function causes only the right motor to spin fowards causing the robot to turn left but not stay in place.
        To keep the robot in place use the RoboBrain.spin_left() function.
        """
        GPIO.output(LEFT_FW_PIN, GPIO.LOW)  # Left motor does not rotate forward
        GPIO.output(LEFT_BK_PIN, GPIO.LOW)  # Left motor does not rotate backwards
        GPIO.output(RIGHT_FW_PIN, GPIO.HIGH)  # Right motor rotates forward
        GPIO.output(RIGHT_BK_PIN, GPIO.LOW)  # Right motor does not rotate backwards
        self.pwm_L_motor_enable.ChangeDutyCycle(
            left_motor_speed
        )  # change the left side motors to the left_motor_speed variable
        self.pwm_R_motor_enable.ChangeDutyCycle(
            right_motor_speed
        )  # change the right side motors to the right_motor_speed variable

    def turn_right(
        self, left_motor_speed: int, right_motor_speed: int
    ):  # Only the left motor goes fowards causing the robot to turn right but not stay in place as it does so
        """
        Move the robot to the right by only activating the left motor to move forwards.

        Parameters:
            - left_motor_speed (int): Speed of the left motor to turn the robot right.
            - right_motor_speed (int): Speed of the right motor; typically lower than left to ensure turning.

        Returns:
            - None: This function does not return any value.

        Example:
            - turn_right(50, 20) -> None

        This function causes only the left motor to spin fowards causing the robot to turn right but not stay in place.
        To keep the robot in place use the RoboBrain.spin_right() function.
        """
        GPIO.output(LEFT_FW_PIN, GPIO.HIGH)  # Left motor does rotate forward
        GPIO.output(LEFT_BK_PIN, GPIO.LOW)  # Left motor does not rotate backwards
        GPIO.output(RIGHT_FW_PIN, GPIO.LOW)  # Right motor does not rotate forward
        GPIO.output(RIGHT_BK_PIN, GPIO.LOW)  # Right motor does not rotate backwards
        self.pwm_L_motor_enable.ChangeDutyCycle(
            left_motor_speed
        )  # change the left side motors to the left_motor_speed variable
        self.pwm_R_motor_enable.ChangeDutyCycle(
            right_motor_speed
        )  # change the right side motors to the right_motor_speed variable

    def spin_left(self, left_motor_speed: int, right_motor_speed: int):
        """
        Rotate the robot in place by spinning the left motor backward and the right motor forward.

        Parameters:
            - left_motor_speed (int): The speed for the left motor to spin in reverse.
            - right_motor_speed (int): The speed for the right motor to spin forward.

        Returns:
            - None

        Example:
            - spin_left(50, 50) -> Rotates the robot at half speed in place.

        This function causes the right motor to spin fowards and the left motor to spin backwards causing the robot to turn left while staying in place.
        To not keep the robot in place use the RoboBrain.turn_left() function.
        """
        GPIO.output(LEFT_FW_PIN, GPIO.LOW)  # Left motor does not rotate forward
        GPIO.output(LEFT_BK_PIN, GPIO.HIGH)  # Left motor does rotate backwards
        GPIO.output(RIGHT_FW_PIN, GPIO.HIGH)  # Right motor rotates forward
        GPIO.output(RIGHT_BK_PIN, GPIO.LOW)  # Right motor does not rotate backwards
        self.pwm_L_motor_enable.ChangeDutyCycle(
            left_motor_speed
        )  # change the left side motors to the left_motor_speed variable
        self.pwm_R_motor_enable.ChangeDutyCycle(
            right_motor_speed
        )  # change the right side motors to the right_motor_speed variable

    def spin_right(self, left_motor_speed: int, right_motor_speed: int):
        """
        Spin the robot to the right in place by adjusting motor directions and speeds.

        Parameters:
            - left_motor_speed (int): Speed at which the left motor should spin forward.
            - right_motor_speed (int): Speed at which the right motor should spin backward.

        Returns:
            - None: This method does not return a value.

        Example:
            - spin_right(50, 50) -> None

        This function causes the left motor to spin fowards and the right motor to spin backwards causing the robot to turn right while staying in place.
        To not keep the robot in place use the RoboBrain.turn_right() function.
        """
        GPIO.output(LEFT_FW_PIN, GPIO.HIGH)  # Left motor rotates forward
        GPIO.output(LEFT_BK_PIN, GPIO.LOW)  # Left motor does not rotate backwards
        GPIO.output(RIGHT_FW_PIN, GPIO.LOW)  # Right motor does not rotate forward
        GPIO.output(RIGHT_BK_PIN, GPIO.HIGH)  # Right motor rotates backwards
        self.pwm_L_motor_enable.ChangeDutyCycle(
            left_motor_speed
        )  # change the left side motors to the left_motor_speed variable
        self.pwm_R_motor_enable.ChangeDutyCycle(
            right_motor_speed
        )  # change the right side motors to the right_motor_speed variable

    def brake(self):
        """
        Stops the movement of both left and right motors by disabling forward and backward rotation.
        Parameters:
            - self (object): The instance of the class the method belongs to, providing access to class attributes.
        Returns:
            - None: This method does not return any value; it directly controls hardware.
        Example:
            - brake() -> None: Stops the motors, effectively halting the movement of the vehicle they control.
        """
        GPIO.output(LEFT_FW_PIN, GPIO.LOW)  # Left motor does not rotate forward
        GPIO.output(LEFT_BK_PIN, GPIO.LOW)  # Left motor does not rotate backwards
        GPIO.output(RIGHT_FW_PIN, GPIO.LOW)  # Right motor does not rotate forward
        GPIO.output(RIGHT_BK_PIN, GPIO.LOW)  # Right motor does not rotate backwards

    def move(self):
        """
        Perform autonomous movement with a maximum measurable distance of 2.5m (250 cm).
        Continuously checks distance while moving forward to ensure dynamic obstacle avoidance.

        Parameters:
            - None

        Returns:
            - None
        """
        MAX_DISTANCE = 250  # Maximum measurable distance in cm
        SAFE_DISTANCE = 40  # Safety threshold in cm for stopping
        MIN_SPEED = 10  # Minimum speed to prevent stalling
        MAX_SPEED = 100  # Maximum speed for forward movement

        if not self.face_detected and not self.stationary_testing:
            while self.running:
                try:
                    distance = None

                    # Select the active sensor
                    if self.use_ultrasonic_sensor and not self.use_TOF_camera:
                        get_distance = self.get_and_validate_ultrasonic_data
                    elif self.use_TOF_camera and not self.use_ultrasonic_sensor:
                        get_distance = self.retrieve_proccess_TOF_frame
                    elif self.use_TOF_camera and self.use_ultrasonic_sensor:
                        print(
                            "Logic for combining TOF and ultrasonic sensors is not implemented."
                        )
                        continue
                    else:
                        print("No active sensor found.")
                        break

                    # Continuously monitor distance while moving forward
                    while True:
                        distance = get_distance()
                        print(f"Measured Distance: {distance} cm")  # Debugging output

                        # Clamp distance to maximum measurable range
                        distance = min(distance, MAX_DISTANCE)

                        # Stop if the distance is below the safety threshold
                        if distance < SAFE_DISTANCE:
                            print("Obstacle detected! Stopping.")
                            self.brake()
                            break

                        # Calculate speed proportional to the distance
                        speed = max(
                            min((distance / MAX_DISTANCE) * MAX_SPEED, MAX_SPEED),
                            MIN_SPEED,
                        )
                        print(f"Moving forward with speed: {speed}")
                        self.move_fowards(speed, speed)

                        time.sleep(0.1)  # Short delay between checks

                    # Obstacle detected, start avoidance maneuver
                    self.brake()
                    self.servo_appointed_detection(self.TOF_or_ultrasonic_servo, 180)
                    time.sleep(1)
                    left_distance = get_distance()

                    self.servo_appointed_detection(self.TOF_or_ultrasonic_servo, 0)
                    time.sleep(1)
                    right_distance = get_distance()

                    if right_distance != left_distance:
                        if right_distance > left_distance:
                            print("Turning right to avoid obstacle.")
                            self.spin_right(40, 40)
                            time.sleep(2)
                        else:
                            print("Turning left to avoid obstacle.")
                            self.spin_left(40, 40)
                            time.sleep(2)
                    else:
                        # Random fallback if both sides are equal
                        if random.choice([True, False]):
                            print("Randomly turning right.")
                            self.spin_right(40, 40)
                            time.sleep(2)
                        else:
                            print("Randomly turning left.")
                            self.spin_left(40, 40)
                            time.sleep(2)

                    self.brake()
                    # Move forward a bit after turning
                    self.move_fowards(50, 50)
                    time.sleep(1)
                    self.brake()

                except KeyboardInterrupt:
                    print("Stopping autonomous movement.")
                    self.stop()
                    break
                except Exception as e:
                    print(f"Error occurred: {e}")
                    self.stop()
                    break

    def manual_control(self):
        """
        Provides real-time manual control for the robot, including movement and camera control.
        Movement keys control the robot, and arrow keys adjust the camera angle if enabled.

        Controls:
            - W: Move forward
            - S: Move backward
            - A: Turn or strafe left
            - D: Turn or strafe right
            - Q: Rotate left (spin in place)
            - E: Rotate right (spin in place)
            - Space: Brake
            - Arrow Keys: Control camera (if enabled)
            - Esc: Exit control
        """
        print("Real-Time Manual Control Activated")
        print("Controls:")
        print("  W: Move forward")
        print("  S: Move backward")
        print("  A: Turn or strafe left")
        print("  D: Turn or strafe right")
        print("  Q: Rotate left (spin in place)")
        print("  E: Rotate right (spin in place)")
        print("  Shift: Hold shift to move faster")
        print("  Space: Brake")
        print("  Arrow Keys: Control camera (if enabled)")
        print("  Esc: Exit control")

        from pynput import keyboard

        # Movement and camera state dictionaries
        movement_state = {
            "forward": False,
            "backward": False,
            "left": False,
            "right": False,
            "rotate_left": False,
            "rotate_right": False,
        }

        speed = 0  # extra speed when holding shift

        # Initialize camera servos if enabled
        if self.use_regular_camera:
            self.servo_appointed_detection(self.camera_horizontal_servo, 90)
            self.servo_appointed_detection(self.camera_vertical_servo, 90)

        # Nested function for handling key presses
        def on_press(key):
            try:
                if key.char == "w":
                    movement_state["forward"] = True
                elif key.char == "s":
                    movement_state["backward"] = True
                elif key.char == "a":
                    movement_state["left"] = True
                elif key.char == "d":
                    movement_state["right"] = True
                elif key.char == "q":
                    movement_state["rotate_left"] = True
                elif key.char == "e":
                    movement_state["rotate_right"] = True
            except AttributeError:
                if key == keyboard.Key.space:
                    print("Braking")
                    self.brake()
                elif self.use_regular_camera:
                    if key == keyboard.Key.up:
                        # Move camera up
                        move = self.servo_pos[self.camera_vertical_servo] + 5

                        print(f"Camera moved up to {move}\u00B0")
                        self.servo_appointed_detection(self.camera_vertical_servo, move)
                    elif key == keyboard.Key.down:
                        # Move camera down
                        move = self.servo_pos[self.camera_vertical_servo] - 5

                        print(f"Camera moved down to {move}\u00B0")
                        self.servo_appointed_detection(self.camera_vertical_servo, move)
                    elif key == keyboard.Key.left:
                        # Move camera left
                        move = max(self.servo_pos[self.camera_horizontal_servo] - 5, 0)
                        print(f"Camera moved left to {move}\u00B0")
                        self.servo_appointed_detection(
                            self.camera_horizontal_servo, move
                        )
                    elif key == keyboard.Key.shift:
                        speed = 50
                    elif key == keyboard.Key.right:
                        # Move camera right
                        move = min(
                            self.servo_pos[self.camera_horizontal_servo] + 5, 180
                        )
                        print(f"Camera moved right to {move}\u00B0")
                        self.servo_appointed_detection(
                            self.camera_horizontal_servo, move
                        )

        # Nested function for handling key releases
        def on_release(key):
            try:
                if key.char == "w":
                    movement_state["forward"] = False
                elif key.char == "s":
                    movement_state["backward"] = False
                elif key.char == "a":
                    movement_state["left"] = False
                elif key.char == "d":
                    movement_state["right"] = False
                elif key.char == "q":
                    movement_state["rotate_left"] = False
                elif key.char == "e":
                    movement_state["rotate_right"] = False
                elif key.char == keyboard.Key.shift.char:
                    speed = 0
            except AttributeError:
                if key == keyboard.Key.esc:
                    print("Exiting control")
                    self.stop()
                    return False  # Exit the listener loop

        # Nested function to continuously update movement based on keys held
        def handle_movement():
            while self.running:
                if movement_state["forward"] and not movement_state["backward"]:
                    if movement_state["left"]:
                        print("Moving forward and turning left")
                        self.turn_left(30 + speed, 50 + speed)
                    elif movement_state["right"]:
                        print("Moving forward and turning right")
                        self.turn_right(50 + speed, 30 + speed)
                    else:
                        print("Moving forward")
                        self.move_fowards(50 + speed, 50 + speed)
                elif movement_state["backward"] and not movement_state["forward"]:
                    if movement_state["left"]:
                        print("Moving backward and turning left")
                        self.turn_left(30 + speed, 50 + speed)
                    elif movement_state["right"]:
                        print("Moving backward and turning right")
                        self.turn_right(50 + speed, 30 + speed)
                    else:
                        print("Moving backward")
                        self.move_backwards(50 + speed, 50 + speed)

                if movement_state["rotate_left"]:
                    print("Rotating left")
                    self.spin_left(50 + speed, 50 + speed)
                elif movement_state["rotate_right"]:
                    print("Rotating right")
                    self.spin_right(50 + speed, 50 + speed)

                # If no movement keys are pressed, brake to stop movement
                if not any(movement_state.values()):
                    self.brake()

                time.sleep(0.1)  # Prevent excessive CPU usage

        def show_frame():
            while self.running:
                # Capture a frame from the camera
                ret, frame = self.reg_cam.read()
                if not ret:
                    print("Error: Unable to capture frame.")
                else:
                    # Flip the frame vertically
                    frame = cv2.flip(frame, 0)
                    cv2.imshow("Robot View", frame)
                    if cv2.waitKey(1) & 0xFF == ord("q"):
                        break

        # Start the movement handler in a separate thread
        import threading

        movement_thread = threading.Thread(target=handle_movement, daemon=True)
        movement_thread.start()

        if self.use_regular_camera:
            capture_thread = threading.Thread(target=show_frame, daemon=True)
            capture_thread.start()

        # Start listening for key events
        with keyboard.Listener(on_press=on_press, on_release=on_release) as listener:
            listener.join()

    def follow_face_camera(self):
        """
        Detects and tracks a face or a green object using the camera.
        Adjusts servos to center the object in view with smoother and more reliable movements.

        Features:
            - Tracks faces or green objects.
            - Smooth servo adjustments for stable tracking.
            - Dynamic adjustments based on object size and position.
            - Displays object position, bounding box, and movement direction.

        Parameters:
            - self: The object instance holding the camera and servo control.

        Returns:
            - None
        """
        # Load the face detection model
        face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        )

        # HSV color range for green detection
        lower_green = np.array([40, 40, 40])
        upper_green = np.array([80, 255, 255])

        # Toggle for tracking green objects
        green_ball_mode = True

        # Sensitivity for servo movement (lower = smoother)
        movement_sensitivity = 2
        position_threshold = 15  # Minimum pixels to consider adjusting

        while self.running:
            try:
                # Capture a frame from the camera
                ret, frame = self.reg_cam.read()
                if not ret:
                    print("Error: Unable to capture frame. Retrying...")
                    continue

                # Flip the frame vertically
                frame = cv2.flip(frame, 0)

                # Initialize tracking variables
                center_of_object = None
                direction_text = ""

                if green_ball_mode:
                    # Green object detection
                    hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
                    mask = cv2.inRange(hsv_frame, lower_green, upper_green)
                    contours, _ = cv2.findContours(
                        mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
                    )

                    if contours:
                        # Find the largest green object
                        largest_contour = max(contours, key=cv2.contourArea)
                        moments = cv2.moments(largest_contour)

                        if moments["m00"] != 0:
                            center_x = int(moments["m10"] / moments["m00"])
                            center_y = int(moments["m01"] / moments["m00"])
                            center_of_object = (center_x, center_y)
                            x, y, w, h = cv2.boundingRect(largest_contour)
                            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                else:
                    # Face detection
                    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                    faces = face_cascade.detectMultiScale(
                        gray_frame, scaleFactor=1.3, minNeighbors=5
                    )

                    if len(faces) > 0:
                        x, y, w, h = faces[0]
                        center_of_object = (x + w // 2, y + h // 2)
                        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)

                # If an object is detected, calculate adjustments
                if center_of_object:
                    frame_center_x = frame.shape[1] // 2
                    frame_center_y = frame.shape[0] // 2

                    horizontal_diff = frame_center_x - center_of_object[0]
                    vertical_diff = frame_center_y - center_of_object[1]

                    # Adjust horizontal servo if the object is significantly off-center
                    if abs(horizontal_diff) > position_threshold:
                        step = (
                            int(horizontal_diff / abs(horizontal_diff))
                            * movement_sensitivity
                        )
                        self.servo_pos[self.camera_horizontal_servo] += step
                        direction_text += "Move Left" if step > 0 else "Move Right"
                        self.servo_appointed_detection(
                            self.camera_horizontal_servo,
                            self.servo_pos[self.camera_horizontal_servo],
                        )

                    # Adjust vertical servo if the object is significantly off-center
                    if abs(vertical_diff) > position_threshold:
                        step = (
                            int(vertical_diff / abs(vertical_diff))
                            * movement_sensitivity
                        )
                        self.servo_pos[self.camera_vertical_servo] -= step
                        direction_text += " | Move Down" if step > 0 else " | Move Up"
                        self.servo_appointed_detection(
                            self.camera_vertical_servo,
                            self.servo_pos[self.camera_vertical_servo],
                        )

                    # Display tracking information
                    cv2.putText(
                        frame,
                        f"Horizontal Diff: {horizontal_diff}",
                        (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6,
                        (255, 255, 255),
                        1,
                    )
                    cv2.putText(
                        frame,
                        f"Vertical Diff: {vertical_diff}",
                        (10, 50),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6,
                        (255, 255, 255),
                        1,
                    )

                # Overlay direction text
                if direction_text:
                    cv2.putText(
                        frame,
                        direction_text,
                        (10, 70),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.8,
                        (0, 255, 255),
                        2,
                    )

                # Show the frame with annotations
                cv2.imshow("Camera Tracking", frame)

                # Exit loop on 'q' key press
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

            except KeyboardInterrupt:
                print("Stopping camera tracking...")
                self.stop()

    def start(self):
        """
        Starts functions that carry out the movement and camera logic of the robot depending on the input sources.
        """

        print("\nTo stop program press ctrl+c while in console.\n")

        if self.use_ultrasonic_sensor or self.use_TOF_camera:
            self.servo_appointed_detection(
                self.TOF_or_ultrasonic_servo, 90
            )  # reset rotation of servo which moves the ultrasonic sensor and/or the TOF camera

        if not self.stationary_testing:
            if (
                self.use_TOF_camera
                or self.use_ultrasonic_sensor
                and not self.control_manualy
            ):
                self.running = True
                self.move()
            elif self.control_manualy:
                self.running = True
                self.manual_control()
            else:
                print(
                    "In order to run movement logic either use_ultrasonic_sensor or use_TOF_camera must be enabled,\n (unless in the future I add AI depth map from the regular camera when I switch to jetson nano)"
                )

        if self.use_regular_camera and not self.control_manualy:
            self.running = True
            self.servo_appointed_detection(
                self.camera_horizontal_servo, 90
            )  # reset camera horizontal servo rotation
            self.servo_appointed_detection(
                self.camera_vertical_servo, 90
            )  # reset camera vertical servo rotation
            if self.stationary_testing:
                self.follow_face_camera()
            else:
                self.follow_face_camera_thread = threading.Thread(
                    target=self.follow_face_camera
                )
                self.follow_face_camera_thread.start()
        elif self.stationary_testing:
            print(
                "Use of a camera was disabled and robot movement was disabled meaning nothing will happen."
            )

    def stop(self):
        """Cleans up all recources to cleanly stop the RoboBrain logic."""
        self.running = False

        self.brake()

        self.cleanup_GPIO()

        self.close_TOF_cam()

        self.close_reg_cam()


if __name__ == "__main__":
    # motor pins
    LEFT_FW_PIN = 20
    LEFT_BK_PIN = 21
    RIGHT_FW_PIN = 19
    RIGHT_BK_PIN = 26

    L_MOTOR_ENABLE_PIN = 16  # Pin for Enabling/disabling the left motor
    R_MOTOR_ENABLE_PIN = 13  # Pin for Enabling/disabling the right motor
    # end of motor pins section

    HCS_PIN = 11  # horizontal camera servo pin number
    VCS_PIN = 9  # verticle camera servo pin number

    ULTRASONIC_IN_PIN = 0  # Ultrasonic sensor input pin
    ULTRASONIC_OUT_PIN = 1  # Ultrasonic sensor output pin

    SENSOR_SERVO_PIN = 25  # the pin for controlling the servo which has the ultrasonic sensor and/or the TOF camera on it

    robot_brain = RoboBrain(
        use_ultrasonic_sensor=False,
        use_regular_camera=True,
        use_TOF_camera=False,
        stationary_testing=False,
        control_manualy=True,
    )

    robot_brain.start()
    robot_brain.stop()  # in case of an error cleanup recources anyways
