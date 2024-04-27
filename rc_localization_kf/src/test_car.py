#!/usr/bin/env python3
import rospy
from std_msgs.msg import Float64, Header
from sensor_msgs.msg import Imu
from geometry_msgs.msg import Quaternion, Vector3, PoseStamped, Pose, Point
import numpy as np
import tf
import tf.transformations

class Ackermann:
    def __init__(self, state, length):
        self.length = length
        self.state = state

        self.velocity_pub = rospy.Publisher(name="/velocity", data_class=Float64, queue_size=10)
        self.imu_pub = rospy.Publisher(name="/imu", data_class=Imu, queue_size=10)
        self.steer_pub = rospy.Publisher(name="/steer", data_class=Float64, queue_size=10)

        rospy.Subscriber(name="/speed_controller/command", data_class=Float64, callback=self.handle_speed_command)
        rospy.Subscriber(name="/steer_controller/command", data_class=Float64, callback=self.handle_steer_command)

        self.truth_pub = rospy.Publisher(name="/truth", data_class=PoseStamped, queue_size=10)

    def handle_steer_command(self, steer_command):
        self.state[3,0] = steer_command.data

    def handle_speed_command(self, speed_command):
        self.state[5,0] = speed_command.data

    def get_next_state(self, dT):
        x, y, theta, psi, v, v_dt = self.state[0,0],self.state[1,0],self.state[2,0],self.state[3,0],self.state[4,0],self.state[5,0]
        return np.array([ [
            x + v * np.cos(theta) * dT,
            y + v * np.sin(theta) * dT,
            theta + ((v * np.tan(psi))/self.length) * dT,
            psi,
            v + v_dt * dT,
            v_dt
        ] ]).T
    
    def update(self, dT):
        self.state = self.get_next_state(dT)

    def upload_velocity_data(self):
        self.velocity_pub.publish(Float64(self.state[4,0]))

    def update_odom_sensors(self):
        self.upload_velocity_data()
        self.upload_steer_data()

    def upload_imu_data(self):
        x, y, theta, psi, v, v_dt = self.state[0,0],self.state[1,0],self.state[2,0],self.state[3,0],self.state[4,0],self.state[5,0]
        orientation_array = tf.transformations.quaternion_from_euler(0, 0, theta)
        orientation = Quaternion(x=orientation_array[0], y=orientation_array[1], z=orientation_array[2], w=orientation_array[3])
        orientation_variances = np.identity(3)

        angular_velocity = Vector3()
        angular_velocity_variances = np.zeros((3,3))

        
        linear_acceleration = Vector3(
            v_dt * np.cos(theta) - (v**2 * np.sin(theta) * np.tan(psi)) / self.length,
            v_dt * np.sin(theta) + (v**2 * np.cos(theta) * np.tan(psi)) / self.length,
            0)
        linear_acceleration_variances = np.identity(3) * 0.1
        
        header = Header()
        header.stamp = rospy.Time.now()
        header.frame_id= "imu"
        self.imu_pub.publish(Imu(
            header,
            orientation, orientation_variances.flatten().tolist(), 
            angular_velocity, angular_velocity_variances.flatten().tolist(), 
            linear_acceleration, linear_acceleration_variances.flatten().tolist()))
    
    def upload_steer_data(self):
        self.steer_pub.publish(Float64(self.state[3,0]))

    def upload_truth(self):
        x, y, theta, psi, v, v_dt = self.state[0,0],self.state[1,0],self.state[2,0],self.state[3,0],self.state[4,0],self.state[5,0]
        header = Header()
        header.frame_id = "odom"
        header.stamp = rospy.Time.now()
        pose = Pose(Point(x, y, 0), Quaternion(*tf.transformations.quaternion_from_euler(0, 0, theta)))
        self.truth_pub.publish(PoseStamped(header, pose))


if __name__ == "__main__":
    rospy.init_node("test_car")

    initial_state = np.array([ [ 0.0, 0.0, 0.0, 0.0, 0.0, 0.0 ]] ).T
    ackermann = Ackermann(initial_state, length=0.3)
    hz = 250.0
    rate = rospy.Rate(hz)
    
    odom_sensor_looper = rospy.Timer(rospy.Duration(nsecs=5_000_000), lambda _: ackermann.update_odom_sensors())
    imu_looper = rospy.Timer(rospy.Duration(nsecs=20_000_000), lambda _: ackermann.upload_imu_data())

    while not rospy.is_shutdown():
        ackermann.update(dT=1/hz)
        ackermann.upload_truth()
        rate.sleep()

    odom_sensor_looper.shutdown()
    imu_looper.shutdown()

    # rospy.spin()