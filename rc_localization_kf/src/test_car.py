#!/usr/bin/env python3
import rospy
from rc_localization_odometry.msg import SensorCollect
from std_msgs.msg import Float32, Header
from sensor_msgs.msg import Imu
from geometry_msgs.msg import Quaternion, Vector3, PoseStamped, Pose, Point
import numpy as np
import tf
import tf.transformations

class Ackermann:
    def __init__(self, state, length):
        self.length = length
        self.state = state

        self.velocity_pub = rospy.Publisher(name="/velocity", data_class=Float32, queue_size=10)
        self.imu_pub = rospy.Publisher(name="/imu", data_class=Imu, queue_size=10)

        self.truth_pub = rospy.Publisher(name="/truth", data_class=PoseStamped, queue_size=10)

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
        self.velocity_pub.publish(Float32(self.state[4,0]))

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
        self.imu_pub.publish(Imu(
            header,
            orientation, orientation_variances.flatten().tolist(), 
            angular_velocity, angular_velocity_variances.flatten().tolist(), 
            linear_acceleration, linear_acceleration_variances.flatten().tolist()))
        
    def upload_truth(self):
        x, y, theta, psi, v, v_dt = self.state[0,0],self.state[1,0],self.state[2,0],self.state[3,0],self.state[4,0],self.state[5,0]
        header = Header()
        header.frame_id = "odom"
        header.stamp = rospy.Time.now()
        pose = Pose(Point(x, y, 0), Quaternion(*tf.transformations.quaternion_from_euler(0, 0, theta)))
        self.truth_pub.publish(PoseStamped(header, pose))


if __name__ == "__main__":
    rospy.init_node("test_car")

    initial_state = np.array([ [ 0, 0, 0, 0, 0.1, 0 ]] ).T
    ackermann = Ackermann(initial_state, length=0.3)
    hz = 250.0
    rate = rospy.Rate(hz)
    
    velocity_looper = rospy.Timer(rospy.Duration(nsecs=5_000_000), lambda _: ackermann.upload_velocity_data())
    imu_looper = rospy.Timer(rospy.Duration(nsecs=20_000_000), lambda _: ackermann.upload_imu_data())

    while not rospy.is_shutdown():
        ackermann.update(dT=1/hz)
        ackermann.upload_truth()
        rate.sleep()

    velocity_looper.shutdown()
    imu_looper.shutdown()

    # rospy.spin()