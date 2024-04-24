#!/usr/bin/env python3
import rospy
import numpy as np
from std_msgs.msg import Float32, Header
from sensor_msgs.msg import Imu
from geometry_msgs.msg import PoseStamped, Pose, Quaternion, Point
import tf
import tf.transformations


class MotionModel:
    def __init__(self, f, Q, F, L):
        self.f = f
        self.Q = Q
        self.F = F
        self.L = L

class MeasurementModel:
    def __init__(self, h, R, H, M):
        self.h = h
        self.R = R
        self.H = H
        self.M = M

class ExtendedKalmanFilter:
    """ Discrete time-extended Kalman filter. """

    def __init__(self, motion_model, x_0, P_0):
        self.motion_model = motion_model
        self.x = x_0
        self.P = P_0

    def predict(self, dT):
        F_k = self.motion_model.F(self.x, dT)
        L_k = self.motion_model.L()
        Q_k = self.motion_model.Q()

        self.P = F_k @ self.P @ F_k.T + L_k @ Q_k @ L_k.T
        self.x = self.motion_model.f(self.x, dT)

    def update(self, measurement, measurement_model):
        H_k = measurement_model.H(self.x)
        M_k = measurement_model.M()
        R_k = measurement_model.R()

        self.K = self.P @ H_k.T @ np.linalg.inv(H_k @ self.P @ H_k.T + M_k @ R_k @ M_k.T)
        self.x = self.x + self.K @ (measurement - measurement_model.h(self.x))
        
        KH = self.K @ H_k
        self.P = (np.identity(KH.shape[0]) - KH) @ self.P

def get_ackermann_motion_model(length, variances):

    def f(state, dT):
        #           0 1     2   3 4  5
        # state = [ x y theta psi v v' ]^T

        x, y, theta, psi, v, v_dt = state[0,0],state[1,0],state[2,0],state[3,0],state[4,0],state[5,0]
        return np.array([ [
            x + v * np.cos(theta) * dT,
            y + v * np.sin(theta) * dT,
            theta + ((v * np.tan(psi))/length) * dT,
            psi,
            v + v_dt * dT,
            v_dt
        ] ]).T

    def F(state, dT):
        x, y, theta, psi, v, v_dt = state[0,0],state[1,0],state[2,0],state[3,0],state[4,0],state[5,0]
        return np.array([
            [1, 0, -v * np.sin(theta) * dT, 0, np.cos(theta) * dT, 0],
            [0, 1, v * np.cos(theta) * dT, 0, np.sin(theta) * dT, 0],
            [0, 0, 1, (v/length * ((1/np.cos(psi)))**2) * dT, (np.tan(psi) * dT)/length, 0 ],
            [0, 0, 0, 1, 0, 0],
            [0, 0, 0, 0, 1, dT],
            [0, 0, 0, 0, 0, 1]
        ])

    def L():
        return np.identity(6)

    def Q():
        return np.array([
            [ variances[0], 0, 0, 0, 0, 0 ],
            [ 0, variances[1], 0, 0, 0, 0 ],
            [ 0, 0, variances[2], 0, 0, 0 ],
            [ 0, 0, 0, variances[3], 0, 0 ],
            [ 0, 0, 0, 0, variances[4], 0 ],
            [ 0, 0, 0, 0, 0, variances[5] ],
        ])
    
    return MotionModel(f, Q, F, L)

def get_ackermann_imu_measurement_model(length, variances):

    def h(state):
        x, y, theta, psi, v, v_dt = state[0,0],state[1,0],state[2,0],state[3,0],state[4,0],state[5,0]
        return np.array([
            [ v_dt * np.cos(theta) - (v**2 * np.sin(theta) * np.tan(psi)) / length],
            [ v_dt * np.sin(theta) + (v**2 * np.cos(theta) * np.tan(psi)) / length],
            [ theta ]
        ])
    
    def H(state):
        # TODO: this is wrong
        x, y, theta, psi, v, v_dt = state[0,0],state[1,0],state[2,0],state[3,0],state[4,0],state[5,0]
        return np.array([
            [ 0, 0, -v_dt * np.sin(theta) - (v**2 * np.cos(theta) * np.tan(psi)) / length, -(v**2 * np.sin(theta) * (1/np.cos(psi))**2)/length, (-2 * v * np.sin(theta) * np.tan(psi))/length, np.cos(theta)],
            [ 0, 0, v_dt * np.cos(theta) - (v**2 * np.sin(theta) * np.tan(psi)) / length, (v**2 * np.cos(theta) * (1/np.cos(psi))**2)/length, (2 * v * np.cos(theta) * np.tan(psi)) / length, np.sin(theta)],
            [ 0, 0, 1, 0, 0, 0 ],
        ])
    
    def M():
        return np.identity(3)
    
    def R():
        return np.array([
            [ variances[0], 0, 0],
            [ 0, variances[1], 0],
            [ 0, 0, variances[2]]
        ])
    
    return MeasurementModel(h, R, H, M)

def get_ackermann_velocity_measurement_model(variance):
    def h(state):
        x, y, theta, psi, v, v_dt = state[0,0],state[1,0],state[2,0],state[3,0],state[4,0],state[5,0]
        return np.array([
            [ v ]
        ])
    
    def H(state):
        x, y, theta, psi, v, v_dt = state[0,0],state[1,0],state[2,0],state[3,0],state[4,0],state[5,0]
        return np.array([[ 0, 0, 0, 0, 1, 0 ]])
    
    def M():
        return np.identity(1)
    
    def R():
        return np.array([ [ variance ] ])
    
    return MeasurementModel(h, R, H, M)

class AckermannFilter:
    def __init__(self):
        length = 0.3
        self.ekf = ExtendedKalmanFilter(
            motion_model=get_ackermann_motion_model(length, variances=[ 0.5, 0.5, 0.5, 0.5, 0.5, 0.5 ]),
            x_0=np.array([ [0, 0, 0, 0, 0, 0]]).T,
            P_0=np.zeros((6,6)))

        self.velocity_measurement_model = get_ackermann_velocity_measurement_model(0.25)
        self.imu_measurement_model = get_ackermann_imu_measurement_model(length, np.array([ 0.1, 0.1, 0.1]))

        self.last_predict_time = None
        rospy.Subscriber("/velocity", data_class=Float32, callback=self.handle_velocity)
        rospy.Subscriber("/imu", data_class=Imu, callback=self.handle_imu)

        self.odom_pub = rospy.Publisher("/odom", data_class=PoseStamped, queue_size=10)

    def predict(self):
        if self.last_predict_time is None:
            self.last_predict_time = rospy.Time.now()
            return
        
        current_time = rospy.Time.now()
        delta_time = (current_time - self.last_predict_time).to_sec()
        self.ekf.predict(delta_time)
        self.last_predict_time = current_time

    def publish_odom(self):
        state = self.ekf.x
        x, y, theta, psi, v, v_dt = state[0,0],state[1,0],state[2,0],state[3,0],state[4,0],state[5,0]
        header = Header()
        header.frame_id = "odom"
        header.stamp = rospy.Time.now()
        pose = Pose(Point(x, y, 0), Quaternion(*tf.transformations.quaternion_from_euler(0, 0, theta)))
        self.odom_pub.publish(PoseStamped(header, pose))


    def handle_velocity(self, velocity):
        self.ekf.update(np.array([ [ velocity.data ] ]), self.velocity_measurement_model)

    def handle_imu(self, imu):
        _, _, theta = tf.transformations.euler_from_quaternion([ imu.orientation.x, imu.orientation.y, imu.orientation.z, imu.orientation.w ] )
        x_accel, y_accel = imu.linear_acceleration.x, imu.linear_acceleration.y

        z = np.array([ [x_accel], [y_accel], [theta] ])
        self.ekf.update(z, self.imu_measurement_model)


if __name__ == "__main__":
    rospy.init_node("ackermann_filter")

    ackermann_filter = AckermannFilter()

    hz = 100
    rate=rospy.Rate(hz)
    while not rospy.is_shutdown():
        ackermann_filter.predict()
        ackermann_filter.publish_odom()
        rate.sleep()