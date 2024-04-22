#!/usr/bin/env python3
import rospy
import numpy as np


class MotionModel:
    """
    x_k = f(x_k-1, w_k-1)
    w_k ~ N(0, Q_k)

    (f is given without w_k-1 as an input, as w is never actually plugged in (e.g. is always 0))
    (as of now, L and Q are assumed to be constant/time-invariant)

    F_k = partial of f w.r.t. x_k 
    L_k = partial of f w.r.t. v_k
    """
    def __init__(self, f, Q, F, L):
        self.f = f
        self.Q = Q
        self.F = F
        self.L = L

class MeasurementModel:
    """
    z_k = h_k(x_k, v_k)
    v_k ~ N(0, R_k)

    (h is given without v_k-1 as an input, as v is never actually plugged in (e.g. is always 0))
    (as of now, M and R are assumed to be constant/time-invariant)
    
    H_k = partial of h w.r.t. x_k
    M_k = parital of h w.r.t. v_k
    """
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

    def predict(self):
        F_k = self.motion_model.F(self.x)
        L_k = self.motion_model.L()
        Q_k = self.motion_model.Q()

        self.P = F_k @ self.P @ F_k.T + L_k @ Q_k @ L_k.T
        self.x = self.motion_model.f(self.x)

    def update(self, measurement, measurement_model):
        H_k = measurement_model.K(self.x)
        M_k = measurement_model.M(self.x)
        R_k = measurement_model.R(self.x)

        self.K = self.P @ H_k.T @ np.linalg.inv(H_k @ self.P @ H_k.T + M_k @ R_k @ M_k.T)
        self.x = self.x + self.K @ (measurement - measurement_model.h(self.x))
        
        KH = self.K @ H_k
        self.P = (np.identity(KH.shape) - KH) @ self.P

def get_ackermann_motion_model(length, dT, variances):

    def f(state):
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

    def F(state):
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
            [ variances[0,0], 0, 0, 0, 0, 0 ],
            [ 0, variances[1,1], 0, 0, 0, 0 ],
            [ 0, 0, variances[2,2], 0, 0, 0 ],
            [ 0, 0, 0, variances[3,3], 0, 0 ],
            [ 0, 0, 0, 0, variances[4,4], 0 ],
            [ 0, 0, 0, 0, 0, variances[5,5] ],
        ])
    
    return MotionModel(f, Q, F, L)

def get_ackermann_imu_measurement_model(length, dT, variances):

    def h(state):
        x, y, theta, psi, v, v_dt = state[0,0],state[1,0],state[2,0],state[3,0],state[4,0],state[5,0]
        return np.array([
            [ v_dt * np.cos(theta) - (v**2 * np.cos(theta) * np.tan(psi)) / length],
            [ v_dt * np.sin(theta) + (v**2 + np.cos(theta) * np.tan(psi)) / length],
            [ theta ]
        ])
    
    def H(state):
        x, y, theta, psi, v, v_dt = state[0,0],state[1,0],state[2,0],state[3,0],state[4,0],state[5,0]
        return np.array([
            [ 0, 0, -v_dt * np.sin(theta) - (v**2 * np.cos(theta) * np.tan(psi)) / length, -(v**2 * np.sin(theta) * (1/np.cos(psi))**2)/length, (-2 * v * np.sin(theta) * np.tan(psi))/length, np.cos(theta)],
            [ 0, 0, v_dt * np.cos(theta) + (v**2 * np.sin(theta) * np.tan(psi)) / length, (v**2 * np.cos(theta) * (1/np.cos(psi))**2)/length, (2 * v * np.cos(theta) * np.tan(psi)) / length, np.sin(theta)],
            [ 0, 0, 1, 0, 0, 0 ],
        ])
    
    def M():
        return np.identity(3)
    
    def R():
        return np.array([
            [ variances[0,0], 0, 0],
            [ 0, variances[1,1], 0],
            [ 0, 0, variances[2,2]]
        ])
    
    return MeasurementModel(h, R, H, M)

if __name__ == "__main__":
    # constants: L, delta T = 0.1, 0.1
    # x = [ x y theta psi v v' ]^T
    length, dT = 0.1, 0.01
    x0 = np.array([ [0, 0, 0.1, 0.1, 1, 0 ]]).T
    P0 = np.zeros((6,6))


    x = x0
    P = P0
    print("[",end="")
    iters = 100

    for i in range(iters):
        print(f"({x[0,0]}, {x[1,0]})", end=("" if i+1 == iters else ", "))
        F_k, L_k, Q_k = F(x, length, dT), L(x, length, dT), Q(x, length, dT)
        P = F_k @ P @ F_k.T + L_k @ Q_k @ L_k.T
        x = f(x, length, dT)
    print("]",end="")
    print()
