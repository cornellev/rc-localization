#!/usr/bin/env python3
import rospy
import numpy as np
import scipy

class MotionModel:
    def __init__(self, f, Q, A, L):
        self.f = f
        self.Q = Q
        self.A = A
        self.L = L

class MeasurementModel:
    def __init__(self, h, v, C, M):
        self.h = h
        self.v = v
        self.C = C
        self.M = M

class ExtendedKalmanFilter:
    def __init__(self, x_0, P_0):
        self.x = x_0
        self.P = P_0

    def predict(self, motion_model, dt):
        # TODO: recompute jacobian matrices at x from motion model
        # numerically integrate to update x with motion model
        # update P with matrices
        
        # self.P = (A @ self.P) + self.P @ A.T + L @ Q
        pass

    def update(self, measurement, measurement_model):
        # TODO: recompute jacobian matrices at x from measurement model
        # compute K with jacobian matrices
        # compute the updated x with x + K(y - h(x))
        # compute next P
        pass

if __name__ == "__main__":
    rospy.init_node('kalman_filter')

    rospy.spin()