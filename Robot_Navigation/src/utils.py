import numpy as np
import cv2
import time

class PIDController:
    def __init__(self, kp, ki, kd, target):
        self.kp, self.ki, self.kd = kp, ki, kd
        self.target = target
        self.last_error = 0
        self.integral = 0
        self.last_time = time.time()

    def update(self, current_value):
        now = time.time()
        dt = max(now - self.last_time, 0.001)
        error = self.target - current_value
        self.integral += error * dt
        derivative = (error - self.last_error) / dt
        self.last_error, self.last_time = error, now
        return (self.kp * error) + (self.ki * self.integral) + (self.kd * derivative)

def trace_full_line(mask, start_x, window_h=15):
    """
    Tighter Crawler: Only looks 50px left/right to prevent 
    grabbing adjacent lanes during curves.
    """
    path = []
    curr_x = start_x
    h, w = mask.shape
    curr_y = h - (window_h // 2)
    patience = 3 
    consecutive_empty = 0

    while curr_y > window_h:
        y1, y2 = int(curr_y - (window_h // 2)), int(curr_y + (window_h // 2))
        if y1 < 0: break
        
        # TIGHT WINDOW (50px) - Prevents lane jumping
        win_w = 50 
        x1, x2 = max(0, curr_x - win_w), min(w, curr_x + win_w)
        
        window_roi = mask[y1:y2, x1:x2]
        M = cv2.moments(window_roi)
        
        if M["m00"] > 250:
            cx = int(M["m10"] / M["m00"]) + x1
            path.append((cx, curr_y))
            curr_x, consecutive_empty = cx, 0
        else:
            consecutive_empty += 1
            if consecutive_empty > patience: break
        
        curr_y -= window_h 
    return path