import numpy as np

WIDTH = 320
HEIGHT = 240

# Move ROI slightly higher to see the line earlier
ROI_TOP = 50
ROI_BOTTOM = 150

# Broader Yellow Range (Lowered Saturation and Value to catch shadows)
LOWER_COLOR = np.array([18, 50, 50]) 
UPPER_COLOR = np.array([30, 255, 255])

# Stricter Lane Lock (Decrease this to prevent jumping to side lanes)
LANE_LOCK_THRESHOLD = 35

LOG_FOLDER = "logs"
VIDEO_OUT_PATH = "logs/tracked_output.mp4"
KP, KI, KD = 0.4, 0.01, 0.1