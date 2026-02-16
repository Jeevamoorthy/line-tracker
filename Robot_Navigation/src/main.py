import cv2
import numpy as np
import time
import os
import config
from utils import trace_full_line, PIDController

def main():
    # 1. Setup Paths
    video_path = os.path.join("assets", "test_line_video.mp4")
    if not os.path.exists(video_path):
        print(f"Error: Video not found at {video_path}")
        return

    cap = cv2.VideoCapture(video_path)
    
    # Video Writer
    if not os.path.exists(config.LOG_FOLDER): os.makedirs(config.LOG_FOLDER)
    output_filename = os.path.join(config.LOG_FOLDER, "lane_locked_output.mp4")
    fourcc = cv2.VideoWriter_fourcc(*'mp4v') 
    out = cv2.VideoWriter(output_filename, fourcc, 30.0, (config.WIDTH, config.HEIGHT))

    pid = PIDController(config.KP, config.KI, config.KD, target=config.WIDTH//2)
    
    # Persistence Variables
    last_cx = config.WIDTH // 2 
    frame_count = 0
    start_time = time.time()

    print("Strict Lane Locking Active. Press 'q' to quit.")

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break
        
        frame_count += 1
        frame_res = cv2.resize(frame, (config.WIDTH, config.HEIGHT))
        roi_top = 80 
        roi = frame_res[roi_top:config.HEIGHT, 0:config.WIDTH]
        
        # 1. Processing
        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, config.LOWER_COLOR, config.UPPER_COLOR)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((5,5), np.uint8))

        # 2. THE FIX: LOCKED START SEARCH
        # We only search a small 'Tunnel' (50px) around last_cx
        search_half_width = 50 
        x1 = max(0, last_cx - search_half_width)
        x2 = min(config.WIDTH, last_cx + search_half_width)
        
        # Crop the mask just for the tunnel search
        tunnel_roi = mask[-40:, x1:x2]
        M = cv2.moments(tunnel_roi)

        if M["m00"] > 400:
            start_x = int(M["m10"] / M["m00"]) + x1
            last_cx = start_x
            status_color = (0, 255, 0) # Green = Locked
        else:
            # If line lost in tunnel, stay where we are, do NOT look at side lanes
            start_x = last_cx 
            status_color = (0, 0, 255) # Red = Lost (Bridging)

        # 3. Trace and Draw
        full_path = trace_full_line(mask, start_x)
        
        # Draw Search Tunnel at bottom
        cv2.rectangle(frame_res, (x1, config.HEIGHT-40), (x2, config.HEIGHT-5), status_color, 1)

        if len(full_path) > 1:
            display_pts = np.array([(p[0], p[1] + roi_top) for p in full_path], np.int32)
            cv2.polylines(frame_res, [display_pts.reshape((-1, 1, 2))], False, (255, 0, 0), 2)
            cv2.circle(frame_res, tuple(display_pts[0]), 6, status_color, -1)

        # FPS calculation
        fps = frame_count / (time.time() - start_time)
        cv2.putText(frame_res, f"FPS: {fps:.1f}", (10, 25), 0, 0.5, (0, 255, 255), 1)

        out.write(frame_res)
        cv2.imshow("Locked Lane Tracker", frame_res)
        if cv2.waitKey(1) & 0xFF == ord('q'): break

    cap.release()
    out.release()
    cv2.destroyAllWindows()
    print(f"Finished. Saved to {output_filename}")

if __name__ == "__main__":
    main()