import cv2
import numpy as np
import time
import os
import config
from utils import trace_full_line, PIDController

def main():
    # 1. Path Setup
    video_path = os.path.join("assets", "test_line_video.mp4")
    if not os.path.exists(video_path):
        print(f"ERROR: Video not found at {video_path}")
        return

    # Ensure logs folder exists
    if not os.path.exists(config.LOG_FOLDER):
        os.makedirs(config.LOG_FOLDER)

    # 2. Initialize Video Source & Writer
    cap = cv2.VideoCapture(video_path)
    
    # Get properties from source video to match output
    fps_input = cap.get(cv2.CAP_PROP_FPS)
    if fps_input == 0: fps_input = 30.0 # Default if metadata missing
    
    # Define the codec and create VideoWriter object
    # Saving at 320x240 as per our optimization config
    output_filename = os.path.join(config.LOG_FOLDER, "tracked_path_output.mp4")
    fourcc = cv2.VideoWriter_fourcc(*'mp4v') 
    out = cv2.VideoWriter(output_filename, fourcc, fps_input, (config.WIDTH, config.HEIGHT))

    pid = PIDController(config.KP, config.KI, config.KD, target=config.WIDTH//2)
    
    # Tracking Variables
    start_time = time.time()
    frame_count = 0
    last_cx = config.WIDTH // 2 
    
    print(f"Processing and saving to: {output_filename}")
    print("Press 'q' to stop early.")

    while cap.isOpened():
        t1 = cv2.getTickCount()
        ret, frame = cap.read()
        if not ret: break
        
        frame_count += 1
        frame_res = cv2.resize(frame, (config.WIDTH, config.HEIGHT))

        # 3. Masking & Processing
        roi_top = 80 
        roi = frame_res[roi_top:config.HEIGHT, 0:config.WIDTH]
        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, config.LOWER_COLOR, config.UPPER_COLOR)
        
        # Bridge gaps using Morphological Closing
        kernel = np.ones((5,5), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

        # 4. Full-Path Tracing
        bottom_slice = mask[-30:, :]
        M = cv2.moments(bottom_slice)
        if M["m00"] > 500:
            start_x = int(M["m10"] / M["m00"])
            last_cx = start_x 
        else:
            start_x = last_cx 

        full_path = trace_full_line(mask, start_x)

        # 5. Drawing Overlays
        if len(full_path) > 1:
            display_pts = np.array([(p[0], p[1] + roi_top) for p in full_path], np.int32)
            
            # Draw Path (Blue)
            cv2.polylines(frame_res, [display_pts.reshape((-1, 1, 2))], False, (255, 0, 0), 2)
            
            # Target (Green) and End (Red)
            cv2.circle(frame_res, tuple(display_pts[0]), 6, (0, 255, 0), -1)
            cv2.circle(frame_res, tuple(display_pts[-1]), 6, (0, 0, 255), -1)

        # Benchmarking & UI
        t2 = cv2.getTickCount()
        latency = (t2 - t1) / cv2.getTickFrequency() * 1000
        fps_curr = frame_count / (time.time() - start_time)
        cv2.putText(frame_res, f"FPS: {fps_curr:.1f} Lat: {latency:.2f}ms", (10, 25), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)

        # 6. SAVE FRAME TO VIDEO
        out.write(frame_res)

        # Show preview
        cv2.imshow("Processing Video...", frame_res)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # 7. Cleanup
    cap.release()
    out.release() # CRITICAL: If you don't release, the video won't save correctly!
    cv2.destroyAllWindows()
    print(f"Success! {frame_count} frames saved to {output_filename}")

if __name__ == "__main__":
    main()