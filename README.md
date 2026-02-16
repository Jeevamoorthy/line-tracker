# Industrial Line Follower Prototype (High-Speed)

### Overview
This system is optimized for sub-10ms latency processing using a focused ROI and Image Moments instead of heavy contour detection.

### Pipeline Logic
1. **Capture**: Grabs 320x240 frame.
2. **Crop**: Extract ROI defined in `config.py`.
3. **Threshold**: Convert to HSV and isolate the line color.
4. **Locate**: Calculate the centroid ($C_x$) of the white pixels.
5. **Control**: Input $C_x$ into PID to generate steering output.

### How to Run
1. Install dependencies: `pip install opencv-python numpy`
2. Place your test video in `/assets/test_line_video.mp4`.
3. Run `python src/main.py`.
4. Press 'Q' to exit.

   <img width="1300" height="966" alt="image" src="https://github.com/user-attachments/assets/72c43ec5-151e-47ef-a93e-a3f96f4eacb1" />

   <img width="1920" height="1080" alt="Screenshot 2026-02-15 133102" src="https://github.com/user-attachments/assets/28ce58c4-3f5d-40b7-9abe-c80f6be20d7d" />



### Benchmarking Targets
- **PC Target**: < 2ms processing time per frame.
- **Jetson Nano Target**: < 5ms processing time per frame.

### Improvements
The version of OpenCV provided by pip is a generic build. To get high-speed performance on the Jetson Nano, you need the version of OpenCV that comes pre-installed with NVIDIA JetPack. The pre-installed version is specifically compiled with:
GStreamer Support: Required for the hardware-accelerated camera pipeline.
CUDA Support: Required if you decide to offload the color masking to the GPU.
V4L2 Support: For low-latency USB camera access.
On the Jetson, you will only need to install numpy:
code
Bash
pip install numpy
