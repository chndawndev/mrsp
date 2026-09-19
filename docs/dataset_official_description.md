# Dataset

C3VDv2 consists of two distinct colon shapes (c1 and c2), each segmented into seven to eight anatomical regions, with each segment further having four unique textures and colors (t1, t2, t3, and t4). C3VDv2 contains 192 videos with a total of 169,371 frames. It comprises three different types of video sequences:

* Pixel-level Ground Truth Videos: [Registered Videos](https://durrlab.github.io/C3VDv2/#registered-1) were acquired with a static, undeformed colon phantom and are provided with per-frame ground truth maps (depth, normals, optical flow, etc.). Up to three videos were recorded per phantom segment:
   * v1: clean colon with a baseline camera trajectory and imaging settings.
   * v2: clean colon with a different camera trajectory and imaging settings as v1.
   * v3: debris-filled colon using the same camera trajectory and imaging settings as v2.
This category includes 169 short videos with a total of 67,886 frames.
* Deformation Videos: [Deformation Videos](https://durrlab.github.io/C3VDv2/#deformation) consist of v4 videos featuring externally induced active phantom deformation, captured with either static or linear camera motion. All videos include debris. Each folder contains all recorded RGB frames and a corresponding pose.txt file (if camera is not stationary). The camera poses are in a frame-wise homogeneous format. This folder contains 15 short videos with a total of 6,185 frames.
* Simulated Screening Videos: [Screening Videos](https://durrlab.github.io/C3VDv2/#screening) comprise full-colon withdrawal sequences performed by a gastroenterologist to capture realistic camera motion. Similar to deformation videos, only RGB frames and camera poses in pose.txt are provided. A total of 8 videos are included, comprising 95,300 frames.
