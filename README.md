## GazeTR Real-time Demo Instructions

### 0. Introduction

We provide a GazeTR real-time demo that runs with live input from a webcam. And this demo is based on the demo of [few-shot-gaze](https://github.com/NVlabs/few_shot_gaze). 

Additionally, we applied the  GazeTR real-time demo to eSports for a competitive level comparison display. 

We used this demo to collect the gaze points of two players of different levels while observing League of Legends group battles. The positions of the heroes in the game were then marked, compared to the player's gaze points, and presented on the original video.

![tester1](https://github.com/GazeTR_demo/raw/master/readme_images/tester1.gif)

![tester2](https://github.com/GazeTR_demo/raw/master/readme_images/tester2.gif)

### 1. Setup

This codebase should run on most standard Linux systems. It is tested with Ubuntu 20.04, pytorch v1.10.1, cuda v10.2, python v3.6.13.

a. This demo uses three external submodules: [EOS](https://pypi.org/project/eos-py/) and
   [HRNET](https://github.com/HRNet/HRNet-Facial-Landmark-Detection) for face and facial landmarks detection, respectively, and [GazeTR](https://github.com/yihuacheng/GazeTR) as the main network.

you are supposed to  pull the submodules by running:

    git submodule update --init --recursive

In addition, please download the pre-trained `GazeTR-H-ETH.pt` model for GazeTR from [google drive](https://drive.google.com/file/d/1WEiKZ8Ga0foNmxM7xFabI4D5ajThWAWj/view?usp=sharing) or [ baidu cloud disk](https://pan.baidu.com/s/1GEbjbNgXvVkisVWGtTJm7g)
   and place it inside default home directory.

Also, please download the pre-trained `HR18-WFLW.pth` model for HRNet from [here](https://1drv.ms/u/s!AiWjZ1LamlxzdTsr_9QZCwJsn5U)
   and place it inside the folder:

    mkdir ext/HRNet-Facial-Landmark-Detection/hrnetv2_pretrained

b. Create a Python virtual environment:

    python3 -m venv venv
    source venv/bin/activate
    # if you have the same version of Ubuntu, python with me, 
    # you can replace "requirements.txt" with "reference_requirements.txt" in the following line.
    pip install -r requirements.txt
    sudo apt-get install -y software-properties-common
    sudo add-apt-repository ppa:ubuntu-toolchain-r/test
    sudo apt update
    sudo apt install g++-7 -y
    CC=`which gcc-7` CXX=`which g++-7` pip3 install eos-py

### 2. Camera and Monitor calibration
  a. Calibrate your camera:

    python calibrate_camera.py

   This should generate a file named `calib_cam<id>.pkl`.

   b. Calibrate your monitor's orientation and the position of its upper-left corner w.r.t. to the
   camera using the [Mirror-based Calibration](https://computer-vision.github.io/takahashi2012cvpr/) routine and
   update the methods `camera_to_monitor` and `monitor_to_camera` in `monitor.py` for your system appropriately.

   If you don't use the [Mirror-based Calibration](https://computer-vision.github.io/takahashi2012cvpr/) routine to update some functions,
   we assume that your camera is in-built camera and is 5mm directly above the upper border of your monitor.

   We recommend using the in-built camera in laptops or attaching an external webcam **rigidly** to your monitor.
   If you move your webcam relative to the monitor you will have to calibrate it again.


### 3. Run demo
    python run_demo.py

Without calibration, the gaze network will be used to continuously compute the user's on-screen point-of-regard and shown on the display.

### Best practices:

* A user should always look directly at the targets when pressing the arrow
keys and not at the keyboard to record accurate calibration data.

* For best results, experiment with the contrast, brightness and sharpness settings of your webcam .
    * see top of `run_demo.py`

* For best results, experiment with the learning rate and number of training steps used for fine-tuning.
    * Adjust the `lr` argument of `fine_tune` as called from `run_demo.py`.

* To change the delay/smoothing of the estimated on-screen point-of-regard modify the Kalman filter settings
in `frame_processor.py`.
