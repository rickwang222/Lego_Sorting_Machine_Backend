# Lego Machine Repo

## Contents

- **DemoWebsite**: Simple combined website/inference web server for our presentation demo.
- **SyntheticData**: Python script for Blender that produces annotated, synthetic images of Legos.
- **UIServer**: Flask server that runs on a Raspberry Pi which captures images and forwards them to the inference machine upon request.
- **website**: Original website code which has since been integrated into the UIServer directory.
- **start_lego_proj.sh**: Bash script to be placed on a Raspberry Pi which creates an SSH tunnel to the inference server and starts both the UIServer and remotely starts the InferenceServer.
- **start_lego_proj_inf.sh**: Bash script to be placed on the inference machine which starts the InferenceServer. Called by start_lego_proj.sh.


## How it works

[software-diagram]: figures/Detailed%20Software%20Diagram.png "Detailed software diagram."
[transmission-diagram]: figures/Remote%20Inference%20Transmission%20Sequence.png "Remote inference transmission diagram."
![alt text][software-diagram]

The client browser runs on a personal computer, interfacing with the UI server running on the Raspberry Pi via HTTP requests. The UI server makes an HTTP request to the inference server when it needs an image to be processed. The Raspberry Pi uses its GPIO ports to (indirectly) control the motors, and uses USB to capture images from the camera.

The client's browser connects to the UI Server via a port forwarded over SSH. Currently this port is set to 9000. The web page provides start and reset buttons. When the start button is pressed, the client makes requests to the /open-hopper, /start-shaker, and /start-conveyor endpoints, and begins regularly making requests to the /process-img endpoint.

![alt text][transmission-diagram]

The /process-img endpoint on the UI server will ultimately return an annotated image and a list of the pieces identified in the image. The Pi is not powerful enough on its own to complete inference in a timely manner, so it sends the last image it captured to the inference server on the much more powerful inference machine to complete the request. The inference machine returns an annotated version of the image, along with a list of objects it detected. The Pi finally forwards this back to the client browser.

When the reset button is pressed, the motors are stopped and no new images are captured.

Many more details can be found in the Final Report pdf, as well as the commented source code.


## Running the server code

### Inference server
[Miniconda](https://docs.conda.io/en/latest/miniconda.html) can be used to easily manage the YOLOv5 environment on the inference server.

1. Create the environment. Tested with Python 3.10. I encountered a resource leak issue on Python 3.7 while running hyperparameter evolution, so I would avoid 3.7.
```
conda create -n py310yolov5 python=3.10
```

2. Install PyTorch.
```
conda activate py310yolov5
conda install pytorch torchvision torchaudio cudatoolkit=11.3 -c pytorch
```

3. Clone YOLOv5 repo and install requirements.
```
cd <wherever you'd like to clone it>
git clone https://github.com/ultralytics/yolov5.git
cd yolov5/
pip install -r requirements.txt
```

4. All done, now you can run the inference server.

```
python3 server.py
```

A script `start_lego_proj_inf.sh` is provided to launch the inference server. This should be placed and ran on the inference machine itself.

### UI server (for Raspberry Pi)

1. Install Python 3 and PIP (if not already installed).
```
sudo apt install python-pip3
```

2. Install dependencies.
```
pip install opencv-python numpy flask RPi.GPIO gpiozero
```

3. All done, now you can run the UI server.
```
python3 server.py
```

A script `start_lego_proj.sh` is provided to launch both the UI server on the local machine AND the inference server on the remote machine. This should be placed and ran on the Raspberry Pi.
