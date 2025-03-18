# DETR - Detection Transformer (DETR) object detection in Python

Detection Transformer (DETR) in Python - a simple minimal inference demonstration code for Detection Transformer (DETR) object detection

### Background

Heavily based on the example code from [https://learnopencv.com/detr-overview-and-inference/](https://learnopencv.com/detr-overview-and-inference/), and making full use of the [Hugging Face Transformers Library](https://pypi.org/project/transformers/), intended for use in teaching within the undergraduate Computer Science programme
at [Durham University](http://www.durham.ac.uk) (UK) by [Prof. Toby Breckon](https://breckon.org/toby/).

Implements DETR object detection based on the ResNet 50 backbone as listed in the original reference implementation from [FAIR](https://github.com/facebookresearch/detr).

### Code Details:

- _detr_\__demo.py_: live DETR object detection from webcam / video file
- _camera_stream.py_: threaded camera capture interface

_detr_\__demo.py_ runs with a webcam connected or from a command line supplied video
file of a format OpenCV supports on your system (otherwise edit the script to provide your own image source).

![Python - PEP8](https://github.com/tobybreckon/detr-example/workflows/Python%20-%20PEP8/badge.svg)

Tested with [OpenCV](http://www.opencv.org) 4.x, Python 3.x and Transformers 4.49.

Performance: ~10 fps on ``NVIDIA GeForce RTX 3080 Laptop GPU 15993Mb, sm_86, Driver/Runtime ver.12.80/12.40``

---

### How to download and run:

Clone repository and run as follows:

```
git clone https://github.com/tobybreckon/detr-example.git
cd detr-example
pip install -r requirements.txt 
python3 ./detr_demo.py [optional video file]
```

Command line usage of the DETR demo is as follows:

```
usage: detr-demo.py [-h] [-c CAMERA_TO_USE] [-r RESCALE]
                    [-s SET_RESOLUTION SET_RESOLUTION] [-fs]
                    [video_file]

Perform ./detr-demo.py example operation on incoming camera/video image

positional arguments:
  video_file            specify optional video file

options:
  -h, --help            show this help message and exit
  -c, --camera_to_use CAMERA_TO_USE
                        specify camera to use
  -r, --rescale RESCALE
                        rescale image by this factor
  -s, --set_resolution SET_RESOLUTION SET_RESOLUTION
                        override default camera resolution as H W
  -fs, --fullscreen     run in full screen mode


```

Once the tracking script is running perform the following steps

- use the slider to adjust detection threshold
- press 'x' to exit, press 'f' to toggle fullscreen

Demo source code is provided _"as is"_ to aid learning and understanding of topics on the course and beyond.

---

If you find any bugs raise an issue (or much better still submit a git pull request with a fix) - toby.breckon@durham.ac.uk

_"may the source be with you"_ - anon.
