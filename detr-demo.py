##########################################################################

# Example : perform DETR object detection live from a video file
# specified on the command line (e.g. python FILE.py video_file) or from an
# attached web camera

# Author : Toby Breckon, toby.breckon@durham.ac.uk

# Copyright (c) 2025 Toby Breckon, Engineering & Computing Science,
#                    Durham University, UK
# License : LGPL - http://www.gnu.org/licenses/lgpl.html

# Acknowledgements: 
# -- based on code at https://learnopencv.com/detr-overview-and-inference/

##########################################################################

import cv2
import argparse
import sys
import math
import torch
import numpy as np
from transformers import DetrForObjectDetection, DetrImageProcessor

##########################################################################

keep_processing = True

# parse command line arguments for camera ID or video file

parser = argparse.ArgumentParser(
    description='Perform ' +
    sys.argv[0] +
    ' example operation on incoming camera/video image')
parser.add_argument(
    "-c",
    "--camera_to_use",
    type=int,
    help="specify camera to use",
    default=0)
parser.add_argument(
    "-r",
    "--rescale",
    type=float,
    help="rescale image by this factor",
    default=1.0)
parser.add_argument(
    "-s",
    "--set_resolution",
    type=int,
    nargs=2,
    help='override default camera resolution as H W')
parser.add_argument(
    "-fs",
    "--fullscreen",
    action='store_true',
    help="run in full screen mode")
parser.add_argument(
    'video_file',
    metavar='video_file',
    type=str,
    nargs='?',
    help='specify optional video file')
args = parser.parse_args()

##########################################################################
# calculate size of text label for image


def calculate_label_size(box_width, box_height, frame_width, frame_height, min_scale=0.4, max_scale=1.2):
    """
    Calculate appropriate font scale based on bounding box size and frame dimensions.
    """
    box_size_ratio = (box_width * box_height) / (frame_width * frame_height)
    font_scale = min_scale + \
        (np.log(1 + box_size_ratio * 100) / 5) * (max_scale - min_scale)
    return np.clip(font_scale, min_scale, max_scale)

##########################################################################
# Predefined colors for object classes


def generate_color_palette(num_classes):
    np.random.seed(42)  # Ensure the same colors are generated every time
    return {label: tuple(np.random.randint(0, 255, 3).tolist()) for label in range(num_classes)}


##########################################################################
# Create a color palette for each class (COCO has 91 classes by default)


colors = generate_color_palette(num_classes=91)

##########################################################################
# Function to get the color for a specific label


def get_label_color(label):
    return colors.get(label, (0, 255, 0))  # Default to green if no label found

##########################################################################
# dummy on trackbar callback function


def on_trackbar(val):
    return

##########################################################################

# define video capture object


try:
    # to use a non-buffered camera stream (via a separate thread)

    if not (args.video_file):
        import camera_stream
        cap = camera_stream.CameraVideoStream()
    else:
        cap = cv2.VideoCapture()  # not needed for video files

except BaseException:
    # if not then just use OpenCV default

    print("INFO: camera_stream class not found - camera input may be buffered")
    cap = cv2.VideoCapture()

# define display window name

window_name = "Live Camera Input"  # window name

# if command line arguments are provided try to read video_name
# otherwise default to capture from attached camera

if (((args.video_file) and (cap.open(str(args.video_file))))
        or (cap.open(args.camera_to_use))):

    # create window by name (as resizable) + trackbar

    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.setWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN,
                          cv2.WINDOW_FULLSCREEN & args.fullscreen)
    trackbarName = 'reporting confidence > (x 0.01)'
    cv2.createTrackbar(trackbarName, window_name, 70, 100, on_trackbar)

   # override default camera resolution

    if (args.set_resolution is not None):
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.set_resolution[1])
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.set_resolution[0])

    print("INFO: input resolution : (",
          int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), "x",
          int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)), ")")

    # initialize the model and image processor; resizes inputto (800x1333)

    model = DetrForObjectDetection.from_pretrained("facebook/detr-resnet-50")
    processor = DetrImageProcessor.from_pretrained("facebook/detr-resnet-50")

    # set model to use GPU if available

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    torch.cuda.empty_cache()
    model.to(device)
    print("INFO: DEVICE in use : ", device)

    while (keep_processing):

        # start a timer (to see how long processing and display takes)

        start_t = cv2.getTickCount()

        # if camera /video file successfully open then read frame

        if (cap.isOpened):
            ret, frame = cap.read()

            # when we reach the end of the video (file) exit cleanly

            if (ret == 0):
                keep_processing = False
                continue

            # rescale if specified

            if (args.rescale != 1.0):
                frame = cv2.resize(
                    frame, (0, 0), fx=args.rescale, fy=args.rescale)

        # convert input to RGB (from BGR) and format as tensor inputs to DETR

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        inputs = processor(images=frame_rgb, return_tensors="pt")
        inputs.to(device)

        # pass this inputs to the model (inference only)

        with torch.no_grad():
            outputs = model(**inputs)

        # convert outputs to standard COCO bounding box format

        # Image size (height, width)
        target_sizes = torch.tensor([frame_rgb.shape[:2]])
        results = processor.post_process_object_detection(
            outputs, target_sizes=target_sizes)[0]

        # Filter boxes based on confidence score (threshold can be adjusted)

        results = results
        confThreshold = cv2.getTrackbarPos(trackbarName, window_name) / 100
        scores = results["scores"].cpu().numpy()
        keep = scores > confThreshold
        boxes = results["boxes"].cpu().numpy()[keep]
        labels = results["labels"].cpu().numpy()[keep]
        scores = scores[keep]

        # draw the bounding boxes and labels on the image

        for box, label, score in zip(boxes, labels, scores):
            xmin, ymin, xmax, ymax = box
            box_width = xmax - xmin
            box_height = ymax - ymin
            img_width, img_height = frame.shape[:2]
            font_scale = calculate_label_size(
                box_width, box_height, img_width, img_height, max_scale=1)

            label_text = f"{model.config.id2label[label]}: {score:.2f}"
            (text_width, text_height), baseline = cv2.getTextSize(
                label_text,
                cv2.FONT_HERSHEY_SIMPLEX,
                font_scale, 1
            )

            # Get the color for this label
            color = get_label_color(label)

            # Draw rectangle and label with the same color for the same class
            cv2.rectangle(frame, (int(xmin), int(ymin)), (int(
                xmax), int(ymax)), color, max(2, int(font_scale * 3)))
            cv2.rectangle(frame, (int(xmin), int(ymin) - text_height - baseline - 5),
                          (int(xmin) + text_width, int(ymin)), color, -1)
            cv2.putText(frame, label_text, (int(xmin), int(ymin) - baseline - 2),
                        cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 0, 0), max(1, int(font_scale * 1.5)), cv2.LINE_AA)

        # stop the timer and convert to ms. (to see how long processing takes)

        stop_t = ((cv2.getTickCount() - start_t) /
                  cv2.getTickFrequency()) * 1000

        # Display efficiency information

        label = ('Inference time: %.2f ms' % stop_t) + \
            (' (Framerate: %.2f fps' % (1000 / stop_t)) + ')'
        cv2.putText(frame, label, (0, 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255))

        # display image

        cv2.imshow(window_name, frame)

        # start the event loop - essential

        # cv2.waitKey() is a keyboard binding function (argument is the time in
        # milliseconds). It waits for specified milliseconds for any keyboard
        # event. If you press any key in that time, the program continues.
        # If 0 is passed, it waits indefinitely for a key stroke.
        # (bitwise and with 0xFF to extract least significant byte of
        # multi-byte response)

        # wait 40ms or less depending on processing time taken (i.e. 1000ms /
        # 25 fps = 40 ms)

        key = cv2.waitKey(max(2, 40 - int(math.ceil(stop_t)))) & 0xFF

        # It can also be set to detect specific key strokes by recording which
        # key is pressed

        # e.g. if user presses "x" then exit  / press "f" for fullscreen
        # display

        if (key == ord('x')):
            keep_processing = False
        elif (key == ord('f')):
            args.fullscreen = not (args.fullscreen)
            cv2.setWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN,
                                  cv2.WINDOW_FULLSCREEN & args.fullscreen)

    # close all windows

    cv2.destroyAllWindows()

else:
    print("No video file specified or camera connected.")

##########################################################################
