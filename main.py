"""People Counter."""
"""
 Copyright (c) 2018 Intel Corporation.
 Permission is hereby granted, free of charge, to any person obtaining
 a copy of this software and associated documentation files (the
 "Software"), to deal in the Software without restriction, including
 without limitation the rights to use, copy, modify, merge, publish,
 distribute, sublicense, and/or sell copies of the Software, and to
 permit person to whom the Software is furnished to do so, subject to
 the following conditions:
 The above copyright notice and this permission notice shall be
 included in all copies or substantial portions of the Software.
 THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
 EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
 MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
 NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE
 LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION
 OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION
 WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
"""


import os
import sys
import time
import socket
import json
import cv2

import logging as log
import paho.mqtt.client as mqtt

from argparse import ArgumentParser
from inference import Network

# MQTT server environment variables
HOSTNAME = socket.gethostname()
IPADDRESS = socket.gethostbyname(HOSTNAME)
MQTT_HOST = IPADDRESS
MQTT_PORT = 3001
MQTT_KEEPALIVE_INTERVAL = 60


def build_argparser():
    """
    Parse command line arguments.
    :return: command line arguments
    """
    parser = ArgumentParser()
    parser.add_argument("-m", "--model", required=True, type=str,
                        help="Path to an xml file with a trained model.")
    parser.add_argument("-i", "--input", required=True, type=str,
                        help="Path to image or video file")
    parser.add_argument("-l", "--cpu_extension", required=False, type=str,
                        default=None,
                        help="MKLDNN (CPU)-targeted custom layers."
                             "Absolute path to a shared library with the"
                             "kernels impl.")
    parser.add_argument("-d", "--device", type=str, default="CPU",
                        help="Specify the target device to infer on: "
                             "CPU, GPU, FPGA or MYRIAD is acceptable. Sample "
                             "will look for a suitable plugin for device "
                             "specified (CPU by default)")
    parser.add_argument("-pt", "--prob_threshold", type=float, default=0.5,
                        help="Probability threshold for detections filtering"
                        "(0.5 by default)")
    return parser


def connect_mqtt():
    ### TODO: Connect to the MQTT client ###
    client = mqtt.Client()
    client.connect(MQTT_HOST, MQTT_PORT, MQTT_KEEPALIVE_INTERVAL)
    return client


def infer_on_stream(args, client):
    """
    Initialize the inference network, stream video to network,
    and output stats and video.
    :param args: Command line arguments parsed by `build_argparser()`
    :param client: MQTT client
    :return: None
    """
    # Initialise the class
    infer_network = Network()
    # Set Probability threshold for detections
    prob_threshold = args.prob_threshold

    ### TODO: Load the model through `infer_network` ###
    infer_network.load_model(args.model, args.device, args.cpu_extension)
    net_input_shape = infer_network.get_input_shape()
    
    single_image_mode = False
    ### TODO: Handle the input stream ###
    if args.input == "CAM":
        input_validated = 0
    elif args.input.endswith('.jpg') or args.input.endswith('.bmp'):
        single_image_mode = True
        input_validated = args.input
    else:
        input_validated = args.input
        assert os.path.isfile(args.input), "File doesn't exist!"

    request_id = 0
    prev_count = 0
    total_count = 0
    start_time = 0
    total_duration = 0
    counter = 0
    dur = 0
    prev_duration = 0
    prev_total_count = 0

    ### TODO: Loop until stream is over ###
    cap = cv2.VideoCapture(args.input)
    cap.open(args.input)

    w = int(cap.get(3))
    h = int(cap.get(4))

    fps = cap.get(cv2.CAP_PROP_FPS)
    while cap.isOpened():
        ### TODO: Read from the video capture ###
        flag, frame = cap.read()
       
        
        if not flag:
            break
        key_pressed = cv2.waitKey(60)

        ### TODO: Pre-process the image as needed ###
        p_frame = cv2.resize(frame, (net_input_shape[3], net_input_shape[2]))
        p_frame = p_frame.transpose((2, 0, 1))
        p_frame = p_frame.reshape(1, *p_frame.shape)

        ### TODO: Start asynchronous inference for specified request ###
        inference_start = time.time()
        infer_network.exec_net(request_id, p_frame)
        
        ### TODO: Wait for the result ###
        if infer_network.wait(request_id) == 0:
            ### TODO: Get the results of the inference request ###
            result = infer_network.get_output(request_id)
            detection_time = time.time() - inference_start
            ### TODO: Extract any desired stats from the results ###
            current_count = 0
            for obj in result[0][0]:
                ### Draw bounding box if exceeding threshold
                if obj[2] > prob_threshold:
                    x_min = int(obj[3] * w)
                    y_min = int(obj[4] * h)
                    x_max = int(obj[5] * w)
                    y_max = int(obj[6] * h)

                    cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (0, 0, 0), 1)
                    current_count += 1
            
            inference_time_msg = "Inference time: {:.3f}ms".format(detection_time * 1000)
            cv2.putText(frame, inference_time_msg, (15, 15), cv2.FONT_HERSHEY_COMPLEX, 0.5, (200, 10, 10), 1)
            
            if current_count != counter:
                prev_count, counter = counter, current_count
                if dur >= 10:
                    prev_duration = dur
                    dur = 0
                else:
                    dur = prev_duration + dur
                    prev_duration = 0
            else:
                dur += 1
                if dur == 10 and counter > prev_count:
                    total_count += counter - prev_count
                elif dur == 10 and counter < prev_count:
                    total_duration = (prev_duration * 100)
            ### TODO: Calculate and send relevant information on ###
            ### current_count, total_count and duration to the MQTT server ###
            ### Topic "person": keys of "count" and "total" ###
            ### Topic "person/duration": key of "duration" ###
            
            
            client.publish("person", json.dumps({"count": current_count, "total": total_count}))
            if total_duration is not None and total_count > prev_total_count:
                client.publish("person/duration", json.dumps({"duration": total_duration}))
            
            prev_total_count = total_count
            
            if key_pressed == 27: ### Esc pressed
                break
        
        ### TODO: Send the frame to the FFMPEG server ###
        sys.stdout.buffer.write(frame)
        sys.stdout.flush()

        ### TODO: Write an output image if `single_image_mode` ###
        if single_image_mode:
            cv2.imwrite("output_img.jpg", frame)
    
    cap.release()
    cv2.destroyAllWindows()
    client.disconnect()


def main():
    """
    Load the network and parse the output.
    :return: None
    """
    # Grab command line args
    args = build_argparser().parse_args()
    # Connect to the MQTT server
    client = connect_mqtt()
    # Perform inference on the input stream
    infer_on_stream(args, client)


if __name__ == '__main__':
    main()