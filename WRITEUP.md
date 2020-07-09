# Project Write-Up

You can use this document as a template for providing your project write-up. However, if you
have a different format you prefer, feel free to use it as long as you answer all required
questions.

# Downloading and Converting the Models
## 1. Tensorflow SSD Mobilenet V2 COCO
## Use the following steps:
        1. Download model
            command: `wget http://download.tensorflow.org/models/object_detection/ssd_mobilenet_v2_coco_2018_03_29.tar.gz`
        2. Unpack the downloaded file
            command: `tar -xvf ssd_mobilenet_v2_coco_2018_03_29.tar.gz`
        3. change directory to the extracted folder
        4. Convert model using following command:
            `python /opt/intel/openvino/deployment_tools/model_optimizer/mo.py --input_model frozen_inference_graph.pb --tensorflow_object_detection_api_pipeline_config pipeline.config --reverse_input_channels --tensorflow_use_custom_operations_config /opt/intel/openvino/deployment_tools/model_optimizer/extensions/front/tf/ssd_v2_support.json`
        5. Run model [check port and model path]:
            `python main.py -i resources/Pedestrian_Detect_2_1_1.mp4 -m ssd_mobilenet_v2_coco_2018_03_29/frozen_inference_graph.xml -l /opt/intel/openvino/deployment_tools/inference_engine/lib/intel64/libcpu_extension_sse4.so -d CPU -pt 0.6 | ffmpeg -v warning -f rawvideo -pixel_format bgr24 -video_size 768x432 -framerate 24 -i - http://0.0.0.0:3004/fac.ffm`

## 2. Tensorflow SSD Inception V2 COCO
## Use the following steps:
        1. Download model
            command: `wget http://download.tensorflow.org/models/object_detection/ssd_inception_v2_coco_2018_01_28.tar.gz`
        2. Unpack the downloaded file
            command: `tar -xvf ssd_inception_v2_coco_2018_01_28.tar.gz`
        3. change directory to the extracted folder
        4. Convert model using following command:
            `python /opt/intel/openvino/deployment_tools/model_optimizer/mo.py --input_model frozen_inference_graph.pb --tensorflow_object_detection_api_pipeline_config pipeline.config --reverse_input_channels --tensorflow_use_custom_operations_config /opt/intel/openvino/deployment_tools/model_optimizer/extensions/front/tf/ssd_v2_support.json`
        5. Run model [check port and model path]:
            `python main.py -i resources/Pedestrian_Detect_2_1_1.mp4 -m ssd_inception_v2_coco_2018_01_28/frozen_inference_graph.xml -l /opt/intel/openvino/deployment_tools/inference_engine/lib/intel64/libcpu_extension_sse4.so -d CPU -pt 0.6 | ffmpeg -v warning -f rawvideo -pixel_format bgr24 -video_size 768x432 -framerate 24 -i - http://0.0.0.0:3004/fac.ffm`
       
## 2. Tensorflow SSD MobileNet V1 0.75 Depth COCO
## Use the following steps:
        1. Download model
            command: `wget http://download.tensorflow.org/models/object_detection/ssd_mobilenet_v1_0.75_depth_300x300_coco14_sync_2018_07_03.tar.gz`
        2. Unpack the downloaded file
            command: `tar -xvf ssd_mobilenet_v1_0.75_depth_300x300_coco14_sync_2018_07_03.tar.gz`
        3. change directory to the extracted folder
        4. Convert model using following command:
            `python /opt/intel/openvino/deployment_tools/model_optimizer/mo.py --input_model frozen_inference_graph.pb --tensorflow_object_detection_api_pipeline_config pipeline.config --reverse_input_channels --tensorflow_use_custom_operations_config /opt/intel/openvino/deployment_tools/model_optimizer/extensions/front/tf/ssd_v2_support.json`
        5. Run model [check port and model path]:
            `python main.py -i resources/Pedestrian_Detect_2_1_1.mp4 -m ssd_mobilenet_v1_0.75_depth_300x300_coco14_sync_2018_07_03/frozen_inference_graph.xml -l /opt/intel/openvino/deployment_tools/inference_engine/lib/intel64/libcpu_extension_sse4.so -d CPU -pt 0.6 | ffmpeg -v warning -f rawvideo -pixel_format bgr24 -video_size 768x432 -framerate 24 -i - http://0.0.0.0:3004/fac.ffm`
       
## Explaining Custom Layers

Model Optimizer searches for each layer of the input model in the list of known layers before building the model's internal representation, optimizing the model, and producing the Intermediate Representation. The list of known layers is different for each of supported frameworks.

Custom layers are layers that are not included into a list of known layers. If your topology contains any layers that are not in the list of known layers, the Model Optimizer classifies them as custom.

You have two options for TensorFlow* models with custom layers:

1. Register those layers as extensions to the Model Optimizer. In this case, the Model Optimizer generates a valid and optimized Intermediate Representation.
2. If you have sub-graphs that should not be expressed with the analogous sub-graph in the Intermediate Representation, but another sub-graph should appear in the model, the Model Optimizer provides such an option. This feature is helpful for many TensorFlow models.

Some of the potential reasons for handling custom layers is that there is a possibility that the Intermediate Representation does not support all the layers in the orginal framework. This might be due to hardware limitation i.e., few IR models are supported on CPU while others may not.

## Comparing Model Performance

My method(s) to compare models before and after conversion to Intermediate Representations
were...

The difference between model accuracy pre- and post-conversion was...

The size of the model pre- and post-conversion was...

1. Model: ssd_mobilenet_v2_coco_2018_03_29
    size-before: 180 Mb (zip file)
    size-after: 65 Mb

2. Model: TensorFlow Resnet v1
    size-before: 351 Mb
    size-after: 129Mb

The inference time of the model pre- and post-conversion was...

1. Model: ssd_mobilenet_v2_coco_2018_03_29
   avg_infer time: 69-70 ms
          
2. Model: TensorFlow Resnet v1
   avg_infer time: 2600 - 2640 ms

## Assess Model Use Cases

Some of the potential use cases of the people counter app are...
This application is best fit for scenarios such as:
1. Monitor social distancing by counting number of people in an enclosed space during situations like COVID-19
2. Monitor the average attendee count in case of conferences or lectures
3. Determine the popularity of product lines in a store

Each of these use cases would be useful because...
1. This will help in taking the necessary precautions to handle the COVID-19 pandemic situation.
2. It can determine the average number of students/attendees who take up the class or attend the conference and help monitor the performance.
3. This is will provide insights as to the popularity among different product sections in a store enabling better business decisions.

## Assess Effects on End User Needs

Lighting, model accuracy, and camera focal length/image size have different effects on a deployed edge model. The potential effects of each of these are as follows...

1. In places like clothing stores, mannequins can be counted as people resulting in incorrect data and insights.
2. In order to be detected properly, people have to stand up straight.
3. There might be confused count in case of a crowded place again resulting in incorrect data and insights.

## Model Research

[This heading is only required if a suitable model was not found after trying out at least three
different models. However, you may also use this heading to detail how you converted 
a successful model.]

In investigating potential people counter models, I tried each of the following three models:

- Model 1: [Name]
  - [Model Source]
  - I converted the model to an Intermediate Representation with the following arguments...
  - The model was insufficient for the app because...
  - I tried to improve the model for the app by...
  
- Model 2: [Name]
  - [Model Source]
  - I converted the model to an Intermediate Representation with the following arguments...
  - The model was insufficient for the app because...
  - I tried to improve the model for the app by...

- Model 3: [Name]
  - [Model Source]
  - I converted the model to an Intermediate Representation with the following arguments...
  - The model was insufficient for the app because...
  - I tried to improve the model for the app by...

