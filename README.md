
# How To Train an Object Detection Classifier for Cigarettes Using TensorFlow GPU on Windows 10

## Brief Summary

This repository tells how to train a Tensorflow model using the Faster R-cnn algorithm for Anaconda virtual environment on windows 10 operating system. There are also photographs that you will use for training in the warehouse and the codes to export the Inference Graph of the completed training model. This model has been prepared to detect cigarettes and its derivatives, but can be reused for different objects with minor changes in the training data and trained model settings.

**I also prepared a video showing how the completed model works in real time.**

[![](https://raw.githubusercontent.com/yavashaktan/Cigarette-Detector-Training-Tutorial-TensorFlow-Object-Detection-API/master/readme.md%20files/video.png?token=AMFEBNM4XQ5WLPWK4UA5NPC6TN2HM)](https://www.youtube.com/watch?v=Vz-mCZyb2Q0)


This readme file describes all the steps needed to prepare your own object detector:

1. [Anaconda, CUDA, and cuDNN Setup]
2. [Preparation of object detection folder and Anaconda virtual environment]
3. [Tagging pictures and using labelImg]
4. [Preparation of training data (train.xml - train.record)]
5. [Creating a label map and making training settings]
6. [Training]
7. [Exporting the inference graph]
8. [using your newly trained object detection classifier with mp4 file and webcam][Appendix: Common Errors]

This repository contains all the files you need to train the model to identify cigarettes and their variants. In this tutorial, I also explained where you need to change the detector you need. After all, you can learn how to test a trained model with an mp4 file or your camera. (All codes are available.)


## Introduction
The purpose of this tutorial is to explain how to train your own convolutional neural network object detection classifier for an object starting from scratch.

At the end of this tutorial, you will have a program that can identify and draw boxes around certain objects in pictures or videos.

The tutorial is written for Windows 10, and it will also work for Windows 7 and 8. I use TensorFlow-GPU v1.14 while writing the initial version of this tutorial.

TensorFlow-GPU allows your PC to use the video card to provide extra processing power while training, so it will be used for this tutorial. In my experience, using TensorFlow-GPU instead of regular TensorFlow reduces training time by a factor of about 8. The CPU-only version of TensorFlow can also be used for this tutorial, but it will take longer. If you use CPU-only TensorFlow, you do not need to install CUDA and cuDNN in Step 1. 

## Steps
### 1. Install Anaconda, CUDA, and cuDNN

There is a nice video by Mark Jay for Anaconda, CUDA and cuDNN installations. You can also get help here.
Since I used TensorFlow 1.14, I installed CUDA 10 and cuDNN 7.6.5.
You can find out which version of CUDA and cuDNN you need to use for a different version of TensorFlow from TenfowFlow's site. 

If you are using an older version of TensorFlow, make sure you use the CUDA and cuDNN versions that are compatible with the TensorFlow version you are using. [Here](https://www.tensorflow.org/install/source#tested_build_configurations) is a table showing which version of TensorFlow requires which versions of CUDA and cuDNN.

Install [Anaconda](https://www.anaconda.com/distribution/#download-section) as instructed in the video. 

Visit [TensorFlow's website](https://www.tensorflow.org/install) for further installation details, including how to install it on other operating systems (like Linux). The [object detection repository](https://github.com/tensorflow/models/tree/master/research/object_detection) itself also has [installation instructions](https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/installation.md).

### 2. Set up TensorFlow Directory and Anaconda Virtual Environment

-   Go to  Start  and Search “environment variables”
-   Click “Edit the system environment variables”. This should open the “System Properties” window
-   In the opened window, click the “Environment Variables…” button to open the “Environment Variables” window.    
-   Under “System variables”, search for and click on the  `Path`  system variable, then click “Edit…”    
-   Add the following paths, then click “OK” to save the changes:    
    > -   `<INSTALL_PATH>\NVIDIA  GPU  Computing  Toolkit\CUDA\v10.0\bin`
    > -   `<INSTALL_PATH>\NVIDIA  GPU  Computing  Toolkit\CUDA\v10.0\libnvvp`
    > -   `<INSTALL_PATH>\NVIDIA  GPU  Computing  Toolkit\CUDA\v10.0\extras\CUPTI\libx64`
    > -   `<INSTALL_PATH>\NVIDIA  GPU  Computing  Toolkit\CUDA\v10.0\cuda\bin`This portion of the tutorial goes over the full set up required. It is fairly meticulous, but follow the instructions closely, because improper setup can cause unwieldy errors down the road.
    
#### Update your GPU drivers (Optional)[](https://tensorflow-object-detection-api-tutorial.readthedocs.io/en/latest/install.html#update-your-gpu-drivers-optional "Permalink to this headline")

If during the installation of the CUDA Toolkit (see  [Install CUDA Toolkit](https://tensorflow-object-detection-api-tutorial.readthedocs.io/en/latest/install.html#cuda-install)) you selected the  Express Installation  option, then your GPU drivers will have been overwritten by those that come bundled with the CUDA toolkit. These drivers are typically NOT the latest drivers and, thus, you may wish to updte your drivers.

-   Go to  [http://www.nvidia.com/Download/index.aspx](http://www.nvidia.com/Download/index.aspx)
-   Select your GPU version to download
-   Install the driver for your chosen OS

#### Create a new Conda virtual environment[](https://tensorflow-object-detection-api-tutorial.readthedocs.io/en/latest/install.html#create-a-new-conda-virtual-environment "Permalink to this headline")
-   Open a new  Terminal  window  
-   Type the following command:    
    > conda create -n tfgpu pip python=3.7    
-   The above will create a new virtual environment with name  `tfgpu`    
-   Now lets activate the newly created virtual environment by running the following in the  Anaconda Promt  window:    
    > activate tfgpu
    
Once you have activated your virtual environment, the name of the environment should be displayed within brackets at the beggining of your cmd path specifier, e.g.:

(tfgpu) C:\Users\haktan>

#### Install TensorFlow GPU for Python[](https://tensorflow-object-detection-api-tutorial.readthedocs.io/en/latest/install.html#install-tensorflow-gpu-for-python "Permalink to this headline")

-   Open a new  Terminal  window and activate the  tfgpu environment.
    
-   Once open, type the following on the command line:
    
    > pip install --upgrade tensorflow-gpu==1.14
    
-   Wait for the installation to finish  
 
#### Test your Installation[](https://tensorflow-object-detection-api-tutorial.readthedocs.io/en/latest/install.html#id9 "Permalink to this headline")
-   Open a new  Terminal  window and activate the  tfgpu environment (if you have not done so already)    
-   Start a new Python interpreter session by running:    
    > python    
-   Once the interpreter opens up, type:    
    >  import tensorflow as tf    
-   If the above code shows an error, then check to make sure you have activated the  tfgpu environment and that tfgpu was successfully installed within it in the previous step.    
-   Then run the following:    
    > hello = tf.constant('Hello, TensorFlow!')
    >  sess = tf.Session()    
-   Once the above is run, you should see a print-out similar (but not identical) to the one bellow:    
    > 2019-11-25 07:20:32.415386: I tensorflow/stream_executor/platform/default/dso_loader.cc:44] Successfully opened dynamic library nvcuda.dll
    > 2019-11-25 07:20:32.449116: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1618] Found device 0 with properties:
    > name: GeForce GTX 1070 Ti major: 6 minor: 1 memoryClockRate(GHz): 1.683
    > pciBusID: 0000:01:00.0
    > 2019-11-25 07:20:32.455223: I tensorflow/stream_executor/platform/default/dlopen_checker_stub.cc:25] GPU libraries are statically linked, skip dlopen check.
    > 2019-11-25 07:20:32.460799: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1746] Adding visible gpu devices: 0
    > 2019-11-25 07:20:32.464391: I tensorflow/core/platform/cpu_feature_guard.cc:142] Your CPU supports instructions that this TensorFlow binary was not compiled to use: AVX2
    > 2019-11-25 07:20:32.472682: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1618] Found device 0 with properties:
    > name: GeForce GTX 1070 Ti major: 6 minor: 1 memoryClockRate(GHz): 1.683
    > pciBusID: 0000:01:00.0
    > 2019-11-25 07:20:32.478942: I tensorflow/stream_executor/platform/default/dlopen_checker_stub.cc:25] GPU libraries are statically linked, skip dlopen check.
    > 2019-11-25 07:20:32.483948: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1746] Adding visible gpu devices: 0
    > 2019-11-25 07:20:33.181565: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1159] Device interconnect StreamExecutor with strength 1 edge matrix:
    > 2019-11-25 07:20:33.185974: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1165]      0
    > 2019-11-25 07:20:33.189041: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1178] 0:   N
    > 2019-11-25 07:20:33.193290: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1304] Created TensorFlow device (/job:localhost/replica:0/task:0/device:GPU:0 with 6358 MB memory) -> physical GPU (device: 0, name: GeForce GTX 1070 Ti, pci bus id: 0000:01:00.0, compute capability: 6.1)
    
-   Finally, run the following:
    
    >  print(sess.run(hello))
    > b'Hello, TensorFlow!'
    

## TensorFlow Models Installation[](https://tensorflow-object-detection-api-tutorial.readthedocs.io/en/latest/install.html#tensorflow-models-installation "Permalink to this headline")


### Install Prerequisites[](https://tensorflow-object-detection-api-tutorial.readthedocs.io/en/latest/install.html#install-prerequisites "Permalink to this headline")

Building on the assumption that you have just created your new virtual environment (whether that’s  tensorflow_cpu,  tensorflow_gpu  or whatever other name you might have used), there are some packages which need to be installed before installing the models.

| Prerequisite packages Name | Tutorial version-build |
|--------------------|---------------------------------|
|pillow             |6.2.1-py37hdc69c19_0 |
|lxml             |4.4.1-py37h1350720_0 |
|jupyter             |1.0.0-py37_7 |
|matplotlib            |3.1.1-py37hc8f65d3_0 |
|opencv            |3.4.2-py37hc319ecb_0 |
|pathlib            |1.0.1-cp37 |



The packages can be installed using  `conda`  by running:


>conda install pillow, 
>conda install lxml
>conda install jupyter
>conda install matplotlib
>conda install opencv
>conda install cython


### Downloading the TensorFlow Models[](https://tensorflow-object-detection-api-tutorial.readthedocs.io/en/latest/install.html#downloading-the-tensorflow-models "Permalink to this headline")

Note

To ensure compatibility with the chosen version of Tensorflow (i.e.  `1.14.0`), it is generally recommended to use one of the  [Tensorflow Models releases](https://github.com/tensorflow/models/releases), as they are most likely to be stable. Release  `v1.13.0`  is the last unofficial release before  `v2.0`  and therefore is the one used here.

-   Create a new folder under a path of your choice and name it  `TensorFlow`. (e.g.  `C:\Users\haktan\Documents\TensorFlow`).
-   From your  Terminal  `cd`  into the  `TensorFlow`  directory.
-   To download the models you can either use  [Git](https://git-scm.com/downloads)  to clone the  [TensorFlow Models v.1.13.0 release](https://github.com/tensorflow/models/tree/r1.13.0)  inside the  `TensorFlow`  folder, or you can simply download it as a  [ZIP](https://github.com/tensorflow/models/archive/r1.13.0.zip)  and extract it’s contents inside the  `TensorFlow`  folder. To keep things consistent, in the latter case you will have to rename the extracted folder  `models-r1.13.0`  to  `models`.
-   You should now have a single folder named  `models`  under your  `TensorFlow`  folder, which contains another 4 folders as such:

        TensorFlow
        └─ models
        ├── official
            ├── research
            ├── samples
            └── tutorials
            

#### 2b. Download the Faster-RCNN-Inception-V2-COCO model from TensorFlow's model zoo
TensorFlow offers several object detection models (specific neural network architectures and pre-trained classifiers) in the model zoo. Some models (such as the SSD-MobileNet model) have an architecture that allows for faster detection but with less accuracy, while some models (such as the Faster-RCNN model) provide slower detection but more accuracy. Since the object we consider in this training is cigarettes, we need to process it in a much more detailed and small area. Searching for a small cigarette in all input is a really difficult and long process. For this reason, we should search only in regions where the object we are looking for is likely to exist. Algorithms such as SSD or YOLO search all input and with one step, while the Faster R-CNN algorithm makes predictions about the areas where the object is likely to be in the input and includes only these prediction fields in the neural network. This feature will enable us to detect cigarettes with higher consistency and consume less resources. I seem to hear your voice, "So what's the downside of Faster R-CNN?" They work remarkably slow because CNN networks operate in 2 stages, . (The frame rate we obtained in training is 14 fps.)



This tutorial will use the Faster-RCNN-Inception-V2 model. [Download the model here.](http://download.tensorflow.org/models/object_detection/faster_rcnn_inception_v2_coco_2018_01_28.tar.gz) 

#### 2c. Download this tutorial's repository from GitHub
Download the full repository located on this page (scroll to the top and click Clone or Download) and extract all the contents directly into the C:\tensorflow1\models\research\object_detection directory. (You can overwrite the existing "README.md" file.) This establishes a specific directory structure that will be used for the rest of the tutorial. 

At this point, here is what your \object_detection folder should look like:


This repository contains the images, annotation data, .csv files, and TFRecords needed to train a "Cigarette Detector". You can use these images and data to practice making your own Cigarette Detector. It also contains Python scripts that are used to generate the training data. It has scripts to test out the object detection classifier on images, videos, or a webcam feed. You can ignore the \doc folder and its files; they are just there to hold the images used for this readme.


If you want to practice training your own "Cigarette Detector". You can leave all the files as they are. You can follow along with this tutorial to see how each of the files were generated, and then run the training. You will still need to generate the TFRecord files (train.record and test.record) as described in Step 4. 


If you want to train your own object detector, delete the following files (do not delete the folders):
- All files in \object_detection\images\train and \object_detection\images\test
- The “test_labels.csv” and “train_labels.csv” files in \object_detection\images
- All files in \object_detection\training
-	All files in \object_detection\inference_graph

Now, you are ready to start from scratch in training your own object detector. This tutorial will assume that all the files listed above were deleted, and will go on to explain how to generate the files for your own training dataset.

### Protobuf Installation/Compilation[](https://tensorflow-object-detection-api-tutorial.readthedocs.io/en/latest/install.html#protobuf-installation-compilation "Permalink to this headline")

The Tensorflow Object Detection API uses Protobufs to configure model and training parameters. Before the framework can be used, the Protobuf libraries must be downloaded and compiled.

This should be done as follows:

-   Head to the  [protoc releases page](https://github.com/google/protobuf/releases)
    
-   Download the latest  `protoc-*-*.zip`  release (e.g.  `protoc-3.11.0-win64.zip`  for 64-bit Windows)
    
-   Extract the contents of the downloaded  `protoc-*-*.zip`  in a directory  `<PATH_TO_PB>`  of your choice (e.g.  `C:\Program  Files\Google  Protobuf`)
    
-   Extract the contents of the downloaded  `protoc-*-*.zip`, inside  `C:\Program  Files\Google  Protobuf`
    
-   Add  `<PATH_TO_PB>`  to your  `Path`  environment variable (see  [Environment Setup](https://tensorflow-object-detection-api-tutorial.readthedocs.io/en/latest/install.html#set-env))
    
-   In a new  Terminal  [[1]](https://tensorflow-object-detection-api-tutorial.readthedocs.io/en/latest/install.html#id11),  `cd`  into  `TensorFlow/models/research/`  directory and run the following command:
    
    > //From within TensorFlow/models/research/
    > protoc object_detection/protos/*.proto --python_out=.


### Adding necessary Environment Variables[](https://tensorflow-object-detection-api-tutorial.readthedocs.io/en/latest/install.html#adding-necessary-environment-variables "Permalink to this headline")

1.  Install the  `Tensorflow\models\research\object_detection`  package by running the following from  `Tensorflow\models\research`:    
    > //From within TensorFlow/models/research/
    > pip install .    
2.  Add  research/slim  to your  `PYTHONPATH`:    
Windows
-   Go to  Start  and Search “environment variables”
-   Click “Edit the system environment variables”. This should open the “System Properties” window    
-   In the opened window, click the “Environment Variables…” button to open the “Environment Variables” window.    
-   Under “System variables”, search for and click on the  `PYTHONPATH`  system variable,    
    > -   If it exists then click “Edit…” and add  `<PATH_TO_TF>\TensorFlow\models\research\slim`  to the list
    > -   If it doesn’t already exist, then click “New…”, under “Variable name” type  `PYTHONPATH`  and under “Variable value” enter  `<PATH_TO_TF>\TensorFlow\models\research\slim`    
-   Then click “OK” to save the changes:    
where, in both cases,  `<PATH_TO_TF>`  replaces the absolute path to your  `TesnorFlow`  folder. (e.g.  `<PATH_TO_TF>`  =  `C:\Users\haktan\Documents`  if  `TensorFlow`  resides within your  `Documents`  folder)

### Test your Installation[](https://tensorflow-object-detection-api-tutorial.readthedocs.io/en/latest/install.html#test-tf-models "Permalink to this headline")
-   Open a new  Terminal  window and activate the  tensorflow_gpu  environment (if you have not done so already)    
-   `cd`  into  `TensorFlow\models\research\object_detection`  and run the following command:    
    > // From within TensorFlow/models/research/object_detection
    > jupyter notebook         
-   This should start a new  `jupyter  notebook`  server on your machine and you should be redirected to a new tab of your default browser.    
-   Once there, simply follow  [sentdex’s Youtube video](https://youtu.be/COlbP62-B-U?t=7m23s)  to ensure that everything is running smoothly.    
-   When done, your notebook should look similar to the image bellow:    

### 3. Gather and Label Pictures
Now that the TensorFlow Object Detection API is all set up and ready to go, we need to provide the images it will use to train a new detection classifier.

## LabelImg Installation[](https://tensorflow-object-detection-api-tutorial.readthedocs.io/en/latest/install.html#labelimg-installation "Permalink to this headline")

There exist several ways to install  `labelImg`. 

### Get from PyPI[](https://tensorflow-object-detection-api-tutorial.readthedocs.io/en/latest/install.html#get-from-pypi-recommended "Permalink to this headline")

1.  Open a new  Terminal  window and activate the  tensorflow_gpu  environment (if you have not done so already)
2.  Run the following command to install  `labelImg`:

pip install labelImg

3.  `labelImg`  can then be run as follows:

labelImg

## Annotating images[](https://tensorflow-object-detection-api-tutorial.readthedocs.io/en/latest/training.html#annotating-images "Permalink to this headline")

To annotate images we will be using the  [labelImg](https://github.com/tzutalin/labelImg)  package. If you haven’t installed the package yet, then have a look at  [LabelImg Installation](https://tensorflow-object-detection-api-tutorial.readthedocs.io/en/latest/install.html#labelimg-install).
-   Once you have collected all the images to be used to test your model (ideally more than 100 per class), place them inside the folder  `training_demo\images`.    
-   Open a new  Anaconda/Command Prompt  window and  `cd`  into  `Tensorflow\addons\labelImg`.    
-   If (as suggested in  [LabelImg Installation](https://tensorflow-object-detection-api-tutorial.readthedocs.io/en/latest/install.html#labelimg-install)) you created a separate Conda environment for  `labelImg`  then go ahead and activate it by running:    
    > activate labelImg    
-   Next go ahead and start  `labelImg`, pointing it to your  `training_demo\images`  folder.    
    > python labelImg.py ..\..\workspace\training_demo\images
-   A File Explorer Dialog windows should open, which points to the  `training_demo\images`  folder.
-   Press the “Select Folder” button, to start annotating your images.
    
Once open, you should see a window similar to the one below:

ResimResimResimResimResimResimResimResimResimResim


## Creating Label Map[](https://tensorflow-object-detection-api-tutorial.readthedocs.io/en/latest/training.html#creating-label-map "Permalink to this headline")

TensorFlow requires a label map, which namely maps each of the used labels to an integer values. This label map is used both by the training and detection processes.

Below I show an example label map (e.g  `label_map.pbtxt`), assuming that our dataset containes 1 label,  `cigarette` :

    item {
        id: 1
        name: 'cigarette'
    }
    

Label map files have the extention  `.pbtxt`  and should be placed inside the  `training_demo\annotations`  folder.


## Creating TensorFlow Records[](https://tensorflow-object-detection-api-tutorial.readthedocs.io/en/latest/training.html#creating-tensorflow-records "Permalink to this headline")

Now that we have generated our annotations and split our dataset into the desired training and testing subsets, it is time to convert our annotations into the so called  `TFRecord`  format.

There are two steps in doing so:

-   Converting the individual  `*.xml`  files to a unified  `*.csv`  file for each dataset.
-   Converting the  `*.csv`  files of each dataset to  `*.record`  files (TFRecord format).

Before we proceed to describe the above steps, let’s create a directory where we can store some scripts. Under the  `TensorFlow`  folder, create a new folder  `TensorFlow\scripts`, which we can use to store some useful scripts. To make things even tidier, let’s create a new folder  `TensorFlow\scripts\preprocessing`, where we shall store scripts that we can use to preprocess our training inputs. Below is out  `TensorFlow`  directory tree structure, up to now:

    TensorFlow
    ├─ addons
    │   └─ labelImg
    ├─ models
    │   ├─ official
    │   ├─ research
    │   ├─ samples
    │   └─ tutorials
    ├─ scripts
    │   └─ preprocessing
    └─ workspace
        └─ training_demo

### Converting  `*.xml`  to  `*.csv`[](https://tensorflow-object-detection-api-tutorial.readthedocs.io/en/latest/training.html#converting-xml-to-csv "Permalink to this headline")

To do this we can write a simple script that iterates through all  `*.xml`  files in the  `training_demo\images\train`  and  `training_demo\images\test`  folders, and generates a  `*.csv`  for each of the two.

kodEKLEkodEKLEkodEKLEkodEKLEkodEKLEkodEKLEkodEKLEkodEKLE

-   Create a new file with name  `xml_to_csv.py`  under  `TensorFlow\scripts\preprocessing`, open it, paste the above code inside it and save.
    
-   Install the  `pandas`  package:
    
    > conda install pandas # Anaconda
    >                      # or
    > pip install pandas   # pip
    
-   Finally,  `cd`  into  `TensorFlow\scripts\preprocessing`  and run:
    
> // Create train data:
> python xml_to_csv.py -i [PATH_TO_IMAGES_FOLDER]/train -o [PATH_TO_ANNOTATIONS_FOLDER]/train_labels.csv
> 
> // Create test data:
> python xml_to_csv.py -i [PATH_TO_IMAGES_FOLDER]/test -o [PATH_TO_ANNOTATIONS_FOLDER]/test_labels.csv
> 
> // For example
> // python xml_to_csv.py -i C:\Users\haktan\Documents\TensorFlow\workspace\training_demo\images\train -o C:\Users\haktan\Documents\TensorFlow\workspace\training_demo\annotations\train_labels.csv
> // python xml_to_csv.py -i C:\Users\haktan\Documents\TensorFlow\workspace\training_demo\images\test -o C:\Users\haktan\Documents\TensorFlow\workspace\training_demo\annotations\test_labels.csv


Once the above is done, there should be 2 new files under the  `training_demo\annotations`  folder, named  `test_labels.csv`  and  `train_labels.csv`, respectively.

### Converting from  `*.csv`  to  `*.record`[](https://tensorflow-object-detection-api-tutorial.readthedocs.io/en/latest/training.html#converting-from-csv-to-record "Permalink to this headline")
---
-   Create a new file with name  `generate_tfrecord.py`  under  `TensorFlow\scripts\preprocessing`, open it, paste the above code inside it and save.
    
-   Once this is done,  `cd`  into  `TensorFlow\scripts\preprocessing`  and run:
    
    > // Create train data:
    > python generate_tfrecord.py --label=<LABEL> --csv_input=<PATH_TO_ANNOTATIONS_FOLDER>/train_labels.csv
    > --img_path=<PATH_TO_IMAGES_FOLDER>/train  --output_path=<PATH_TO_ANNOTATIONS_FOLDER>/train.record
    > 
    > // Create test data:
    > python generate_tfrecord.py --label=<LABEL> --csv_input=<PATH_TO_ANNOTATIONS_FOLDER>/test_labels.csv
    > --img_path=<PATH_TO_IMAGES_FOLDER>/test
    > --output_path=<PATH_TO_ANNOTATIONS_FOLDER>/test.record
    > 
    > // For example
    > // python generate_tfrecord.py --label=ship --csv_input=C:\Users\haktan\Documents\TensorFlow\workspace\training_demo\annotations\train_labels.csv --output_path=C:\Users\haktan\Documents\TensorFlow\workspace\training_demo\annotations\train.record --img_path=C:\Users\haktan\Documents\TensorFlow\workspace\training_demo\images\train
    > // python generate_tfrecord.py --label=ship --csv_input=C:\Users\haktan\Documents\TensorFlow\workspace\training_demo\annotations\test_labels.csv --output_path=C:\Users\haktan\Documents\TensorFlow\workspace\training_demo\annotations\test.record --img_path=C:\Users\haktan\Documents\TensorFlow\workspace\training_demo\images\test
    

Once the above is done, there should be 2 new files under the  `training_demo\annotations`  folder, named  `test.record`  and  `train.record`, respectively.

## Configuring a Training Pipeline[](https://tensorflow-object-detection-api-tutorial.readthedocs.io/en/latest/training.html#configuring-a-training-pipeline "Permalink to this headline")

For the purposes of this tutorial we will not be creating a training job from the scratch, but rather we will go through how to reuse one of the pre-trained models provided by TensorFlow. If you would like to train an entirely new model, you can have a look at  [TensorFlow’s tutorial](https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/configuring_jobs.md).

The model we shall be using in our examples is the  `faster_rcnn_inception_v2_coco`  model, since it provides a relatively good trade-off between performance and speed, however there are a number of other models you can use, all of which are listed in  [TensorFlow’s detection model zoo](https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/detection_model_zoo.md). More information about the detection performance, as well as reference times of execution, for each of the available pre-trained models can be found  [here](https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/detection_model_zoo.md#coco-trained-models-coco-models).

First of all, we need to get ourselves the sample pipeline configuration file for the specific model we wish to re-train. You can find the specific file for the model of your choice  [here](https://github.com/tensorflow/models/tree/master/research/object_detection/samples/configs). In our case, since we shall be using the  `faster_rcnn_inception_v2_coco`  model, we shall be downloading the corresponding  [faster_rcnn_inception_v2_coco.config](https://github.com/tensorflow/models/blob/master/research/object_detection/samples/configs/faster_rcnn_inception_v2_coco.config)  file.

Apart from the configuration file, we also need to download the latest pre-trained NN for the model we wish to use. This can be done by simply clicking on the name of the desired model in the tables found in  [TensorFlow’s detection model zoo](https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/detection_model_zoo.md#coco-trained-models-coco-models). Clicking on the name of your model should initiate a download for a  `*.tar.gz`  file.

Once the  `*.tar.gz`  file has been downloaded, open it using a decompression program of your choice (e.g. 7zip, WinZIP, etc.). Next, open the folder that you see when the compressed folder is opened (typically it will have the same name as the compressed folded, without the  `*.tar.gz`  extension), and extract it’s contents inside the folder  `training_demo\pre-trained-model`.

## Training the Model[](https://tensorflow-object-detection-api-tutorial.readthedocs.io/en/latest/training.html#training-the-model "Permalink to this headline")

Before we begin training our model, let’s go and copy the  `TensorFlow/models/research/object_detection/legacy/train.py`  script and paste it straight into our  `training_demo`  folder. We will need this script in order to train our model.

Now, to initiate a new training job,  `cd`  inside the  `training_demo`  folder and type the following:

    python train.py --logtostderr --train_dir=training/ --pipeline_config_path=training/faster_rcnn_inception_v2_coco.config

Once the training process has been initiated, you should see a series of print outs similar to the one below (plus/minus some warnings):

    INFO:tensorflow:depth of additional conv before box predictor: 0
    INFO:tensorflow:depth of additional conv before box predictor: 0
    INFO:tensorflow:depth of additional conv before box predictor: 0
    INFO:tensorflow:depth of additional conv before box predictor: 0
    INFO:tensorflow:depth of additional conv before box predictor: 0
    INFO:tensorflow:depth of additional conv before box predictor: 0
    INFO:tensorflow:Restoring parameters from faster_rcnn_inception_v2_coco/model.ckpt
    INFO:tensorflow:Running local_init_op.
    INFO:tensorflow:Done running local_init_op.
    INFO:tensorflow:Starting Session.
    INFO:tensorflow:Saving checkpoint to path training\model.ckpt
    INFO:tensorflow:Starting Queues.
    INFO:tensorflow:global_step/sec: 0
    INFO:tensorflow:global step 1: loss = 1.3456 (12.339 sec/step)
    INFO:tensorflow:global step 2: loss = 1.0432 (0.937 sec/step)
    INFO:tensorflow:global step 3: loss = 0.9786 (0.904 sec/step)
    INFO:tensorflow:global step 4: loss = 0.8754 (0.894 sec/step)
    INFO:tensorflow:global step 5: loss = 0.7497 (0.922 sec/step)
    INFO:tensorflow:global step 6: loss = 0.7563 (0.936 sec/step)
    INFO:tensorflow:global step 7: loss = 0.7245 (0.910 sec/step)
    INFO:tensorflow:global step 8: loss = 0.7993 (0.916 sec/step)
    INFO:tensorflow:global step 9: loss = 0.1277 (0.890 sec/step)
    INFO:tensorflow:global step 10: loss = 0.3972 (0.919 sec/step)
    INFO:tensorflow:global step 11: loss = 0.9487 (0.897 sec/step)
    INFO:tensorflow:global step 12: loss = 0.7954 (0.884 sec/step)
    INFO:tensorflow:global step 13: loss = 0.4329 (0.906 sec/step)
    INFO:tensorflow:global step 14: loss = 0.8270 (0.897 sec/step)
    INFO:tensorflow:global step 15: loss = 0.4877 (0.894 sec/step)
    ...

If you ARE observing a similar output to the above, then CONGRATULATIONS, you have successfully started your first training job. Following what people have said online, it seems that it is advisable to allow you model to reach a  `TotalLoss`  of at least 0.0500 (ideally 0.0100  and lower) if you want to achieve “fair” detection results. Obviously, lower  `TotalLoss`  is better, however very low  `TotalLoss`  should be avoided, as the model may end up overfitting the dataset, meaning that it will perform poorly when applied to images outside the dataset.


## Exporting a Trained Inference Graph[](https://tensorflow-object-detection-api-tutorial.readthedocs.io/en/latest/training.html#exporting-a-trained-inference-graph "Permalink to this headline")

Once your training job is complete, you need to extract the newly trained inference graph, which will be later used to perform the object detection. This can be done as follows:
-   Open a new  Anaconda/Command Prompt    
-   Activate your TensorFlow conda environment (if you have one), e.g.:    
    > activate tfgpu    
-   Copy the  `TensorFlow/models/research/object_detection/export_inference_graph.py`  script and paste it straight into your  `training_demo`  folder.    
-   Check inside your  `training_demo/training`  folder for the  `model.ckpt-*`  checkpoint file with the highest number following the name of the dash e.g.  `model.ckpt-200000`). This number represents the training step index at which the file was created.    
-   Alternatively, simply sort all the files inside  `training_demo/training`  by descending time and pick the  `model.ckpt-*`  file that comes first in the list.    
-   Make a note of the file’s name, as it will be passed as an argument when we call the  `export_inference_graph.py`  script.    
-   Now,  `cd`  inside your  `training_demo`  folder, and run the following command:
    
>python export_inference_graph.py --input_type image_tensor --pipeline_config_path training/faster_rcnn_inception_v2_coco.config`

