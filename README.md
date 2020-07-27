# Readme

## Install Cuda and cuDNN

Below is the setup process. Assuming you have Ubuntu 20.04 installed with an Nvidia GPU available.

To make pytorch work with GPU, it is necessary to install CUDA and cuDNN. If working from Ubuntu 20.04 the [following guide](https://askubuntu.com/questions/1230645/when-is-cuda-gonna-be-released-for-ubuntu-20-04) is suitable.
  

**After installing both CUDA and cuDNN check CUDA and cuDNN are version 10.1 and 7.6.5 respectively:**

**CUDA:**

    $ nvcc -V

**cuDNN:**

get where cuda installed:

    $ whereis cuda
then navigate to that folder's /include folder and: 

    $ cat cudnn.h | grep CUDNN_MAJOR -A 2

  
  

## Activate conda venv with tensorflow-gpu==2.2.0

ensure that conda is initialised with Python 3.8.3 (set it within conda environment creation).

See [cheatsheet](https://docs.conda.io/projects/conda/en/4.6.0/_downloads/52a95608c49671267e40c689e0bc00ca/conda-cheatsheet.pdf) !
  

To install correct pytorch and torchvision do this with conda activated environment:

    $ conda install pytorch torchvision cudatoolkit=10.1 -c pytorch

Check that GPU is available and working. In virtual environment `$ python`:

    import torch
    torch.cuda.current_device()
    torch.cuda.get_device_name(0)

Then install some Detectron2 dependencies:

    $ pip install cython pyyaml==5.1
    $ pip install pycocotools>=2.0.1
Then install Detectron2:

    $ pip install detectron2==0.1.3 -f https://dl.fbaipublicfiles.com/detectron2/wheels/cu101/torch1.5/index.html

See also Detectron2 [installation guide](https://github.com/facebookresearch/detectron2/blob/master/INSTALL.md). 

  

**Note:**
Ensure that the eval set contains at least 1 instance of all classes that can be encountered in the training dataset:

    cfg.DATASETS.TEST = ("my_dataset_val",


Otherwise the validation step will fail at EVAL_PERIOD.

Also make sure that your data is marked according to COCO standard. A good tool to use is the [VGG annotator](http://www.robots.ox.ac.uk/~vgg/software/via/via_demo.html) (export annotations to COCO json).
  
**Run from conda virtual env:**

    $ python main.py

  
**Output:**
to graphically visualise and get data about model performance, after training completes, run:

    $ tensorboard --logdir output

**To Launch Webserver (Express)**

    $ npm install
then:

    $ npm run dev

APP SERVES ON PORT 3000






