# You may need to restart your runtime prior to this, to let your installation take effect
# Some basic setup:
# Setup detectron2 and logger
import detectron2
from detectron2.utils.logger import setup_logger
setup_logger()

# import some common libraries
import numpy as np
from cv2 import cv2
import random
import os
import glob

# import some common detectron2 utilities
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog
from detectron2.data.catalog import DatasetCatalog
from detectron2.data.datasets import register_coco_instances
from detectron2.utils.visualizer import Visualizer
from detectron2.utils.visualizer import ColorMode


register_coco_instances("my_dataset_train", {}, "content/train/_annotations.coco.json", "content/train")
register_coco_instances("my_dataset_val", {}, "content/valid/_annotations.coco.json", "content/valid")
register_coco_instances("my_dataset_test", {}, "content/test/_annotations.coco.json", "content/test")

#Get metadata from training set, to provide metadata during inference. This metadata allows the predictor to assign predictions to human-readable classes for output.
my_dataset_train_metadata = MetadataCatalog.get("my_dataset_train")
dataset_dicts = DatasetCatalog.get("my_dataset_train")

#setup config for inference.
cfg = get_cfg()
cfg.merge_from_file(model_zoo.get_config_file("COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml"))
cfg.DATASETS.TRAIN = ("my_dataset_train",)
cfg.DATASETS.TEST = ("my_dataset_val",)
cfg.DATALOADER.NUM_WORKERS = 2
cfg.SOLVER.IMS_PER_BATCH = 2
cfg.SOLVER.BASE_LR = 0.001
cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 32
cfg.MODEL.ROI_HEADS.NUM_CLASSES = 31 #your number of classes + 1

# Inference with Detectron2 Saved Weights on a previously unknown image

# specify trained weights file
cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR, "model_final.pth")
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.65   # set the testing threshold for this model
predictor = DefaultPredictor(cfg)


print("Starting to output inference results. Predictions below confidence score of 0.65 score ignored.")
#Set the correct format. If using JPG files set *.JPG instead of *.PNG !


for imageName in glob.glob('inferenceContent/input/*.png'):

    print("{}".format(imageName))
   # print("{}".format(my_dataset_train_metadata))

    im = cv2.imread(imageName)
    outputs = predictor(im)
    v = Visualizer(im[:, :, ::-1],
                    #Get class names from dataset train metadata to put on visualization.
                    metadata=my_dataset_train_metadata, 
                    scale=1
                    )
    out = v.draw_instance_predictions(outputs["instances"].to("cpu"))
    # Show images with predictions
    cv2.imshow('Inference Preview',out.get_image()[:, :, ::-1])
    # Save images with predictions to savePath folder and imageName with path to image removed from name.
    savePath = './inferenceContent/output'
    cv2.imwrite(os.path.join(savePath , '{}'.format(os.path.basename(imageName))), out.get_image()[:, :, ::-1])
    cv2.waitKey(1000)

#Todo: Run test and inference on the same image (1), to get the text output of findings and the image visualization