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
from detectron2.utils.visualizer import _create_text_labels
from detectron2.data import MetadataCatalog
from detectron2.data.catalog import DatasetCatalog
from detectron2.data.datasets import register_coco_instances
from detectron2.utils.visualizer import ColorMode

#Register dataset annotations in coco format. This is important for metadata used by Detectron, for example in inference.
register_coco_instances("my_dataset_train", {}, "content/train/_annotations.coco.json", "content/train")
register_coco_instances("my_dataset_val", {}, "content/valid/_annotations.coco.json", "content/valid")
register_coco_instances("my_dataset_test", {}, "content/test/_annotations.coco.json", "content/test")

#Get metadata from training set, to provide metadata during inference. 
#This metadata allows the predictor to assign predictions to human-readable classes for output.
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
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.65   # set the class confidence threshold for this model!
predictor = DefaultPredictor(cfg)


print("Starting to output inference results. Predictions below confidence score of 0.65 score ignored.")

#Initialise file for inference output data:
import json
#Get all available classes for this dataset from training metadata. Needed to make counts for each class in inference results.
AllClasses = my_dataset_train_metadata.thing_classes


#Set the correct format. If using JPG files set *.JPG instead of *.PNG !
#Currently, when inferencing, all images in folder are inferenced every time.
#This is not optimal.
for imageName in glob.glob('inferenceContent/input/*.png'):
    imageBaseName = format(os.path.basename(imageName))
    print(imageBaseName)
   # print("{}".format(my_dataset_train_metadata))

    im = cv2.imread(imageName)
    outputs = predictor(im)
    v = Visualizer(im[:, :, ::-1],
                    #Get class names from dataset train metadata to put on visualization.
                    metadata=my_dataset_train_metadata, 
                    scale=1
                    )
    out = v.draw_instance_predictions(outputs["instances"].to("cpu"))
    #As described in draw_instance_predictions function of Visualizer, we can get data about each image. 
    #We might need these, but currently only interested in labels as they contain predicted classes for each image and prediction score.
    imagePredClasses = outputs["instances"].pred_classes
    imagePredBoxes = outputs["instances"].pred_boxes
    imagePredScores = outputs["instances"].scores

    #Initialise a dictionary and store inference data (labels with class name and prediction score) for each:
    imageDataArray = {imageBaseName: []}
    imageDataArray[imageBaseName] = _create_text_labels(outputs["instances"].pred_classes, outputs["instances"].scores, my_dataset_train_metadata.get("thing_classes", None))
    print(imageDataArray)

    #data in it. As it loops through all images it will fill the JSON with needed data.

    # Show images with predictions in system window. If not using host, then:
    #cv2.imshow('Inference Preview',out.get_image()[:, :, ::-1])
    # If running inference locally, to view result with timer in system window, uncomment:
    #cv2.waitKey(10000)
    # Save images with predictions to savePath folder and imageName with path to image removed from name.
    savePath = './inferenceContent/output'
    cv2.imwrite(os.path.join(savePath , imageBaseName), out.get_image()[:, :, ::-1])
    
    # Create a JSON object and append results for each image.


# Save JSON object.
#with open('inferenceContent/output/infOutputData.json', 'w', encoding='utf-8') as f:  
#json.dump(data, f, ensure_ascii=False, indent=4)
#Todo: Run test and inference on the same image (1), to get the text output of findings and the image visualization