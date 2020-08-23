# This file is meant to train the model on the sets provided in the /content folder
# This file will also test on the resulting weights file and run inference for /content/valid images.
# After the training finishes and it starts inferencing you can just escape the process ^C in shell
# After the training finishes you can launch the webserver and it will be fully functional. See Readme for more details.

# Basic setup
# Setup detectron2 logger
import detectron2
from detectron2.utils.logger import setup_logger
setup_logger()

# import common libraries
import numpy as np
from cv2 import cv2
import random
import os

# import common detectron2 utilities
from detectron2 import model_zoo
from detectron2.engine import DefaultTrainer
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog
from detectron2.data.catalog import DatasetCatalog
from detectron2.data.datasets import register_coco_instances
from detectron2.evaluation import COCOEvaluator
from detectron2.data import build_detection_test_loader
from detectron2.evaluation import inference_on_dataset

register_coco_instances("my_dataset_train", {}, "content/train/_annotations.coco.json", "content/train")
register_coco_instances("my_dataset_val", {}, "content/valid/_annotations.coco.json", "content/valid")

#visualize training data
my_dataset_train_metadata = MetadataCatalog.get("my_dataset_train")
dataset_dicts = DatasetCatalog.get("my_dataset_train")

for d in random.sample(dataset_dicts, 3):
    img = cv2.imread(d["file_name"])
    visualizer = Visualizer(img[:, :, ::-1], metadata=my_dataset_train_metadata, scale=0.5)
    vis = visualizer.draw_dataset_dict(d)
    cv2.imshow('preview',vis.get_image()[:, :, ::-1])
    cv2.waitKey(100)

#Importing our own Trainer Module to use the COCO validation evaluation 
#during training. Otherwise no validation eval occurs.

class CocoTrainer(DefaultTrainer):

  @classmethod
  def build_evaluator(cls, cfg, dataset_name, output_folder=None):

    if output_folder is None:
        os.makedirs("coco_eval", exist_ok=True)
        output_folder = "coco_eval"

    return COCOEvaluator(dataset_name, cfg, False, output_folder)

# An overview of the various available models is available: https://github.com/facebookresearch/detectron2/blob/master/MODEL_ZOO.md
# Selecting the one with least VRAM requirement
cfg = get_cfg()
cfg.merge_from_file(model_zoo.get_config_file("COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml"))
cfg.DATASETS.TRAIN = ("my_dataset_train",)
cfg.DATASETS.TEST = ("my_dataset_val",)

#Number of workers and 'images per batch' are effectively multipliers for training speed and VRAM requirement. 
# With this setup the model requires approximately 5GB VRAM during training
cfg.DATALOADER.NUM_WORKERS = 2
cfg.SOLVER.IMS_PER_BATCH = 2
cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml")  # Let training initialize from model zoo
cfg.SOLVER.BASE_LR = 0.001
cfg.SOLVER.WARMUP_ITERS = 1000
cfg.SOLVER.MAX_ITER = 1500 #Suitable for proof-of-concept and quick training. Can raise if mAP is still rising at final epochs. If not rising, can overfit.
cfg.SOLVER.STEPS = (1000, 1500)
cfg.SOLVER.GAMMA = 0.05


cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 62
cfg.MODEL.ROI_HEADS.NUM_CLASSES = 12 #your number of classes + 1

cfg.TEST.EVAL_PERIOD = 100 #Adjust this to set at what interval the model performance should be
                          #evaluated. Currently adjusted for eval debug.


os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
trainer = CocoTrainer(cfg)


# Set (resume=False) if running for the first time

trainer.resume_or_load(resume=True)

trainer.train()

#Should now have "model_final.pth" weights file.



# Run evaluation on the test set. Specify the sets appropriately if using multiple.
"""
cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR, "model_final.pth")
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.85
predictor = DefaultPredictor(cfg)
evaluator = COCOEvaluator("my_dataset_val", cfg, False, output_dir="./output/")
val_loader = build_detection_test_loader(cfg, "my_dataset_val")
inference_on_dataset(trainer.model, val_loader, evaluator)

print("Evaluation on test set finished")
"""
# Inference with Detectron2 Saved Weights on a previously unknown image

cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR, "model_final.pth")
cfg.DATASETS.TEST = ("my_dataset_val", )
# set the testing threshold for this model
threshold = 0.65
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = threshold   
predictor = DefaultPredictor(cfg)
test_metadata = MetadataCatalog.get("my_dataset_val")

from detectron2.utils.visualizer import ColorMode
import glob

print("Starting to output inference results. Predictions below confidence score of {} score ignored.".format(threshold))
#Set the correct format. If using JPG files set *.JPG instead of *.PNG ! Also it's case sensitive!
for imageName in glob.glob('content/valid/*.PNG'):

    print("{}".format(imageName))

    im = cv2.imread(imageName)
    outputs = predictor(im)
    v = Visualizer(im[:, :, ::-1],
                    #Get class names from dataset train metadata to put on visualization.
                    metadata=my_dataset_train_metadata, 
                    scale=1
                    )
    out = v.draw_instance_predictions(outputs["instances"].to("cpu"))
    cv2.imshow('Inference Preview',out.get_image()[:, :, ::-1])
    cv2.waitKey(10000)

#For single image inference see inferenceSingle.py file. 