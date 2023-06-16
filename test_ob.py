"""
Main file for training Yolo model on Pascal VOC and COCO dataset
"""
import os
ROOT = os.getcwd()
import sys
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))
import config
import torch
import torch.optim as optim
import numpy as np
from model import YOLOv3
from tqdm import tqdm
import torch.nn.functional as F
from utils import (
    mean_average_precision,
    cells_to_bboxes,
    get_evaluation_bboxes,
    save_checkpoint,
    load_checkpoint,
    check_class_accuracy,
    get_loaders,
    plot_couple_examples,
    getseg_loaders,
    changeto3
)

from loss import YoloLoss
import warnings
warnings.filterwarnings("ignore")

torch.backends.cudnn.benchmark = True

def main():
    model = YOLOv3(num_classes=config.NUM_CLASSES).to(config.DEVICE)
    optimizer = optim.Adam(
        model.parameters(), lr=config.LEARNING_RATE, weight_decay=config.WEIGHT_DECAY
    )
    
    scaler = torch.cuda.amp.GradScaler()

    
    train_loader, test_loader = get_loaders(
            train_csv_path=config.DATASET + "/train.txt", test_csv_path=config.DATASET + "/test.txt"
        )
    if config.LOAD_MODEL:
        load_checkpoint(
            config.CHECKPOINT_FILE, model, optimizer, config.LEARNING_RATE
        )


        # val set
        #if epoch > 0 and epoch % 3 == 0:
        check_class_accuracy(model, test_loader, threshold=config.CONF_THRESHOLD)
        pred_boxes, true_boxes = get_evaluation_bboxes(
            test_loader,
            model,
            iou_threshold=config.NMS_IOU_THRESH,
            anchors=config.ANCHORS,
            threshold=config.CONF_THRESHOLD,
        )
        mapval = mean_average_precision(
            pred_boxes,
            true_boxes,
            iou_threshold=config.MAP_IOU_THRESH,
            box_format="midpoint",
            num_classes=config.NUM_CLASSES,
        )
        print(f"VAL_MAP: {mapval.item()}")

def main_1():
    model = YOLOv3(num_classes=config.NUM_CLASSES).to(config.DEVICE)
    optimizer = optim.Adam(
        model.parameters(), lr=config.LEARNING_RATE, weight_decay=config.WEIGHT_DECAY
    )
    

    
    train_loader, test_loader = getseg_loaders(
            train_csv_path='ADE' + "/train.txt", test_csv_path='ADE' + "/test.txt"
        )

    load_checkpoint(
        'checkpoint_seg_40.pt', model, optimizer, config.LEARNING_RATE
    )


    model.eval()
    loop = tqdm(test_loader, leave=True)
    acc=[]
    for batch_idx, (x, y) in enumerate(loop):
        x = x.to(config.DEVICE)
        y = y.to(config.DEVICE)


        out_ob ,out_sg = model(x)
        pre=F.interpolate(out_sg.cpu(), (config.IMAGE_SIZE,config.IMAGE_SIZE), mode='bilinear', align_corners=False)  

        #resu=changeto3(pre)

        #accuracy
        pre = torch.argmax(pre,dim=1)


        pre=pre.view(1,-1).cpu().numpy()
        y=y.view(1,-1).cpu().numpy()
        acc.append(np.sum(pre==y)/pre.shape[1])

        torch.cuda.empty_cache()
    print(sum(acc)/len(acc))
if __name__ == "__main__":
    main_1()