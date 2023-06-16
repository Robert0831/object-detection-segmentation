"""
Creates a Pytorch dataset to load the Pascal VOC & MS COCO datasets
"""

import config
import numpy as np
import os
import pandas as pd
import torch
import cv2
from PIL import Image, ImageFile
from torch.utils.data import Dataset, DataLoader
from utils import (
    cells_to_bboxes,
    iou_width_height as iou,
    non_max_suppression as nms,
    plot_image
)
import xml.etree.ElementTree as ET

ImageFile.LOAD_TRUNCATED_IMAGES = True

def read_pascal_xml(xml_file,w,h):
    tree = ET.parse(xml_file)
    root = tree.getroot()
    
    annotations = []
    sx=config.IMAGE_SIZE/w
    sy=config.IMAGE_SIZE/h

    for obj in root.findall('object'):
        name = obj.find('name').text
        name=config.PASCAL_CLASSES.index(name)
        bbox = obj.find('bndbox')

        xmin = float(bbox.find('xmin').text)*sx/config.IMAGE_SIZE
        ymin = float(bbox.find('ymin').text)*sy/config.IMAGE_SIZE
        xmax = float(bbox.find('xmax').text)*sx/config.IMAGE_SIZE
        ymax = float(bbox.find('ymax').text)*sy/config.IMAGE_SIZE


        
        temp=[(xmax+xmin)/2,(ymax+ymin)/2,xmax-xmin,ymax-ymin,name]
        
        annotations.append(temp)
    
    return annotations




class YOLODataset(Dataset):
    def __init__(
        self,
        csv_file,
        img_dir,
        label_dir,
        anchors,
        image_size=416,
        S=[13, 26, 52],
        C=20,
        transform=None,
    ):
        self.annotations =pd.read_csv(csv_file,sep=" ",header=None,names=["name"],dtype={"name": str})
        self.img_dir = img_dir
        self.label_dir = label_dir
        self.image_size = image_size
        self.transform = transform
        self.S = S
        self.anchors = torch.tensor(anchors[0] + anchors[1] + anchors[2])  # for all 3 scales
        self.num_anchors = self.anchors.shape[0]
        self.num_anchors_per_scale = self.num_anchors // 3
        self.C = C
        self.ignore_iou_thresh = 0.5

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, index):
        img_path = os.path.join(self.img_dir, str(self.annotations["name"][index])+'.jpg')
        image = np.array(Image.open(img_path).convert("RGB"))
        h,w =image.shape[:2]
        label_path = os.path.join(self.label_dir, str(self.annotations["name"][index])+'.xml')
        bboxes=read_pascal_xml(label_path,w,h)
        #print(bboxes)

        if self.transform:
            augmentations = self.transform(image=image, bboxes=bboxes)
            image = augmentations["image"]
            bboxes = augmentations["bboxes"]

        # Below assumes 3 scale predictions (as paper) and same num of anchors per scale
        targets = [torch.zeros((self.num_anchors // 3, S, S, 6)) for S in self.S]
        for box in bboxes:
            iou_anchors = iou(torch.tensor(box[2:4]), self.anchors)
            anchor_indices = iou_anchors.argsort(descending=True, dim=0)
            x, y, width, height, class_label = box
            has_anchor = [False] * 3  # each scale should have one anchor
            for anchor_idx in anchor_indices:
                scale_idx = anchor_idx // self.num_anchors_per_scale
                anchor_on_scale = anchor_idx % self.num_anchors_per_scale
                S = self.S[scale_idx]
                i, j = int(S * y), int(S * x)  # which cell
                anchor_taken = targets[scale_idx][anchor_on_scale, i, j, 0]
                if not anchor_taken and not has_anchor[scale_idx]:
                    targets[scale_idx][anchor_on_scale, i, j, 0] = 1
                    x_cell, y_cell = S * x - j, S * y - i  # both between [0,1]
                    width_cell, height_cell = (
                        width * S,
                        height * S,
                    )  # can be greater than 1 since it's relative to cell
                    box_coordinates = torch.tensor(
                        [x_cell, y_cell, width_cell, height_cell]
                    )
                    targets[scale_idx][anchor_on_scale, i, j, 1:5] = box_coordinates
                    targets[scale_idx][anchor_on_scale, i, j, 5] = int(class_label)
                    has_anchor[scale_idx] = True

                elif not anchor_taken and iou_anchors[anchor_idx] > self.ignore_iou_thresh:
                    targets[scale_idx][anchor_on_scale, i, j, 0] = -1  # ignore prediction

        return image, tuple(targets)



class ADEdataset(Dataset):
    def __init__(
        self,
        csv_file,
        img_dir,
        label_dir,
        image_size=416,
        C=150, #沒用到
        transform=None,
    ):
        self.annotations =pd.read_csv(csv_file,sep=" ",header=None,names=["name"],dtype={"name": str})
        self.img_dir = img_dir
        self.label_dir = label_dir
        self.image_size = image_size
        self.transform = transform
        self.C = C

    def __len__(self):
        return len(self.annotations)
    @staticmethod
    def _convert_to_segmentation_mask(mask):
        height, width = mask.shape[:2] # H,W,C
        segmentation_mask = np.zeros((height, width, len(config.ADE)), dtype=np.float32)
        for label_index, label in enumerate(config.ADE):
            segmentation_mask[:, :, label_index] = np.all(mask == label, axis=-1).astype(float)   #-1 is channel
        #print(np.sum(segmentation_mask))
        return segmentation_mask



    def __getitem__(self, index):
        img_path = os.path.join(self.img_dir, str('ADE_val_'+self.annotations["name"][index])+'.jpg')
        image = np.array(Image.open(img_path).convert("RGB"))
        mask_path = os.path.join(self.label_dir, str('ADE_val_'+self.annotations["name"][index])+'_seg.png')


        mask = cv2.imread(mask_path)
        mask=cv2.resize(mask,(config.IMAGE_SIZE ,config.IMAGE_SIZE))
        mask=cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
        #mask = cv2.cvtColor(mask, cv2.COLOR_BGR2RGB)
        mask=mask.astype(np.uint8)
        #img = Image.fromarray(mask.astype(np.uint8)).show()

        
        #mask = self._convert_to_segmentation_mask(mask)        
        #mask=np.transpose(mask,(2,0,1))
        #mask = np.argmax(mask, axis = 0) 
        if self.transform:
            augmentations = self.transform(image=image)
            image = augmentations["image"]

        return image, mask

def test():
    anchors = config.ANCHORS

    transform = config.my_transforms

    dataset = YOLODataset(
        "ttt.txt",
        "./VOC/imgs",
        "./VOC/labs",
        S=[13, 26, 52],
        anchors=anchors,
        transform=transform,
    )
    S = [13, 26, 52]
    scaled_anchors = torch.tensor(anchors) / (
        1 / torch.tensor(S).unsqueeze(1).unsqueeze(1).repeat(1, 3, 2)
    )
    loader = DataLoader(dataset=dataset, batch_size=1, shuffle=True)
    for x, y in loader:
        boxes = []

        for i in range(y[0].shape[1]):
            anchor = scaled_anchors[i]
            # print(anchor.shape)
            # print(y[i].shape)
            boxes += cells_to_bboxes(
                y[i], is_preds=False, S=y[i].shape[2], anchors=anchor
            )[0]
        boxes = nms(boxes, iou_threshold=1, threshold=0.7, box_format="midpoint")
        #print(boxes)
        plot_image(x[0].permute(1, 2, 0).to("cpu")*255, boxes)





if __name__ == "__main__":
    test()

    
