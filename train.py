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
# def seg_loss(x,y):
#     lo=torch.nn.CrossEntropyLoss()
#     x=F.interpolate(x, (config.IMAGE_SIZE,config.IMAGE_SIZE), mode='bilinear', align_corners=False)

#     loss=lo(x,y)
#     return loss


def train_fn(train_loader, model, optimizer, loss_fn, scaler, scaled_anchors):
    loop = tqdm(train_loader, leave=True)
    losses = []
    for batch_idx, (x, y) in enumerate(loop):
        x = x.to(config.DEVICE)
        y0, y1, y2 = (
            y[0].to(config.DEVICE),
            y[1].to(config.DEVICE),
            y[2].to(config.DEVICE),
        )

        with torch.cuda.amp.autocast():
            out_ob ,out_sg = model(x)
            loss = (
                loss_fn(out_ob[0], y0, scaled_anchors[0])
                + loss_fn(out_ob[1], y1, scaled_anchors[1])
                + loss_fn(out_ob[2], y2, scaled_anchors[2])
            )

        losses.append(loss.item())
        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        # update progress bar
        mean_loss = sum(losses) / len(losses)
        loop.set_postfix(loss=mean_loss)

        torch.cuda.empty_cache()

    logger = open('train_acc.txt', 'a')
    logger.write('%f '%(mean_loss))
    logger.close()  
def train_seg(train_loader, model, optimizer, scaler):
    lo=torch.nn.CrossEntropyLoss()
    loop = tqdm(train_loader, leave=True)
    losses = []
    acc=[]
    for batch_idx, (x, y) in enumerate(loop):
        x = x.to(config.DEVICE)
        y = y.to(config.DEVICE)

        with torch.cuda.amp.autocast():
            out_ob ,out_sg = model(x)
            pre=F.interpolate(out_sg, (config.IMAGE_SIZE,config.IMAGE_SIZE), mode='bilinear', align_corners=False)
            loss=lo(pre,y.long())

        # if batch_idx%20==0:
        #     resu=changeto3(pre.detach().cpu())


        losses.append(loss.cpu().item())
        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        # update progress bar
        mean_loss = sum(losses) / len(losses)
        loop.set_postfix(loss=mean_loss)

        #accuracy
        pre = torch.argmax(pre,dim=1)
        pre=pre.view(1,-1).cpu().numpy()
        y=y.view(1,-1).cpu().numpy()
        acc.append(np.sum(pre==y)/pre.shape[1])

        # print(np.sum(pre==y))
        # print(np.sum(pre==y)/pre.shape[1])

        torch.cuda.empty_cache()

    logger = open('train_seg_acc.txt', 'a')
    logger.write('%f '%(mean_loss))
    logger.close()

    logger = open('train_seg_acc.txt', 'a')
    logger.write('%f\n'%(sum(acc)/len(acc)))
    logger.close()

def main():
    model = YOLOv3(num_classes=config.NUM_CLASSES).to(config.DEVICE)
    optimizer = optim.Adam(
        model.parameters(), lr=config.LEARNING_RATE, weight_decay=config.WEIGHT_DECAY
    )
    
    scaler = torch.cuda.amp.GradScaler()

    if config.TRAIN_WHAT=="bac_obj":
        train_loader, test_loader = get_loaders(
            train_csv_path=config.DATASET + "/train.txt", test_csv_path=config.DATASET + "/val.txt"
        )
    else:
        train_loader, test_loader = getseg_loaders(
        train_csv_path=config.DATASET + "/train.txt", test_csv_path=config.DATASET + "/val.txt"
    )

    if config.LOAD_MODEL:
        load_checkpoint(
            config.CHECKPOINT_FILE, model, optimizer, config.LEARNING_RATE
        )

    if config.TRAIN_WHAT=="bac_obj":
        for name, param in model.named_parameters():
            if 'seg' in name:
                param.requires_grad = False
    elif config.TRAIN_WHAT=="seg":
        for name, param in model.named_parameters():
            if 'pred' in name:
                param.requires_grad = False
    if config.FIX_BAC=="tt":
        for name, param in model.named_parameters():
            if 'pred' not in name and not 'seg' in name:
                param.requires_grad = False

    # for name, param in model.named_parameters():
    #     if param.requires_grad:
    #         print(name)

    if config.TRAIN_WHAT=="bac_obj":
        scaled_anchors = (
            torch.tensor(config.ANCHORS)
            * torch.tensor(config.S).unsqueeze(1).unsqueeze(1).repeat(1, 3, 2)
        ).to(config.DEVICE)
    


    for epoch in range(config.NUM_EPOCHS):
        plot_couple_examples(model, train_loader, 0.7, 0.5, scaled_anchors)
        print(f"Currently epoch {epoch}")

        if config.TRAIN_WHAT=="bac_obj":
            logger = open('train_acc.txt', 'a')
            logger.write('%d '%(epoch))
            logger.close() 
        else:
            logger = open('train_seg_acc.txt', 'a')
            logger.write('%d '%(epoch))
            logger.close() 


        if config.TRAIN_WHAT=="bac_obj":
            loss_fn = YoloLoss()
            train_fn(train_loader, model, optimizer, loss_fn, scaler, scaled_anchors)
        else:
            train_seg(train_loader, model, optimizer, scaler)

        if epoch %10 ==0:
            if config.SAVE_MODEL:
                save_checkpoint(model, optimizer, filename=f"checkpoint_seg_{epoch}.pt")

        #save_checkpoint(model, optimizer, filename=f"checkpoint_seg_{epoch}.pth")



        #print("On Train Eval loader:")
        #print("On Train loader:")
        #check_class_accuracy(model, train_loader, threshold=config.CONF_THRESHOLD)


        ### training set acc
        if config.TRAIN_WHAT=="seg":
            model.eval()
            loop = tqdm(test_loader, leave=True)
            acc=[]
            for batch_idx, (x, y) in enumerate(loop):
                x = x.to(config.DEVICE)
                y = y.to(config.DEVICE)


                out_ob ,out_sg = model(x)
                pre=F.interpolate(out_sg, (config.IMAGE_SIZE,config.IMAGE_SIZE), mode='bilinear', align_corners=False)  

                #resu=changeto3(pre)

                #accuracy
                pre = torch.argmax(pre,dim=1)


                pre=pre.view(1,-1).cpu().numpy()
                y=y.view(1,-1).cpu().numpy()
                acc.append(np.sum(pre==y)/pre.shape[1])

                torch.cuda.empty_cache()

            logger = open('val_seg_acc.txt', 'a')
            logger.write('%d %f\n'%(epoch,sum(acc)/len(acc)))
            logger.close()



        if config.TRAIN_WHAT=="bac_obj":
            check_class_accuracy(model, train_loader, threshold=config.CONF_THRESHOLD)
            pred_boxes, true_boxes = get_evaluation_bboxes(
                train_loader,
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
            print(f"TRAIN_MAP: {mapval.item()}")
            model.train()

            logger = open('train_acc.txt', 'a')
            logger.write('%f\n'%(mapval.item()))
            logger.close()  



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
            model.train()

            logger = open('val_acc.txt', 'a')
            logger.write('%d %f\n'%(epoch,mapval.item()))
            logger.close()  

    save_checkpoint(model, optimizer, filename=f"checkpoint_final.pth")

if __name__ == "__main__":
    main()