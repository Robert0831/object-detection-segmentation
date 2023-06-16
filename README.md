# object-detection-segmentation

Yolov3+3-layer segmentation head


Trining
----------------------------------------------------------------------------
The flag can see train_run.py

For object detection

python train_run.py --trainwhat bac_obj --fixbac tt --data-path ./VOC --batch 16 --epoch 40 

For segmentation

python train_run.py --trainwhat seg --fixbac tt --data-path ./ADE --batch 8 --epoch 6 
