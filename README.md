# object-detection-segmentation

Yolov3+3-layer segmentation head


Trining
----------------------------------------------------------------------------
The flag can see train_run.py

For object detection

python train_run.py --trainwhat bac_obj --data-path ./VOC --batch 16 --epoch 40 

For segmentation

python train_run.py --trainwhat seg --fixbac tt --data-path ./ADE --batch 8 --epoch 6 

Test
----------------------------------------------------------------------------
Run test_ob.py to test accuracy

Reference YOLOV3
----------------------------------------------------------------------------
https://github.com/aladdinpersson/Machine-Learning-Collection/tree/master/ML/Pytorch/object_detection/YOLOv3
