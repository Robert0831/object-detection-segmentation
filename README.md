# object-detection-segmentation

Trining
----------------------------------------------------------------------------
The flag can see train_run.py

For object detection

python train_run.py --trainwhat bac_obj --fixbac tt --data-path ./VOC --batch 16 --epoch 40 

for segmentation

python train_run.py --trainwhat seg --fixbac tt --data-path ./ADE --batch 8 --epoch 6 
