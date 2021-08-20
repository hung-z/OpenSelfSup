CUDA_VISIBLE_DEVICES=0,1 bash tools/dist_train.sh configs/selfsup/moco/yolov4tiny.py 2 --resume_from work_dirs/selfsup/moco/yolov4tiny/latest.pth
