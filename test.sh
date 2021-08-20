CHECKPOINT='work_dirs/selfsup/moco/yolov4tiny/epoch_178.pth'
# CHECKPOINT='model_zoo/moco_r50_v1-4ad89b5c.pth'
WEIGHT_FILE='pretrains/moco_yolov4tiny'
CONFIG_FILE='configs/benchmarks/linear_classification/imagenet/yolov4tiny_last.py'
GPUS=2
python tools/extract_backbone_weights.py ${CHECKPOINT} ${WEIGHT_FILE}
bash benchmarks/dist_train_linear.sh ${CONFIG_FILE} ${WEIGHT_FILE}
# bash tools/dist_test.sh ${CONFIG_FILE} ${GPUS} ${CHECKPOINT}
