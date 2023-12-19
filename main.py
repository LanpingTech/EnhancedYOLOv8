from ultralytics.models import YOLO

# Load a model
model = YOLO("ultralytics/cfg/models/v8/yolov8n-spd.yaml")  # build a new model from scratch
hyp = {
    'name': 'endovis-spd-aug',
    'data': 'datasets/endovis.yaml',
    'epochs': 100,
    'device': '2,3',
    'workers': 8,
    'mixup':0.5,
}
model.train(**hyp)  # train the model

# nohup python -u main.py > endovis-spd-aug.log 2>&1 &