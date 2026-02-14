from ultralytics import YOLO

import fire

def validate(version, size):
    model = YOLO(f"./models/yolo{version}{size}.pt")
    model.info()
    results = model.val(data="coco.yaml")

def export_to_onnx(version, size):
    model = YOLO(f"./models/yolo{version}{size}.pt")
    model.export(format="onnx")

if __name__ == '__main__':
  fire.Fire()