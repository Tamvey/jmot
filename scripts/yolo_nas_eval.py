from ultralytics import NAS

import fire

def validate(size):
    model = NAS(f"./models/yolo_nas_{size}.pt")
    model.info()
    results = model.val(data="coco.yaml")

def export_to_onnx(size):
   model = NAS(f"./models/yolo_nas_{size}.pt")
   model.export(format="onnx")

if __name__ == '__main__':
  fire.Fire()
