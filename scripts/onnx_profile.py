import onnxruntime as ort
import numpy as np
import json
import fire

def onnx_profile_dump_json(model_path="./models/yolo11l.onnx", iterations=100):
    options = ort.SessionOptions()
    options.enable_profiling = True
    sess = ort.InferenceSession(model_path, options, providers=["CUDAExecutionProvider"])

    for _ in range(iterations):
        sess.run(None, {"images": np.random.randn(1, 3, 640, 640).astype(np.float32)})

    prof_file = sess.end_profiling()
    print(f"Profile saved: {prof_file}")  

def example_work_with_json(prof_file="result.json"):
        # Example of working with profile.json file of onnx
    with open(prof_file, 'r') as f:
        prof_data = json.load(f)
        
    conv_times = [e['dur'] for e in prof_data if 'Conv' in e['name']]
    print(f"Conv слои: {len(conv_times)} штук")

if __name__ == "__main__":
    fire.Fire()