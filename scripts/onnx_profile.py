import onnxruntime as ort
import numpy as np
import json
import fire
import time

def onnx_profile_dump_json(model_path="./models/yolo11l.onnx", iterations=100):
    options = ort.SessionOptions()
    options.enable_profiling = True
    options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

    sess = ort.InferenceSession(model_path, options, providers=[("CUDAExecutionProvider", {"device_id": 0})])
    for _ in range(1):
        sess.run(None, {"images": np.random.randn(1, 3, 640, 640).astype(np.float32)})

    start = time.perf_counter()
    for _ in range(iterations):
        sess.run(None, {"images": np.random.randn(1, 3, 640, 640).astype(np.float32)})
    end = time.perf_counter()
    print("Time: ", end - start)
    prof_file = sess.end_profiling()
    print(f"Profile saved: {prof_file}")  

def example_work_with_json(prof_file="result.json"):
        # Example of working with profile.json file of onnx
    with open(prof_file, 'r') as f:
        prof_data = json.load(f)
    
    dur = 0
    al = []
    for lst in prof_data:
        if lst["dur"] == 0: continue
        dur += lst["dur"]
        al.append(lst["dur"] / 1000)
    print(sorted(al, reverse=True)[:100])
    print(sum(al[1:]))

if __name__ == "__main__":
    fire.Fire()