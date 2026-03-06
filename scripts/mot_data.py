import cv2
import torch
import numpy as np
import os
import multiprocessing as mp
from pathlib import Path
from boxmot import BotSort, OcSort, ByteTrack
from ultralytics import YOLO
from sahi import AutoDetectionModel
from sahi.predict import get_sliced_prediction
import numpy as np


# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

trackers = ["bytetrack", "ocsort", "bot-sort"]

def coco_to_kitti_type(coco_class_id):
    mapping = {
        0: 'Pedestrian', 1: 'Cyclist', 3: 'Cyclist',
        2: 'Car', 5: 'Van', 7: 'Truck', 6: "Tram"
    }
    res = mapping.get(coco_class_id)
    return "Misc" if res is None else res

def process_single_sequence(args):
    seq_imgs_path, seq_res_path, tracker_name, detector_path = args
    
    import cv2
    import torch
    import numpy as np
    from boxmot import BotSort, OcSort, ByteTrack
    from ultralytics import YOLO
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    detector = YOLO(detector_path, verbose=False)
    
    sahi_model = AutoDetectionModel.from_pretrained(
        model_type='ultralytics',
        model_path=detector_path,  
        confidence_threshold=0.1,   
        device='cuda',              
        image_size=640,
    )
    
    if tracker_name == "bytetrack":
        tracker = ByteTrack(
            reid_weights=None, device=device, half=True, with_reid=False,
            track_thresh=0.2, match_thresh=0.5, track_buffer=25, frame_rate=30
        )
    elif tracker_name == "ocsort":
        tracker = OcSort(reid_weights=None, device=device, half=True, with_reid=False)
    else:
        tracker = BotSort(reid_weights=None, device=device, half=True, with_reid=False)
    
    cap = cv2.VideoCapture(seq_imgs_path)
    if not cap.isOpened():
        print(f"Failed to open {seq_imgs_path}")
        return
    
    idx = 0
    results = []
    
    print(f"Processing {seq_imgs_path} in process {os.getpid()}")
    
    with torch.inference_mode():
        while True:
            success, frame = cap.read()
            if not success:
                break
            
            # output = get_sliced_prediction(
            #     frame,
            #     sahi_model,
            #     slice_height=256,
            #     slice_width=256,
            #     overlap_height_ratio=0.2,
            #     overlap_width_ratio=0.2,
            # ).to_coco_annotations()
            # scores = []
            # boxes = []
            # labels = []                
            # for o in output:
            #     x, y, w, h = o["bbox"]
            #     scores.append(o["score"])
            #     boxes.append([x, y, x + w, y + h])
            #     labels.append(o["category_id"])
            # scores = np.array(scores)
            # boxes = np.array(boxes)
            # labels = np.array(labels)
            output = detector([frame], verbose=False)[0]
            scores = output.boxes.conf.cpu().numpy()
            boxes = output.boxes.data.cpu().numpy()[:, :4]
            labels = output.boxes.cls.cpu().numpy()
            
            mask = scores >= 0.2
            filtered_boxes = boxes[mask]
            filtered_labels = labels[mask]
            filtered_scores = scores[mask]
            
            if len(filtered_boxes) > 0:
                detections = np.concatenate([
                    filtered_boxes, 
                    filtered_scores[:, None], 
                    filtered_labels[:, None]
                ], axis=1)
            else:
                detections = np.empty((0, 6))
            
            res = tracker.update(detections, frame)
            
            with open(seq_res_path, "a") as fw:
                for track in res:
                    line = [
                        idx, int(track[4]), 
                        coco_to_kitti_type(int(track[6])),
                        -1, -1, -1,
                        float(track[0]), float(track[1]), float(track[2]), float(track[3]),
                        -1, -1, -1, -1, -1, -1, -1
                    ]
                    fw.write(" ".join(str(x) for x in line) + "\n")
            
            idx += 1
    
    cap.release()
    print(f"Finished {seq_imgs_path}, total frames: {idx-1}")
    return f"Done {seq_imgs_path}"

def run_tracker(
    tracker_name,
    detector_path="/home/matvey/projects/jmot/scripts/models/yolo11s.pt",
    capture_path="/media/matvey/EB6B-E36F/diploma/kitti_metrics/data_tracking_image_2/training",
    out_path="/home/matvey/projects/TrackEval/data/trackers/kitti/kitti_2d_box_train"
):
    train_path = os.path.join(capture_path, "image_02")
    sequences = []
    
    for seq_name in sorted(os.listdir(train_path)):
        seq_dir = os.path.join(train_path, seq_name)
        if not os.path.isdir(seq_dir):
            continue
            
        imgs_path = os.path.join(seq_dir, "%06d.png")
        
        tracker_dir = os.path.join(out_path, tracker_name)
        data_dir = os.path.join(tracker_dir, "data")
        os.makedirs(data_dir, exist_ok=True)
        
        res_path = os.path.join(data_dir, f"{seq_name}.txt")
        if os.path.exists(res_path):
            os.remove(res_path)
        
        sequences.append((imgs_path, res_path, tracker_name, detector_path))
        print(f"Added sequence: {seq_name} -> {res_path}")
    
    mp.set_start_method('spawn', force=True)
    for seq in sequences:
        p = mp.Process(target=process_single_sequence, args=(seq,))
        p.start()
        p.join()

if __name__ == "__main__":
    for tracker_name in trackers:
        run_tracker(tracker_name)
