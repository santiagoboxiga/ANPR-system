import os
import csv
import cv2
import numpy as np
from decord import VideoReader, cpu
from ultralytics import YOLO
from sort.sort import Sort
from utils.snapshot_utils import get_car, update_best_plates, save_best_plates

# Output folder for snapshots
os.makedirs("snaps", exist_ok=True)

# Initialize models and tracker
coco_model = YOLO("models\\yolov8n.pt").to('cuda')
lpr_detector = YOLO("models\\license_plate.pt").to('cuda')

# Prepare to write a single CSV file covering all videos
csv_filename = "snap_log.csv"
fieldnames = ["video_name", "car_id", "frame_num", "lp_confidence", "snap_path"]

with open(csv_filename, "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()

    videos_folder = "videos"
    for filename in os.listdir(videos_folder):
        if not filename.lower().endswith((".mp4", ".mov", ".avi", ".mkv")):
            continue

        video_path = os.path.join(videos_folder, filename)
        print(f"Processing: {video_path}")

        try:
            vr = VideoReader(video_path, ctx=cpu(0))
        except Exception as e:
            print(f"Could not open {video_path}. Skipping... Error: {e}")
            continue

        mot_tracker = Sort()
        best_plates = {}
        vehicles = [2, 3, 5, 7]
        batch_size = 4

        for i in range(0, len(vr), batch_size):
            batch_indices = list(range(i, min(i + batch_size, len(vr))))
            batch_frames = [vr[j].asnumpy() for j in batch_indices]
            bgr_frames = [cv2.cvtColor(f, cv2.COLOR_RGB2BGR) for f in batch_frames]

            coco_results = coco_model(bgr_frames, stream=False)
            lp_results = lpr_detector(bgr_frames, stream=False)

            for idx_in_batch, frame_num in enumerate(batch_indices):
                frame = bgr_frames[idx_in_batch]
                detections = coco_results[idx_in_batch].boxes.data.tolist()
                detections_ = [d[:5] for d in detections if int(d[5]) in vehicles]

                if detections_:
                    track_ids = mot_tracker.update(np.array(detections_))
                else:
                    track_ids = np.empty((0, 5))

                license_plates = lp_results[idx_in_batch].boxes.data.tolist()
                for license_plate in license_plates:
                    x1, y1, x2, y2, lp_score, lp_class_id = license_plate
                    xcar1, ycar1, xcar2, ycar2, car_id = get_car(license_plate, track_ids)

                    print(f"Frame: {frame_num}, Car ID: {car_id}, LP Score: {lp_score}")

                    if car_id != -1:
                        update_best_plates(
                            best_plates, car_id, lp_score, frame_num, frame,
                            (xcar1, ycar1, xcar2, ycar2),
                            (x1, y1, x2, y2)
                        )

        save_best_plates(best_plates, filename, writer)

print("Done! Best snapshots are in 'snaps',")
print("snap_log.csv summarizes them.")

