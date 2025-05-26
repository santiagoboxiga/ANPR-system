import os
import cv2

def get_car(license_plate, vehicle_track_ids):
    """
    Retrieve the vehicle coordinates and ID based on the license plate coordinates.

    Args:
        license_plate (tuple): Tuple containing the coordinates of the license plate (x1, y1, x2, y2, score, class_id).
        vehicle_track_ids (list): List of vehicle track IDs and their corresponding coordinates.

    Returns:
        tuple: Tuple containing the vehicle coordinates (x1, y1, x2, y2) and ID.
    """
    x1, y1, x2, y2, score, class_id = license_plate
    for xcar1, ycar1, xcar2, ycar2, car_id in vehicle_track_ids:
        if x1 > xcar1 and y1 > ycar1 and x2 < xcar2 and y2 < ycar2:
            return xcar1, ycar1, xcar2, ycar2, car_id
    return -1, -1, -1, -1, -1

def update_best_plates(best_plates, car_id, lp_score, frame_num, frame, car_bbox, lp_bbox):
    if (car_id not in best_plates) or (lp_score > best_plates[car_id]["score"]):
        best_plates[car_id] = {
            "score": lp_score,
            "frame_num": frame_num,
            "frame": frame.copy(),
            "car_bbox": car_bbox,
            "lp_bbox": lp_bbox,
        }

def save_best_plates(best_plates, filename, writer):
    for car_id, data in sorted(best_plates.items(), key=lambda item: item[1]["frame_num"]):
        frame = data["frame"]
        conf = data["score"]
        frame_num = data["frame_num"]
        xcar1, ycar1, xcar2, ycar2 = data["car_bbox"]
        xlp1, ylp1, xlp2, ylp2 = map(int, data["lp_bbox"])
        crop = frame[ylp1:ylp2, xlp1:xlp2]

        snap_filename = f"{os.path.splitext(filename)[0]}_car_{car_id}_frame_{frame_num}_lp.jpg"
        snap_path = os.path.join("snaps", snap_filename)
        cv2.imwrite(snap_path, crop)

        writer.writerow({
            "video_name": filename,
            "car_id": car_id,
            "frame_num": frame_num,
            "lp_confidence": f"{conf:.4f}",
            "snap_path": snap_path
        })
