# Automatic Number Plate Recognition (ANPR) System

This project implements a complete Automatic Number Plate Recognition (ANPR) pipeline using YOLO-based vehicle and license plate detection, SORT tracking, and FastALPR OCR. It processes videos to extract high-confidence license plate snapshots, then analyzes and exports results to an Excel file.

## 📂 Project Structure

```
├── videos/                     # Input videos to be analyzed
├── models/                     # YOLO detection models (vehicle + license plate)
├── sort/                       # Kalman filter
├── utils/
│   ├── plate_utils.py          # Utilities for OCR processing and Excel generation
│   └── snapshot_utils.py       # Utilities for snapshot cropping and selection
├── snap_log.csv                # Log of snapshots with metadata
├── plate_recognition_results.xlsx # Final summary of license plates
├── plate_snapshot.py           # Main script to extract license plate snapshots from video
├── plate_reader.py             # Script to run OCR on snapshots and generate Excel output
```

## 🚀 Features

- **YOLOv8 vehicle detection**
- **YOLO license plate detection**
- **SORT tracking algorithm** to assign consistent car IDs across frames
- **FastALPR OCR** for license plate text recognition
- **Duplicate removal** based on plate text
- **Excel export** with embedded cropped images and confidence levels

## 🧰 Dependencies

Install dependencies via pip:

```bash
pip install -r requirements.txt
```

Required libraries include:
- `opencv-python`
- `decord`
- `ultralytics`
- `fast-alpr`
- `openpyxl`
- `numpy`
- `torch`

## 💻 How to Use

1. Place your video files inside the `videos/` folder.
2. Run the snapshot extraction:
   ```bash
   python plate_snapshot.py
   ```
   This will detect vehicles and plates, crop high-confidence results, and save them to the `snaps/` folder with a summary in `snap_log.csv`.

3. Run OCR and save results to Excel:
   ```bash
   python plate_reader.py
   ```

   This will perform OCR on the snapshots, remove duplicates, and generate an annotated Excel file: `plate_recognition_results.xlsx`.

## 🧪 Sample Output

- Snapshots: `snaps/video1_car_7_frame_245_lp.jpg`
- Log file: `snap_log.csv`
- Excel report: Includes plate text, confidence, and image thumbnails.
