import cv2
import numpy as np
from ultralytics import YOLO

def detect_plates_with_characters(
    image_path, 
    plate_model_path="license_plate.pt", 
    char_model_path="Charcter-LP.pt",
    conf_plate=0.25,
    conf_char=0.25
):
    """
    1) Detect license plates using the plate_model_path (e.g., 'license_plate.pt').
    2) Crop the plate region from the image.
    3) Use the char_model_path (e.g., 'Character-LP.pt') to detect individual characters.
    4) Sort the characters by x-coordinate and build the recognized license plate string.
    5) Draw bounding boxes around plates and recognized text on the original image (for visualization).

    :param image_path: Path to your input image.
    :param plate_model_path: YOLO model for detecting entire license plates.
    :param char_model_path: YOLO model for detecting individual characters.
    :param conf_plate: Confidence threshold for plate detection.
    :param conf_char: Confidence threshold for character detection.
    :return: List of recognized plate texts.
    """

    # -------------------
    # 1. Load models
    # -------------------
    plate_model = YOLO(plate_model_path)
    char_model = YOLO(char_model_path)

    # -------------------
    # 2. Read input image
    # -------------------
    image = cv2.imread(image_path)
    if image is None:
        raise FileNotFoundError(f"Could not read image from {image_path}")

    # -------------------
    # 3. Detect plates
    # -------------------
    plate_results = plate_model.predict(image, conf=conf_plate)
    
    # This will hold all recognized plates in the image
    recognized_plates = []

    # Process each result in the batch
    for result in plate_results:
        # Each result can have multiple bounding boxes (plates)
        for plate_box in result.boxes:
            x1, y1, x2, y2 = map(int, plate_box.xyxy[0])

            # Crop the plate region
            cropped_plate = image[y1:y2, x1:x2]

            # Optional: skip tiny bounding boxes that may be false positives
            # e.g. if (x2 - x1) < 5 or (y2 - y1) < 5: continue

            # -------------------------------------------
            # 4. Use char_model to detect characters
            # -------------------------------------------
            # YOLO expects images in BGR or RGB format. We'll pass the cropped region directly.
            char_results = char_model.predict(cropped_plate, conf=conf_char)

            # We’ll collect all (x_min, char_name) so we can sort them left-to-right
            char_detections = []

            for char_result in char_results:
                # The char_result.boxes might contain multiple character detections
                for char_box in char_result.boxes:
                    # Coordinates relative to `cropped_plate`
                    cx1, cy1, cx2, cy2 = map(int, char_box.xyxy[0])
                    
                    # The predicted class index. e.g. 0 for '0', 1 for '1', ...
                    class_idx = int(char_box.cls[0])
                    
                    # Retrieve the class name from the model’s .names dictionary
                    char_name = char_model.names[class_idx]
                    
                    # Save for sorting. Use cx1 (left bounding box coordinate) as key
                    # to keep the correct left→right order.
                    char_detections.append((cx1, char_name))

            # Sort characters by their x-coordinate
            char_detections.sort(key=lambda tup: tup[0])

            # Build the final text for this plate
            plate_text = "".join([ch[1] for ch in char_detections])
            recognized_plates.append(plate_text)

            # -------------------------------------------
            # 5. Draw bounding box + recognized text
            # -------------------------------------------
            cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(
                image,
                plate_text,
                (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.9,
                (0, 255, 0),
                2
            )
    
    # -------------------------------------------
    # Show the image with drawn detections
    # -------------------------------------------
    cv2.imshow("License Plate + Characters", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    return recognized_plates

if __name__ == "__main__":
    plate_model_path = "license_plate.pt"     
    char_model_path  = "Charcter-LP.pt"      
    image_path       = "snaps\car_15.0_frame_47_lp.jpg"        # Sample image

    recognized_texts = detect_plates_with_characters(
        image_path, 
        plate_model_path, 
        char_model_path
    )

    if not recognized_texts:
        print("No plates detected.")
    else:
        for i, text in enumerate(recognized_texts, start=1):
            print(f"Plate {i} recognized text: {text}")
