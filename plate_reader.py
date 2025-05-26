import os
from fast_alpr import ALPR
from utils.plate_utils import extract_car_id, analyze_plate, remove_duplicate_plates, write_to_excel

# Initialize ALPR
alpr = ALPR(
    detector_model="yolo-v9-t-384-license-plate-end2end",
    ocr_model="global-plates-mobile-vit-v2-model",
)

# Input and output setup
snaps_folder = "snaps"
output_file = "plate_recognition_results.xlsx"

# Get image files from snaps folder
image_files = [
    f for f in os.listdir(snaps_folder)
    if f.lower().endswith(('png', 'jpg', 'jpeg'))
]

# Collect [car_id, plate_text, image_path, confidence]
data = []
for image_file in image_files:
    image_path = os.path.join(snaps_folder, image_file)
    car_id = extract_car_id(image_file)
    plate_text, confidence = analyze_plate(alpr, image_path)
    data.append([car_id, plate_text, image_path, confidence])

# Remove duplicate plates
unique_data = remove_duplicate_plates(data)

# Write everything to Excel
write_to_excel(unique_data, output_file)

print(f"Processing completed. Results saved to {output_file}")
