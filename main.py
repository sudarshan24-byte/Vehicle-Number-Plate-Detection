from ultralytics import YOLO
import cv2
from sort.sort import *
from util import get_car, read_license_plate, write_csv


results = {}
tracker = Sort()

# load models
yolo = YOLO('yolov8n.pt')
license_plate_detector = YOLO('./models/best.pt')

frame_mr = -1
vehicle_class = [2, 5, 7] #  list of vehicle classes detected in the frame (vehicle_class.txt)

cap = cv2.VideoCapture('./Videos/sample_video-1.mp4')

while True and frame_mr < 100:
    frame_mr += 1
    ret, frame = cap.read()
    if ret:
        results[frame_mr] = {}
        detections = yolo(frame)[0]
        detections_ = []

        # Vehicle Detection
        for detection in detections.boxes.data.tolist():
            x1, y1, x2, y2, score, class_id = detection
            if int(class_id) in vehicle_class:
                detections_.append([x1, y1, x2, y2, score])

        
        # Track Vehicles
        track_ids = tracker.update(np.asarray(detections_))

        # Licence Plate Detection
        license_plates = license_plate_detector(frame)[0]
        for license_plate in license_plates.boxes.data.tolist():
            x1, y1, x2, y2, score, class_id = license_plate

            # Assign Licence plate to the car
            xcar1, ycar1, xcar2, ycar2, car_id = get_car(license_plate, track_ids)

            # crop license plate
            license_plate_crop = frame[int(y1):int(y2), int(x1): int(x2), :]

            license_plate_crop_gray = cv2.cvtColor(license_plate_crop, cv2.COLOR_BGR2GRAY)
            _, license_plate_crop_thresh = cv2.threshold(license_plate_crop_gray, 80, 255, cv2.THRESH_BINARY_INV)

            license_plate_text, license_plate_text_score = read_license_plate(license_plate_crop_thresh)
            print('license_plate_text_score: ', license_plate_text_score)
            print('license_plate_text: ', license_plate_text)

            if license_plate_text is not None:
                results[frame_mr][car_id] = {'car': {'bbox': [xcar1, ycar1, xcar2, ycar2]},
                                              'license_plate': {'bbox': [x1, y1, x2, y2],
                                                                'text': license_plate_text,
                                                                'bbox_score': score,
                                                                'text_score': license_plate_text_score}}

            # cv2.imshow('original', license_plate_crop)
            # cv2.imshow('threshold', license_plate_crop_thresh)
            
            # cv2.waitKey(0)
            # cv2.destroyAllWindows()


write_csv(results, 'output-1.csv')
# cap.release()
# cv2.destroyAllWindows()