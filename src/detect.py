import cv2, torch
import numpy as np
from constants import *
from ultralytics import YOLO
import time, math, argparse, os

model = YOLO(MODEL)

def close_windows(cap, out):
    cap.release()
    if out != None: # exporting video
        out.release()
        
    print("[INFO] Closing Windows")
    cv2.destroyAllWindows()
    print("Detections Stopped")

def calculate_distance(pt1, pt2):
    '''Calculating euclidean distance between 2 points'''
    x1, x2 = pt1[0], pt2[0]
    y1, y2 = pt1[1], pt2[1]

    delta_x = x2 - x1
    delta_y = y2 - y1
    add = math.pow(delta_x, 2) + math.pow(delta_y, 2)
    return math.sqrt(add)

def draw_label(image: cv2.Mat, label: str, pts, colour):
    x_min, y_min = pts
    (text_w, text_h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_COMPLEX_SMALL, 0.7, 1)

    outside = y_min - text_h >= text_h
    bg_start = x_min, y_min
    bg_end = x_min + text_w, y_min - text_h - text_h - 2 if outside else y_min + text_h + text_h + 2
    
    cv2.rectangle(image, bg_start,  bg_end, colour, cv2.FILLED)
    cv2.putText(image, label, (x_min, y_min - text_h), cv2.FONT_HERSHEY_COMPLEX_SMALL, 0.7, WHITE, 1)

def draw_bounding_box(image: cv2.Mat, pts, label: str, colour):
    cv2.rectangle(image, (pts[0], pts[1]), (pts[2], pts[3]), colour, 2)
    draw_label(image, label, (pts[0], pts[1]), colour)

def annotate(image: cv2.Mat, detections):
    for result in detections:        
        for bbox in result.boxes:
            x_min, y_min, x_max, y_max = bbox.xyxy.numpy().ravel().astype(np.int64)
            height = y_max - y_min
            conf = bbox.conf.item()
            
            class_no = bbox.cls.type(torch.int64).item()
            class_name = CLASSES[class_no]
            
            # detected objec is not a vehicle/person
            if class_name not in VALID_DETECTIONS:
                continue

            if conf < 0.5:
                continue

            bbox_center = int((x_min + x_max)/2), int((y_min + y_max)/2)
            if class_name not in VEHICLE_SIZES: # display the bounding box and label only if the detected object is not a vehicle                
                draw_bounding_box(image, (x_min, y_min, x_max, y_max), f'{class_name}: {conf:.0%}', MAGENTA)
                continue

            # detecting vehicles only if the object is approximetly in the same lane
            if RIGHT_BOUNDARY >= bbox_center[0] >= LEFT_BOUNDARY and bbox_center[1] <= BOTTOM_BOUNDARY:
                current_time = time.localtime()
                print(f"{time.strftime('%H:%M:%S', current_time)} [INFO]: {class_name} {conf:.0%} x1, y1: {x_min, y_min} x2, y2: {x_max, y_max}")

                # estimating distance from camera to vehicle
                distance_px = calculate_distance(DANGER_ZONE, bbox_center)
                cv_factor = VEHICLE_SIZES[class_name] / height
                distance_m = distance_px * cv_factor
                
                close = distance_m > VEHICLE_SIZES[class_name] and distance_m < VEHICLE_SIZES[class_name] * 5 # vehicle is further than 1 car length but closer than 5
                brake = distance_m < VEHICLE_SIZES[class_name] # vehicle is closer than 1 car length
                
                colour = ORANGE if close else RED if brake else None
                if colour is None:
                    continue
                
                # displays a line from the danger zone to the center of the bounding box
                # cv2.line(image, DANGER_ZONE, bbox_center, (255,255,255), 2) 
                # cv2.circle(image, DANGER_ZONE, 10, WHITE, -1)
                
                draw_label(image, f'{distance_m:.1f}m', (x_min, y_min + height), colour)
                msg = "Obstacle Close" if close else "Imminent Collision - Brake!" if brake else ""
                draw_bounding_box(image, (x_min, y_min, x_max, y_max), f'{class_name}: {conf:.0%} {msg}', colour)

def get_options() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', default=f'{VIDEOS}/input.mp4', help='video to perform detection on')
    parser.add_argument('--nosave', help='do not save the video')
    return parser.parse_args()

def start_video_capture(source: str):
    cap = cv2.VideoCapture(source)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, RESOLUTION[0])
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, RESOLUTION[1])
    cap.set(cv2.CAP_PROP_SHARPNESS, 50)
    return cap

def get_exporter():
    '''Returns OpenCV VideoWriter with an incremented filename.'''
    fourcc = cv2.VideoWriter_fourcc(*'MP4V')
    path = f'{VIDEOS}/output.mp4'
    
    if not os.path.exists(path):
        out = cv2.VideoWriter(path, fourcc, 20.0, RESOLUTION)
    else:
        i = 1
        while os.path.exists(f'{path}_{i}.mp4'):
            i += 1
            
        base, ext = os.path.splitext(path)
        out = cv2.VideoWriter(f'{base}_{i}{ext}', fourcc, 20.0, RESOLUTION)
        return out

def main(source=f'{VIDEOS}/input.mp4', nosave=False):
    cap = start_video_capture(source)
    out = get_exporter() if not nosave else None
    kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
    i = 0

    try:
        while True:
            ret = cap.grab()
            i += 1

            if i % 2 == 1:
                continue

            ret, frame = cap.retrieve()
            if not ret:
                break

            frame = cv2.filter2D(frame, ddepth=-1, kernel=kernel)            
            detections = model.predict(frame, stream=True)
            annotate(frame, detections)

            cv2.imshow('Footage', frame)
            if out != None:
                out.write(frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    except KeyboardInterrupt:
        pass
    finally:
        close_windows(cap, out)

if __name__ == "__main__":
    options = get_options()
    main(**vars(options))