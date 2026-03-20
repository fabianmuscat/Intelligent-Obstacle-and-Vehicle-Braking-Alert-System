ROOT = '../resources'
IMAGES = ROOT + '/images'
VIDEOS = ROOT + '/videos'
MODEL = ROOT + '/yolo/yolov8n.pt'
RESOLUTION = (1280, 720)

CLASSES = [line.strip().capitalize() for line in open(ROOT + '/yolo/coco.names', 'r').readlines()]
VALID_DETECTIONS = ['Car', 'Truck', 'Bus', 'Bicycle', 'Motorbike', 'Person']
VEHICLE_SIZES = {
    'Car': 4.48,
    'Truck': 6.04,
    'Bus': 6.50,
    'Bicycle': 1.75,
    'Motorbike': 2.22
}

# Colours
ORANGE = (5, 145, 255)
RED = (0, 0, 255)
WHITE = (255, 255, 255)
MAGENTA = (255, 0, 255)

# Boundaries
LEFT_BOUNDARY = 570
RIGHT_BOUNDARY = 800
BOTTOM_BOUNDARY = 550

# Point where distance to the next car is calculated from
DANGER_ZONE = int(RESOLUTION[0] / 2), BOTTOM_BOUNDARY