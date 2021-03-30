from src.label              import dknet_label_conversion, Label
from src.utils 				import im2single, nms, crop_region, draw_boxes, draw_licence_plate, bbox2points
from src.keras_utils 		import load_model, detect_lp
from copy                   import deepcopy
from collections            import defaultdict
import darknet.darknet
import numpy as np
import datetime
import cv2


def vehicle_detection(image_name, vehicle_net, vehicle_meta):
    '''
    Use darknet framework for identifying vehicle bounding boxes
    in image frame and cropping them for licence plate detection
    '''
    vehicle_threshold = .5
    detections , _ = darknet.darknet.detect(vehicle_net, vehicle_meta, image_name ,thresh=vehicle_threshold)

    detections = [r for r in detections if r[0] in [b'car', b'bus', b"motorbike", b"truck"]]

    vehicle_bbox_image = draw_boxes(detections, image_name)

    return(detections, vehicle_bbox_image)


def licence_plate_detection(image_original, cropped_car, wpod_net):
    '''
    Returns image crop containing the number plate of a vehicle 
    '''
    lp_threshold = .5

    Iorig = deepcopy(image_original)
    original_width = np.array(Iorig.shape[1::-1],dtype=float)

    # Extracting image crop containing a vehicle from camera frame
    cx,cy,w,h = (np.array(cropped_car[2])/np.concatenate( (original_width,original_width))).tolist()
    tl = np.array([cx - w/2., cy - h/2.])
    br = np.array([cx + w/2., cy + h/2.])
    label = Label(0,tl,br)
    Ivehicle = crop_region(Iorig,label)
    Ivehicle = Ivehicle.astype(np.uint8)

    # Computing vehicle dimensions relative to image crop
    ratio = float(max(Ivehicle.shape[:2]))/min(Ivehicle.shape[:2])
    side  = int(ratio*288.)
    bound_dim = min(side + (side%(2**4)),608)
    
    # Run model inference for returning licence plate bounding boxes
    licence_coords, licence_images ,_ = detect_lp(wpod_net, im2single(Ivehicle), bound_dim, 2**4, (240,80), lp_threshold)

    return(licence_coords, licence_images)


def licence_plate_ocr(licence_image, ocr_net, ocr_meta):
    '''
    Extracts characters of the licence plate and orders them
    to return the predicted licence plate of a vehicle
    '''
    ocr_threshold = .4
    licence_image *= 255.

    ocr_net_width = darknet.darknet.network_width(ocr_net)
    ocr_net_height = darknet.darknet.network_height(ocr_net)

    image_resized = cv2.resize(licence_image, (ocr_net_width, ocr_net_height), interpolation=cv2.INTER_NEAREST)
    image_resized = cv2.cvtColor(image_resized, cv2.COLOR_RGB2BGR)
    image_resized = image_resized.astype(np.uint8)

    plate_characters,(width,height) = darknet.darknet.detect(ocr_net, ocr_meta, image_resized, thresh=ocr_threshold, nms=None)
    
    lp_str = ""
    if(len(plate_characters)):
        L = dknet_label_conversion(plate_characters,width,height)
        L = nms(L,.45)

        L.sort(key=lambda x: x.tl()[0])
        lp_str = ''.join([chr(l.cl()) for l in L])
    
    return(lp_str)


def initialize_weights():
    # Initialize Vehicle Detection Model
    vehicle_weights = b'./data/vehicle-detector/yolov4-tiny.weights'
    vehicle_netcfg  = b'./data/vehicle-detector/yolov4-tiny.cfg'
    vehicle_dataset = b'./darknet/cfg/coco.data'

    vehicle_net  = darknet.darknet.load_net(vehicle_netcfg, vehicle_weights, 0)
    vehicle_meta = darknet.darknet.load_meta(vehicle_dataset)

    #Initialize Licence Plate Detection model
    wpod_net = load_model("./data/lp-detector/wpod-net_update1.h5")

    # Initiailze Licence Plate OCR Model
    ocr_weights = b'data/ocr/ocr-net.weights'
    ocr_netcfg  = b'data/ocr/ocr-net.cfg'
    ocr_dataset = b'data/ocr/ocr-net.data'

    ocr_net  = darknet.darknet.load_net(ocr_netcfg, ocr_weights, 0)
    ocr_meta = darknet.darknet.load_meta(ocr_dataset)

    return(vehicle_net, vehicle_meta, wpod_net, ocr_net, ocr_meta)


def vehicle_detection_video(video_path, vehicle_net, vehicle_meta, wpod_net, ocr_net, ocr_meta):

    # Use datetime to generate unique string for naming result video
    start = datetime.datetime.now()
    save_path = 'output-{}.webm'.format(datetime.datetime.timestamp(start))

    # Store identified vehicle numberplates
    identified_cars_numberplates = {}
    identified_licence_freq = defaultdict(int)

    # Read Video frame by frame from path
    cap = cv2.VideoCapture(video_path)
    
    frame = None
    if cap.isOpened():
        hasFrame, frame = cap.read()
    else:
        hasFrame = False

    # Write Result Video in .webm format for display in Browser window
    video_height, video_width = frame.shape[:2]
    save = cv2.VideoWriter("./static/results/"+save_path, cv2.VideoWriter_fourcc('V','P','8','0'), 30, (video_width,video_height))

    while hasFrame:
        image_original = frame

        # Perform Vehicle Detection
        detected_cars, vehicle_bbox_image = vehicle_detection(image_original, vehicle_net, vehicle_meta)

        if len(detected_cars):
            for i, cropped_car in enumerate(detected_cars):

                # Perform Licence Plate Detection for every detected vehicle
                licence_coords, licence_images = licence_plate_detection(image_original, cropped_car, wpod_net)
                
                if len(licence_coords):
                    licence_plate = licence_images[0]

                    # Perform Licence Plate Character Recognition
                    licence_str = licence_plate_ocr(licence_plate, ocr_net, ocr_meta)
                    vehicle_bbox_image = draw_licence_plate(cropped_car, vehicle_bbox_image, licence_str)

                    identified_cars_numberplates[licence_str] = cropped_car[0].decode('ascii')
                    identified_licence_freq[licence_str] += 1
        
        save.write(vehicle_bbox_image)
        hasFrame, frame = cap.read()

    cap.release()
    save.release()

    result_str = ""

    # Write Results to String for Display
    for key in identified_cars_numberplates:
        if(len(key) >= 8 and identified_licence_freq[key] > 30):
            result_str += key+"\n "

    return(save_path, result_str)


def vehicle_detection_image(image_path, vehicle_net, vehicle_meta, wpod_net, ocr_net, ocr_meta):
    # Use datetime to generate unique string for naming result image
    start = datetime.datetime.now()
    save_path = 'output-{}.jpg'.format(datetime.datetime.timestamp(start))

    # Store identified vehicle numberplates
    identified_cars_numberplates = {}
    identified_licence_freq = defaultdict(int)

    # Read Image by frame from path
    image_original = cv2.imread(image_path)

    # Perform Vehicle Detection
    detected_cars, vehicle_bbox_image = vehicle_detection(image_original, vehicle_net, vehicle_meta)

    if len(detected_cars):
        for i, cropped_car in enumerate(detected_cars):

            # Perform Licence Plate Detection for every detected vehicle
            licence_coords, licence_images = licence_plate_detection(image_original, cropped_car, wpod_net)
            
            if len(licence_coords):
                licence_plate = licence_images[0]

                # Perform Licence Plate Character Recognition
                licence_str = licence_plate_ocr(licence_plate, ocr_net, ocr_meta)
                vehicle_bbox_image = draw_licence_plate(cropped_car, vehicle_bbox_image, licence_str)

                identified_cars_numberplates[licence_str] = cropped_car[0].decode('ascii')
                identified_licence_freq[licence_str] += 1

    result_str = ""

    # Write Results to String for Display
    for key in identified_cars_numberplates:
        if(len(key) >= 8 and identified_licence_freq[key] > 30):
            result_str += key+"\n "
    
    cv2.imwrite("./static/results/"+save_path, vehicle_bbox_image)

    return(save_path, result_str)
