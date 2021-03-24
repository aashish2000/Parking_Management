from numpy.lib.type_check 	import imag
from darknet.darknet_images import load_images, image_detection
from src.label				import dknet_label_conversion, Label
from src.utils 				import im2single, nms, crop_region
from src.keras_utils 		import load_model, detect_lp
import imutils
import darknet.darknet
import numpy as np
import sys
from copy import deepcopy
import cv2
import random

def bbox2points(bbox):
    '''
    From bounding box yolo format
    to corner points cv2 rectangle
    '''
    x, y, w, h = bbox
    xmin = int(round(x - (w / 2)))
    xmax = int(round(x + (w / 2)))
    ymin = int(round(y - (h / 2)))
    ymax = int(round(y + (h / 2)))
    return xmin, ymin, xmax, ymax

def resize_frame(input_frame,dimensions):
    '''
    Resize input frame to model input layer dimensions
    original_widthile preserving aspect ratio of the image.
    If the resized image is smaller than the required dimensions,
    black borders are added to equalize the dimensions.
    Else if the resized image is larger than the required dimensions,
    the frame is centered and cropped to size.
    '''

    aspect_resized_frame = imutils.resize(input_frame, width = dimensions[1], inter=cv2.INTER_LINEAR)

    if(aspect_resized_frame.shape[0] > dimensions[0]):
        crop_len = (aspect_resized_frame.shape[0] - dimensions[0]) // 2
        aspect_resized_frame = aspect_resized_frame[crop_len : crop_len + aspect_resized_frame.shape[0]][0 : dimensions[1]]
    
    elif(aspect_resized_frame.shape[0] < dimensions[0]):
        border_len = (dimensions[0] - aspect_resized_frame.shape[0]) // 2
        aspect_resized_frame = cv2.copyMakeBorder(aspect_resized_frame, border_len, border_len, 0, 0, cv2.BORDER_CONSTANT)
    
    return (aspect_resized_frame)


def vehicle_detection(image_name, vehicle_net, vehicle_meta, vehicle_threshold):
    '''
    Use darknet framework for identifying vehicle bounding boxes
    in image frame and cropping them for licence plate detection
    '''
    detections , _ = darknet.darknet.detect(vehicle_net, vehicle_meta, bytes(image_name, encoding='utf-8') ,thresh=vehicle_threshold)

    detections = [r for r in detections if r[0] in [b'car', b'bus', b"motorbike", b"truck"]]

    print('\t\t%d cars found' % len(detections))

    return(detections)

def licence_plate_detection(image_original, cropped_car, wpod_net, lp_threshold):
    '''
    Returns image crop containing the number plate of a vehicle 
    '''

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
    print("\t\tBound dim: %d, ratio: %f" % (bound_dim,ratio))
    
    # Run model inference for returning licence plate bounding boxes
    licence_coords, licence_images ,_ = detect_lp(wpod_net, im2single(Ivehicle), bound_dim, 2**4, (240,80), lp_threshold)

    return(licence_coords, licence_images)

def licence_plate_ocr(licence_image, ocr_net, ocr_meta, ocr_threshold, ocr_net_width, ocr_net_height):
    '''
    Extracts characters of the licence plate and orders them
    to return the predicted licence plate of a vehicle
    '''
    licence_image *= 255.
    image_resized = cv2.resize(licence_image, (ocr_net_width, ocr_net_height), interpolation=cv2.INTER_NEAREST)
    cv2.imwrite("./tmp/file.jpg", image_resized)
    
    plate_characters,(width,height) = darknet.darknet.detect(ocr_net, ocr_meta, bytes("./tmp/file.jpg", encoding='utf-8') ,thresh=ocr_threshold, nms=None)

    if(len(plate_characters)):
        L = dknet_label_conversion(plate_characters,width,height)
        L = nms(L,.45)

        L.sort(key=lambda x: x.tl()[0])
        lp_str = ''.join([chr(l.cl()) for l in L])

        print("LICENCE_PLATE: ", lp_str)


# Initialize Vehicle Detection Model
vehicle_threshold = .5
vehicle_weights = b'../yolov4/yolov4-tiny.weights'
vehicle_netcfg  = b'../yolov4/yolov4-tiny.cfg'
vehicle_dataset = b'./darknet/cfg/coco.data'

vehicle_net  = darknet.darknet.load_net(vehicle_netcfg, vehicle_weights, 0)
vehicle_meta = darknet.darknet.load_meta(vehicle_dataset)

#Initialize Licence Plate Detection model
lp_threshold = .5
wpod_net = load_model("./data/lp-detector/wpod-net_update1.h5")

# Initiailze Licence Plate OCR Model
ocr_threshold = .4
ocr_weights = b'data/ocr/ocr-net.weights'
ocr_netcfg  = b'data/ocr/ocr-net.cfg'
ocr_dataset = b'data/ocr/ocr-net.data'

ocr_net  = darknet.darknet.load_net(ocr_netcfg, ocr_weights, 0)
ocr_meta = darknet.darknet.load_meta(ocr_dataset)
ocr_net_width = darknet.darknet.network_width(ocr_net)
ocr_net_height = darknet.darknet.network_height(ocr_net)

# Load test images from input directory
images = load_images(sys.argv[1])
width = darknet.darknet.network_width(vehicle_net)
height = darknet.darknet.network_width(vehicle_net)


for index in range(len(images)):
    image_name = images[index]
    image_original = cv2.imread(image_name)
    print(image_name)

    # Perform Vehicle Detection
    detected_cars = vehicle_detection(image_name, vehicle_net, vehicle_meta, vehicle_threshold)

    if len(detected_cars):
        for i, cropped_car in enumerate(detected_cars):
            # Perform Licence Plate Detection for every detected vehicle
            licence_coords, licence_images = licence_plate_detection(image_original, cropped_car, wpod_net, lp_threshold)
            
            if len(licence_coords):
                licence_plate = licence_images[0]
                
                # Display Detected Licence Plate
                # cv2.imshow("window",licence_plate)
                # if cv2.waitKey() & 0xFF == ord('q'):
                #     break

                # Perform Licence Plate Character Recognition
                licence_plate_ocr(licence_plate, ocr_net, ocr_meta, ocr_threshold, ocr_net_width, ocr_net_height)


                



    

