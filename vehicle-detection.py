import sys
import cv2
import numpy as np
import traceback

import darknet.darknet as dn

from src.label 				import Label, lwrite
from os.path 				import splitext, basename, isdir
from os 					import makedirs
from src.utils 				import crop_region, image_files_from_folder


if __name__ == '__main__':

	try:
	
		input_dir  = sys.argv[1]
		output_dir = sys.argv[2]

		vehicle_threshold = .5

		vehicle_weights = b'./data/vehicle-detector/yolov4-tiny.weights'
		vehicle_netcfg  = b'./data/vehicle-detector/yolov4-tiny.cfg'
		vehicle_dataset = b'./darknet/cfg/coco.data'

		vehicle_net  = dn.load_net(vehicle_netcfg, vehicle_weights, 0)
		vehicle_meta = dn.load_meta(vehicle_dataset)

		imgs_paths = image_files_from_folder(input_dir)
		imgs_paths.sort()

		if not isdir(output_dir):
			makedirs(output_dir)

		print('Searching for vehicles using YOLO...')

		for i,img_path in enumerate(imgs_paths):

			print('\tScanning %s' % img_path)

			bname = basename(splitext(img_path)[0])

			R,_ = dn.detect(vehicle_net, vehicle_meta, bytes(img_path, encoding='utf-8') ,thresh=vehicle_threshold)
            
			R = [r for r in R if r[0] in [b'car',b'bus']]

			print('\t\t%d cars found' % len(R))

			if len(R):

				Iorig = cv2.imread(img_path)
				WH = np.array(Iorig.shape[1::-1],dtype=float)
				Lcars = []

				for i,r in enumerate(R):

					cx,cy,w,h = (np.array(r[2])/np.concatenate( (WH,WH) )).tolist()
					tl = np.array([cx - w/2., cy - h/2.])
					br = np.array([cx + w/2., cy + h/2.])
					label = Label(0,tl,br)
					Icar = crop_region(Iorig,label)

					Lcars.append(label)

					cv2.imwrite('%s/%s_%dcar.png' % (output_dir,bname,i),Icar)

				lwrite('%s/%s_cars.txt' % (output_dir,bname),Lcars)

	except:
		traceback.print_exc()
		sys.exit(1)

	sys.exit(0)
	

# from numpy.lib.type_check import imag
# from darknet.darknet_images import load_images, image_detection
# import darknet.darknet
# import cv2
# import random
# from src.utils import im2single
# from src.keras_utils import load_model, detect_lp
# import imutils

# def bbox2points(bbox):
#     """
#     From bounding box yolo format
#     to corner points cv2 rectangle
#     """
#     x, y, w, h = bbox
#     xmin = int(round(x - (w / 2)))
#     xmax = int(round(x + (w / 2)))
#     ymin = int(round(y - (h / 2)))
#     ymax = int(round(y + (h / 2)))
#     return xmin, ymin, xmax, ymax

# def resize_frame(input_frame,dimensions):
#     '''
#     Resize input frame to model input layer dimensions
#     while preserving aspect ratio of the image.
#     If the resized image is smaller than the required dimensions,
#     black borders are added to equalize the dimensions.
#     Else if the resized image is larger than the required dimensions,
#     the frame is centered and cropped to size.
#     '''

#     aspect_resized_frame = imutils.resize(input_frame, width = dimensions[1], inter=cv2.INTER_LINEAR)

#     if(aspect_resized_frame.shape[0] > dimensions[0]):
#         crop_len = (aspect_resized_frame.shape[0] - dimensions[0]) // 2
#         aspect_resized_frame = aspect_resized_frame[crop_len : crop_len + aspect_resized_frame.shape[0]][0 : dimensions[1]]
    
#     elif(aspect_resized_frame.shape[0] < dimensions[0]):
#         border_len = (dimensions[0] - aspect_resized_frame.shape[0]) // 2
#         aspect_resized_frame = cv2.copyMakeBorder(aspect_resized_frame, border_len, border_len, 0, 0, cv2.BORDER_CONSTANT)
    
#     return (aspect_resized_frame)


# random.seed(3)  # deterministic bbox colors
# network, class_names, class_colors = darknet.darknet.load_network(
#     "../yolov4/yolov4-tiny.cfg",
#     "./darknet/cfg/coco.data",
#     "../yolov4/yolov4-tiny.weights",
#     batch_size=1
# )


# # images = load_images("../Dark_Dataset/Car")
# images = load_images("./test_images")
# width = darknet.darknet.network_width(network)
# height = darknet.darknet.network_height(network)

# print(width,height)
# lp_threshold = .5
# wpod_net = load_model("./data/lp-detector/wpod-net_update1.h5")

# for index in range(len(images)):
#     image_name = images[index]
#     print(image_name)
#     image, detections = image_detection(image_name, network, class_names, class_colors, 0.5)
#     img_read = cv2.imread(image_name)
#     img_read = resize_frame(img_read, (height, width))

#     for label, confidence, bbox in detections:
#         if(label == 'car' or label == "motorbike" or label == "truck" or label == "bus"):
#             left, top, right, bottom = bbox2points(bbox)
#             img_bound = cv2.rectangle(img_read, (left,top), (right,bottom), (255,0,0), 2)
#             Ivehicle = img_read[top:bottom, left:right]
#             ratio = float(max(Ivehicle.shape[:2]))/min(Ivehicle.shape[:2])
#             side  = int(ratio*288.)
#             bound_dim = min(side + (side%(2**4)),608)
#             print("\t\tBound dim: %d, ratio: %f" % (bound_dim,ratio))
#             Llp,LlpImgs,_ = detect_lp(wpod_net,im2single(Ivehicle),bound_dim,2**4,(240,80),lp_threshold)
#             if len(LlpImgs):
#                 Ilp = LlpImgs[0]
#                 Ilp = cv2.cvtColor(Ilp, cv2.COLOR_BGR2GRAY)
#                 Ilp = cv2.cvtColor(Ilp, cv2.COLOR_GRAY2BGR)
#                 Ilp *= 255.
#                 print(Llp[0].pts)

#                 cv2.imshow("window",Ilp)
#                 if cv2.waitKey() & 0xFF == ord('q'):
#                     break



    

