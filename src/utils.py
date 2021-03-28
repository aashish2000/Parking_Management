
import numpy as np
import cv2

def IOU_labels(l1,l2):
	return IOU(l1.tl(),l1.br(),l2.tl(),l2.br())

def draw_boxes(detections, image):
    '''
    Draws Bounding Boxes for all 
    detected vehicles in frame
    '''
    for label, confidence, bbox in detections:
        left, top, right, bottom = bbox2points(bbox)
        cv2.rectangle(image, (left, top), (right, bottom), (0,255,0), 1)
        cv2.putText(image, "{}".format(label.decode('ascii')),
                    (left, top - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    (0,0,255), 2)
    return image

def draw_licence_plate(car_detection, image, licence_str):
    '''
    Writes Licence Plate String of
    given car inside its Bounding Box
    '''
    left, top, right, bottom = bbox2points(car_detection[2])
    cv2.putText(image, "{}".format(licence_str), ((left+right)//2, (top+bottom)//2), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,0,255), 2)
    return image

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

def im2single(I):
	'''
	Normailze pixel values in image
	'''
	assert(I.dtype == 'uint8')
	return I.astype('float32')/255.

def IOU(tl1,br1,tl2,br2):
	'''
	Calculates the amount of overlap 
	between	2 given Bounding Boxes 
	'''

	wh1,wh2 = br1-tl1,br2-tl2
	assert((wh1>=.0).all() and (wh2>=.0).all())
	
	intersection_wh = np.maximum(np.minimum(br1,br2) - np.maximum(tl1,tl2),0.)
	intersection_area = np.prod(intersection_wh)
	area1,area2 = (np.prod(wh1),np.prod(wh2))
	union_area = area1 + area2 - intersection_area;
	return intersection_area/union_area

def nms(Labels,iou_threshold=.5):
	SelectedLabels = []
	Labels.sort(key=lambda l: l.prob(),reverse=True)
	
	for label in Labels:

		non_overlap = True
		for sel_label in SelectedLabels:
			if IOU_labels(label,sel_label) > iou_threshold:
				non_overlap = False
				break

		if non_overlap:
			SelectedLabels.append(label)

	return SelectedLabels

def crop_region(I,label,bg=0.5):
	'''
	Returns image crop containing the licence plate
	of a car inside its bounding box area
	'''
	wh = np.array(I.shape[1::-1])

	ch = I.shape[2] if len(I.shape) == 3 else 1
	tl = np.floor(label.tl()*wh).astype(int)
	br = np.ceil (label.br()*wh).astype(int)
	outwh = br-tl

	if np.prod(outwh) == 0.:
		return None

	outsize = (outwh[1],outwh[0],ch) if ch > 1 else (outwh[1],outwh[0])
	if (np.array(outsize) < 0).any():
		pause()
	Iout  = np.zeros(outsize,dtype=I.dtype) + bg

	offset 	= np.minimum(tl,0)*(-1)
	tl 		= np.maximum(tl,0)
	br 		= np.minimum(br,wh)
	wh 		= br - tl

	Iout[offset[1]:(offset[1] + wh[1]),offset[0]:(offset[0] + wh[0])] = I[tl[1]:br[1],tl[0]:br[0]]

	return Iout
