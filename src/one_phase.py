from collections import defaultdict
import numpy as np
import src.darknet as dn
import cv2

def deploy_func_one_phase(imageList, thresh, nms, approach_obj):
    """object detection pipeline for one-phase,augmented,augmented2,cropped approach
    
    Args:
        imageList (list): list of image paths to predict
        thresh (float): threshold value for detector
        nms (float): nms threshold value
        approach_obj (object): information about the experiment
        
    Returns:
        dict: dict of predicted bounding boxes
    """

    if (approach_obj.heuristic=="scale" or approach_obj.scaled_dataset):
	    scale = "_scale"
    elif approach_obj.segment_dataset:
	    scale = "_segment" 
    else:
	    scale = ""
    scale_val = approach_obj.scale_val if approach_obj.scaled_dataset else ""
    yolo_model = (approach_obj.name+"/darknet/yolov3_"+str(approach_obj.num_classes)+"c.cfg.test").encode()
    yolo_weights = (approach_obj.name+"/weights/"
                    +str(approach_obj.num_classes)+"c"+scale+"/"+
                    str(scale_val)+"/yolov3_"+str(approach_obj.yolo_weight)+".weights").encode()
    yolo_data = (approach_obj.name+"/darknet/graffiti_"+str(approach_obj.num_classes)+".data").encode()
    
    
    dn.init_net(approach_obj.cpu)
    if approach_obj.cpu==False:
        dn.set_gpu(0)
    net = dn.load_net(yolo_model, yolo_weights, 0)
    meta = dn.load_meta(yolo_data)

    box_id = 0
    pred_dict = defaultdict(dict)
    for idx, img_file in enumerate(imageList):
        dets = dn.detect(net, meta,img_file.encode('utf-8'), thresh=thresh, nms=nms)
        pred_box_list = []
        for bbox in dets:   
            [x,y,w,h] = bbox[2]
            #https://github.com/pjreddie/darknet/issues/243
            y = y-(h/2)
            x = x-(w/2)
            img = cv2.imread(img_file)
            y1_unscaled = int(max(0,y))
            y2_unscaled = int(min((y+h),img.shape[0]))
            x1_unscaled = int(max(0,x))
            x2_unscaled = int(min((x+w),img.shape[1]))
            crop_img = img[y1_unscaled:y2_unscaled,x1_unscaled:x2_unscaled]
            if crop_img.size > 0:
                pred_box_list.append(np.array([box_id, approach_obj.name_to_id[bbox[0].decode("utf-8")], bbox[1],x1_unscaled,y1_unscaled,x2_unscaled,y2_unscaled]))
                box_id += 1
        pred_dict[img_file] = np.array(pred_box_list)
    print("total images predicted:", len(pred_dict))
    print("total bbox predicted:", box_id)
    return pred_dict
