from collections import defaultdict
import numpy as np
import src.darknet as dn
import cv2

def classify_detections(imageList, return_dict, num_images_for_clf, approach_obj=None):
    model_dir = "segment_dataset" if approach_obj.segment_dataset else approach_obj.heuristic
    level1_file_name = os.path.join(approach_obj.name,"caffe","models", model_dir+"_models",
                                    "test_"+str(len(approach_obj.level1_classes))+"c.temp")
    f = open(level1_file_name,"w")
    for img_file in imageList:
        if img_file not in return_dict:
            continue
        for bbox in return_dict[img_file]:
            segment_path = bbox[0]
            f.write(segment_path+" "+str(int(bbox[2][0]))+"\n")
    f.close()
    
    # Classifier
    model = os.path.join("two-phase/caffe/models/",model_dir+"_models","train_val_"+str(approach_obj.num_classes)+"c.prototxt.test")
    print("using caffe model: {}".format(model))
    weights = os.path.join("two-phase/caffe/weights/",
                           model_dir+"_weights",str(approach_obj.num_classes)+"c",
                           "bvlc_googlenet_iter_"+str(approach_obj.surface_caffe_weight)+".caffemodel")
    print("using caffe weights: {}".format(weights))
    assert(model==approach_obj.level1_model)
    assert(weights==approach_obj.level1_weights)

    clf = CaffePredictor(model,weights)
    y_preds, surface_pred_confs = clf.predict(num_images_for_clf, meta=True)
    print("total predicted boxesfrom caffe:",len(y_preds),"pred boxes from yolo:",num_images_for_clf)
    assert(len(y_preds)==num_images_for_clf)

    segment_predictions = {}
    surface_segment_predictions = {}
    with open(os.path.join(approach_obj.name,"caffe","models", model_dir+"_models",
                           "test_"+str(approach_obj.num_classes)+"c.temp"),"r") as f:
        lines = f.readlines()
        for i in range(len(lines)):
            segment_path = lines[i].strip().split(" ")[0]
            segment_class_id = y_preds[i]
            segment_predictions[segment_path] = (segment_class_id,surface_pred_confs[i][y_preds[i]])
            surface_segment_predictions[segment_path] = surface_pred_confs[i]                       
        
    #map the result of each segment to pred_dict:
    for img_file in imageList:
        if img_file not in return_dict: continue    
        segment_list = return_dict[img_file]
        for idx,segment in enumerate(segment_list):
            segment[2] = list(segment[2])
            segment[2][1] = segment_predictions[segment[0]][0]
            #for hybrid this step is done in combine_results
            if not approach_obj.hybrid:
                segment[2][2] *= segment_predictions[segment[0]][1]
    
    
    #to keep the return format same as one-phase
    for img_file in return_dict:    
        for idx,segment in enumerate(return_dict[img_file]):
            return_dict[img_file][idx] = segment[2]
    
    return return_dict


def deploy_func_two_phase(imageList, thresh, nms, approach_obj):
    """object detection pipeline for two-phase approach
    
    Args:
        imageList (list): list of image paths to predict
        thresh (float): threshold value for detector
        nms (float): nms threshold value
        approach_obj (object): information about the experiment
                
    Returns:
        dict: dict of predicted bounding boxes
    """
    if approach_obj.scaled_dataset:
        dataset = "scaled_dataset"
    elif approach_obj.segment_dataset:
        dataset = "segment_dataset"
    else:
        dataset = "orig_dataset"
    yolo_model = (approach_obj.name+"/darknet/yolov3.cfg.test").encode()
    yolo_weights = (os.path.join(approach_obj.name+"/weights/",dataset,"yolov3_"+str(approach_obj.yolo_weight)+".weights")).encode()
    yolo_data = (approach_obj.name+"/darknet/graffiti.data").encode()

    dn.init_net(approach_obj.cpu)
    if approach_obj.cpu==False:
        dn.set_gpu(0)
    net = dn.load_net(yolo_model, yolo_weights, 0)
    meta = dn.load_meta(yolo_data)
    
    os.system("rm -rf ./two-phase/predictions")
    os.system("mkdir ./two-phase/predictions")

    # pipeline: object detection
    y_pred_temp = {}
    scale = approach_obj.scale_val if (approach_obj.heuristic=="scale" or approach_obj.heuristic=="baseline") else 0.0
    box_id = 0
    return_dict = defaultdict(dict)
    for idx, img_file in enumerate(imageList):
        dets = dn.detect(net, meta, img_file.encode('utf-8'), thresh=thresh, nms=nms)
        return_list = []
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
            
            if scale > 0.0 and (not approach_obj.scaled_dataset) and (not approach_obj.segment_dataset):
                y1 = max(0,int((1-scale)*y1_unscaled))
                y2 = min(int((y2_unscaled)*(1+scale)),img.shape[0])
                x1 = max(0,int((1-scale)*x1_unscaled))
                x2 = min(int((x2_unscaled)*(1+scale)),img.shape[1])
            else:
                y1 = y1_unscaled
                y2 = y2_unscaled
                x1 = x1_unscaled
                x2 = x2_unscaled

            crop_img = img[y1:y2,x1:x2]
            crop_img_name = os.path.basename(img_file).rsplit('.',1)
            crop_img_name = crop_img_name[0]+"_predicted_crop_"+str(int(box_id))+"."+crop_img_name[1]
            crop_img_file = os.path.abspath(os.path.join("./two-phase/predictions",crop_img_name))
            
            if crop_img.size > 0:
                pred = box_id
                cv2.imwrite(crop_img_file,crop_img)
                if approach_obj.heuristic == "segment" and (not approach_obj.segment_dataset):
                    return_list.append(np.array([box_id,pred,bbox[1],x1_unscaled,y1_unscaled,x2_unscaled,y2_unscaled])) 
                else:
                    return_list.append([crop_img_file,[x1_unscaled,y1_unscaled,x2_unscaled,y2_unscaled],
                                    np.array([box_id,pred,bbox[1],x1_unscaled,y1_unscaled,x2_unscaled,y2_unscaled])])
                box_id += 1
        
        if approach_obj.heuristic == "segment":
            return_dict[img_file] = np.array(return_list)
        else:
            return_dict[img_file] = return_list
    
    print("Detection done")
    if approach_obj.heuristic == "segment":
        if not approach_obj.segment_dataset:
            from src.deeplab_segmentation import Segmentor
            s = 0.0 if approach_obj.scaled_dataset else approach_obj.scale_val
            mrcnn_segmentor = Segmentor()
            return_dict = mrcnn_segmentor.test(return_dict, scale=s)

    return_dict = classify_detections(imageList, return_dict, box_id, approach_obj=approach_obj)
    

    return return_dict