import argparse

# import for caffe
## import caffe first, always. yolo loads a different(old) version of <forgot name> library, which is not supported by caffe
import os,sys
import time
import numpy as np
try:
    caffe_root = "/home/ubuntu/caffe-0.15.9/"
    sys.path.insert(0, caffe_root + 'python')
    os.environ['GLOG_minloglevel'] = '2' 
    import caffe
    from src.deploy_street import CaffePredictor
    caffe_support = True
except ImportError as e:
    print("caffe not installed, disable two-phase approach")
    caffe_support = False

# imports for darknet
import cv2
from collections import defaultdict
from src.approach import Approach
from src.one_phase import *
from src.two_phase import *
import xml.etree.ElementTree as ET


def deploy_func(input_file, thresh, nms, approach_obj):
    """main function to deploy road damage detection model
    
    This function
    1) loads the darknet model based on the approach, architecture suppport
    2) calls appropriate functions to get predictions
    3) saves predictions in required format
    
    Args:
        input_file (str): contains list of image paths that needs detection.
        thresh (float): threshold value for detector
        nms (float): nms threshold value
        approach_obj (object): stores information about the experiment
    """
    print("using scale value (only if applicable): ",approach_obj.scale_val)

    #list of test images
    with open(input_file,"r") as f:
        lines = f.readlines()

    imageList = [line.strip() for line in lines]

    csvFile = open("output/sample_submission.csv","w")
    if approach_obj.name=="one-phase":
        pred_dict = deploy_func_one_phase(imageList, thresh, nms, approach_obj)
    elif approach_obj.name=="two-phase":
        if caffe_support == True:
            pred_dict = deploy_func_two_phase(imageList, thresh, nms, approach_obj)
        else:
            print("Caffe not supported on this system, can't run two-phase approach.")
            return
    else:
        assert(0)

    #save predictions
    for img_file in pred_dict:
        csvFile.write(os.path.basename(img_file)+",")
        for bbox in pred_dict[img_file]:
            csvFile.write(str(int(bbox[1]))+" "+str(int(bbox[3]))+" "+str(int(bbox[4]))+" "+str(int(bbox[5]))+" "+str(int(bbox[6]))+" ")
        csvFile.write("\n")
    csvFile.close()
    
    os.system("rm -rf "+os.path.join(approach_obj.name,"predicted"))
    os.system("mkdir "+os.path.join(approach_obj.name,"predicted"))
    for img_file in pred_dict:
        name = os.path.join(approach_obj.name,"predicted",os.path.basename(img_file).rsplit('.',1)[0]+".txt")
        with open(os.path.join(name),"w") as f:
            for bbox in pred_dict[img_file]:
                f.write(str(approach_obj.id_to_name[int(bbox[1])])+" "+str(bbox[2])+" "+str(int(bbox[3]))+" "+str(int(bbox[4]))+" "+str(int(bbox[5]))+" "+str(int(bbox[6]))+"\n")

    return pred_dict
                
def main():    
    parser = argparse.ArgumentParser(description='run phase2.')
    parser.add_argument('--approach', type=str,
                        help='name of the approach ["one-phase","two-phase","ensemble"]',
                        default='one-phase', choices=["one-phase","two-phase","ensemble"])
    parser.add_argument('--heuristic', type=str, help='input to classifier in two-phase ["base","scale","segment","baseline"]',
                        default='base', choices=["base","scale","segment","baseline"])
    parser.add_argument('--num-classes', type=int, help='number of classes in dataset',default=25, choices=[25,18])
    parser.add_argument('--yolo', type=int, help='yolo iteration number for weights',default=40000)
    parser.add_argument('--surface-caffe-weights', type=int, help='caffe iteration number for weights of surface classifer', default=0)
    parser.add_argument('--nms', type=float, help='nms threshold value', default=0.45)
    parser.add_argument('--thresh', type=float, help='threshold value for detector', default=0.5)
    parser.add_argument('--gpu', type=bool, help='want to run on GPU?', default=True)
    parser.add_argument('--input-file', type=str, help='location to the input list of test images',default='test_25c.txt')
    parser.add_argument('--scaled-dataset', help='using scaled dataset', action="store_true", default=False)
    parser.add_argument('--segment-dataset', help='using segment dataset', action="store_true", default=False)
    parser.add_argument('--scale-val', type=float, help='scale value',default=0.3)
    args = parser.parse_args()
    
    app = args.approach
    heuristic = args.heuristic
    num_classes = args.num_classes
    y = args.yolo
    t = args.thresh
    nms = args.nms
    cpu = not args.gpu
    test_file = args.input_file
    
    
    t1 = time.time()
    approach = Approach(args)
    pred_dict = deploy_func(test_file,t,nms,approach)    
    t2 = time.time()
    print("Time taken to run this script: {} s".format(t2-t1))

if __name__=="__main__":
    main()
