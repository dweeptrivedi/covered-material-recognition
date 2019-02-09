class Approach(object):
    def __init__(self,args):
        self.name = args.approach
        self.yolo_weight = args.yolo
        self.cpu = not args.gpu
        self.heuristic = args.heuristic
        self.num_classes = args.num_classes
        self.surface_caffe_weight = args.surface_caffe_weights
        self.scaled_dataset = args.scaled_dataset
        self.segment_dataset = args.segment_dataset
        self.scale_val = args.scale_val
        
        #class name to int mapping
        self.name_to_id = {}
        self.id_to_name = {}
        self.class_list = []
        with open("one-phase/darknet/graffiti_"+str(self.num_classes)+".names","r") as f:
            names = f.readlines()
            for i in range(len(names)):
                self.class_list.append(names[i].strip())
                self.name_to_id[names[i].strip()] = i
                self.id_to_name[i] = names[i].strip()
                
        model_dir = "segment_dataset" if self.segment_dataset else self.heuristic
        self.level1_classes = self.class_list
        self.level2_clf = None            
            
        if self.name == "two-phase":
            self.level1_model = os.path.join(
                "two-phase/caffe/models/",model_dir+"_models",
                "train_val_"+str(self.num_classes)+"c.prototxt.test")
            self.level1_weights = os.path.join(
                "two-phase/caffe/weights/",model_dir+"_weights",
                str(self.num_classes)+"c","bvlc_googlenet_iter_"+str(self.surface_caffe_weight)+".caffemodel")
            print("using caffe model: {}".format(self.level1_model))
            print("using caffe weights: {}".format(self.level1_weights))