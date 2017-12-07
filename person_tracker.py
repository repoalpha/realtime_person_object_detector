
# coding: utf-8

# # Object Detection Demo
# Welcome to the object detection inference walkthrough!  This notebook will walk you step by step through the process of using a pre-trained model to detect objects in an image.

# # Imports

# In[15]:


import numpy as np
import os
import six.moves.urllib as urllib
import sys
import tarfile
import tensorflow as tf
import zipfile

from collections import defaultdict
from io import StringIO
from matplotlib import pyplot as plt
from PIL import Image
import cv2
from skimage.measure import compare_ssim
import imutils
import subprocess

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("CAMERA NOT OPEN ")
    sys.exit()

# ## Env setup

# In[16]:


# This is needed to display the images.

# This is needed since the notebook is stored in the object_detection folder.
sys.path.append("..")


# In[17]:


sys.path


# ## Object detection imports
# Here are the imports from the object detection module.

# In[18]:


from utils import label_map_util

from utils import visualization_utils as vis_util


# # Model preparation 

# ## Variables
# 
# Any model exported using the `export_inference_graph.py` tool can be loaded here simply by changing `PATH_TO_CKPT` to point to a new .pb file.  
# 
# By default we use an "SSD with Mobilenet" model here. See the [detection model zoo](g3doc/detection_model_zoo.md) for a list of other models that can be run out-of-the-box with varying speeds and accuracies.



# What model to download.
MODEL_NAME = 'faster_rcnn_resnet101_coco_11_06_2017' # this model needs GPU for realtime detection I am using a GTX1080 Ti
# MODEL_FILE = MODEL_NAME + '.tar.gz'
# DOWNLOAD_BASE = 'http://download.tensorflow.org/models/object_detection/'

# Path to frozen detection graph. This is the actual model that is used for the object detection.
PATH_TO_CKPT = MODEL_NAME + '/frozen_inference_graph.pb'

# List of the strings that is used to add correct label for each box.
PATH_TO_LABELS = os.path.join('data', 'mscoco_label_map.pbtxt')

NUM_CLASSES = 90


# ## Download Model

'''opener = urllib.request.URLopener()
opener.retrieve(DOWNLOAD_BASE + MODEL_FILE, MODEL_FILE)
tar_file = tarfile.open(MODEL_FILE)
for file in tar_file.getmembers():
    file_name = os.path.basename(file.name)
    if 'frozen_inference_graph.pb' in file_name:
        tar_file.extract(file, os.getcwd())'''


# ## Load a (frozen) Tensorflow model into memory.


detection_graph = tf.Graph()
with detection_graph.as_default():
    od_graph_def = tf.GraphDef()
    with tf.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
        serialized_graph = fid.read()
        od_graph_def.ParseFromString(serialized_graph)
        tf.import_graph_def(od_graph_def, name='')


# ## Loading label map
''' Label maps map indices to category names, so that when our convolution network predicts `5`, we know that this corresponds to `airplane`.  
Here we use internal utility functions, but anything that returns a dictionary mapping integers to appropriate string labels would be fine'''


label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=NUM_CLASSES, use_display_name=True)
category_index = label_map_util.create_category_index(categories)


# ## Helper code

def load_image_into_numpy_array(image):
  (im_width, im_height) = image.size
  return np.array(image.getdata()).reshape(
      (im_height, im_width, 3)).astype(np.uint8)

# # Detection

with detection_graph.as_default():
  with tf.Session(graph=detection_graph) as sess:
      while True:
      
        ret, frame = cap.read()
        _, original = cap.read()

        image_np = (frame)
        # Expand dimensions since the model expects images to have shape: [1, None, None, 3]
        image_np_expanded = np.expand_dims(image_np, axis=0)
        image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')
        # Each box represents a part of the image where a particular object was detected.
        boxes = detection_graph.get_tensor_by_name('detection_boxes:0')
        # Each score represent how level of confidence for each of the objects.
        # Score is shown on the result image, together with the class label.
        scores = detection_graph.get_tensor_by_name('detection_scores:0')
        classes = detection_graph.get_tensor_by_name('detection_classes:0')
        num_detections = detection_graph.get_tensor_by_name('num_detections:0')
        

        # Actual detection.
        (boxes, scores, classes, num_detections) = sess.run(
            [boxes, scores, classes, num_detections],
            feed_dict={image_tensor: image_np_expanded})
        
        
        # Visualization of the results of a detection.
        vis_util.visualize_boxes_and_labels_on_image_array(
            image_np,
            np.squeeze(boxes),
            np.squeeze(classes).astype(np.int32),
            np.squeeze(scores),
            category_index,
            min_score_thresh=.9,
            use_normalized_coordinates=True,
            line_thickness=2)

        
        #print([category_index.get(value) for index,value in enumerate(classes[0]) if scores[0,index] > 0.9])
        for index,value in enumerate(classes[0]):
            if scores[0,index] > 0.9 and value == 1:         
        # This section grabs bounding box coordinates from processed tensorflow 'image_np'
        # [0,0,0] is first ranked box, [0,1,0] is second ranked box, [0,2,0] 3rd etc. if second ranked object is not a person
        # slicing the middle index gives a list of ranked objects
                ymin = boxes[0,:1,0]
                xmin = boxes[0,:1,1]
                ymax = boxes[0,:1,2]
                xmax = boxes[0,:1,3]
                # Change the coordinates of a bounding box from normalized coordinates
                # and convert these to absolute coordinates.
                xminn = np.int_(xmin * 640)
                xmaxx = np.int_(xmax * 640)
                yminn = np.int_(ymin * 480)
                ymaxx = np.int_(ymax * 480)
                
                
                mask = np.zeros(image_np.shape[:2], dtype="uint8")
                
                for i in range(1): # iterate through objects 
                    #cv2.rectangle(original, (xminn[i],yminn[i]), (xmaxx[i],ymaxx[i]), (0,255,0), 2)
                    cv2.rectangle(mask, (xminn[i],yminn[i]), (xmaxx[i],ymaxx[i]), 255, -1)
                        # grabs selected object
                    square = original[yminn[i]:ymaxx[i], xminn[i]: xmaxx[i]]
                    #cv2.rectangle(mask, (xminn2,yminn2), (xmaxx2,ymaxx2), 255, -1)
                    # Apply our mask -- notice how only the person in the image is cropped out
                    masked = cv2.bitwise_and(original, original, mask=mask)
                #flip image vertically so they face the direction of movement    
                flipped = cv2.flip(masked,1)
                cv2.imshow('original', cv2.resize(flipped, (640,480)))
            elif scores[0,index] > 0.9 and value != 1:
                break
        # cv2.imshow('square', cv2.resize(square, (320,240)))
        #cv2.imshow('object detection', cv2.resize(masked2, (640,480)))
        if cv2.waitKey(1) & 0xFF == ord('q'):
          cv2.destroyAllWindows()
          cap.release
          break

sys.exit()





