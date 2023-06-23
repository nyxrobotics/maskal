'''
Resized cocodataset
'''
import albumentations as A
import cv2
import json
import numpy as np
import copy
import os
import matplotlib.pyplot as plt
from glob import glob
import pandas as pd
from tqdm import tqdm

main_path = "dataset_coco\\"
image_folder_path = main_path 
json_path = main_path +"annotations.json"
save_path = main_path + 'resized\\'
converted_json_path = save_path + "annotations_resized.json"
image_save_path = save_path + "PNGImages/"
size = (360,960)
H,W = size


##############################################################################
################################FUNCTIONS#####################################
def func_image_name(path_image):
    """
    Return the name of image 
    """
    basename = os.path.basename(path_image)
    name, ext = os.path.splitext(basename)
    return name

def pad_image(img, aspect_width=960, aspect_height=960, constant_values = 0):
  """
  Paddind with zeros the image
  """  
  img_width = img.shape[1]
  img_height = img.shape[0]

  original_w2h_ratio = float(img_width) / img_height
  target_w2h_ratio = float(aspect_width) / aspect_height
  is_skinny_image = original_w2h_ratio < target_w2h_ratio

  if is_skinny_image:
    dx = int((target_w2h_ratio * img_height) - img_width)
    pad = ((0, 0), (0, dx))    
    if(img.ndim == 3):      
      img = np.stack([np.pad(img[:,:,c], pad, mode='constant', constant_values=constant_values) for c in range(3)], axis=2)
    else:
      img = np.pad(img[:,:], pad, mode='constant', constant_values=constant_values)
  else:
    dy = int((img_width / target_w2h_ratio) - img_height)
    pad = ((0, dy), (0, 0))
    if(img.ndim == 3):
      img = np.stack([np.pad(img[:,:,c], pad, mode='constant', constant_values=constant_values) for c in range(3)], axis=2)
    else:
      img = np.pad(img[:,:], pad, mode='constant', constant_values=constant_values)
  return img

##############################################################################
################################ MAIN ########################################

#open annotation cocodataset.json
with open(json_path) as json_file:
    json_data = json.loads(json_file.read())
    json_file.close()
#copy json
tjson = copy.deepcopy(json_data)
#define transformation
transform = A.Compose([A.Resize(height =H,width=W, interpolation = cv2.INTER_CUBIC)], 
                      keypoint_params=A.KeypointParams(format='xy'),
                      bbox_params = A.BboxParams(format='coco',label_fields=['class_labels'])) 
n_images = json_data['annotations'][-1]['image_id'] + 1
n_seg = len(json_data['annotations'])
id_seg = 0
for id_image in tqdm(range(0,n_images), total=len(range(0,n_images))):
    file_name = json_data['images'][id_image]['file_name']
    image_name = func_image_name(file_name)
    image = cv2.imread(image_folder_path + 'PNGImages\\' +  image_name + '.png')  
    list_xy = []
    bboxes = []   
    class_labels = []
    len_segs = []
    for i in range(0,n_seg): 
        if json_data['annotations'][i]['image_id'] == id_image:
            len_segs.append(len(json_data['annotations'][i]['segmentation'][0]))           
            x = json_data['annotations'][i]['segmentation'][0][0::2]
            y = json_data['annotations'][i]['segmentation'][0][1::2]
            bboxes.append(json_data['annotations'][i]['bbox'])
            class_labels.append('nouse')            
            for i in range(len(x)):
                list_xy.append((x[i],y[i]))
    keypoints = list_xy
    transformed = transform(image = image, keypoints=keypoints,bboxes=bboxes,class_labels=class_labels)
    timage = transformed["image"]
    timage = pad_image(timage)
    tkeypoints = transformed["keypoints"]
    tkbboxes = transformed["bboxes"]    
    #change news annotations
    tjson['images'][id_image]['file_name'] = 'PNGImages/' + image_name + '.png'
    tjson['images'][id_image]['height'] = timage.shape[0]
    tjson['images'][id_image]['width'] = timage.shape[1]    
    s = 0    
    for n,len_seg in enumerate(len_segs):
        e = int(len_seg/2) + s 
        txys = list(tkeypoints[s:e])
        new_segmentation = []
        for txy in txys:
            new_segmentation.append(txy[0])
            new_segmentation.append(txy[1])        
        tjson['annotations'][id_seg]['segmentation'] = [new_segmentation] 
        tjson['annotations'][id_seg]['bbox'] = list(tkbboxes[n])
        id_seg += 1 
        s = e
    #save new coco annotations
    with open(converted_json_path, 'w') as fp:
        json.dump(tjson, fp)
    cv2.imwrite(image_save_path+image_name + '.png' ,timage)