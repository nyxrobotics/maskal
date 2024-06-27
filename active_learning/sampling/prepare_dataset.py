# @Author: Pieter Blok
# @Date:   2021-03-26 14:30:31
# @Last Modified by:   Pieter Blok
# @Last Modified time: 2023-01-20 13:59:25

import sys
import io
import random
import os, traceback
import numpy as np
import shutil
import cv2
from scipy.interpolate import splprep, splev
from PIL import Image
import json
import xml.etree.cElementTree as ET
import math
import datetime
import time
import json
import zlib
import base64
from tqdm import tqdm
import xmltodict
import logging
from zipfile import ZipFile

from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.modeling import build_model
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog

# Initialize the logging
log_format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s \n'
logger = logging.getLogger(__name__)
logger.setLevel('DEBUG')
file_handler = logging.StreamHandler()
formatter = logging.Formatter(log_format)
file_handler.setFormatter(formatter)
logger.addHandler(file_handler)

supported_cv2_formats = (".bmp", ".dib", ".jpeg", ".jpg", ".jpe", ".jp2", ".png", ".pbm", ".pgm", ".ppm", ".sr", ".ras", ".tiff", ".tif")

def print_exception(e):
    tb = sys.exc_info()[-1]
    stk = traceback.extract_tb(tb, 2)
    fname = stk[-1][2]
    logger.error(f'Error in function {fname}: '+ repr(e))

def list_files(rootdir):
    images = []
    annotations = []

    if os.path.isdir(rootdir):
        for root, dirs, files in list(os.walk(rootdir)):
            for name in files:
                subdir = root.split(rootdir)
                all('' == s for s in subdir)
                
                if subdir[1].startswith('/'):
                    subdirname = subdir[1][1:]
                else:
                    subdirname = subdir[1]

                if name.lower().endswith(supported_cv2_formats):
                    if all('' == s for s in subdir):
                        images.append(name)
                    else:
                        images.append(os.path.join(subdirname, name))

                if name.endswith(".json") or name.endswith(".xml"):
                    if all('' == s for s in subdir):
                        annotations.append(name)
                    else:
                        annotations.append(os.path.join(subdirname, name))
    
        images.sort()
        annotations.sort()

    return images, annotations

def find_tiff_files(traindir, valdir=[], testdir=[], initial_traindir=[]):
    tiff_images = []
    tiff_annotations = []

    all_dirs = [traindir, valdir, testdir, initial_traindir]
    for d in range(len(all_dirs)):
        if all_dirs[d] != []:
            if os.path.isdir(all_dirs[d]):
                for root, dirs, files in list(os.walk(all_dirs[d])):
                    for name in files:
                        subdir = root.split(all_dirs[d])
                        all('' == s for s in subdir)
                        
                        if subdir[1].startswith('/'):
                            subdirname = subdir[1][1:]
                        else:
                            subdirname = subdir[1]

                        if name.lower().endswith(".tiff") or name.lower().endswith(".tif"):
                            if all('' == s for s in subdir):
                                tiff_images.append(os.path.join(all_dirs[d], name))
                            else:
                                tiff_images.append(os.path.join(all_dirs[d], subdirname, name))

                        if name.endswith(".tiff.json") or name.endswith(".tif.json"):
                            if all('' == s for s in subdir):
                                tiff_annotations.append(os.path.join(all_dirs[d], name))
                            else:
                                tiff_annotations.append(os.path.join(all_dirs[d], subdirname, name))
            
                tiff_images.sort()
                tiff_annotations.sort()

    return tiff_images, tiff_annotations

def matching_images_and_annotations(rootdir):
    images = []
    images_basenames = []
    annotations = []
    annotations_basenames = []

    matching_images = []
    matching_annotations = []

    if os.path.isdir(rootdir):
        for root, dirs, files in list(os.walk(rootdir)):
            for name in files:
                subdir = root.split(rootdir)
                all('' == s for s in subdir)
                
                if subdir[1].startswith('/'):
                    subdirname = subdir[1][1:]
                else:
                    subdirname = subdir[1]

                if name.lower().endswith(supported_cv2_formats):
                    if all('' == s for s in subdir):
                        images.append(name)
                        images_basenames.append(os.path.splitext(name)[0])
                    else:
                        images.append(os.path.join(subdirname, name))
                        images_basenames.append(os.path.splitext(os.path.join(subdirname, name))[0])

                if name.endswith(".json") or name.endswith(".xml"):
                    if all('' == s for s in subdir):
                        annotations.append(name)
                        annotations_basenames.append(os.path.splitext(name)[0])
                    else:
                        annotations.append(os.path.join(subdirname, name))
                        annotations_basenames.append(os.path.splitext(os.path.join(subdirname, name))[0])
    
        images.sort()
        images_basenames.sort()
        annotations.sort()
        annotations_basenames.sort()

        matching_images_annotations = list(set(images_basenames) & set(annotations_basenames))
        matching_images = [img for img in images if os.path.splitext(img)[0] in matching_images_annotations]
        matching_annotations = [annot for annot in annotations if os.path.splitext(annot)[0] in matching_images_annotations]

    return matching_images, matching_annotations

def rename_xml_files(annotdir):
    if os.path.isdir(annotdir): 
        all_files = os.listdir(annotdir)
        annotations = [x for x in all_files if ".json" in x or ".xml" in x]
        
        for a in range(len(annotations)):
            annotation = annotations[a]
            if annotation.endswith(".xml"):
                if "item_" in annotation:
                    with open(os.path.join(annotdir, annotation)) as xml_file:
                        data_dict = xmltodict.parse(xml_file.read())
                        xml_file.close()
                        json_data = json.dumps(data_dict)
                        json_data = json.loads(json_data)
                        
                        img_name = json_data['annotation']['filename']
                        file_ext = os.path.splitext(img_name)[1]
                        xml_name = img_name.replace(file_ext, '.xml')

                        os.rename(os.path.join(annotdir, annotation), os.path.join(annotdir, xml_name))

def process_labelme_json(jsonfile, classnames):
    group_ids = []

    with open(jsonfile, 'r') as json_file:
        data = json.load(json_file)
        for p in data['shapes']:
            group_ids.append(p['group_id'])

    only_group_ids = [x for x in group_ids if x is not None]
    unique_group_ids = list(set(only_group_ids))
    no_group_ids = sum(x is None for x in group_ids)
    total_masks = len(unique_group_ids) + no_group_ids

    all_unique_masks = np.zeros(total_masks, dtype = object)

    if len(unique_group_ids) > 0:
        unique_group_ids.sort()

        for k in range(len(unique_group_ids)):
            unique_group_id = unique_group_ids[k]
            all_unique_masks[k] = unique_group_id

        for h in range(no_group_ids):
            all_unique_masks[len(unique_group_ids) + h] = "None" + str(h+1)
    else:
        for h in range(no_group_ids):
            all_unique_masks[h] = "None" + str(h+1)    

    category_ids = []
    masks = []
    crowd_ids = []

    for i in range(total_masks):
        category_ids.append([])
        masks.append([])
        crowd_ids.append([])

    none_counter = 0 

    for p in data['shapes']:
        group_id = p['group_id']

        if group_id is None:
            none_counter = none_counter + 1
            fill_id = int(np.where(np.asarray(all_unique_masks) == (str(group_id) + str(none_counter)))[0][0])
        else:
            fill_id = int(np.where(np.asarray(all_unique_masks) == group_id)[0][0])

        classname = p['label']

        try:
            category_id = int(np.where(np.asarray(classnames) == classname)[0][0] + 1)
            category_ids[fill_id] = category_id
            run_further = True
        except:
            print("Cannot find the class name (please check the annotation files)")
            run_further = False

        if run_further:
            if p['shape_type'] == "circle":
                # https://github.com/wkentaro/labelme/issues/537
                bearing_angles = [0, 15, 30, 45, 60, 75, 90, 105, 120, 135, 150, 165, 
                180, 195, 210, 225, 240, 255, 270, 285, 300, 315, 330, 345, 360]
                            
                orig_x1 = p['points'][0][0]
                orig_y1 = p['points'][0][1]

                orig_x2 = p['points'][1][0]
                orig_y2 = p['points'][1][1]

                cx = (orig_x2 - orig_x1)**2
                cy = (orig_y2 - orig_y1)**2
                radius = math.sqrt(cx + cy)

                circle_polygon = []
            
                for k in range(0, len(bearing_angles) - 1):
                    ad1 = math.radians(bearing_angles[k])
                    x1 = radius * math.cos(ad1)
                    y1 = radius * math.sin(ad1)
                    circle_polygon.append( (orig_x1 + x1, orig_y1 + y1) )

                    ad2 = math.radians(bearing_angles[k+1])
                    x2 = radius * math.cos(ad2)  
                    y2 = radius * math.sin(ad2)
                    circle_polygon.append( (orig_x1 + x2, orig_y1 + y2) )

                pts = np.asarray(circle_polygon).astype(np.float32)
                pts = pts.reshape((-1,1,2))
                points = np.asarray(pts).flatten().tolist()
                
            if p['shape_type'] == "rectangle":
                (x1, y1), (x2, y2) = p['points']
                x1, x2 = sorted([x1, x2])
                y1, y2 = sorted([y1, y2])
                points = [x1, y1, x2, y1, x2, y2, x1, y2]

            if p['shape_type'] == "polygon":
                points = p['points']
                pts = np.asarray(points).astype(np.float32).reshape(-1,1,2)   
                points = np.asarray(pts).flatten().tolist()

                # Ensure polygon points are within image boundaries
                for i in range(0, len(points), 2):
                    if points[i] < 0 or points[i] >= width:
                        logger.warning(f"X-coordinate {points[i]} is out of bounds. Clamping to image boundary.")
                        points[i] = max(0, min(points[i], width - 1))
                    if points[i+1] < 0 or points[i+1] >= height:
                        logger.warning(f"Y-coordinate {points[i+1]} is out of bounds. Clamping to image boundary.")
                        points[i+1] = max(0, min(points[i+1], height - 1))

            masks[fill_id].append(points)

            ## labelme version 4.5.6 does not have a crowd_id, so fill it with zeros
            crowd_ids[fill_id] = 0
            status = "successful"
        else:
            status = "unsuccessful"

    return category_ids, masks, crowd_ids, status

def process_darwin_json(jsonfile, classnames):
    
    with open(jsonfile, 'r') as json_file:
        data = json.load(json_file)

    total_masks = len(data['annotations'])
    category_ids = []
    masks = []
    crowd_ids = []

    for i in range(total_masks):
        category_ids.append([])
        masks.append([])
        crowd_ids.append([])

    fill_id = 0 

    for p in data['annotations']:
        classname = p['name']

        try:
            category_id = int(np.where(np.asarray(classnames) == classname)[0][0] + 1)
            category_ids[fill_id] = category_id
            run_further = True
        except:
            print("Cannot find the class name (please check the annotation files)")
            run_further = False

        if run_further:
            if 'polygon' in p:
                if 'path' in p['polygon']:
                    points = []
                    path_points = p['polygon']['path']
                    for h in range(len(path_points)):
                        points.append(path_points[h]['x'])
                        points.append(path_points[h]['y'])

                    masks[fill_id].append(points)

            if 'complex_polygon' in p:
                if 'path' in p['complex_polygon']:
                    for k in range(len(p['complex_polygon']['path'])):
                        points = []
                        path_points = p['complex_polygon']['path'][k]
                        for h in range(len(path_points)):
                            points.append(path_points[h]['x'])
                            points.append(path_points[h]['y'])

                        masks[fill_id].append(points)
                    
            crowd_ids[fill_id] = 0
            status = "successful"
        else:
            status = "unsuccessful"

        fill_id += 1

    return category_ids, masks, crowd_ids, status

def process_cvat_xml(xmlfile, classnames):
    group_ids = []

    with open(xmlfile) as xml_file:
        data_dict = xmltodict.parse(xml_file.read())
        xml_file.close()
        data = json.dumps(data_dict)
        data = json.loads(data)

    group_ids = []

    ## if 'object' is a list, then there are multiple masks
    if isinstance(data['annotation']['object'], list):
        for q in range(len(data['annotation']['object'])):
            obj = data['annotation']['object'][q]

            if obj['parts']['ispartof'] is not None:
                group_ids.append(int(obj['parts']['ispartof']))
            else:
                group_ids.append(int(obj['id']))

    ## if 'object' is a dict, then it only has 1 mask  
    if isinstance(data['annotation']['object'], dict):
        obj = data['annotation']['object']
        group_ids.append(int(obj['id']))            
            
    unique_group_ids = list(set(group_ids))
    unique_group_ids.sort()
    total_masks = len(unique_group_ids)        

    category_ids = []
    masks = []
    crowd_ids = []

    for i in range(total_masks):
        category_ids.append([])
        masks.append([])
        crowd_ids.append([])

    ## if 'object' is a list, then there are multiple masks
    if isinstance(data['annotation']['object'], list):
        for p in range(len(data['annotation']['object'])):
            obj = data['annotation']['object'][p]
            fid = group_ids[p]
            fill_id = unique_group_ids.index(fid)
            classname = obj['name']

            try:
                category_id = int(np.where(np.asarray(classnames) == classname)[0][0] + 1)
                category_ids[fill_id] = category_id
                run_further = True
            except:
                print("Cannot find the class name (please check the annotation files)")
                run_further = False

            if run_further:
                if 'polygon' in obj:
                    if 'pt' in obj['polygon']:
                        points = []
                        path_points = obj['polygon']['pt']
                        for h in range(len(path_points)):
                            points.append(float(path_points[h]['x']))
                            points.append(float(path_points[h]['y']))

                        # Ensure polygon points are within image boundaries
                        for i in range(0, len(points), 2):
                            if points[i] < 0 or points[i] >= width:
                                logger.warning(f"X-coordinate {points[i]} is out of bounds. Clamping to image boundary.")
                                points[i] = max(0, min(points[i], width - 1))
                            if points[i+1] < 0 or points[i+1] >= height:
                                logger.warning(f"Y-coordinate {points[i+1]} is out of bounds. Clamping to image boundary.")
                                points[i+1] = max(0, min(points[i+1], height - 1))

                        masks[fill_id].append(points)

                crowd_ids[fill_id] = 0
                status = "successful"
            else:
                status = "unsuccessful"

    ## if 'object' is a dict, then it only has 1 mask  
    if isinstance(data['annotation']['object'], dict):
        obj = data['annotation']['object']
        fill_id = 0
        classname = obj['name']

        try:
            category_id = int(np.where(np.asarray(classnames) == classname)[0][0] + 1)
            category_ids[fill_id] = category_id
            run_further = True
        except:
            print("Cannot find the class name (please check the annotation files)")
            run_further = False

        if run_further:
            if 'polygon' in obj:
                if 'pt' in obj['polygon']:
                    points = []
                    path_points = obj['polygon']['pt']
                    for h in range(len(path_points)):
                        points.append(float(path_points[h]['x']))
                        points.append(float(path_points[h]['y']))

                    # Ensure polygon points are within image boundaries
                    for i in range(0, len(points), 2):
                        if points[i] < 0 or points[i] >= width:
                            logger.warning(f"X-coordinate {points[i]} is out of bounds. Clamping to image boundary.")
                            points[i] = max(0, min(points[i], width - 1))
                        if points[i+1] < 0 or points[i+1] >= height:
                            logger.warning(f"Y-coordinate {points[i+1]} is out of bounds. Clamping to image boundary.")
                            points[i+1] = max(0, min(points[i+1], height - 1))

                    masks[fill_id].append(points)

            crowd_ids[fill_id] = 0
            status = "successful"
        else:
            status = "unsuccessful"

    return category_ids, masks, crowd_ids, status

def mkdir_supervisely(img_dir, write_dir, supervisely_meta_json):
    out_dir =  os.path.join(write_dir, "out_supervisely")
    if os.path.exists(out_dir):
        shutil.rmtree(out_dir)
        
    ds_dir = os.path.join(out_dir, "ds0")
    out_img_dir =  os.path.join(ds_dir, "img")
    out_ann_dir =  os.path.join(ds_dir, "ann")

    dirs = [out_dir, ds_dir, out_img_dir, out_ann_dir]
    for d in range(len(dirs)):
        if not os.path.exists(dirs[d]):
            os.mkdir(dirs[d])

    out_meta_path = os.path.join(out_dir, "meta.json")
    shutil.copy(supervisely_meta_json, out_meta_path)
    files = os.listdir(img_dir)

    for cur_file in files:
        if cur_file.lower().endswith(supported_cv2_formats):
            img_path = os.path.join(img_dir, cur_file)
            out_img_path = os.path.join(out_img_dir, cur_file)
            if not os.path.exists(out_img_path):
                shutil.copy(img_path, out_img_path)

    print("Output folder:", out_dir)
    return out_ann_dir  

def process_supervisely_json(jsonfile, classnames):
    with open(jsonfile, 'r') as json_file:
        data = json.load(json_file)

    total_masks = len(data['objects'])
    category_ids = []
    masks = []
    crowd_ids = []

    for i in range(total_masks):
        category_ids.append([])
        masks.append([])
        crowd_ids.append([])

    fill_id = 0 

    for p in data['objects']:
        classname = p['classTitle']

        try:
            category_id = int(np.where(np.asarray(classnames) == classname)[0][0] + 1)
            category_ids[fill_id] = category_id
            run_further = True
        except:
            print("Cannot find the class name (please check the annotation files)")
            run_further = False

        if run_further:
            if p['geometryType'] == 'bitmap':
                if 'data' in p['bitmap']:
                    ## https://legacy.docs.supervise.ly/ann_format/
                    final_mask = np.zeros((data['size']['height'], data['size']['width']), dtype=bool)
                    z = zlib.decompress(base64.b64decode(p['bitmap']['data']))
                    n = np.fromstring(z, np.uint8)
                    mask = cv2.imdecode(n, cv2.IMREAD_UNCHANGED)[:, :, 3].astype(bool)
                    ox = p['bitmap']['origin'][0]
                    oy = p['bitmap']['origin'][1]
                    final_mask[oy:oy+mask.shape[0], ox:ox+mask.shape[1]] = mask
                    final_mask = np.multiply(final_mask.astype(np.uint8), 255)

                    num_labels, labels, stats, _  = cv2.connectedComponentsWithStats(final_mask)
                    region_areas = []

                    for label in range(1,num_labels):
                        region_areas.append(stats[label, cv2.CC_STAT_AREA])

                    # procedure to remove the very small masks
                    if len(region_areas) > 1:
                        for w in range(len(np.asarray(region_areas))):
                            region_area = np.asarray(region_areas)[w]
                            if region_area < 50:
                                final_mask = np.where(labels==(w+1), 0, final_mask)

                    contours, _ = cv2.findContours(final_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                    for cnt in range(len(contours)):
                        contour = contours[cnt]
                        contour = contour.reshape(contour.shape[0], contour.shape[-1])
                        points = []
                        for h in range(len(contour)):
                            points.append(int(contour[h][0]))
                            points.append(int(contour[h][1]))
                        masks[fill_id].append(points)

            if p['geometryType'] == 'polygon':
                exterior_points = p['points']['exterior']
                points = []
                for h in range(len(exterior_points)):
                    points.append(exterior_points[h][0])
                    points.append(exterior_points[h][1])

                    # Ensure polygon points are within image boundaries
                    if points[-2] < 0 or points[-2] >= width:
                        logger.warning(f"X-coordinate {points[-2]} is out of bounds. Clamping to image boundary.")
                        points[-2] = max(0, min(points[-2], width - 1))
                    if points[-1] < 0 or points[-1] >= height:
                        logger.warning(f"Y-coordinate {points[-1]} is out of bounds. Clamping to image boundary.")
                        points[-1] = max(0, min(points[-1], height - 1))

                masks[fill_id].append(points)

            crowd_ids[fill_id] = 0
            status = "successful"
        else:
            status = "unsuccessful"

        fill_id += 1

    return category_ids, masks, crowd_ids, status

def bounding_box(masks):
    areas = []
    boxes = []

    for _ in range(len(masks)):
        areas.append([])
        boxes.append([])

    for i in range(len(masks)):
        points = masks[i]
        all_points = np.concatenate(points)

        pts = np.asarray(all_points).astype(np.float32).reshape(-1,1,2)
        bbx,bby,bbw,bbh = cv2.boundingRect(pts)

        area = bbw*bbh 
        areas[i] = area                      
        boxes[i] = [bbx,bby,bbw,bbh]

    return areas, boxes

def visualize_annotations(img, category_ids, masks, boxes, classes):
    colors = [(0, 255, 0), (255, 0, 0), (255, 0, 255), (0, 0, 255), (0, 255, 255), (255, 255, 255)]
    color_list = np.remainder(np.arange(len(classes)), len(colors))
    
    font_face = cv2.FONT_HERSHEY_DUPLEX
    font_scale = 1
    font_thickness = 1
    thickness = 3
    text_color1 = [255, 255, 255]
    text_color2 = [0, 0, 0]

    img_vis = img.copy()

    for i in range(len(masks)):
        points = masks[i]
        bbx,bby,bbw,bbh = boxes[i]
        category_id = category_ids[i]
        class_id = category_id-1
        _class = classes[class_id]
        color = colors[color_list[class_id]]

        for j in range(len(points)):
            point_set = points[j]
            pntset = np.asarray(point_set).astype(np.int32).reshape(-1,1,2) 
            img_vis = cv2.polylines(img_vis, [pntset], True, color, thickness)

        img_vis = cv2.rectangle(img_vis, (bbx, bby), ((bbx+bbw), (bby+bbh)), color, thickness)

        text_str = "{:s}".format(_class)
        text_w, text_h = cv2.getTextSize(text_str, font_face, font_scale, font_thickness)[0]

        if bby < 100:
            text_pt = (bbx, bby+bbh)
        else:
            text_pt = (bbx, bby)

        img_vis = cv2.rectangle(img_vis, (text_pt[0], text_pt[1] + 7), (text_pt[0] + text_w, text_pt[1] - text_h - 7), text_color1, -1)
        img_vis = cv2.putText(img_vis, text_str, (text_pt[0], text_pt[1]), font_face, font_scale, text_color2, font_thickness, cv2.LINE_AA)

    return img_vis

def visualize_mrcnn(img_np, classes, scores, masks, boxes, class_names):
    masks = masks.astype(dtype=np.uint8)
    font_scale = 0.6
    font_thickness = 1
    text_color = [0, 0, 0]

    if masks.any():
        maskstransposed = masks.transpose(1,2,0) # transform the mask in the same format as the input image array (h,w,num_dets)
        red_mask = np.zeros((maskstransposed.shape[0],maskstransposed.shape[1]),dtype=np.uint8)
        blue_mask = np.zeros((maskstransposed.shape[0],maskstransposed.shape[1]),dtype=np.uint8)
        green_mask = np.zeros((maskstransposed.shape[0],maskstransposed.shape[1]),dtype=np.uint8)
        all_masks = np.zeros((maskstransposed.shape[0],maskstransposed.shape[1],3),dtype=np.uint8) # BGR

        colors = [(0, 255, 0), (255, 0, 0), (255, 0, 255), (0, 0, 255), (0, 255, 255), (255, 255, 255)]
        color_list = np.remainder(np.arange(len(class_names)), len(colors))
        imgcopy = img_np.copy()

        for i in range (maskstransposed.shape[-1]):
            color = colors[color_list[classes[i]]]
            x1, y1, x2, y2 = boxes[i, :]
            cv2.rectangle(imgcopy, (int(x1), int(y1)), (int(x2), int(y2)), color, 1)

            _class = class_names[classes[i]]
            text_str = '%s: %.2f' % (_class, scores[i])

            font_face = cv2.FONT_HERSHEY_DUPLEX

            text_w, text_h = cv2.getTextSize(text_str, font_face, font_scale, font_thickness)[0]
            text_pt = (int(x1), int(y1) - 3)

            cv2.rectangle(imgcopy, (int(x1), int(y1)), (int(x1) + text_w, int(y1) - text_h - 4), color, -1)
            cv2.putText(imgcopy, text_str, text_pt, font_face, font_scale, text_color, font_thickness, cv2.LINE_AA)

            mask = maskstransposed[:,:,i]

            if colors.index(color) == 0: # green
                green_mask = cv2.add(green_mask,mask)
            elif colors.index(color) == 1: # blue
                blue_mask = cv2.add(blue_mask,mask)
            elif colors.index(color) == 2: # magenta
                blue_mask = cv2.add(blue_mask,mask)
                red_mask = cv2.add(red_mask,mask)
            elif colors.index(color) == 3: # red
                red_mask = cv2.add(red_mask,mask)
            elif colors.index(color) == 4: # yellow
                green_mask = cv2.add(green_mask,mask)
                red_mask = cv2.add(red_mask,mask)
            else: #white
                blue_mask = cv2.add(blue_mask,mask)
                green_mask = cv2.add(green_mask,mask)
                red_mask = cv2.add(red_mask,mask)

        all_masks[:,:,0] = blue_mask
        all_masks[:,:,1] = green_mask
        all_masks[:,:,2] = red_mask
        all_masks = np.multiply(all_masks,255).astype(np.uint8)

        img_mask = cv2.addWeighted(imgcopy,1,all_masks,0.5,0)
    else:
        img_mask = img_np

    return img_mask

def write_file(imgdir, images, name):
    with open(os.path.join(imgdir, "{:s}.txt".format(name)), 'w') as f:
        for img in images:
            f.write("{:s}\n".format(img))

def split_datasets_randomly(rootdir, images, train_val_test_split, initial_datasize):
    all_ids = np.arange(len(images))
    random.shuffle(all_ids)

    train_slice = int(train_val_test_split[0]*len(images))
    val_slice = int(train_val_test_split[1]*len(images))

    train_ids = all_ids[:train_slice]
    val_ids = all_ids[train_slice:train_slice+val_slice]
    test_ids = all_ids[train_slice+val_slice:]

    train_images = np.array(images)[train_ids].tolist()
    initial_train_images = random.sample(train_images, initial_datasize)
    val_images = np.array(images)[val_ids].tolist()
    test_images = np.array(images)[test_ids].tolist()

    write_file(rootdir, train_images, "train")
    write_file(rootdir, initial_train_images, "initial_train")
    write_file(rootdir, val_images, "val")
    write_file(rootdir, test_images, "test") 
    
    return [initial_train_images, val_images, test_images], ["train", "val", "test"]

def create_json(rootdir, imgdir, images, classes, name):
    date_created = datetime.datetime.now()
    year_created = date_created.year

    ## initialize the final json file
    writedata = {}
    writedata['info'] = {"description": "description", "url": "url", "version": str(1), "year": str(year_created), "contributor": "contributor", "date_created": str(date_created)}
    writedata['licenses'] = []
    writedata['licenses'].append({"url": "license_url", "id": "license_id", "name": "license_name"})
    writedata['images'] = []
    writedata['type'] = "instances"
    writedata['annotations'] = []
    writedata['categories'] = []

    for k in range(len(classes)):
        superclass = classes[k]
        writedata['categories'].append({"supercategory": superclass, "id": (k+1), "name": superclass})

    annotation_id = 1   ## see: https://github.com/cocodataset/cocoapi/issues/507
    output_file = name + ".json"

    print("")
    print(output_file)

    for j in tqdm(range(len(images))):
        imgname = images[j]
        img = cv2.imread(os.path.join(imgdir, imgname))

        height, width = img.shape[:2]

        try:
            modTimesinceEpoc = os.path.getmtime(os.path.join(imgdir, imgname))
            modificationTime = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(modTimesinceEpoc))
            date_modified = modificationTime
        except:
            date_modified = None

        basename, fileext = os.path.splitext(imgname)
        base_name = basename.split(fileext)
        bn = base_name[0]

        write = False
        file_is_json = False
        file_is_xml = False

        if os.path.exists(os.path.join(imgdir, bn + ".json")):
            file_is_json = True
            annot_filename = os.path.join(imgdir, bn + ".json")

        ## supervisely annotation files include the file extension of the image
        if os.path.exists(os.path.join(imgdir, imgname + ".json")):
            file_is_json = True
            annot_filename = os.path.join(imgdir, imgname + ".json")
            
        if os.path.exists(os.path.join(imgdir, bn + ".xml")):
            file_is_xml = True
            annot_filename = os.path.join(imgdir, bn + ".xml")

        if file_is_json:
            with open(annot_filename, 'r') as json_file:
                try:
                    data = json.load(json_file)

                    ## labelme
                    if 'version' in data and 'shapes' in data:
                        if len(data['shapes']) > 0:
                            annot_format = 'labelme'
                            write = True

                    ## v7-darwin                
                    if 'annotations' in data:
                        if len(data['annotations']) > 0:
                            annot_format = 'darwin'
                            write = True

                    ## supervisely               
                    if 'objects' in data:
                        if len(data['objects']) > 0:
                            bitmap_or_poly = np.zeros(len(data['objects']), dtype=bool)
                            for ob in range(len(data['objects'])):
                                obj = data['objects'][ob]
                                if obj['geometryType'] == "bitmap" or obj['geometryType'] == "polygon":
                                    bitmap_or_poly[ob] = True

                        if all(bitmap_or_poly): 
                            annot_format = 'supervisely'
                            write = True
                        else:
                            logger.error("Only bitmap-annotations or polygon-annotations are supported for Supervisely")
                except:
                    continue

        if file_is_xml:
            with open(annot_filename) as xml_file:
                data_dict = xmltodict.parse(xml_file.read())
                xml_file.close()
                data = json.dumps(data_dict)
                data = json.loads(data)

                ## cvat-labelme3.0 xml files
                if 'annotation' in data:
                    if 'object' in data['annotation']:
                        annot_format = 'cvat'
                        write = True
            
        
        if write:
            writedata['images'].append({
                            'license': 0,
                            'url': None,
                            'file_name': imgname,
                            'height': height,
                            'width': width,
                            'date_captured': None,
                            'id': j
                        })

            # Procedure to store the annotations in the final JSON file
            if annot_format == 'labelme':
                category_ids, masks, crowd_ids, status = process_labelme_json(annot_filename, classes)
            
            if annot_format == 'darwin':
                category_ids, masks, crowd_ids, status = process_darwin_json(annot_filename, classes)

            if annot_format == 'cvat':
                category_ids, masks, crowd_ids, status = process_cvat_xml(annot_filename, classes)

            if annot_format == 'supervisely':
                category_ids, masks, crowd_ids, status = process_supervisely_json(annot_filename, classes)

            areas, boxes = bounding_box(masks)
            img_vis = visualize_annotations(img, category_ids, masks, boxes, classes)

            for q in range(len(category_ids)):
                category_id = category_ids[q]
                mask = masks[q]
                bb_area = areas[q]
                bbpoints = boxes[q]
                crowd_id = crowd_ids[q]

                writedata['annotations'].append({
                        'id': annotation_id,
                        'image_id': j,
                        'category_id': category_id,
                        'segmentation': mask,
                        'area': bb_area,
                        'bbox': bbpoints,
                        'iscrowd': crowd_id
                    })
        
                annotation_id = annotation_id+1
            
    with open(os.path.join(rootdir, output_file), 'w') as outfile:
        json.dump(writedata, outfile)

def highlight_missing_annotations(annot_folder, cur_annot_diff):
    rename_xml_files(annot_folder)
    images, annotations = list_files(annot_folder)
    img_basenames = [os.path.splitext(img)[0] for img in images]
    
    ## supervisely puts the json extension behind the image extension
    annotation_basenames = [os.path.splitext(os.path.splitext(annot)[0])[0] if os.path.splitext(annot)[0].lower().endswith(supported_cv2_formats) else os.path.splitext(annot)[0] for annot in annotations]
    
    diff_img_annot = []
    for c in range(len(img_basenames)):
        img_basename = img_basenames[c]
        if img_basename not in annotation_basenames:
            diff_img_annot.append(img_basename)
    diff_img_annot.sort()

    if len(diff_img_annot) > 0:
        if len(diff_img_annot) != cur_annot_diff:
            print("Go to the folder {:s}".format(annot_folder))
            print("and annotate the following images:")
            for i in range(len(diff_img_annot)):
                print(diff_img_annot[i])
            cur_annot_diff = len(diff_img_annot)
            print("")
    return diff_img_annot, cur_annot_diff
    
def check_json_presence(config, imgdir, image_list, dataset_type):
    try:
        print(f"\nChecking {dataset_type} annotations...")
        annotations = []
        missing_annotations = []

        for image in image_list:
            base_name = os.path.splitext(image)[0]
            json_file = f"{base_name}.json"
            json_path = os.path.join(imgdir, json_file)

            if not os.path.exists(json_path):
                print(f"Warning: Annotation file {json_path} not found for image {image}")
                missing_annotations.append(image)
            else:
                with open(json_path, 'r') as f:
                    data = json.load(f)
                    annotations.append(data)

        if len(annotations) == 0:
            raise IndexError("No annotations found")

        print(f"{len(annotations)} annotations found out of {len(image_list)} images.")
        if len(missing_annotations) > 0:
            print(f"Missing annotations for {len(missing_annotations)} images.")
            for missing in missing_annotations:
                print(f"Missing annotation for image: {missing}")

    except FileNotFoundError as fnf_error:
        logger.error(f"FileNotFoundError: {fnf_error}")
        sys.exit("Closing application")
    except IndexError as ie:
        logger.error(f"IndexError: {ie}")
        sys.exit("Closing application")
    except json.JSONDecodeError as json_error:
        logger.error(f"JSONDecodeError: {json_error}")
        sys.exit("Closing application")
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        sys.exit("Closing application")



def smooth_contours(contours):
    ## thanks to: https://agniva.me/scipy/2016/10/25/contour-smoothing.html
    smoothened = []

    for contour in contours:
        x,y = contour.T
        x = x.tolist()[0]
        y = y.tolist()[0]
        tck, u = splprep([x,y], u=None, s=1.0, per=1)
        u_new = np.linspace(u.min(), u.max(), 50) ## maximum number of contour points: 50
        x_new, y_new = splev(u_new, tck, der=0)
        res_array = [[[int(i[0]), int(i[1])]] for i in zip(x_new,y_new)]
        smoothened.append(np.asarray(res_array, dtype=np.int32))

    return smoothened

def write_labelme_annotations(write_dir, basename, class_names, masks, height, width):
    masks = masks.astype(np.uint8)

    if masks.any():
        writedata = {}
        writedata['version'] = "4.5.6"
        writedata['flags'] = {}
        writedata['shapes'] = []
        writename = basename

        md, mh, mw = masks.shape
        maskstransposed = masks.transpose(1,2,0) # transform the mask in the same format as the input image array (h,w,num_dets)
        useful_masks = False

        for i in range (maskstransposed.shape[-1]):
            groupid = 1
            masksel = maskstransposed[:,:,i] # select the individual masks
            class_name = class_names[i]
            contours_unfiltered, hierarchy = cv2.findContours((masksel*255).astype(np.uint8),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
            contours = []
            for cnts in range(len(contours_unfiltered)):
                area = cv2.contourArea(contours_unfiltered[cnts]) 
                if area > 50:
                    contours_lowpoly = cv2.approxPolyDP(contours_unfiltered[cnts], 0.004*cv2.arcLength(contours_unfiltered[cnts], True), True)
                    contours.append(contours_lowpoly)

            try:
                contours = smooth_contours(contours)
            except:
                pass
               
            if len(contours) > 0:
                useful_masks = True
                if len(contours) == 1:
                    segm = np.vstack(contours).squeeze()
                    if segm.ndim > 1:
                        x = [int(segm[idx][0]) for idx in range(len(segm))]
                        y = [int(segm[idx][1]) for idx in range(len(segm))]
                        xy = list(zip(x, y))

                        writedata['shapes'].append({
                            'label': class_name,
                            'line_color': None,
                            'fill_color': None,
                            'points': xy,
                            'group_id': None,
                            'shape_type': "polygon",
                            'flags': {}
                        })

                elif len(contours) > 1:
                    for s in range(len(contours)):
                        cnt = contours[s]
                        segm = np.vstack(cnt).squeeze()
                        if segm.ndim > 1:
                            x = [int(segm[idx][0]) for idx in range(len(segm))]
                            y = [int(segm[idx][1]) for idx in range(len(segm))]
                            xy = list(zip(x, y))

                            writedata['shapes'].append({
                                'label': class_name,
                                'line_color': None,
                                'fill_color': None,
                                'points': xy,
                                'group_id': groupid,
                                'shape_type': "polygon",
                                'flags': {}
                            })

                    groupid = groupid + 1
                        
        writedata['lineColor'] = [0,255,0,128]
        writedata['fillColor'] = [255,0,0,128]
        writedata['imagePath'] = writename
        writedata['imageData'] = None
        writedata['imageHeight'] = height
        writedata['imageWidth'] = width

        jn = os.path.splitext(basename)[0] +'.json'
        if useful_masks:
            with open(os.path.join(write_dir, jn), 'w') as outfile:
                json.dump(writedata, outfile)

def write_darwin_annotations(write_dir, basename, class_names, masks, height, width):
    masks = masks.astype(np.uint8)

    if masks.any():
        writedata = {}
        writedata['dataset'] = "maskAL_export"
        writedata['image'] = {
            "width": width,
            "height": height,
            "original_filename": basename,
            "filename": basename,
            "url": "https://darwin.v7labs.com/",
            "thumbnail_url": "https://darwin.v7labs.com/",
            "path": "/",
            "workview_url": "https://darwin.v7labs.com/"
        }
        writedata['annotations'] = []
        writename = basename

        md, mh, mw = masks.shape
        maskstransposed = masks.transpose(1,2,0) # transform the mask in the same format as the input image array (h,w,num_dets)
        useful_masks = False

        for i in range (maskstransposed.shape[-1]):
            masksel = maskstransposed[:,:,i] # select the individual masks
            class_name = class_names[i]
            contours_unfiltered, hierarchy = cv2.findContours((masksel*255).astype(np.uint8),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
            contours = []
            for cnts in range(len(contours_unfiltered)):
                area = cv2.contourArea(contours_unfiltered[cnts])
                if area > 50:
                    contours.append(contours_unfiltered[cnts])

            try:
                contours = smooth_contours(contours)
            except:
                pass
               
            if len(contours) > 0:
                useful_masks = True
                if len(contours) == 1:
                    segm = np.vstack(contours).squeeze()
                    if segm.ndim > 1:
                        x = [float(segm[idx][0]) for idx in range(len(segm))]
                        y = [float(segm[idx][1]) for idx in range(len(segm))]
                        xy = list(zip(x, y))

                        writedata['annotations'].append({
                            'instance_id': {"value": i+1},
                            'name': class_name,
                            'polygon': {"path": []},
                        })

                        for r in range(len(xy)):
                            writedata['annotations'][i]['polygon']["path"].append({"x": xy[r][0], "y": xy[r][1]})

                elif len(contours) > 1:
                    writedata['annotations'].append({
                            'instance_id': {"value": i+1},
                            'name': class_name,
                            'complex_polygon': {"path": []},
                        })
                    vc = 0
                    for s in range(len(contours)):
                        cnt = contours[s]
                        segm = np.vstack(cnt).squeeze()
                        if segm.ndim > 1:
                            writedata['annotations'][i]['complex_polygon']["path"].append([])
                            x = [float(segm[idx][0]) for idx in range(len(segm))]
                            y = [float(segm[idx][1]) for idx in range(len(segm))]
                            xy = list(zip(x, y))

                            for r in range(len(xy)):
                                writedata['annotations'][i]['complex_polygon']["path"][vc].append({"x": xy[r][0], "y": xy[r][1]})
                            vc += 1

        jn = os.path.splitext(basename)[0] +'.json'
        if useful_masks:
            with open(os.path.join(write_dir, jn), 'w') as outfile:
                json.dump(writedata, outfile)

def write_cvat_annotations(write_dir, basename, class_names, masks, height, width):
    masks = masks.astype(np.uint8)

    if masks.any():
        annot = ET.Element("annotation")
        ET.SubElement(annot, "filename").text = basename
        ET.SubElement(annot, "folder")

        source = ET.SubElement(annot, "source")
        ET.SubElement(source, "sourceImage")
        ET.SubElement(source, "sourceAnnotation").text = "Datumaro"

        imagesize = ET.SubElement(annot, "imagesize")
        ET.SubElement(imagesize, "nrows").text = str(height).rstrip()
        ET.SubElement(imagesize, "ncols").text = str(width).rstrip()

        md, mh, mw = masks.shape
        maskstransposed = masks.transpose(1,2,0) # transform the mask in the same format as the input image array (h,w,num_dets)
        polygons = 0
        useful_masks = False

        for i in range (maskstransposed.shape[-1]):
            masksel = maskstransposed[:,:,i] # select the individual masks
            class_name = class_names[i]
            contours_unfiltered, hierarchy = cv2.findContours((masksel*255).astype(np.uint8),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
            contours = []
            for cnts in range(len(contours_unfiltered)):
                area = cv2.contourArea(contours_unfiltered[cnts])
                if area > 50:
                    contours.append(contours_unfiltered[cnts])

            try:
                contours = smooth_contours(contours)
            except:
                pass
               
            if len(contours) > 0:
                useful_masks = True
                if len(contours) == 1:
                    xmlobj = ET.SubElement(annot, "object")
                    ET.SubElement(xmlobj, "name").text = class_name
                    ET.SubElement(xmlobj, "deleted").text = str(0).rstrip()
                    ET.SubElement(xmlobj, "verified").text = str(0).rstrip()
                    ET.SubElement(xmlobj, "occluded").text = "no"
                    ET.SubElement(xmlobj, "date")
                    ET.SubElement(xmlobj, "id").text = str(polygons).rstrip()
                    
                    parts = ET.SubElement(xmlobj, "parts")
                    ET.SubElement(parts, "hasparts")
                    ET.SubElement(parts, "ispartof")

                    segm = np.vstack(contours).squeeze()
                    if segm.ndim > 1:                
                        polygon = ET.SubElement(xmlobj, "polygon")
                        for j in range(len(segm)):
                            pt = ET.SubElement(polygon, "pt")
                            ET.SubElement(pt, "x").text = str(segm[j][0]).rstrip()
                            ET.SubElement(pt, "y").text = str(segm[j][1]).rstrip()
                        ET.SubElement(polygon, "username")
                        ET.SubElement(xmlobj, "attributes")    

                        polygons += 1

                elif len(contours) > 1:
                    for s in range(len(contours)):
                        xmlobj = ET.SubElement(annot, "object")
                        ET.SubElement(xmlobj, "name").text = class_name
                        ET.SubElement(xmlobj, "deleted").text = str(0).rstrip()
                        ET.SubElement(xmlobj, "verified").text = str(0).rstrip()
                        ET.SubElement(xmlobj, "occluded").text = "no"
                        ET.SubElement(xmlobj, "date")
                        ET.SubElement(xmlobj, "id").text = str(polygons).rstrip()

                        parts = ET.SubElement(xmlobj, "parts")
                        if s == 0:
                            hasparts_str = ''
                            for ss in range(1, len(contours)):
                                part_id = polygons + ss
                                hasparts_str = hasparts_str + str(part_id) + ","
                            hasparts_str = hasparts_str[:-1]
                            ET.SubElement(parts, "hasparts").text = hasparts_str.rstrip()
                            ET.SubElement(parts, "ispartof")
                            polygon_id = polygons
                        else:
                            ET.SubElement(parts, "hasparts")
                            ET.SubElement(parts, "ispartof").text = str(polygon_id).rstrip()

                        cnt = contours[s]
                        segm = np.vstack(cnt).squeeze()
                        if segm.ndim > 1:     
                            polygon = ET.SubElement(xmlobj, "polygon")
                            for j in range(len(segm)):
                                pt = ET.SubElement(polygon, "pt")
                                ET.SubElement(pt, "x").text = str(segm[j][0]).rstrip()
                                ET.SubElement(pt, "y").text = str(segm[j][1]).rstrip()
                            ET.SubElement(polygon, "username")
                            ET.SubElement(xmlobj, "attributes")

                            polygons += 1

        tree = ET.ElementTree(annot)
        xmln = os.path.splitext(basename)[0] +'.xml'
        if useful_masks:
            tree.write(os.path.join(write_dir, xmln))

def create_zipfile(annot_folder):
    zipObj = ZipFile(os.path.join(annot_folder, 'cvat_annotations.zip'), 'w')
    for file in os.listdir(annot_folder):
        if file.endswith(".xml"):
            zipObj.write(os.path.join(annot_folder, file))
    zipObj.close()

def write_supervisely_annotations(write_dir, basename, class_names, masks, height, width, meta_json):
    with open(meta_json, 'r') as json_file:
        data = json.load(json_file)

    masks = masks.astype(np.uint8)

    if masks.any():
        writedata = {}
        writedata['description'] = ""
        writedata['tags'] = []
        writedata['size'] = {"height": height, "width": width}
        writedata['objects'] = []

        md, mh, mw = masks.shape
        maskstransposed = masks.transpose(1,2,0) # transform the mask in the same format as the input image array (h,w,num_dets)
        useful_masks = False

        for i in range (maskstransposed.shape[-1]):
            masksel = maskstransposed[:,:,i] # select the individual masks
            class_name = class_names[i]

            for cn in range(len(data['classes'])):
                meta_class_name = data['classes'][cn]['title']
                if class_name == meta_class_name:
                    class_id = int(data['classes'][cn]['id'])

            contours, _ = cv2.findContours((masksel*255).astype(np.uint8),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
            cnt = np.concatenate(contours)
            xx,yy,w,h = cv2.boundingRect(cnt)

            time = datetime.datetime.now()
            time_str = str(time)[:-3].replace(' ','T') + "Z"

            num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(masksel, 4, cv2.CV_32S)
            for s in range(1, num_labels):
                area = stats[s, cv2.CC_STAT_AREA]
                if area <= 50:
                    labels = np.where(labels==s, 0, labels)

            masksel = np.where(labels > 0, 1, labels)
               
            if cv2.contourArea(cnt) > 50:
                useful_masks = True
                
                ## https://legacy.docs.supervise.ly/ann_format/
                maskclip = masksel[yy:yy+h, xx:xx+w]
                img_pil = Image.fromarray(np.array(maskclip, dtype=np.uint8))
                img_pil.putpalette([0,0,0,255,255,255])
                bytes_io = io.BytesIO()
                img_pil.save(bytes_io, format='PNG', transparency=0, optimize=0)
                bytes = bytes_io.getvalue()
                bitmap = base64.b64encode(zlib.compress(bytes)).decode('utf-8')

                writedata['objects'].append({
                    'id': i+1,
                    'classId': class_id,
                    'description': "", 
                    'geometryType': "bitmap",
                    'labelerLogin': "auto_annotate",
                    'createdAt': time_str,
                    'updatedAt': time_str,
                    'tags': [],
                    'classTitle': class_name,
                    'bitmap': {'data': bitmap, 'origin': []}
                })

                writedata['objects'][-1]['bitmap']['origin'].append(xx)
                writedata['objects'][-1]['bitmap']['origin'].append(yy)

        jn = basename +'.json'
        if useful_masks:
            with open(os.path.join(write_dir, jn), 'w') as outfile:
                json.dump(writedata, outfile)

def convert_tiffs(tiff_files, tiff_annotations=[]):
    print("\nConverting the tif/tiff images to png")
    for tf in tqdm(range(len(tiff_files))):
        img = cv2.imread(tiff_files[tf])
        basename = os.path.splitext(tiff_files[tf])[0]
        writename = basename + '.png'
        cv2.imwrite(writename, img)
        os.remove(tiff_files[tf])

    print("\nConverting the tif/tiff annotations")
    for ta in tqdm(range(len(tiff_annotations))):
        annot_file = tiff_annotations[ta]
        if annot_file.lower().endswith(".tiff.json"):
            output_json = annot_file.split(".tiff.json")[0] + ".png.json"
        if annot_file.lower().endswith(".tif.json"):
            output_json = annot_file.split(".tif.json")[0] + ".png.json"
        os.rename(annot_file, output_json)

## the function below is heavily inspired by the function "repeat_factors_from_category_frequency" in maskAL/detectron2/data/samplers/distributed_sampler.py
def calculate_repeat_threshold(config, dataset_dicts_train):
    images_with_class_annotations = np.zeros(len(config['classes'])).astype(np.int16)
    for d in range(len(dataset_dicts_train)):
        data_point = dataset_dicts_train[d]
        classes_annot = []
        for k in range(len(data_point["annotations"])):
            classes_annot.append(data_point["annotations"][k]['category_id'])
        unique_classes = list(set(classes_annot))
        for c in unique_classes:
            images_with_class_annotations[c] += 1

    for mc in range(len(config['minority_classes'])):
        minorty_class = config['minority_classes'][mc]
        search_id = config['classes'].index(minorty_class)
        image_count = images_with_class_annotations[search_id]

        try:
            if image_count < min_value:
                min_value = image_count
        except:
            min_value = image_count
    
    repeat_threshold = np.power(config['repeat_factor_smallest_class'], 2) * (min_value / len(dataset_dicts_train))
    repeat_threshold = np.clip(repeat_threshold, 0, 1)
    return float(repeat_threshold)

def calculate_iterations(config, dataset_dicts_train):
    div_factor = math.ceil(len(dataset_dicts_train)/config['step_image_number'])
    if div_factor == 1:
        max_iterations = config['train_iterations_base']
    else:
        max_iterations = config['train_iterations_base'] + ((div_factor - 1) * config['train_iterations_step_size'])
    steps = [int(s * max_iterations) for s in config['step_ratios']]
    return int(max_iterations), steps

def read_train_file(txt_file):
    initial_train_files = []
    txtfile = open(txt_file, 'r')
    lines = txtfile.readlines()
    
    for line in lines:
        if line != "\n":
            full_line = line.strip()
            train_file = full_line.split(",")[0]
            initial_train_files.append(train_file)
    txtfile.close()
    return initial_train_files 

def prepare_all_dataset_randomly(rootdir, imgdir, classes, train_val_test_split, initial_datasize):
    rename_xml_files(imgdir)
    images, annotations = list_files(imgdir)
    print("{:d} images found!".format(len(images)))
    print("{:d} annotations found!".format(len(annotations)))

    datasets, names = split_datasets_randomly(rootdir, images, train_val_test_split, initial_datasize)
    for dataset, name in zip(datasets, names):
        check_json_presence(imgdir, dataset, name)

    print("Converting annotations...")
    for dataset, name in zip(datasets, names):
        create_json(rootdir, imgdir, dataset, classes, name)   

def prepare_complete_dataset(rootdir, classes, traindir, valdir, testdir):
    try:
        for imgdir, name in zip([traindir, valdir, testdir], ['train', 'val', 'test']):
            print("")
            print("Processing {:s}-dataset: {:s}".format(name, imgdir))
            rename_xml_files(imgdir)
            images, annotations = matching_images_and_annotations(imgdir)
            print("{:d} matching images found!".format(len(images)))
            print("{:d} matching annotations found!".format(len(annotations)))

            write_file(rootdir, images, name)
            check_json_presence(imgdir, images, name)
            create_json(rootdir, imgdir, images, classes, name)
    except:
        logger.error("Cannot create complete-dataset")
        sys.exit("Closing application")   


def prepare_initial_dataset_randomly(config):
    try:
        all_annotated_images = []
        datasets = [
            (config['traindir'], 'train'),
            (config['valdir'], 'val'),
            (config['testdir'], 'test')
        ]
        
        for imgdir, name in datasets:
            print(f"\nProcessing {name}-dataset: {imgdir}")
            rename_xml_files(imgdir)
            images, annotations = list_files(imgdir)
            print(f"{len(images)} images found!")
            print(f"{len(annotations)} annotations found!")

            annotated_images = [img for img in images if os.path.exists(os.path.join(imgdir, os.path.splitext(img)[0] + '.json'))]
            print(f"{len(annotated_images)} annotated images found!")
            
            all_annotated_images.extend(annotated_images)

            if name != 'train' and name != 'test' and name != 'val':
                write_file(config['dataroot'], images, name)
                if not check_json_presence(config, imgdir, images, name):
                    logger.warning(f"No annotations found in the {name} set.")
                create_json(config['dataroot'], imgdir, images, config['classes'], name)

        if not all_annotated_images:
            raise ValueError("No annotations found in the initial train set.")

        write_file(config['dataroot'], all_annotated_images, "initial_train")
        if not check_json_presence(config, config['traindir'], all_annotated_images, 'train'):
            raise ValueError("No annotations found in the initial train set.")
        create_json(config['dataroot'], config['traindir'], all_annotated_images, config['classes'], 'train')

    except Exception as e:
        print_exception(e)
        logger.error("Cannot create initial-datasets")
        sys.exit("Closing application")


def prepare_initial_dataset(config):
    try:
        for i, (imgdir, name) in enumerate(zip([config['traindir'], config['initial_train_dir'], config['valdir'], config['testdir']], ['train', 'initial_train', 'val', 'test'])):
            print("")
            print("Processing {:s}-dataset: {:s}".format(name, imgdir))
            rename_xml_files(imgdir)
            images, annotations = list_files(imgdir)
            print("{:d} images found!".format(len(images)))
            print("{:d} annotations found!".format(len(annotations)))
            write_file(config['dataroot'], images, name)

            if i != 0:
                if "train" in name:
                    name = "train"
                check_json_presence(config, imgdir, images, name)
                create_json(config['dataroot'], imgdir, images, config['classes'], name)

    except Exception as e:
        print_exception(e)
        logger.error("Cannot create initial-datasets")
        sys.exit("Closing application")

def prepare_initial_dataset_from_list(config, train_list):
    try:
        for i, (imgdir, name) in enumerate(zip([config['traindir'], config['valdir'], config['testdir']], ['train', 'val', 'test'])):
            print("")
            print("Processing {:s}-dataset: {:s}".format(name, imgdir))
            rename_xml_files(imgdir)
            images, annotations = list_files(imgdir)
            print("{:d} images found!".format(len(images)))
            print("{:d} annotations found!".format(len(annotations)))
            write_file(config['dataroot'], images, name)

            if name == "train":
                check_json_presence(config, imgdir, train_list, name)
                create_json(config['dataroot'], imgdir, train_list, config['classes'], name)
            else:
                check_json_presence(config, imgdir, images, name)
                create_json(config['dataroot'], imgdir, images, config['classes'], name)

    except Exception as e:
        print_exception(e)
        logger.error("Cannot create initial-datasets")
        sys.exit("Closing application")      

def update_train_dataset(config, cfg, train_list):
    try:
        rename_xml_files(config['traindir'])
        images, annotations = list_files(config['traindir'])
        print("{:d} images found!".format(len(images)))
        print("{:d} annotations found!".format(len(annotations)))

        check_json_presence(config, config['traindir'], train_list, "train", cfg)
        print("Converting annotations...")
        create_json(config['dataroot'], config['traindir'], train_list, config['classes'], "train")
        
    except Exception as e:
        print_exception(e)
        logger.error("Cannot update train-dataset")
        sys.exit("Closing application")

def train_model(config, cfg, dataset_dicts_train):
    try:
        print("Starting training...")
        cfg.DATASETS.TRAIN = ("train_dataset",)
        cfg.DATASETS.TEST = ()
        cfg.DATALOADER.NUM_WORKERS = config['num_workers']
        cfg.SOLVER.IMS_PER_BATCH = config['train_batch_size']
        cfg.SOLVER.BASE_LR = config['learning_rate']
        cfg.SOLVER.MAX_ITER = config['train_iterations_base']
        cfg.SOLVER.STEPS = config['train_iterations_step_size']
        cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 128
        cfg.MODEL.ROI_HEADS.NUM_CLASSES = len(config['classes'])

        trainer = DefaultTrainer(cfg)
        trainer.resume_or_load(resume=False)
        trainer.train()

        print("Training completed.")

    except Exception as e:
        print_exception(e)
        logger.error("Error during training")
        sys.exit("Closing application")
