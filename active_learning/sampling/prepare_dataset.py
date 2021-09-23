# @Author: Pieter Blok
# @Date:   2021-03-26 14:30:31
# @Last Modified by:   Pieter Blok
# @Last Modified time: 2021-09-23 16:09:59

import sys
import random
import os
import numpy as np
import shutil
import cv2
import json
import xml.etree.cElementTree as ET
import math
import datetime
import time
from tqdm import tqdm
import xmltodict
import logging
from detectron2.engine import DefaultPredictor


## initialize the logging
log_format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s \n'
logger = logging.getLogger(__name__)
logger.setLevel('DEBUG')
file_handler = logging.StreamHandler()
formatter = logging.Formatter(log_format)
file_handler.setFormatter(formatter)
logger.addHandler(file_handler)


supported_cv2_formats = (".bmp", ".dib", ".jpeg", ".jpg", ".jpe", ".jp2", ".png", ".pbm", ".pgm", ".ppm", ".sr", ".ras", ".tiff", ".tif")


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

                    masks[fill_id].append(points)

            crowd_ids[fill_id] = 0
            status = "successful"
        else:
            status = "unsuccessful"

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


def visualize(img, category_ids, masks, boxes, classes):
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

    annotation_id = 0
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

            areas, boxes = bounding_box(masks)
            img_vis = visualize(img, category_ids, masks, boxes, classes)

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
    annotation_basenames = [os.path.splitext(annot)[0] for annot in annotations]
    
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
    

def check_json_presence(imgdir, dataset, name, cfg=[], all_classes=[], pre_annotate=False, export_format=[]):
    print("")
    print("Checking {:s} annotations...".format(name))
    rename_xml_files(imgdir)
    all_images, annotations = list_files(imgdir)
    img_basenames = [os.path.splitext(img)[0] for img in dataset]
    annotation_basenames = [os.path.splitext(annot)[0] for annot in annotations]
    
    diff_img_annot = []
    for c in range(len(img_basenames)):
        img_basename = img_basenames[c]
        if img_basename not in annotation_basenames:
            diff_img_annot.append(img_basename)
    diff_img_annot.sort()
    
    ii32 = np.iinfo(np.int32)
    cur_annot_diff = ii32.max
    annot_folder = os.path.join(imgdir, "annotate")

    ## copy the images that lack an annotation to the "annotate" subdirectory so that we can annotate them easily
    if len(diff_img_annot) > 0:
        annot_folder_present = os.path.isdir(annot_folder)
        
        if not annot_folder_present:
            os.makedirs(annot_folder)
        else:
            shutil.rmtree(annot_folder)
            os.makedirs(annot_folder)

        for p in range(len(diff_img_annot)):
            search_idx = img_basenames.index(diff_img_annot[p])
            image_copy = dataset[search_idx]

            if not os.path.isdir(os.path.dirname(os.path.join(annot_folder, image_copy))):
                os.makedirs(os.path.dirname(os.path.join(annot_folder, image_copy)))

            shutil.copyfile(os.path.join(imgdir, image_copy), os.path.join(annot_folder, image_copy))

    ## check whether all images have been annotated in the "annotate" subdirectory
    if not pre_annotate:
        while len(diff_img_annot) > 0:
            diff_img_annot, cur_annot_diff = highlight_missing_annotations(annot_folder, cur_annot_diff)

    else:
        if len(diff_img_annot) > 0:
            rename_xml_files(annot_folder)
            images, annotations = list_files(annot_folder)
            predictor = DefaultPredictor(cfg)

            for i in tqdm(range(len(images))):
                # Load the RGB image
                imgname = images[i]
                basename = os.path.basename(imgname)
                img = cv2.imread(os.path.join(annot_folder, imgname))
                height, width, _ = img.shape

                # Do the image inference and extract the outputs from Mask R-CNN
                outputs = predictor(img)
                instances = outputs["instances"].to("cpu")
                classes = instances.pred_classes.numpy()
                masks = instances.pred_masks.numpy()

                class_names = []
                for h in range(len(classes)):
                    class_id = classes[h]
                    class_name = all_classes[class_id]
                    class_names.append(class_name)

                if export_format == 'labelme':
                    write_labelme_annotations(annot_folder, basename, class_names, masks, height, width)
                elif export_format == 'cvat':
                    write_cvat_annotations(annot_folder, basename, class_names, masks, height, width)
                else:
                    logger.error("unsupported export_format in the maskAL.yaml file")
                    sys.exit("Closing application")

            diff_img_annot, cur_annot_diff = highlight_missing_annotations(annot_folder, cur_annot_diff)
            input("Press Enter when all annotations have been checked in folder: {:s}".format(annot_folder))

    if os.path.isdir(annot_folder):
        ## copy the annotations back to the imgdir
        rename_xml_files(annot_folder)
        images, annotations = list_files(annot_folder)
        for a in range(len(annotations)):
            annotation = annotations[a]
            shutil.copyfile(os.path.join(annot_folder, annotation), os.path.join(imgdir, annotation))

        ## remove the annotation-folder again
        annot_folder_present = os.path.isdir(annot_folder)
        if annot_folder_present:
            time.sleep(1)
            shutil.rmtree(annot_folder)


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
            contours, hierarchy = cv2.findContours((masksel*255).astype(np.uint8),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
            cnt = np.concatenate(contours)
               
            if cv2.contourArea(cnt) > 50:
                useful_masks = True
                if len(contours) == 1:
                    segm = np.vstack(contours).squeeze()
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
            contours, hierarchy = cv2.findContours((masksel*255).astype(np.uint8),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
            cnt = np.concatenate(contours)

            if cv2.contourArea(cnt) > 50:
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
    

def prepare_initial_dataset_randomly(rootdir, imgdir, classes, train_val_test_split, initial_datasize):
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


def prepare_initial_dataset(rootdir, classes, traindir, valdir, testdir, initial_datasize):
    try:
        for imgdir, name, init_ds in zip([traindir, valdir, testdir], ['train', 'val', 'test'], [initial_datasize, 0, 0]):
            print("")
            print("Processing {:s}-dataset: {:s}".format(name, imgdir))
            rename_xml_files(imgdir)
            images, annotations = list_files(imgdir)
            print("{:d} images found!".format(len(images)))
            print("{:d} annotations found!".format(len(annotations)))

            if init_ds > 0:
                initial_train_images = random.sample(images, initial_datasize)
                write_file(rootdir, images, "train")
                write_file(rootdir, initial_train_images, "initial_train")
                check_json_presence(imgdir, initial_train_images, "train")
                create_json(rootdir, imgdir, initial_train_images, classes, "train")
            else:
                write_file(rootdir, images, name)
                check_json_presence(imgdir, images, name)
                create_json(rootdir, imgdir, images, classes, name)
    except:
        logger.error("Cannot create initial-datasets")
        sys.exit("Closing application")


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


def update_train_dataset(cfg, rootdir, imgdir, classes, train_list, pre_annotate, export_format):
    try:
        rename_xml_files(imgdir)
        images, annotations = list_files(imgdir)
        print("{:d} images found!".format(len(images)))
        print("{:d} annotations found!".format(len(annotations)))

        check_json_presence(imgdir, train_list, "train", cfg, classes, pre_annotate, export_format)
        print("Converting annotations...")
        create_json(rootdir, imgdir, train_list, classes, "train")
    except:
        logger.error("Cannot update train-dataset")
        sys.exit("Closing application")