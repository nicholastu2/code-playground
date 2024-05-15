import io
import os
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
import torch
from ultralytics import YOLO
from PIL import Image, ImageDraw
import numpy as np
import pandas as pd
from IPython.display import clear_output
import socket
import time
from datetime import datetime
import shutil
from random import randint
import random
import re
import colorsys
import math
from scipy.spatial import Voronoi
import matplotlib.colors as mcolors
import matplotlib.path as mpltPath

def find_first_file(m_folder, m_contains):
    """
    ok so basically this is looking at each subdirectory (including the current directory) and looking at the files in it
    and then checking if m_contains is in each subdirectory
    
    suggestion: you don't have to do for file in files: if m_contains in file. you can just do if m_contains in files
        - file is a string
        - m_contains is also a string
        - you can just check if m_contains is in the list, files
    Oh wait, no. m_contains in file checks if file has the string m_contains as a substring
        
    so basically this function finds the path of where m_contains is located
    actually, this function finds the path of the first file in m_folder that contains the substring m_contains
    """
    for root, dirs, files in os.walk(m_folder):
        for file in files:
            if m_contains in file:
                return os.path.join(root, file)
    return None

def create_multichannel_array(folder_path):
    image_arrays = []
    image_names = []
    for root, dirs, files in os.walk(folder_path):
        for file in files:
            if file.lower().endswith(('.png', '.jpg', '.jpeg','.bmp','.tif')):
                file_path = os.path.join(root, file)
                t_img = Image.open(file_path).convert('L')  # Convert to grayscale if not already
                t_arr = np.array(t_img)
                if t_arr.ndim == 2:  # Ensure the image is grayscale
                    image_arrays.append(t_arr)
                    image_names.append(file)
    
    if not image_arrays:
        return None  # Or raise an exception if you prefer

    # Stack the arrays along a new axis to create a multi-channel array
    multi_channel_array = np.stack(image_arrays, axis=-1)
    return multi_channel_array, image_names

def find_mask_intensities(img_data, image_array, file_name, shift_x = 0, shift_y = 0, include_headers = True, meta_name = "NA", tile_name = "NA"):
    sto = io.StringIO()
    sth = ''; d = '\t'

    def bstr_h(sth1):
        nonlocal sth
        sth += sth1

    def bstr_m(st1):
        sto.write(st1)

    def bstr_m_start():
        nonlocal sth, sto
        st = sth + '\r' + sto.getvalue()
        sto.close()
        sto = io.StringIO()
        sto.write(st)

    def get_mask(vertices):
        polygon_path = mpltPath.Path(vertices) # Create a path object from the vertices
        inside_polygon = polygon_path.contains_points(class_points)
        mask = inside_polygon.reshape(xx.shape) # Reshape the mask back to the image shape
        return mask

    width =image_array.shape[1]; height = image_array.shape[0]; channels = image_array.shape[2]
    boxes = img_data.boxes.cpu()
    img_box_centers = boxes.xywh 
    img_mask_coords = None if img_data.masks is None else img_data.masks.xy
    img_vor_coords = get_vor_boundaries(img_box_centers)

    first = include_headers; masks = {}
    print("width =",width,"height =",height,"chs =",channels,"boxes =",len(img_box_centers),"vor =",len(img_vor_coords))
    xx, yy = np.meshgrid(np.arange(width),np.arange(height)) # Create a mesh grid of coordinate values
    x_flat = xx.flatten(); y_flat = yy.flatten()
    class_points = np.vstack((x_flat, y_flat)).T # Create a list of (x, y) points from the flattened grid
    for idx in range(len(img_box_centers)):
        if (idx % 250 == 0): print("Measuring Intensities",idx)
        bbox_xywh = img_box_centers[idx]
        bbox_corners = [[bbox_xywh[0] - bbox_xywh[2], bbox_xywh[1] + bbox_xywh[3]],[bbox_xywh[0] + bbox_xywh[2], bbox_xywh[1] + bbox_xywh[3]] ,[bbox_xywh[0] + bbox_xywh[2], bbox_xywh[1] - bbox_xywh[3]], [bbox_xywh[0] - bbox_xywh[2], bbox_xywh[1] - bbox_xywh[3]]]
        vor_corners = img_vor_coords[idx]
        polys = { "box": bbox_corners, "poly": img_mask_coords, "vor": vor_corners }
        masks = {key: get_mask(value) for key, value in polys.items() if value}

        # want to add parentID here
        if (first): bstr_h('FileName' + d + 'MetaName' + d + 'TileName' + d + 'ObjectID' + d + 'Class'                      + d + 'Confidence'                  + d + 'cx' + d + 'cy' + d)
        bstr_m(             file_name + d + meta_name  + d +  tile_name + d + str(idx)   + d + str(boxes[idx].cls.item()) + d + str(boxes[idx].conf.item()) + d + str(bbox_xywh[0].item() + shift_x) + d + str(bbox_xywh[1].item() + shift_y))

        # Look at each mask for each channel
        for c in range(channels):
            cs = str(c)
            for key in masks:
                selected_pixels = image_array[:, :, c][masks[key]]
                area = len(selected_pixels)
                if (first and c==0): bstr_h(key + ' AreaP' + d)
                if (c==0): bstr_m(               str(area) + d)

                sum = np.sum(selected_pixels)
                avg = np.average(selected_pixels)
                std = np.std(selected_pixels)
                if (first): bstr_h(key + ' Total Intensity wv' + cs + d + key + ' Avg Intensity wv' + cs + d + key + ' Std Intensity wv' + cs + d)
                bstr_m(                    str(sum)                 + d + str(avg)                       + d + str(std)                       + d)

        if (first): bstr_m_start(); first = False
        bstr_m('\r')
    return sto.getvalue()

def Predict_OnPartsOfImage(model, original_image_name, full_image_arr_predict, full_image_arr_measure = None, save_path = None, new_w = 256, new_h = 256, 
                           overlap_amount = 0, fill_edges = False, include_headers = True, meta_name = "NA", maxdets = 6666, minconf = 0.25):
    """_summary_

    Args:
        model (YOLO model): the model used to make predictions
        original_image_name (string): this is like the path to the image I think
        full_image_arr_predict (numpy array): _description_
        full_image_arr_measure (numpy array, optional): _description_. Defaults to None.
        save_path (_type_, optional): _description_. Defaults to None.
        new_w (int, optional): _description_. Defaults to 256.
        new_h (int, optional): _description_. Defaults to 256.
        overlap_amount (int, optional): _description_. Defaults to 0.
        fill_edges (bool, optional): _description_. Defaults to False.
        include_headers (bool, optional): _description_. Defaults to True.
        meta_name (str, optional): _description_. Defaults to "NA".
        maxdets (int, optional): _description_. Defaults to 6666.
        minconf (float, optional): _description_. Defaults to 0.25.
    
    Returns:
        
    """
    def get_piece(t_arr, x, y):
        piece = t_arr[y:min(y + new_h, t_arr.shape[0]), x:min(x + new_w, t_arr.shape[1])] # Calculate the dimensions of the piece
        if fill_edges: # Create a new array filled with zeros (black) of the desired final size
            filled_piece = np.zeros((new_h, new_w), dtype=t_arr.dtype)
            filled_piece[:piece.shape[0], :piece.shape[1]] = piece
            piece = filled_piece
        return piece
            
    t_arr = full_image_arr_predict
    first = include_headers
    st = io.StringIO()
    for y in range(0, t_arr.shape[0], new_h - overlap_amount):
        for x in range(0, t_arr.shape[1], new_w - overlap_amount):
            piece_pred = get_piece(t_arr, x, y)
            piece_meas = get_piece(full_image_arr_measure, x, y) if (full_image_arr_measure is not None) else piece_pred
            tilename = str(x) + "," + str(y); print("Region:",tilename)
            predictions = model.predict(piece_pred, show=False, max_det=maxdets) #minconf
            #img_array=predictions[0].plot(labels=False, boxes=True, masks=True); display(Image.fromarray(img_array[..., ::-1]))
            st.write(find_mask_intensities(predictions[0], piece_meas, original_image_name, x, y, first, meta_name, tilename))
            first = False

    strRet = st.getvalue()
    if (save_path is not None):
        with open(save_path, 'a') as file: file.write(strRet)
        st.close()
    print("Done with File")
    return strRet

def work_on_folder(model, SubFolder, PredContains, IncludeHeaders = True, save_path = None, maxdet = 6000):
    file_pred = find_first_file(SubFolder, PredContains)
    pred_arr = np.array(Image.open(file_pred).convert('RGB'))  # Convert to RGB
    meas_arr, names = create_multichannel_array(SubFolder) # with all the images in the subfolder?
    st = Predict_OnPartsOfImage(model, file_pred, pred_arr, meas_arr, None, 553, 553, 0, False, IncludeHeaders, SubFolder, maxdet)
    if (save_path is not None): 
        with open(save_path, 'a') as file: file.write(st)
    print("Done with Files")
    return st, names

model_path = r"S:\Phys\FIV925 XSection\Datasets\Creed\01a\YO 553 0328 MAXI\map75=0296662 yolov9c  idx=1 ep=8 btch=16 rnd=4717152\weights\best.pt"
m_folder = r"S:\Phys\FIV925 XSection\Datasets\Creed\20240406"
m_contains = "T0_DAPI"
res_append = "2"

model = YOLO(model_path) 

"""
what is os.walk(m_folder)?
    ok idk, it's like a generator object, whatever that means
    but next(os.walk(m_folder)) is a tuple where:
        - next(os.walk(m_folder))[0] is a folder name
        - next(os.walk(m_folder))[1] is a list of subdirectories in the folder
        - next(os.walk(m_folder))[2] is another list, but it's of the files in the directory that aren't folders
"""
first_level_subfolders = next(os.walk(m_folder))[1]  # Get first level of folders only
First = True # ok this must make it so that they're saving out the header
stio = io.StringIO()
namedict = {}
for subfolder in first_level_subfolders:
    print(subfolder,"---------------------------------")
    subfolder_path = os.path.join(m_folder, subfolder)
    st, names = work_on_folder(model, subfolder_path, m_contains, First)
    stio.write(st)
    namedict[subfolder] = names
    First = False

# Save out the main data
save_path = os.path.join(m_folder, "Res00"+res_append+".txt")
strRet = stio.getvalue(); stio.close()
with open(save_path, 'a') as file: file.write(strRet)

# Now save out the name information
save_path = os.path.join(m_folder, "Res00"+res_append+"_Names.txt")
rows = [f"{subfolder}\t{idx}\t{name}" for subfolder, names in namedict.items() for idx, name in enumerate(names)]
with open(save_path, 'w') as txtfile:
    txtfile.write("Subfolder\tIndex\tName\n") 
    txtfile.write("\n".join(rows))

print("Done with Folder")