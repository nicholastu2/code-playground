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
from image_data_export_functions import *

# variables
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
stio = io.StringIO() # an object to write strings to
namedict = {} # initializing dict I guess

# for loop
for subfolder in first_level_subfolders: # for each subfolder in the list of subfolders in m_folder
    print(subfolder,"---------------------------------") # first_level_subfolders is a list of strings, so subfolder is a string, and this line prints it
    subfolder_path = os.path.join(m_folder, subfolder) # creates a string to contain the path of the subfolder
    st, names = work_on_folder(model, subfolder_path, m_contains, First) # ok so now apparently we're running the work_on_folder function
    # passing m_contains makes work_on_folder create a numpy array of the first image in subfolder_path with m_contains in the name
    # so, we're only looking at images with TO_DAPI in its name
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