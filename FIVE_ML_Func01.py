# - - - - - - - Imports
import matplotlib.pyplot as plt
import seaborn as sns

import os 
import time
from datetime import datetime
import json
from typing import Dict
import re #regular Expressions
import math
import csv
from random import randint
import random
from PIL import Image
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.utils import class_weight
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
from scipy.ndimage import binary_dilation, shift
import matplotlib.lines as lines
#import umap.umap_ as umap ###### - - - - this may be an issue
import tensorflow as tf
from tensorflow import keras
from tensorflow.python.framework.convert_to_constants import convert_variables_to_constants_v2
from tensorflow.keras.metrics import MeanAbsoluteError, MeanSquaredError, Mean
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras import backend as K
from tensorflow.keras.losses import binary_crossentropy
import tf2onnx

#from IPython.display import clear_output
def clear_screen():
    try:
        # Attempt to detect if we are in a Jupyter notebook environment
        shell = get_ipython().__class__.__name__
        if shell == 'ZMQInteractiveShell':
            # We are in Jupyter, use the IPython display clear function
            from IPython.display import clear_output
            clear_output(wait=True)
        else:
            # Not in Jupyter, define clear_output as a no-op
            def clear_output(wait=False):
                pass
    except NameError:
        # get_ipython doesn't exist, likely not in a Jupyter environment
        pass
    
    # If we are not in a Jupyter environment, clear the screen using the system's clear command
    os.system('cls' if os.name == 'nt' else 'clear')

# - - - - - - - Function Definitions - - Standard set - - - - - - - - - - - - - - - - - - - - - - -- - - - - - - - - - 

# summarize history for accuracy, loss
def QPlot_Loss(hst, vS, log_scale=True):
    plt.figure()
    plt.plot(hst.history[vS])
    try:
        plt.plot(hst.history['val_'+vS]) #may not be a val version
    except KeyError:
        pass
    plt.title('model ' + vS)
    plt.ylabel(vS)
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.grid()
    #if log_scale:
    #    plt.yscale('log')
    plt.show(block=False)

def QPlot_Loss2(ax, hst, vS, log_scale=True, ScaleEpochsAfter=1):
    ax.plot(range(len(hst.history[vS])), hst.history[vS])
    try:
        ax.plot(range(len(hst.history['val_'+vS])), hst.history['val_'+vS]) 
    except KeyError:
        pass

    ax.set_title('model ' + vS)
    ax.set_ylabel(vS)
    ax.set_xlabel('epoch')
    ax.legend(['train', 'test'], loc='upper left')
    ax.grid()
    
    y_values = hst.history[vS] if ScaleEpochsAfter <= 0 else hst.history[vS][ScaleEpochsAfter:]
    y_values_test = hst.history.get('val_'+vS, len(y_values))
    y_all = y_values + y_values_test

    if log_scale and all(i > 0 for i in y_all):
        ax.set_yscale('log')

    y_lower, y_upper = np.percentile(y_all, [0, 100])
    ax.set_ylim(y_lower, y_upper)

def QGetBestEpoch_Val(hst, metric_check, mectric_report):
    try:
        min_v = min(hst.history[metric_check])
        min_idx = [i for i, j in enumerate(hst.history[metric_check]) if j == min_v]
        min_idx = min_idx[0]
        best_val_acc = hst.history[mectric_report][min_idx]
        return min_idx, best_val_acc, min_v
    except:
        return -1,0,0

def QWorstCase_Accuracy(hst, FinalEpochToAverage, NumberToAverage, metric):
    n = min(len(hst.history['loss']), FinalEpochToAverage)
    min_idx, best_val_acc, _ = QGetBestEpoch_Val(hst,'val_loss', 'val_' + metric)
    last_accuracy_L = last_accuracy_E = 0
    for i in range(0,NumberToAverage):
        last_accuracy_L += hst.history[metric][n-i-1]
        last_accuracy_E += hst.history['val_'+metric][n-i-1]
    return min(last_accuracy_E, last_accuracy_L) / NumberToAverage, n, best_val_acc

def QPlot_AEVal5e(n_examples, orig_dims, in_train, in_test = None, rec_train = None, rec_test = None, out_train = None, out_test = None, fullSavePath = None, width = None, dilation = 0, shift_val = 0, row_labels=None, column_labels=None, font_size=16):
 
    def preprocess_image(image, dilation = 0, shift_val = 0): 
        if tf.is_tensor(image):
            image = tf.reshape(image, [orig_dims, orig_dims, -1])
            image = tf.cast(image, tf.float32) if image.dtype not in [tf.float32, tf.float64] else image
        else:
            image = image.reshape(orig_dims, orig_dims, -1)
            image = image.astype(np.float32) if image.dtype not in [np.float32, np.float64] else image
        image = (image - np.min(image)) / (np.max(image) - np.min(image) + np.finfo(float).eps) # Min-Max normalize to 0-1 range
        if dilation > 0: # Expand positive pixels if requested
            structuring_element = np.ones((dilation+1, dilation+1)) # create structuring element
            for i in range(image.shape[-1]): # Assuming last dimension is channels
                image[..., i] = binary_dilation(image[..., i], structure=structuring_element)
        if shift_val != 0: # Shift the image channel
            image[..., 0] = shift(image[..., 0], [shift_val, 0])
            image[..., 0] = (image[..., 0] - np.min(image[..., 0])) / (np.max(image[..., 0]) - np.min(image[..., 0]) + np.finfo(float).eps) # Min-Max normalize to 0-1 range
        return image
    
    def display_image(image, subplot_location, rows):
        ax = plt.subplot(rows, columnsets * n_examples, subplot_location)
        reshaped_image = preprocess_image(image)
        if reshaped_image.shape[2] == 1:
            plt.imshow(reshaped_image.squeeze(), cmap='gray')
        else:
            plt.imshow(reshaped_image[..., [0,2,1]], interpolation='none')
        ax.get_xaxis().set_visible(False); ax.get_yaxis().set_visible(False)

    def display_two_color_image(image1, image2, subplot_location, rows):
        ax = plt.subplot(rows, columnsets * n_examples, subplot_location)
        combined_image = np.zeros((orig_dims, orig_dims, 3))
        combined_image[..., 1] = preprocess_image(image1)[..., 0]
        if (image2 is not None):
            combined_image[..., 0] = preprocess_image(image2, dilation, shift_val)[..., 0]
        else:
            combined_image[..., 0] = preprocess_image(image1)[..., 0]
            combined_image[..., 2] = preprocess_image(image1)[..., 0]
        plt.imshow(combined_image, interpolation='none')
        ax.get_xaxis().set_visible(False); ax.get_yaxis().set_visible(False)
    
    pltIdx = 0
    def inc():
        nonlocal pltIdx
        pltIdx += 1
        return pltIdx

    columnsets = 1 if in_test is None else 2
    ndv = 2 if columnsets == 1 else 1
    columns = n_examples * columnsets
    width = columns * 3 if width is None else width
    channels = rec_train[0].shape[-1] if tf.is_tensor(rec_train[0]) else rec_train[0].reshape(orig_dims, orig_dims, -1).shape[2]
    expandChannels = channels > 3
    output_rows = 1 if (out_train is None) else 2
    rows = channels + 1 if expandChannels else 1 + output_rows
    height = width * rows / (columns)
    fig = plt.figure(figsize=(width, height), facecolor='dimgrey')
    check = ""

    for i in range(n_examples):
        display_image(in_train[i], inc(), rows); check += f"{i} {pltIdx} tr or | "
    if (columnsets == 2):
        for i in range(n_examples): #Changed 8/24
            display_image(in_test[i], inc(), rows); check += f"{i} {pltIdx} va or | "
    check += '\n'

    if (expandChannels): #FIV823 Style
        for channel in range(channels): 
            for i in range(n_examples):
                UseImg = None if out_train is None else out_train[i][..., channel] #New way 8/24/2023
                display_two_color_image(rec_train[i][..., channel], UseImg, inc(), rows); check += f"{i} {pltIdx} tr  {channel} | "
            if (columnsets == 2):
                for i in range(n_examples): #Changed 8/24
                    UseImg = None if out_train is None else out_test[i][..., channel]
                    display_two_color_image(rec_test[i][..., channel], UseImg, inc(), rows); check += f"{i} {pltIdx} va  {channel} | "
            check += '\n'

    else:                #FIV906 Style
        for i in range(n_examples):
            display_image(rec_train[i], inc(), rows); check += f"{i} {pltIdx} tr rc | "
        if (columnsets == 2 and rec_test is not None):
            for i in range(n_examples):
                display_image(rec_test[i], inc(), rows); check += f"{i} {pltIdx} va rc | "
        check += '\n'
        if (out_train is not None): 
            for i in range(n_examples):
                display_image(out_train[i], inc(), rows); check += f"{i} {pltIdx} tr ot | "
        if (columnsets == 2 and out_test is not None):
            for i in range(n_examples):
                display_image(out_test[i], inc(), rows); check += f"{i} {pltIdx} va ot | "
    
    #print(check)
    if row_labels: # Display row labels if provided
        for idx, label in enumerate(row_labels):
            plt.text(-width/10, idx/rows + 1/rows/2, label, ha='center', va='center', color='white', fontsize=font_size)
    if column_labels: # Display col labels if probided
        for idx, label in enumerate(column_labels):
            fig.text((idx + 1.97)/(columns+3.35), 0.24, label, ha='left', va='center', color='lightgray', fontsize=font_size)
    plt.subplots_adjust(wspace=0.05, hspace=0.05)
    if (columnsets == 2):
        l1 = lines.Line2D([0.5125, 0.5125], [0.2, 0.8], transform=plt.gcf().transFigure, figure=plt.gcf(), color='cyan', linewidth=2)
        plt.gcf().lines.extend([l1])
    if (fullSavePath != None):
        plt.savefig(fullSavePath, bbox_inches='tight', pad_inches=0.1)
    plt.show(block=False)

# - - - - - - - - - - - - - - - - - - - - - - -- - - - - - - - - -  Save model stuff  - - - - - - - - - - - - - - - - - - - - - - -- - - - - - - - - - 

def save_metrics_to_csv_raft(hist, best_epoch, model_name, output_file_name, modelSettings = "", AdditionalNotes = ""):
    # Extract the metrics and their values at the best epoch
    headers = ['model_name'] + ['Epoch'] + ['Settings'] + ['Notes'] + list(hist.history.keys())
    row_data = [model_name] + [best_epoch] + [modelSettings] + [AdditionalNotes] + [hist.history[key][best_epoch - 1] for key in hist.history.keys()]
    file_exists = os.path.isfile(output_file_name)
    # Open the file in append mode, write headers if the file is new, and append the row
    with open(output_file_name, 'a', newline='') as csvfile:
        writer = csv.writer(csvfile)
        if not file_exists:
            writer.writerow(headers)
        writer.writerow(row_data)

def SaveAsFrozen(frozen_out_path, frozen_graph_filename, saveModel, writeTextRep = False, saveONNX = None):
    #https://medium.com/@sebastingarcaacosta/how-to-export-a-tensorflow-2-x-keras-model-to-a-frozen-and-optimized-graph-39740846d9eb
    # Convert Keras model to ConcreteFunction
    full_model = tf.function(lambda x: saveModel(x))
    full_model2 = full_model.get_concrete_function(
        tf.TensorSpec(saveModel.inputs[0].shape, saveModel.inputs[0].dtype))
    
    # Get frozen ConcreteFunction
    frozen_func = convert_variables_to_constants_v2(full_model2)
    frozen_func.graph.as_graph_def()
    layers = [op.name for op in frozen_func.graph.get_operations()]

    # Save frozen graph to disk
    saveName =((f"{frozen_graph_filename[:22]}").strip())
    tf.io.write_graph(graph_or_graph_def=frozen_func.graph, logdir=frozen_out_path, name=saveName + ".pb", as_text=False)
    
    if (writeTextRep): # Save its text representation- Optional
        tf.io.write_graph(graph_or_graph_def=frozen_func.graph, logdir=frozen_out_path, name=saveName+".txt", as_text=True)
    
    # Save as ONNX model
    if (saveONNX == False): return
    print("Save ONNX = ",saveONNX)
    onnx_model_path = os.path.join(frozen_out_path, saveName + ".onnx")
    with tf.device('/cpu:0'): # Make sure to run the conversion on the CPU
        onnx_model_proto, _ = tf2onnx.convert.from_function( full_model, 
            input_signature=[tf.TensorSpec(saveModel.inputs[0].shape, saveModel.inputs[0].dtype)],
            opset=13, output_path=onnx_model_path
        ) # Convert the frozen graph to ONNX format and save

def MultiModelSave(saveModel, folderToSave, saveName, modelType = "", totalTraininCount = 0, classDict = None, classCounts = None, scalingFactors = None, featureNames = [""], input_metadata=None, useModelNameAsSubfolder = True, saveONNX = None):
    if (saveONNX is None): saveONNX = True
    #Lets save both the pb and the frozen format in a subfolder together
    #TODO: Implement a metadata system to also save here
    folderToSaveSub = os.path.join(folderToSave, saveName if useModelNameAsSubfolder else "m")
    saveModel.save(os.path.join(folderToSaveSub, "KSM")) #folder_infer + r"\M_" + prefix)
    SaveAsFrozen(folderToSaveSub, saveName + " F", saveModel, saveONNX = saveONNX)

    if (classDict!=None or input_metadata!=None):
        SaveModelMetadata(classDict, classCounts, scalingFactors, totalTraininCount,  saveName, modelType, folderToSaveSub, "", featureNames, input_metadata)

    ## Optimizing the Frozen Model (from the same article) ------------------------------------------

    #conda create -n tf15 python=3.7
    #conda activate tf15
    #conda install tensorflow=1.5.0
    #python -m tensorflow.python.tools.optimize_for_inference --input ./model_20K_96_soft_f1/frozen_model/frozen_graph.pb --output ./model_20K_96_soft_f1/optimized/optmized_graph.pb --frozen_graph=True --input_names=x --output_names=Identity


def PostRunTasks(folder_save, Name, ModelType, Settings, randomSettings, predSet, pSave, classDict, classCounts, scalingFactors, i, denseSizes, mModel, hst, metric, trainingCount=0):
    #Save the training/testing epoch images

    last_accuracy_Show, mEpochsActual, best_accuracy = QWorstCase_Accuracy(hst, Settings.mEpochs, 3, metric) 
    szName = "acc=" + f'{best_accuracy:.4f}' + " " + Name + " CNNs=" + str(randomSettings.conv_layers) + " kn=" + str(randomSettings.kernel_sz) + " flt=" + str(randomSettings.filter_nm) + " iRt=" + str(randomSettings.cnn_filter_increase_rate) + " ds=" + str(denseSizes) + " nE=" + str(mEpochsActual)+","+str(Settings.mEpochs) + " bSz=" + str(Settings.BatchSize) + " bNm=" + str(randomSettings.used_batch_norm_conv) + " i=" + str(i)
    szName = re.sub(r'[^\w\s\-_\.,]', '_', szName) #Regular expression to clean this Prefix from any unsafe characters
    szName = szName if len(szName) < 86 else szName[:86]
    if szName.endswith(','): szName = szName[:-1]
    szName = szName.strip()
    
    #plt.figure(figsize=(14, 7)); ax = plt.subplot(1, 2, 2); QPlot_Loss(hst, metric); ax = plt.subplot(1, 2, 1); QPlot_Loss(hst, 'loss') #Late 2022 style
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 7)); QPlot_Loss2(ax1, hst, metric); QPlot_Loss2(ax2, hst, 'loss');  #6/2023 style

    plt.savefig(os.path.join(folder_save, "Plts " + szName + ".jpg")); plt.clf()
    save_metrics_to_csv_raft(hst, mEpochsActual, szName, os.path.join(folder_save,'metrics01.csv'))
    MultiModelSave(mModel, folder_save, szName, ModelType, trainingCount, classDict, classCounts, scalingFactors)

    if (predSet == None):
        return

    #Predict stuf with the new model
    nClasses = len(classDict)
    p = pd.DataFrame(np.squeeze(mModel.predict(predSet)))
    p.drop(columns = p.columns[0], inplace=True) #Get rid of the G1 so only have G2
    if (False): #Turn this on if we want a class prediction, otherwise we don't need it
        pMax = p.iloc[: , :nClasses] #This is the set of columns where we want to find the max
        p["Pred"] = pMax.idxmax(axis=1) #This is the prediction, if you do this later, it won't be numeric

    #Rename the new columns so the contain the original labels, and the prefix
    try:
        p.rename(columns=classDict, inplace=True); 
        p = p.add_prefix(szName + " ")
    except: 
        pass
    
    pSave = pSave.join(p)
    pSave.to_csv(os.path.join(folder_save , "Res_Save_00.csv")) #Save each time so we can stop anytime
    AppLog(szName)
    return pSave

def predict_class_label_number(dataset, classNames, mModel):
  """Runs inference and returns predictions as class label numbers."""
  #https://www.tensorflow.org/hub/tutorials/cropnet_on_device
  rev_label_names = {l: i for i, l in enumerate(classNames)}
  return [
    rev_label_names[o[0][0]]
    for o in mModel.predict(dataset, batch_size=128)
  ]

def show_confusion_matrix(cm, labels):
  plt.figure(figsize=(10, 8))
  sns.heatmap(cm, xticklabels=labels, yticklabels=labels, 
              annot=True, fmt='g')
  plt.xlabel('Prediction')
  plt.ylabel('Label')
  plt.show(block=False)

def MitoCounts(sourceFolder, multer, adder):
    labls_S = []
    for root, dirnames, filenames in os.walk(sourceFolder):
        for filename in filenames:
            if filename.endswith(('bmp')):
                str = filename.split("_")[0]
                l = len(str)
                st1 = str[l-4:l-2]
                st2 = str[l-2:l]
                preCount = int(st1)
                mito = (multer*preCount) + adder
                labls_S.append(mito)
    return labls_S

def OccasionalWait(Period, ShortAmountSeconds, LongAmountSeconds):
    if (randint(0,Period) == 1):
        print("taking a longer break for " + str(LongAmountSeconds) + " sec . . .")
        time.sleep(LongAmountSeconds) # Every so often, give this computer a 30 second break
    else:
        print("taking a quick break for " + str(ShortAmountSeconds) + " sec . . .")
        time.sleep(ShortAmountSeconds) # Always give it at least 2 seconds

def AppLog(Text):
    file1 = open(r"R:\dB\Software\FIVE_Tools\Settings\TF_NB_Log.txt", "a")  # append mode
    file1.write(str(datetime.now()) +"\t"+ str(Text) + "\n")
    file1.close()
    

# - - - - - - - - - 16-bit FIV823 Functions - - - - - - - - - - - - - - - - - - - - - - -- - - - - - - - - - 


### 16-bit AE/Target version 6/4/2023

def ScanDirectories16a(sourceFolder, csv_file, max_size = None):
    print("Loading data . . ")
    df = pd.read_csv(csv_file, skiprows=[0]) #Expects the first row to be the GroundTruth (GTx) reference, then the next to be the column headers, we may have to skip more than 1
    with open(csv_file, 'r') as f: # Open the file and read the first line
        first_line = f.readline().strip()
    metadata = first_line.split(',') # Split the first line into a list of metadata
    metadata_dict = {} # Create a dictionary mapping column names to metadata
    for (column, item) in zip(df.columns, metadata):
        try:
            metadata_dict[column] = int(item)
        except ValueError: 
            pass  #metadata_dict[column] = -1
    metadata = set(metadata_dict.values()) #Recreate

    target_columns = df.columns[1:]  # All columns except the first one
    scaling_factors = {}; classDict = {}; classCount = {}
    for idx, column in enumerate(target_columns):
        scaling_factors[column] = { 'min': df[column].min(), 'max': df[column].max() }
        classDict[idx] = column
        classCount[column] = df[column].max()

    unique_WVx = set()
    unique_GTx = set()
    basename_to_WVx = {}  # map from basename to list of 'WVx' files
    basename_to_GTx = {}  # map from basename to list of 'GTx' files
    print("Checking files . . ")
    for root, dirnames, filenames in os.walk(sourceFolder):
        print("Walking . . ")
        for filename in filenames:
            if filename.endswith(('tif')):
                basename, suffix = filename.split('_')[0], filename.split('_')[1]  # Get the base name (like '000001') and suffix (like 'WVx') from the filename
                if "WV" in suffix:
                    unique_WVx.add(int(suffix[2]))  # Add the index to the set of unique 'WVx' indices
                    if basename not in basename_to_WVx:
                        if max_size is not None and len(basename_to_WVx) >= max_size:
                            break
                        basename_to_WVx[basename] = []
                    basename_to_WVx[basename].append(filename)
                if "GT" in suffix:
                    unique_GTx.add(int(suffix[2]))  # Add the index to the set of unique 'GTx' indices
                    if basename not in basename_to_GTx:
                        if max_size is not None and len(basename_to_GTx) >= max_size:
                            break
                        basename_to_GTx[basename] = []
                    basename_to_GTx[basename].append(filename)
            
    file_label_tuples = []
    for basename, WVx_files in basename_to_WVx.items():
        matching_row = df.loc[df['Idx'].apply(lambda x: f"{x:06}") == basename]
        if not matching_row.empty:
            target_values = []           
            for column in target_columns:
                value = matching_row[column].values[0]
                scaled_value = (value - scaling_factors[column]['min']) / (scaling_factors[column]['max'] - scaling_factors[column]['min'])
                target_values.append(scaled_value)
            GTx_files = basename_to_GTx.get(basename, [])
            file_label_tuples.append((basename, WVx_files, GTx_files, *target_values))

    print("GTX ", len(basename_to_GTx), " WVX ", len(basename_to_WVx), " FLTs " ,(file_label_tuples))

    return file_label_tuples, classDict, classCount, scaling_factors, unique_WVx, unique_GTx, metadata_dict


### 6/21/2023 U-net version - Definitions - - - - - - - - - - - -  - - - - - - - - - - - - - - - - - - - - - - -- - - - - - - - - - 

def get_image_arrays_un(src_path, dim_Load, color_mode, channel_order=[0,1,2]):
    data = []; files = []
    for file in os.listdir(src_path):
        if file.split('.')[-1] == 'jpg' or file.split('.')[-1] == 'bmp' or file.split('.')[-1] == 'png':
            img = Image.open(os.path.join(src_path , file))
            img_cs = img.convert(color_mode) 
            img_rs = img_cs.resize([dim_Load, dim_Load])
            img_array = np.asarray(img_rs)
            if len(img_array.shape) == 3:  # if it's a color image
                img_array = img_array[..., channel_order]  # change channel order
            data.append(img_array)
            files.append(file)
    img_X = np.array(data)
    img_X = img_X.astype('float16') / 255  #img_X = img_X.reshape(img_X.shape[0], dim_Load, dim_Load, 1)
    print('Shape of Images:\n', img_X.shape, src_path)
    return img_X, files


def image_generator_un(file_list, dim_load, color_mode_x, color_mode_y, batch_size):
    batch_x = []; batch_y = []
    
    random.shuffle(file_list)
    for x_file, y_file in file_list:
        x_img = Image.open(x_file).convert(color_mode_x).resize((dim_load, dim_load))
        y_img = Image.open(y_file).convert(color_mode_y).resize((dim_load, dim_load))
        x_img = np.array(x_img).astype('float16') / 255
        y_img = np.array(y_img).astype('float16') / 255
        batch_x.append(x_img); batch_y.append(y_img)

        if len(batch_x) == batch_size: # If we've reached the batch size, yield the batch and reset
            yield (np.array(batch_x), np.array(batch_y))
            batch_x = []; batch_y = []

    if batch_x: # If there are any leftover images not yet yielded, yield them now
        yield (np.array(batch_x), np.array(batch_y))

def get_image_xy_list(x_dir, y_dir, file_types=['jpg', 'bmp']):
    x_files = os.listdir(x_dir); y_files = os.listdir(y_dir)

    x_files_dict = {os.path.splitext(x)[0]: os.path.join(x_dir, x) for x in x_files if x.split('.')[-1] in file_types}
    y_files_dict = {os.path.splitext(y)[0]: os.path.join(y_dir, y) for y in y_files if y.split('.')[-1] in file_types}

    common_files = set(x_files_dict.keys()) & set(y_files_dict.keys())
    matching_files = [(x_files_dict[base_name], y_files_dict[base_name]) for base_name in common_files]
    return matching_files

def split_train_test_un(matching_files, max_images, train_ratio = 0.75):    
    train_images = round(max_images * train_ratio)
    train_images = min(train_images, max_images)
    test_images = max_images - train_images

    train_list = matching_files[:train_images]
    test_list = matching_files[train_images:(train_images + test_images)]

    return train_list, test_list

def get_samples_from_gen(generator, num_samples):
    samples_x = []
    samples_y = []
    while len(samples_x) < num_samples:
        x, y = next(generator)
        samples_x.extend(x); samples_y.extend(y)

    return np.array(samples_x[:num_samples]), np.array(samples_y[:num_samples]) # Make sure to return only the number of samples required

def prime_factors(n):
    factors = []; d = 2
    while d * d <= n:
        while (n % d) == 0:
            factors.append(d)
            n //= d
        d += 1
    if n > 1:
        factors.append(n)
    return factors

def build_unet(input_width, input_channels, output_channels, kernel_size, stride, filters_start, filter_rate, dropout_rate, useBatchNorm = False):
    def down_block(x, filters, kernel_size_i, stride_i, padding="same"): # Contraction path
        c = keras.layers.Conv2D(filters, (kernel_size_i, kernel_size_i), padding=padding, activation="relu")(x)
        if useBatchNorm:
            c = keras.layers.BatchNormalization()(c)
        c = keras.layers.Dropout(dropout_rate)(c)
        p = keras.layers.MaxPooling2D(stride_i)(c)
        return c, p

    def up_block(x, skip, filters, kernel_size_i, stride_i, padding="same"): # Expansion path
        us = keras.layers.UpSampling2D(stride_i)(x)
        concat = keras.layers.Concatenate()([us, skip])
        c = keras.layers.Conv2D(filters, (kernel_size_i, kernel_size_i), padding=padding, activation="relu")(concat)
        if useBatchNorm:
            c = keras.layers.BatchNormalization()(c)
        c = keras.layers.Dropout(dropout_rate)(c)
        return c

    s = prime_factors(input_width); depth = len(s)
    f = [int(filters_start * filter_rate ** i) for i in range(depth)]; print("Filters: ",f," Strides:",s);
    inputs = keras.layers.Input((input_width, input_width, input_channels)); p = inputs

    skips = []
    for i in range(depth): # Create contraction path
        c, p = down_block(p, f[i], kernel_size, s[i])
        skips.append(c)

    bn = p
    for i in reversed(range(depth)): # Create expansion path
        bn = up_block(bn, skips[i], f[i], kernel_size, s[i])

    # This is the final layer. You can change the number of output classes.
    outputs = keras.layers.Conv2D(output_channels, (1, 1), padding="same", activation="sigmoid")(bn)
    model = keras.Model(inputs, outputs)

    return model


## - - - - - - - - - 

def split_data_un(data, train_fraction=0.8):
    # Shuffle the data
    random.shuffle(data)
    split_idx = int(len(data) * train_fraction) # Calculate the index at which to split the data
    train_data = data[split_idx:]
    val_data = data[:split_idx]
    return train_data, val_data

def dice_loss(y_true, y_pred):
    smooth = 1.
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)

def combined_loss_generator(weight):
    def combined_dice_bce_loss(y_true, y_pred):
        return (1-weight)*binary_crossentropy(y_true, y_pred) + weight*dice_loss(y_true, y_pred)
    return combined_dice_bce_loss


# - - - - - - -  Training Settings - - - - - - -- - - - - - -  


class TrainingSettings:
    def __init__(self, img_height=256, img_width=256, ksize_Pool=2, nStrides_CNN=1, regulariz=1E-6,
                 mConvDropOut=0.0, mEarlyStopPatience=18, mPenultimateNeuronsCutOff=12, mPostConvDropOut=0.4,
                 nRuns=99999, mEpochs=180, batchSize=32, mTransferName="", numColumns=40, tf_seed= 13):
        self.img_height = img_height
        self.img_width = img_width
        self.ksize_Pool = ksize_Pool
        self.nStrides_CNN = nStrides_CNN
        self.regulariz = regulariz
        self.mConvDropOut = mConvDropOut
        self.mEarlyStopPatience = mEarlyStopPatience
        self.mPenultimateNeuronsCutOff = mPenultimateNeuronsCutOff
        self.mPostConvDropOut = mPostConvDropOut
        self.nRuns = nRuns
        self.mEpochs = mEpochs
        self.mTransferName = mTransferName
        self.BatchSize = batchSize
        self.numColumns = numColumns
        self.TF_seed = tf_seed
        

    def to_json(self, file_path):
        with open(file_path, 'w') as f:
            json.dump(self.__dict__, f)

    @classmethod
    def from_json(cls, file_path):
        with open(file_path, 'r') as f:
            data = json.load(f)
        return cls(**data)
    

class RandomizedSettings:
    def __init__(self, conv_max, conv_layers, filter_nm, cnn_filter_increase_rate, kernel_sz, used_batch_norm_conv, dense_dropouts, dense_batch_norm, cnn_residual_style, UseMultiLoss, dense_min=40, dense_max=80, denseSizesTabular = None):
        self.conv_max = conv_max
        self.conv_layers = conv_layers
        self.filter_nm = filter_nm
        self.cnn_filter_increase_rate = cnn_filter_increase_rate
        self.kernel_sz = kernel_sz
        self.used_batch_norm_conv = used_batch_norm_conv
        self.dense_dropouts = dense_dropouts
        self.dense_batch_norm = dense_batch_norm
        self.cnn_residual_style = cnn_residual_style
        self.use_multi_loss = UseMultiLoss
        self.dense_min = dense_min
        self.dense_max = dense_max
        self.denseSizesTabular = denseSizesTabular

    @classmethod
    def get_randomized_settings(cls, settings, filter_low=18, filter_high=36, dense_min=40, dense_max=80, denseSizesTabularInput = None):
        conv_max = max(1, int(math.log(settings.img_height, settings.ksize_Pool)))
        conv_layers = randint(max(1, conv_max - 2), conv_max)
        filter_nm = randint(filter_low, filter_high)
        cnn_filter_increase_rate = 1 + (randint(1, 3) / 4)
        kernel_sz = randint(2, 4)
        used_batch_norm_conv = randint(0, 1) * randint(0, 1) * randint(0, 1)
        dense_dropouts = randint(0, 3) / 6
        dense_batch_norm = randint(0, 1) * randint(0, 1)
        cnn_residual_style = randint(0,1) #2 is also an option, but turned off
        use_multi_loss = 1 - (randint(0,1) * randint(0,1)) #0 means don't, 1 means do - - more likely that we do
        if (denseSizesTabularInput == None):
            denseSizesTabularA = [randint(dense_min, dense_max), randint(dense_min, dense_max), randint(dense_min, dense_max)]

        return cls(conv_max, conv_layers, filter_nm, cnn_filter_increase_rate, kernel_sz, used_batch_norm_conv, dense_dropouts, dense_batch_norm, cnn_residual_style, use_multi_loss, denseSizesTabular=denseSizesTabularA)

#  - - - - - - - Model Pre Metadata - - - - - 

class TF_Model_Input_Meta:
    def __init__(self):
        self.ModelType = ""
        self.ColumnsIncluded = ""
        self.OutputMapping = {}
        self.AdditionalNote = ""
        self.ModelParams = ""
        self.PreferredInputShape = ""
        self.InputChannelNames = []
        self.WillModelRescale = False
        self.WillModelResize = False
        self.ResizeByCropTrue_ZoomFalse = True

# - - - - - - -  Model Metadata - - - - - - - - - - - - - - - - 

class TF_Model_Metadata:
    def __init__(self):
        self.Index2ClassInfo = {}
        self.ModelName = ""
        self.ModelPath = ""
        self.ModelType = ""
        self.ModelDate = ""
        self.ThresholdDefault = 0.7
        self.TrainingSize = 0
        self.ColumnsIncluded = ""
        self.OutputMapping = {}
        self.AdditionalNote = ""
        self.ModelParams = ""
        
        self.PreferredInputShape = ""
        self.InputChannelNames = []
        self.WillModelRescale = False
        self.WillModelResize = False
        self.ResizeByCropTrue_ZoomFalse = True

    def copy_from_input_meta(self, input_meta):
        if input_meta is None:
            return
        for attr in vars(input_meta):
            if hasattr(self, attr):
                setattr(self, attr, getattr(input_meta, attr))
        self.OutputMapping = input_meta.OutputMapping #Didn't seem to copy automatically

    def add_class(self, idx: int, name: str, training_count: int, rgb: bytes = None, comment: str = "", scalerMin: float = 0, scalerMax: float = 1):
        tfci = TF_Model_ClassInfo(idx, name, training_count, rgb, comment, scalerMin, scalerMax)
        self.Index2ClassInfo[idx] = tfci

    def to_dict(self) -> Dict:
        exclude = {'Index2ClassInfo'}
        result = {key: getattr(self, key) for key in self.__dict__ if key not in exclude}
        result["Index2ClassInfo"] = {idx: tfci.to_dict() for idx, tfci in self.Index2ClassInfo.items()}
        return result

    def to_json(self) -> str:
        def default(o):
            if isinstance(o, np.integer):
                return int(o)
            raise TypeError
        return json.dumps(self.to_dict(), indent=4, default=default)

    @classmethod
    def from_dict(cls, data: Dict):
        obj = cls()
        for key, value in data.items():
            if key == "Index2ClassInfo":
                obj.Index2ClassInfo = {idx: TF_Model_ClassInfo.from_dict(tfci_data) for idx, tfci_data in value.items()}
            else:
                setattr(obj, key, value)
        return obj

    @classmethod
    def load_from_json_file(cls, filepath: str):
        with open(filepath, 'r') as file:
            data = json.load(file)
        return cls.from_dict(data)

class TF_Model_ClassInfo:
    def __init__(self, idx: int, name: str, training_count: int, rgb: bytes = None, comment: str = "", scalerMin: float = 0, scalerMax: float = 1):
        self.Index = idx
        self.Name = name
        self.TrainingCount = training_count
        self.RGB = list(rgb if not rgb==None else bytes([randint(0,255),randint(0,255),randint(0,255)]))
        self.Comment = comment
        self.Ignore = False
        self.ThresholdMultiplier = 1
        self.ScalerMin = scalerMin
        self.ScalerMax = scalerMax

    def to_dict(self) -> Dict:
        return {key: getattr(self, key) for key in self.__dict__}
    
    @classmethod
    def from_dict(cls, data: Dict):
        obj = cls.__new__(cls)  # Create a new instance without calling __init__
        for key, value in data.items():
            if hasattr(obj, key):  # Check if the object has the attribute
                setattr(obj, key, value)
        return obj

def TestTFUsage(): # Example usage:
    tfmm = TF_Model_Metadata()
    tfmm.add_class(0, "Empty", -1, bytes([184, 118, 232]))
    tfmm.add_class(1, "Neurites No Cell", -1, bytes([193, 73, 75]))
    tfmm.add_class(2, "Cell Tub No Neurite", -1, bytes([40, 251, 57]))
    tfmm.add_class(3, "Dead (no tubulin)", -1, bytes([113, 206, 14]))
    tfmm.add_class(4, "Neuron", -1, bytes([174, 172, 144]))
    tfmm.add_class(5, "Neuron Poor Qual", -1, bytes([203, 14, 5]))
    tfmm.add_class(6, "Cells Tub No Neurite", -1, bytes([167, 122, 36]))
    tfmm.add_class(7, "Nuc 1+ Neuron", -1, bytes([136, 33, 162]))
    tfmm.add_class(8, "Nuc 1+ Neuron Poor Qual", -1, bytes([87, 138, 227]))
    tfmm.add_class(9, "3-4 Nuclei", -1, bytes([250, 245, 255]))
    tfmm.add_class(10, "5+ Nuclei", -1, bytes([62, 45, 26]))
    tfmm.add_class(11, "Focus (only if you really cant tell)", -1, bytes([253, 246, 41]))

    tfmm.ModelName = "acc=0.8177 CNN Cnns=6.5"
    tfmm.ModelType = "Multiclass"
    tfmm.ModelDate = "2023-03-10T14:45:00Z"

    json_str = tfmm.to_json()
    print(json_str)


def GetImageCountsPerClass(tfImageDataset, classNamesA=None):
    classCounts = {}
    
    # Try to get the built-in class names, use provided class names if not available
    classNames = getattr(tfImageDataset, 'class_names', classNamesA)
    
    for images, labels in tfImageDataset:
        for label in labels:
            class_name = classNames[label.numpy()] # Use numpy() to convert label to a Python scalar
            if class_name not in classCounts:
                classCounts[class_name] = 0
            classCounts[class_name] += 1
    return classCounts


def SaveModelMetadata(ClassDict, ClassCounts, ScalingValues, TotalTrainingSize, ModelName, ModelType, ModelPath, output_file = "", featureNames = [""], input_metadata = None):
    #print(output_file + " " + str(featureNames))
    tfmm = TF_Model_Metadata()
    tfmm.ModelDate = str(datetime.now())
    tfmm.ModelName = ModelName
    tfmm.ModelType = ModelType
    tfmm.ModelPath = ModelPath
    tfmm.TrainingSize = TotalTrainingSize
    tfmm.ColumnsIncluded = '|'.join(featureNames)
    
    tfmm.copy_from_input_meta(input_metadata)
    
    if (ClassDict is not None):
        for key, value in ClassDict.items():
            tfmm.add_class(
                key, value, int(ClassCounts[value]) if not ClassCounts==None else -1, None, "", 
                ScalingValues[value]['min'] if not ScalingValues==None else 0,
                ScalingValues[value]['max'] if not ScalingValues==None else 1)
    json_str = tfmm.to_json()
    if (output_file == ""):
        #isFile = os.path.isfile(fpath)
        #isDirectory = os.path.isdir(fpath)
        #dirname, fname = os.path.split(fullpath)
        output_file = os.path.join(ModelPath, "FIVE_metadata.json")
    with open(output_file, 'w') as f:
        f.write(json_str)


# - - - - - - - - - Multi Target Functions - - - - - - - - - - - - - - - - - 


# Experimental . . Ignore for production

# Function Definition

def LoadTargetData(sourceFolder, csvfile):
    df = pd.read_csv(csvfile)
    file_label_tuples = []
    target_columns = df.columns[1:]  # All columns except the first one

    scaling_factors = {}; classDict = {}; classCount = {}
    for idx, column in enumerate(target_columns):
        scaling_factors[column] = { 'min': df[column].min(), 'max': df[column].max() }
        classDict[idx] = column
        classCount[column] = df[column].max()

    for root, dirnames, filenames in os.walk(sourceFolder):
        for filename in filenames:
            if filename.endswith(('bmp')):
                matching_row = df.loc[df['Filename'] == filename]
                if not matching_row.empty:
                    target_values = []           
                    for column in target_columns:
                        value = matching_row[column].values[0]
                        scaled_value = (value - scaling_factors[column]['min']) / (scaling_factors[column]['max'] - scaling_factors[column]['min'])
                        target_values.append(scaled_value)
                    file_label_tuples.append((filename, *target_values))

    return file_label_tuples, classDict, classCount, scaling_factors 

def rescale_predictions(predictions, scaling_factors):
    rescaled_targets = []
    for i, column in enumerate(scaling_factors):
        column_min = scaling_factors[column]['min']
        column_max = scaling_factors[column]['max']
        rescaled_target = predictions[i] * (column_max - column_min) + column_min
        rescaled_targets.append(rescaled_target)
    return rescaled_targets

#Only runs when the image is loaded
def data_generator_multiTarget(file_label_tuples, sourceFolder, batch_size, for_inference=False):
    num_samples = len(file_label_tuples)
    num_targets = len(file_label_tuples[0]) - 1 if not for_inference else 0

    while True:
        if not for_inference:
            np.random.shuffle(file_label_tuples)

        # Generate batches
        for start_idx in range(0, num_samples, batch_size):
            end_idx = min(start_idx + batch_size, num_samples)
            batch_file_label_tuples = file_label_tuples[start_idx:end_idx]

            images = []
            labels = []

            for file_label_tuple in batch_file_label_tuples:
                file = file_label_tuple[0]
                values = file_label_tuple[1:] if not for_inference else []
                img_path = os.path.join(sourceFolder, file)
                img = load_img(img_path)
                img_array = img_to_array(img)  # / 255.0

                images.append(img_array)
                labels.append(values)

            X_batch = np.array(images)
            if for_inference:
                yield X_batch
            else:
                y_batch = np.array(labels)
                yield X_batch, y_batch

## Experimenting with better per-target metrics

# Custom metric for individual target MAE
class CustomMAE(MeanAbsoluteError):
    def __init__(self, target_index, **kwargs):
        super().__init__(**kwargs)
        self.target_index = target_index

    def update_state(self, y_true, y_pred, sample_weight=None):
        y_true_slice = y_true[:, self.target_index]
        y_pred_slice = y_pred[:, self.target_index]
        super().update_state(y_true_slice, y_pred_slice, sample_weight)

class CustomR2(tf.keras.metrics.Metric):
    def __init__(self, target_index, **kwargs):
        super().__init__(**kwargs)
        self.target_index = target_index
        self.mean_y_true = Mean()
        self.mean_y_pred = Mean()
        self.mean_product = Mean()
        self.mean_y_true_square = Mean()
        self.mean_y_pred_square = Mean()

    def update_state(self, y_true, y_pred, sample_weight=None):
        y_true_slice = y_true[:, self.target_index]
        y_pred_slice = y_pred[:, self.target_index]

        self.mean_y_true.update_state(y_true_slice, sample_weight)
        self.mean_y_pred.update_state(y_pred_slice, sample_weight)
        self.mean_product.update_state(y_true_slice * y_pred_slice, sample_weight)
        self.mean_y_true_square.update_state(tf.square(y_true_slice), sample_weight)
        self.mean_y_pred_square.update_state(tf.square(y_pred_slice), sample_weight)

    def result(self):
        mean_y_true = self.mean_y_true.result()
        mean_y_pred = self.mean_y_pred.result()
        mean_product = self.mean_product.result()
        mean_y_true_square = self.mean_y_true_square.result()
        mean_y_pred_square = self.mean_y_pred_square.result()

        covariance = mean_product - mean_y_true * mean_y_pred
        variance_y_true = mean_y_true_square - tf.square(mean_y_true)
        variance_y_pred = mean_y_pred_square - tf.square(mean_y_pred)

        correlation_coefficient = covariance / tf.sqrt(variance_y_true * variance_y_pred)
        return correlation_coefficient

    def reset_state(self):
        self.mean_y_true.reset_states()
        self.mean_y_pred.reset_states()
        self.mean_product.reset_states()
        self.mean_y_true_square.reset_states()
        self.mean_y_pred_square.reset_states()



# - - - - - - - -  Functions for Model Parameters - - - - - - - - - - - - - - 



# - - - - - - - Function Definitions - - Feature Set - - - - - - - - - - - - - - 



def LimitColumns(dataframe_X, Keep_ifInName_List, KeepOtherColumn = "", AddBackInAreaInt = False):
    Xp = dataframe_X
    X_cols_keep = [col for col in Xp.columns if [x for x in Keep_ifInName_List if x in col]] 

    #'SPACING', 'NEIGHBOR', 
    prohibited_substrings = ['CG', 'GLOBAL', 'BORDER', 'ProbPr', 'PrPro', '(SOI)', 'FAKE', 'X UM', 'Y UM', 'Well Idx', 'PlateIdx', 'DispenseIdx','Replicate','Tip']
    X_cols_throw = [col for col in Xp.columns if any(substring in col for substring in prohibited_substrings)]
    X_cols_keep = [x for x in X_cols_keep if x not in X_cols_throw]
    X = Xp[X_cols_keep].copy()
    #--The following line puts back the two main WVH features
    if (AddBackInAreaInt):
        X[['NUCLEI AREA WVH','NUCLEI INTENSITY WVH']] = dataframe_X[['NUCLEI AREA WVH','NUCLEI INTENSITY WVH']]
    if (KeepOtherColumn != ""):
        X[KeepOtherColumn] = Xp[KeepOtherColumn]
    return X

# Norm_SubsetBy is used to equalize all the plates for instance, generally good for intensity measures but not for other measures
def NormalizeF(dataframe_X, Norm_SubsetBy = ""):
    X = dataframe_X.fillna(0)

    if (Norm_SubsetBy != ""):
        groups = dataframe_X.groupby(Norm_SubsetBy)
        #mean, std = groups.transform("mean"), groups.transform("std") #Gets both StDev and Mean
        #df.groupby('indx').transform(lambda x: (x - x.mean()) / x.std()) Slower way
        mean = groups.transform("mean", numeric_only=True)
        X = dataframe_X[mean.columns] / mean
        #normalized = normalized.join(df3[Norm_SubsetBy])
        #normalized.to_csv(r"s:\Feature\FIV836\try.csv")
    
    X=(X-X.min())/(X.max()-X.min()) 
    return X

def NewG2From(specificClassForG2, ListOfClassesPerCell, classesUnique):
    Y = ListOfClassesPerCell
    for cl in classesUnique:
        if (cl==specificClassForG2):
            Y = Y.replace(cl,"G2")    
        else:
            Y = Y.replace(cl,"Mix")    
    return Y

def PrepXY(dataframe_X, dataframe_Y):
    #Remove non-trainable rows
    Y = dataframe_Y.replace('L76P HET A3','G2').replace('L76P C6','G2').replace('R94Q D4','G2')
    Y = Y.replace('R94Q E7','G2').replace('R94Q D10','G2').replace('WTC11 GEN2C','G1')
    rows_Remove = Y.index[Y.str.contains('MIX')].tolist()
    #pd.DataFrame(rows_Remove).to_clipboard()
    X = dataframe_X.drop(dataframe_X.index[rows_Remove]).fillna(0)
    Y = Y.drop(Y.index[rows_Remove])

    #UnComment below if you want to turn the columns numeric
    #Y = Y.replace('Pathogenic',1).replace('Benign',0) #turn the labels into numbers
    mClasses = Y.unique()
    mClassesCount = len(mClasses)
    values, counts = np.unique(Y, return_counts=True)
    RowsPerY = dict(zip(values, counts))

    #Y = pd.get_dummies(Y) #This hot-encodes the classes
    le = LabelEncoder(); le.fit(Y)
    Y = le.transform(Y) #This turns the labeled columns into numbers
    Y_Mapping = dict(zip(le.transform(le.classes_),le.classes_))
    #Y_Mapping = [le.classes_, le.transform(le.classes_)]
    #Y_Mapping = np.transpose(Y_Mapping)

    ## Get Labels and figure out balance (this sets the class weights)
    class_weights = class_weight.compute_class_weight('balanced', classes=np.unique(Y), y=Y) #https://stackoverflow.com/questions/44716150/how-can-i-assign-a-class-weight-in-keras-in-a-simple-way
    class_weights = {i : class_weights[i] for i in range(len(class_weights))} #https://stackoverflow.com/questions/61261907/on-colab-class-weight-is-causing-a-valueerror-the-truth-value-of-an-array-wit
    return X, Y, mClasses, RowsPerY, Y_Mapping, class_weights

def calculate_metrics_2class(y_true, y_pred, n_classes):
    is_binary = n_classes == 2
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, average='weighted')
    recall = recall_score(y_true, y_pred, average='weighted')
    f1 = f1_score(y_true, y_pred, average='weighted')

    if is_binary:
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
        sensitivity = tp / (tp + fn)
        specificity = tn / (tn + fp)
        auc = roc_auc_score(y_true, y_pred)
    else:
        sensitivity, specificity, auc = None, None, None

    metrics = {
        'Accuracy': accuracy,
        'Precision': precision,
        'Sensitivity': sensitivity,
        'Specificity': specificity,
        'AUC': auc,
        'F1_score': f1
    }

    return metrics

def save_metrics_to_csv(Xp, predictions, Y_Class_Mapping, col_TrainingLayouts, col_GenotypedAs, ModelName, csv_file_path, featureNames):
    # Reverse the class mapping
    inv_Y_Class_Mapping = {v: k for k, v in Y_Class_Mapping.items()}
    metrics_dict = {}
    n_classes = len(Y_Class_Mapping)
    TrainingSets = Xp[col_TrainingLayouts].unique()

    # Calculate metrics for each subset
    calculate_metrics = True if col_GenotypedAs != "" else False
    if calculate_metrics:
        for value in TrainingSets:
            subset_indices = Xp[Xp[col_TrainingLayouts] == value].index

            # Filter out rows with class labels not present in inv_Y_Class_Mapping
            subset_true_labels = Xp.loc[subset_indices, col_GenotypedAs]
            valid_indices = subset_true_labels.isin(inv_Y_Class_Mapping.keys())
            valid_subset_indices = subset_indices[valid_indices]
            true_labels = subset_true_labels[valid_indices].map(inv_Y_Class_Mapping).values

            # Get the predictions for the current subset
            subset_predictions = predictions.loc[valid_subset_indices, :]
            subset_pred_labels = np.argmax(subset_predictions.values, axis=-1)

            # Calculate and store the metrics
            metrics_dict[value] = calculate_metrics_2class(true_labels, subset_pred_labels, n_classes)

    # Write metrics to CSV
    file_exists = os.path.isfile(csv_file_path)
    with open(csv_file_path, 'a', newline='') as f:
        writer = csv.writer(f)
        if not file_exists:
            # Write headers if the file doesn't exist
            headers = ['ModelName', 'Subset', 'Features']
            if calculate_metrics:
                first_item_key = next(iter(metrics_dict))
                headers += list(metrics_dict[first_item_key].keys())
            writer.writerow(headers)
        for key, metrics in metrics_dict.items():
            row = [ModelName, key, '|'.join(featureNames)]
            if calculate_metrics:
                row += list(metrics.values())
            writer.writerow(row)


# - - - - - - -More CNN functions - - - - - - - - - -


def get_transfer_model(transfer_num, img_width, img_height):
    if transfer_num == 1:
        conv_base = keras.applications.nasnet.NASNetMobile(weights='imagenet', include_top=False, input_shape=(img_width, img_height, 3))
        transfer_name = "NasNetM"
    elif transfer_num == 2:
        conv_base = keras.applications.xception.Xception(weights='imagenet', include_top=False, input_shape=(img_width, img_height, 3))
        transfer_name = "Xcption"
    elif transfer_num == 3:
        conv_base = keras.applications.MobileNetV3Small(weights='imagenet', include_top=False, input_shape=(img_width, img_height, 3))
        transfer_name = "MobV3s"
    elif transfer_num == 4:
        conv_base = keras.applications.ResNet50(weights='imagenet', include_top=False, input_shape=(img_width, img_height, 3))
        transfer_name = "RsNt50"
    elif transfer_num == 5:
        conv_base = keras.applications.efficientnet_v2.EfficientNetV2B0(weights='imagenet', include_top=False, input_shape=(img_width, img_height, 3))
        transfer_name = "EffNtV2B0"
    elif transfer_num == 6:
        conv_base = keras.applications.inception_v3.InceptionV3(weights='imagenet', include_top=False, input_shape=(img_width, img_height, 3))
        transfer_name = "IncepV3"
    elif transfer_num == 7:
        conv_base = keras.applications.convnext.ConvNeXtTiny(weights='imagenet', include_top=False, input_shape=(img_width, img_height, 3))
        transfer_name = "ConvNTi"
    else:
        conv_base = None
        transfer_name = ""

    return conv_base, transfer_name


def build_conv_block_with_residuals(mNext, settings, random_settings, AddResiduals): ##ResNet Style residuals
    filter_nm_use = random_settings.filter_nm
    
    for mC in range(0, random_settings.conv_layers):
        mInput = mNext
        
        if AddResiduals and mNext.shape[-1] != filter_nm_use:
            mInput = keras.layers.Conv2D(filters=filter_nm_use, kernel_size=1, strides=settings.nStrides_CNN, padding="same")(mInput)
            mInput = keras.layers.BatchNormalization()(mInput)
        
        mNext = keras.layers.Conv2D(filters=filter_nm_use, kernel_size=random_settings.kernel_sz, strides=settings.nStrides_CNN, padding="same", use_bias=True, kernel_initializer='glorot_uniform', bias_initializer='zeros')(mNext)
        if (random_settings.used_batch_norm_conv == 1):
            mNext = keras.layers.BatchNormalization()(mNext)
        mNext = keras.layers.Activation("relu")(mNext)
        
        if AddResiduals:
            mNext = keras.layers.Add()([mNext, mInput])
            mNext = keras.layers.Activation("relu")(mNext)
        
        mNext = keras.layers.MaxPool2D(settings.ksize_Pool)(mNext)
        mNext = keras.layers.Dropout(settings.mConvDropOut)(mNext)
        print("Conv Pooled " + str(mC) + ": " + str(mNext.shape))
        
        filter_nm_use = int(filter_nm_use * random_settings.cnn_filter_increase_rate)
    
    return mNext


def add_conv_layers_with_residuals(mNext, settings, random_settings, AddResiduals=True): ##Style that is more similar to my original CNN architecture (works much better)
    filter_nm_use = random_settings.filter_nm

    for mC in range(0, random_settings.conv_layers):
        mInput = mNext #Store the input tensor
        mNext = keras.layers.Conv2D(filters=filter_nm_use, kernel_size=random_settings.kernel_sz, strides=settings.nStrides_CNN, padding="same", use_bias=True, kernel_initializer='glorot_uniform', bias_initializer='zeros', activation=None)(mNext)

        if AddResiduals:
            if mInput.shape[-1] != mNext.shape[-1]:
                mInput = keras.layers.Conv2D(filters=mNext.shape[-1], kernel_size=1, strides=1, padding="same", use_bias=False, kernel_initializer='glorot_uniform')(mInput)

            mNext = keras.layers.Add()([mNext, mInput])

        mNext = keras.layers.Activation("relu")(mNext)
        mNext = keras.layers.MaxPool2D(settings.ksize_Pool)(mNext)

        if (random_settings.used_batch_norm_conv == 1):
            mNext = keras.layers.BatchNormalization()(mNext)

        mNext = keras.layers.Dropout(settings.mConvDropOut)(mNext)
        print("Conv Pooled " + str(mC) + ": " + str(mNext.shape))

        filter_nm_use = int(filter_nm_use * random_settings.cnn_filter_increase_rate)

    return mNext


# - - - - - - - - Tabular NAS Run  - - - - - - - 

def NAS_Run(randomized_settings, training_settings, dfSaveRes, FolderSave, Prefix, Input_X, Input_Y, PreDrop_X, Y_Class_Mapping, rowsPerClass, dctClassWeigts, Xp_Full, 
            col_Label, col_TrainingLayouts="", col_GenotypedAs="", CalcUMAP=True, saveModels=True, TF_seed=13):
    
    Prefix = re.sub(r'[^\w\s\-_\.,]', '_', Prefix)  # Regular expression to clean this Prefix from any unsafe characters
    ClassesCount = len(Y_Class_Mapping); tf.random.set_seed(training_settings.TF_seed)
    callBack = keras.callbacks.EarlyStopping(monitor="val_loss", patience=training_settings.mEarlyStopPatience, baseline=True)
    for featureRS in range(training_settings.nRuns):
        denseSizes = randomized_settings.denseSizesTabular
        print("denses=" + str(denseSizes))

        ### --- Extract features and get train/test split
        Xs = Input_X.sample(n=min(Input_X.shape[1], training_settings.numColumns), axis='columns', random_state=featureRS)
        column_headers = Xs.columns.tolist(); print("Cols sampled: " + str(column_headers))
        X_train, X_test, Y_train, Y_test = train_test_split(Xs, Input_Y, test_size=0.3, random_state=119)
        print("TrainTest sizes = " + str(X_train.shape) + " " + str(X_test.shape))

        ### --- Define Model
        mInput = keras.Input(shape=(Xs.shape[1])); mNext = mInput
        for dS in denseSizes:
            mNext = keras.layers.Dense(units=dS, kernel_regularizer=keras.regularizers.L2(training_settings.regulariz), activation='relu')(mNext)
            mNext = keras.layers.Dropout(training_settings.mPostConvDropOut)(mNext)
        mOut = keras.layers.Dense(ClassesCount, activation='softmax')(mNext)
        mModel = keras.Model(mInput, mOut)
        mModel.compile(loss=tf.losses.SparseCategoricalCrossentropy(), optimizer=keras.optimizers.Adam(), metrics=["accuracy"])

        hist = mModel.fit(X_train, Y_train, validation_data=(X_test, Y_test), epochs=training_settings.mEpochs, batch_size=training_settings.BatchSize, verbose=2, class_weight=dctClassWeigts, callbacks=[callBack])
        last_accuracy_Show, mEpochsActual, bestAcc = QWorstCase_Accuracy(hist, training_settings.mEpochs, 4, "accuracy")

        szName = Prefix + " acc=" + f'{bestAcc:.4f}' + " nC=" + str(training_settings.numColumns) + " Feat_rs=" + str(featureRS) + " nE=" + str(mEpochsActual) + "," + str(training_settings.mEpochs) + " lay=" + str(denseSizes) + " BS=" + str(training_settings.BatchSize) + " TF_rs=" + str(TF_seed)

        plt.figure(figsize=(12, 5))
        ax = plt.subplot(1, 2, 2); QPlot_Loss(hist, 'accuracy'); ax = plt.subplot(1, 2, 1); QPlot_Loss(hist, 'loss')
        plt.savefig(os.path.join(FolderSave,szName + ".jpg")); plt.clf()

        #Predict with the new model
        fullrows_X = pd.DataFrame(PreDrop_X[column_headers]) #Xs.columns
        pred = mModel.predict(fullrows_X)
        p = pd.DataFrame(np.squeeze(pred)); 
        #y2 = np.expand_dims(predLabels,1)
        p = p.rename(columns=Y_Class_Mapping) ; p = p.add_prefix(szName + " ")
        inv_Y_Class_Mapping = {v: k for k, v in Y_Class_Mapping.items()}
        if (saveModels):
            MultiModelSave(mModel, FolderSave, szName + "M", classDict=inv_Y_Class_Mapping, featureNames=column_headers) 
        dfSaveRes = dfSaveRes.join(p)
        save_metrics_to_csv(Xp_Full, p, Y_Class_Mapping, col_TrainingLayouts, col_GenotypedAs , szName, os.path.join(FolderSave,"evalMetrics01.csv"), featureNames = column_headers)
        #save_metrics_to_csv(Xp_Full, p, Y_Class_Mapping, col_TrainingLayouts, col_GenotypedAs if col_GenotypedAs != "" else col_Label, szName, os.path.join(FolderSave,"evalMetrics01.csv"), featureNames = column_headers)

        if (CalcUMAP):
            print("Calculating UMAP . .")
            toUMAP = fullrows_X.join(p)
            uMAPembed = pd.DataFrame(umap.UMAP().fit_transform(toUMAP)).add_prefix(szName + " ")
            dfSaveRes = dfSaveRes.join(uMAPembed)
        
        print("Saving ResSave00 . . ")
        dfSaveRes.to_csv(os.path.join(FolderSave,"ResSave00.csv")) #Save each time for safety
        OccasionalWait(10 if CalcUMAP else 3, 2, 20)
        clear_output(wait=True)
    return dfSaveRes, mModel, Xs.columns


def NAS_Run_Old(dfSaveRes, FolderSave, Prefix, Input_X, Input_Y, PreDrop_X, Y_Class_Mapping, rowsPerClass, dctClassWeigts, numRuns, numEpochs, numColumns, RegularizeAmount, 
            DropOutAmount, EarlyStopPatience, Batch_SIZE, Xp_Full, col_TrainingLayouts = "", col_GenotypedAs = "", CalcUMAP = True, saveModels = True, TF_seed = 13):
    
    Prefix = re.sub(r'[^\w\s\-_\.,]', '_', Prefix) #Regular expression to clean this Prefix from any unsafe characters
    ClassesCount = len(Y_Class_Mapping)
    callBack = keras.callbacks.EarlyStopping(monitor="val_loss", patience=EarlyStopPatience, baseline=True)
    for featureRS in range(numRuns):
        #denseSizes = [randint(100,220),randint(400,400),randint(4,80)]
        denseSizes = [randint(40,80), randint(40,80), randint(2,40)]; tf.random.set_seed(TF_seed)
        print("denses=" + str(denseSizes))

        ### --- Extract features and get train/test split
        Xs = Input_X.sample(n=min(Input_X.shape[1],numColumns), axis='columns', random_state=featureRS)
        column_headers = Xs.columns.tolist(); print("Cols sampled: " + str(column_headers))
        X_train, X_test, Y_train, Y_test = train_test_split(Xs, Input_Y, test_size=0.3, random_state=119)
        print("TrainTest sizes = " + str(X_train.shape) + " " + str(X_test.shape))

        ### --- Define Model
        mInput = keras.Input(shape = (Xs.shape[1])); mNext = mInput
        for dS in denseSizes: 
            mNext = keras.layers.Dense(units=dS, kernel_regularizer=keras.regularizers.L2(RegularizeAmount), activation='relu')(mNext)
            mNext = keras.layers.Dropout(DropOutAmount)(mNext)
        mOut = keras.layers.Dense(ClassesCount, activation='softmax')(mNext)
        mModel = keras.Model(mInput, mOut)
        mModel.compile(loss=tf.losses.SparseCategoricalCrossentropy(), optimizer=keras.optimizers.Adam(), metrics=["accuracy"])

        hist = mModel.fit(X_train, Y_train,validation_data=(X_test, Y_test), epochs=numEpochs, batch_size=Batch_SIZE, verbose=2, class_weight=dctClassWeigts, callbacks=[callBack])
        last_accuracy_Show, mEpochsActual, bestAcc = QWorstCase_Accuracy(hist, numEpochs, 4, "accuracy")

        szName = Prefix + " acc=" + f'{bestAcc:.4f}' + " nC=" + str(numColumns) + " Feat_rs=" + str(featureRS) + " nE=" + str(numEpochs) +"," + str(mEpochsActual) + " lay=" + str(denseSizes) + " BS=" + str(Batch_SIZE) + " TF_rs=" + str(TF_seed)

        plt.figure(figsize=(12, 5))
        ax = plt.subplot(1, 2, 2); QPlot_Loss(hist, 'accuracy') ; ax = plt.subplot(1, 2, 1); QPlot_Loss(hist, 'loss')
        plt.savefig(os.path.join(FolderSave,szName + ".jpg")); plt.clf()

        #Predict with the new model
        fullrows_X = pd.DataFrame(PreDrop_X[column_headers]) #Xs.columns
        pred = mModel.predict(fullrows_X)
        p = pd.DataFrame(np.squeeze(pred)); 
        #y2 = np.expand_dims(predLabels,1)
        p = p.rename(columns=Y_Class_Mapping) ; p = p.add_prefix(szName + " ")
        inv_Y_Class_Mapping = {v: k for k, v in Y_Class_Mapping.items()}
        if (saveModels):
            MultiModelSave(mModel, FolderSave, szName + "M", classDict=inv_Y_Class_Mapping, featureNames=column_headers) 
        dfSaveRes = dfSaveRes.join(p)
        if (col_TrainingLayouts != "" and col_GenotypedAs != ""):
            save_metrics_to_csv(Xp_Full, p, Y_Class_Mapping, col_TrainingLayouts, col_GenotypedAs, szName, os.path.join(FolderSave,"evalMetrics01.csv"), featureName = column_headers)

        if (CalcUMAP):
            print("Calculating UMAP . .")
            toUMAP = fullrows_X.join(p)
            uMAPembed = pd.DataFrame(umap.UMAP().fit_transform(toUMAP)).add_prefix(szName + " ")
            dfSaveRes = dfSaveRes.join(uMAPembed)
        
        dfSaveRes.to_csv(os.path.join(FolderSave,"ResSave00.csv")) #Save each time for safety
        OccasionalWait(3 if CalcUMAP else 20, 2, 20)
        clear_output(wait=True)
    return dfSaveRes, mModel, Xs.columns