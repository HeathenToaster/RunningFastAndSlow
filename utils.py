"""This module contains a set of low-level/utility functions
that are used in the analysis of the data and shared across
different scripts."""


import numpy as np
import matplotlib.pyplot as plt
import os
import pickle
import pandas as pd
import copy
from itertools import chain
import csv


# conversion
def inch2cm(value): return value * 2.54
def cm2inch(value): return value / 2.54


def matchsession(animal, sessionlist, AMPM=False):
    """Function to get the sessions of a specific animal in a list of sessions
    animal: string, name of the animal
    sessionlist: list of strings, list of all the sessions
    AMPM: string, "AM" or "PM" to filter the sessions by time of the day"""
    list = [session for session in sessionlist if animal == session[0:len(animal)]]
    if AMPM:
        list = [session for session in list if int(session[-8:-6]) < 14] if AMPM == "AM" else [session for session in list if int(session[-8:-6]) > 14]
    return (list)


def movinavg(interval, window_size):
    """Function to compute the moving average of a list of values
    interval: list of values
    window_size: int, size of the window to compute the average"""
    if window_size != 0:
        window = np.ones(int(window_size))/float(window_size)
    else:
        print("Error: Window size == 0")
    return np.convolve(interval, window, 'same')


def movinmedian(interval, window_size):
    """same as moving average but with median"""
    if window_size != 0:
        window = int(window_size)
    else:
        print("Error: Window size == 0")
    val = pd.Series(interval)
    return val.rolling(window).median()


def reversemovinmedian(interval, window_size):
    """same as moving average but with median, starting from the end (aquisition drift)"""
    if window_size != 0:
        window = int(window_size)
    else:
        print("Error: Window size == 0")
    val = pd.Series(interval[::-1])
    return list(reversed(val.rolling(window).median()))


# function to read the parameters for each rat for the session in the
# behav.param file. Specify the name of the parameter that you want
# to get from the file and optionally the value type that you want.
# File path is not an option, maybe change that. Dataindex is in case
# you don't only want the last value in line, so you can choose which
# value you want using its index --maybe add the
# option to choose a range of values.
def read_params(root, animal, session, paramName, dataindex=-1, valueType=str):
    # define path of the file
    behav = root + os.sep+animal + os.sep+"Experiments" + \
            os.sep + session + os.sep + session + ".behav_param"
    # check if it exists
    if not os.path.exists(behav):
        print("No file %s" % behav)
    # check if it is not empty
    # if os.stat(behav).st_size == 0:
        # print("File empty %s" % behav)
    with open(behav, "r") as f:
        # scan the file for a specific parameter, if the name of
        # the parameter is there, get the value
        for line in f:
            if valueType is str:
                if paramName in line:
                    # get the last value of the line [-1], values are
                    # separated with _blanks_ with the .split() function
                    return int(line.split()[dataindex])
            if valueType is float:
                if paramName in line:
                    return float(line.split()[dataindex])
            else:
                if paramName in line:
                    return str(line.split()[dataindex])


# function to open and read from the .position files using pandas, specify
# the path of the file to open, the column that you want
# to extract from, and the extension of the file
def read_csv_pandas(path, Col=None, header=None):
    #  verify that the file exists
    if not os.path.exists(path):
        print("No file %s" % path)
        return []
    try:  # open the file
        csvData = pd.read_csv(path, header=header,
                              delim_whitespace=True, low_memory=False)
    except ValueError:
        print("%s not valid (usually empty)" % path)
        return []
        # verify that the column that we specified is not empty,
        # and return the values
    if Col is not None:
        return csvData.values[:, Col[0]]
    else:
        return csvData


# new px to cm conversion. To correct camera lens distorsion (pixels in the
# center of the treadmill are more precise than the ones located at the
# extremities), filter applied in LabView, and conversion should be uniform
# now, 11 px is equal to 1 cm at every point of the treadmill.
def datapx2cm(list):
    array = []
    for pos in list:
        if pos == 0:
            array.append(pos)
        elif pos > 0 and pos < 1300:
            array.append(pos/11)
        else:
            array.append(pos)
            print("might have error in position", pos)
    return array


# function to compute speed array based on position and time arrays
def compute_speed(dataPos, dataTime):  # speed only computed along X axis. Compute along X AND Y axis?
    rawdata_speed = {}
    deltaXPos = (np.diff(dataPos))
    deltaTime = (np.diff(dataTime))
    rawdata_speed = np.divide(deltaXPos, deltaTime)
    rawdata_speed = np.append(rawdata_speed, 0)
    return rawdata_speed.astype('float32')
    # working on ragged arrays so type of the array may
    # have to be modified from object to float32


# save data as pickle
def save_as_pickle(root, data, animal, session, name):
    sessionPath = root+os.sep+animal+os.sep+"Experiments"+os.sep+session
    folderPath = os.path.join(sessionPath, "Analysis")
    if not os.path.exists(folderPath):
        os.mkdir(folderPath)
    filePath = os.path.join(folderPath, name)
    pickle.dump(data, open(filePath, "wb"))


# load data that has been pickled
def get_from_pickle(root, animal, session, name):
    sessionPath = root+os.sep+animal+os.sep+"Experiments"+os.sep+session
    analysisPath = os.path.join(sessionPath, "Analysis")
    picklePath = os.path.join(analysisPath, name)
    # if not re.do, and there is a pickle, try to read it
    if os.path.exists(picklePath):
        try:
            data = pickle.load(open(picklePath, "rb"))
            return data
        except Exception:
            print("error loading pickle")
            pass
    else:
        print(f"no pickle found {session}")
        return None


# save plot as png
def save_plot_as_png(filename, dpi='figure',
                     transparent=True, background='auto'):
    folderPath = os.path.join(os.getcwd(), "Figures")
    if not os.path.exists(folderPath):
        os.mkdir(folderPath)
    filePath = os.path.join(folderPath, filename)
    plt.savefig(filePath, dpi=dpi, transparent=transparent,
                facecolor=background, edgecolor=background)


# only display one legend when there is duplicates
# (e.g. we don't want one label per run)
def legend_without_duplicate_labels(ax):
    handles, labels = ax.get_legend_handles_labels()
    unique = [(h, l) for i, (h, l) in enumerate(zip(handles, labels)) if l not in labels[:i]]
    ax.legend(*zip(*unique))


def dirty_acceleration(array):
    return [np.diff(a)/0.04 for a in array]


# in action sequence, get block number based on beginning of action
def get_block(t_0):
    block = None
    if 0 <= t_0 <= 300:
        block = 0
    elif 300 < t_0 <= 600:
        block = 1
    elif 600 < t_0 <= 900:
        block = 2
    elif 900 < t_0 <= 1200:
        block = 3
    elif 1200 < t_0 <= 1500:
        block = 4
    elif 1500 < t_0 <= 1800:
        block = 5
    elif 1800 < t_0 <= 2100:
        block = 6
    elif 2100 < t_0 <= 2400:
        block = 7
    elif 2400 < t_0 <= 2700:
        block = 8
    elif 2700 < t_0 <= 3000:
        block = 9
    elif 3000 < t_0 <= 3300:
        block = 10
    elif 3300 < t_0 <= 3600:
        block = 11
    return block


# function to stitch together all the bins of a variable to form the full session variable.
def stitch(input):
    dataSession = copy.deepcopy(input)
    for i, data in enumerate(input):
        dataSession[i] = list(chain.from_iterable(list(data.values())))
    return dataSession



####################################
# Histology functions
####################################

def read_ROI_from_csv(file_path):
    ROI = []
    try:
        with open(file_path, mode='r', newline='') as csv_file:
            csv_reader = csv.reader(csv_file)
            for row in csv_reader:
                if len(row) == 2:
                    try:
                        x = round(float(row[0]))
                        y = round(float(row[1]))
                        ROI.append((x, y))
                    except ValueError:
                        if row[0] == 'X' and row[1] == 'Y':
                            continue
                        else:
                            print(f"Skipping invalid row: {row}")
                else:
                    print(f"Skipping row with incorrect number of columns: {row}")
    except FileNotFoundError:
        print(f"File not found: {file_path}")
    except Exception as e:
        print(f"An error occurred while reading the file: {str(e)}")

    return np.array(ROI)

def compute_area(ROI):
    '''Compute the area of a polygonal ROI using the shoelace formula'''
    n = len(ROI)
    area = 0.0

    for i in range(n):
        x1, y1 = ROI[i]
        x2, y2 = ROI[(i + 1) % n]
        area += (x1 * y2 - x2 * y1)

    area = abs(area) / 2.0
    return area

def compute_centroid(ROI):
    '''Compute the centroid of a polygonal ROI using the shoelace formula'''
    n = len(ROI)
    if n == 0:
        return None

    sum_x = 0
    sum_y = 0
    signed_area = 0

    for i in range(n):
        x1, y1 = ROI[i]
        x2, y2 = ROI[(i + 1) % n]  # Wrap around for the last point

        cross_product = (x1 * y2 - x2 * y1)
        signed_area += cross_product
        sum_x += (x1 + x2) * cross_product
        sum_y += (y1 + y2) * cross_product

    signed_area /= 2.0
    centroid_x = sum_x / (6 * signed_area)
    centroid_y = sum_y / (6 * signed_area)

    return (centroid_x, centroid_y)




############################################
# copy only test.p files for all sessions
# compress and upload to google drive
# add animal, session = 'RatM01', 'RatM01_2021_07_22_17_14_48'  to the list
# import fnmatch
# from os.path import isdir, join
# from shutil import copytree, rmtree

# def include_patterns(*patterns):
#     def _ignore_patterns(path, all_names):
#         # Determine names which match one or more patterns (that shouldn't be
#         # ignored).
#         keep = (name for pattern in patterns for name in fnmatch.filter(all_names, pattern))
#         # Ignore file names which *didn't* match any of the patterns given that
#         # aren't directory names.
#         dir_names = (name for name in all_names if isdir(join(path, name)))
#         return set(all_names) - set(keep) - set(dir_names)
#     return _ignore_patterns


# src = "/home/david/Desktop/DATA"
# dst = "/home/david/Desktop/testcopy"
# copytree(src, dst, ignore=include_patterns('test.p'))
############################################


# weights = {'RatF00': 219.20, 'RatF01': 215.31, 'RatF02': 200.54, 
#             'RatM00': 277.65, 'RatM01': 295.46, 'RatM02': 271.19}

# compute the average weight for each animal from all sessions
# need params.p for each session
# avgweight = {}
# for animal in animalList:
#     avgweight[animal] = []
#     for session in matchsession(animal,  dist60+dist90+dist120 + TM20+TM10+TM2+TMrev2+TMrev10+TMrev20):
#         _params = get_from_pickle(root, animal[0:6], session, name="params.p")
#         avgweight[animal].extend([_params['weight']])
#     avgweight[animal] = sum(avgweight[animal]) / len(avgweight[animal])
