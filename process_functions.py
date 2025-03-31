"""This module contains a set of functions that are used
 in the analysis and preprocessing of the data."""
"""This code is quite old, it runs fine and processes the data
as expected but deserves a good rewrite for better readability,
maintainability and performance."""


import copy
import datetime
import fnmatch
from IPython.display import Image, display, clear_output
import itertools
from itertools import groupby
import matplotlib.cbook
import matplotlib.pyplot as plt
import numpy as np
import os
import re
from scipy import stats
from scipy.ndimage import gaussian_filter as smooth
from scipy.signal import find_peaks
import time
import warnings
matplotlib.rcParams['axes.unicode_minus'] = False
warnings.filterwarnings("ignore", category=matplotlib.cbook.mplDeprecation)
startTimeNotebook = datetime.datetime.now()

from utils import *
from plotting import *

"""
###----------------------------------------------------------------------------
### Utility/low_level computation functions
###----------------------------------------------------------------------------
"""


# function to split lists --> used to split the raw X position array into
# smaller arrays (runs and stays). Later in the code we modify the
# array and change some values to 0, which will be used as cutting points.
def split_a_list_at_zeros(List):
    return [list(g) for k, g in groupby(List, key=lambda x:x != 0) if k]


# cuts session in bins
def bin_session(animal, session, data_to_cut, data_template, bins):
    output = {}
    bincount = 0
    for timebin in bins:
        if timebin[0] == 0:
            start_of_bin = 0
        else:
            start_of_bin = int(np.where(data_template[animal, session] == timebin[0])[0])+1
        end_of_bin = int(np.where(data_template[animal, session] == timebin[1])[0])+1
        output[bincount] = data_to_cut[animal, session][start_of_bin:end_of_bin]
        bincount += 1
    return output



# 2022_05_04 LV: added brain status of the animal (normal, lesion, cno, saline) in behav.params. 
# This is a fix to consign it in antecedent sessions. I think it fixed them all.
def FIXwrite_params(root, animal, session):
    # animal = "RatF02"
    # for session in sorted(matchsession(animal, lesionrev20+lesionrev10+lesionrev2+lesion2+lesion10+lesion20)): #lesiontrain+lesion60+lesion90+lesion120
    #     FIXwrite_params(root, animal, session)
    behav = root + os.sep+animal + os.sep+"Experiments" + os.sep + session + os.sep + session + ".behav_param"
    if not os.path.exists(behav):
        print("No file %s" % behav)
    alreadywritten=False
    with open(behav, "r") as f:
        for line in f:
            if "brainstatus" in line:
                alreadywritten = True
    if not alreadywritten:
        with open(behav, "a") as f: f.write("\nbrainstatus normal")


# save animal plot as png
def save_sessionplot_as_png(root, animal, session, filename,
                            dpi='figure', transparent=True, background='auto'):
    sessionPath = root+os.sep+animal+os.sep+"Experiments"+os.sep+session
    folderPath = os.path.join(sessionPath, "Figures")
    if not os.path.exists(folderPath):
        os.mkdir(folderPath)
    filePath = os.path.join(folderPath, filename)
    plt.savefig(filePath, dpi=dpi, transparent=transparent,
                facecolor=background, edgecolor=background)


# group bin data by reward%
def poolByReward(data, proba, blocks, rewardproba):
    output = []
    for i in range(0, len(blocks)):
        if rewardproba[i] == proba:
            if len(data) == 1:
                output.append(data[0][i])
            if len(data) == 2:  # usually for data like dataLeft+dataRight
                output.append(data[0][i]+data[1][i])
            if len(data) > 2:
                print("too much data, not intended")
    return output


# in action sequence, cut full action sequence into corresponding blocks
def recut(data_to_cut, data_template):
    output = []
    start_of_bin = 0
    for i, _ in enumerate(data_template):
        end_of_bin = start_of_bin + len(data_template[i])
        output.append(data_to_cut[start_of_bin: end_of_bin])
        start_of_bin = end_of_bin
    return output


"""
###----------------------------------------------------------------------------
### INDIVIDUAL FIGURES
###----------------------------------------------------------------------------
"""


def plot_BASEtrajectoryV2(animal, session, time, running_Xs, idle_Xs, lickL, lickR,
                          rewardProbaBlock, blocks, barplotaxes,
                          xyLabels=[" ", " ", " ", " "],
                          title=[None], linewidth=1):
    ax1 = plt.gca()
    for i in range(0, len(blocks)):
        plt.axvspan(blocks[i][0], blocks[i][1],
                    color='grey', alpha=rewardProbaBlock[i]/250,
                    label="%reward: " + str(rewardProbaBlock[i])
                    if (i == 0 or i == 1) else "")
    plt.plot(time, running_Xs, label="run", color="dodgerblue", linewidth=1)
    plt.plot(time, idle_Xs, label="wait", color="orange", linewidth=1)
    plt.plot(time, [None if x == 0 else x for x in lickL],
             color="b", marker="o", markersize=1)
    plt.plot(time, [None if x == 0 else x for x in lickR],
             color="b", marker="o", markersize=1)
    ax1.set_xlabel(xyLabels[0], fontsize=xyLabels[6])
    ax1.set_ylabel(xyLabels[2], fontsize=xyLabels[6])
    ax1.set_xlim([barplotaxes[0], barplotaxes[1]])
    ax1.set_ylim([barplotaxes[0], barplotaxes[3]+30])
    ax1.spines['bottom'].set_linewidth(linewidth[0])
    ax1.spines['left'].set_linewidth(linewidth[0])
    ax1.spines['top'].set_color("none")
    ax1.spines['right'].set_color("none")
    ax1.tick_params(width=2, labelsize=xyLabels[7])
    # x_ticks = np.arange(1800, 3300, 300)
    # ax1.set_xticks(x_ticks)
    # ax1.set_xticklabels([int(val / 60) for val in ax1.get_xticks().tolist()])
    return ax1


# plot rat acceleration along run
def plot_acc(animal, session, posdataRight, timedataRight, bounds,
             xylim, xyLabels=[" ", " ", " ", " "], title=[None], linewidth=1):
    ax = plt.gca()
    for i, j in zip(posdataRight, timedataRight):
        plt.plot(np.subtract(j, j[0]), i, color='g', linewidth=0.3)
    ax.set_title(title[0], fontsize=title[1])
    ax.set_xlabel(xyLabels[0], fontsize=xyLabels[2])
    ax.set_ylabel(xyLabels[1], fontsize=xyLabels[2])
    ax.set_xlim([xylim[0], xylim[1]])
    ax.set_ylim([xylim[2], xylim[3]])
    ax.spines['top'].set_color("none")
    ax.spines['right'].set_color("none")
    xline1 = [bounds[0], bounds[0]]
    xline2 = [bounds[1], bounds[1]]
    yline = [0, 20]
    plt.plot(yline, xline1, ":", color='k')
    plt.plot(yline, xline2, ":", color='k')
    ax.legend()
    return ax

# This function plots the base trajectory of the rat. Parameters are time:
# time data, position : X position data, lickL/R, lick data,
# maxminstep for x and y axis, color and marker of the plot,
# width of the axis, and x y labels
def plot_BASEtrajectory(time, position, lickLeft, lickRight, maxminstep,
                        maxminstep2, color=[], marker=[],
                        linewidth=[], xyLabels=["N", "Bins"]):

    plt.plot(time, position, color=color[0],
             marker=marker[0], linewidth=linewidth[0])
    # lick data, plot position in which the animal licked else None
    plt.plot(time, [None if x == 0 else x for x in lickLeft], color=color[1],
             marker=marker[1], markersize=marker[2])
    plt.plot(time, [None if x == 0 else x for x in lickRight], color=color[1],
             marker=marker[1], markersize=marker[2])

    # plot parameters
    traj = plt.gca()
    traj.set_xlim(maxminstep[0] - maxminstep[2],
                  maxminstep[1] + maxminstep[2])
    traj.set_ylim(maxminstep2[0] - maxminstep2[2],
                  maxminstep2[1] + maxminstep2[2])
    traj.set_xlabel(xyLabels[1], fontsize=12, labelpad=0)
    traj.set_ylabel(xyLabels[0], fontsize=12, labelpad=-1)
    traj.xaxis.set_ticks_position('bottom')
    traj.yaxis.set_ticks_position('left')
    traj.get_xaxis().set_tick_params(direction='out', pad=2)
    traj.get_yaxis().set_tick_params(direction='out', pad=2)
    traj.spines['top'].set_color("none")
    traj.spines['right'].set_color("none")
    return traj

# plot block per %reward
def plot_figBinMean(ax, dataLeft, dataRight, color, ylim):
    mean_left = np.mean(dataLeft)
    mean_right = np.mean(dataRight)
    se_left = np.std(dataLeft, ddof=1) / np.sqrt(len(dataLeft))
    se_right = np.std(dataRight, ddof=1) / np.sqrt(len(dataRight))

    ax.errorbar(0, mean_left, yerr=se_left, fmt='o', color=color[0],
                markersize=7, capsize=5)
    ax.errorbar(1, mean_right, yerr=se_right, fmt='o', color=color[0],
                markersize=7, capsize=5)

    ax.plot([0, 1], [mean_left, mean_right], color='black', linestyle='--', linewidth=0.8)

    diff = mean_right - mean_left
    ax.text(0.5, (mean_left + mean_right) / 2, f"diff = {diff:.2f}",
            ha='center', va='center', fontsize=10)

    ax.axvspan(-0.5, 0.5, color='grey', alpha=10/250)
    ax.axvspan(0.5, 1.5, color='grey', alpha=90/250)

    ax.set_xlim(-0.5, 1.5)
    ax.set_ylim(ylim)
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["10%\nreward", "90%\nreward"])
    ax.set_ylabel("Value")
    ax.set_title("Mean Difference")

    return ax


"""
###------------------------------------------------------------------------------------------------------------------
### DATA PROCESSING FUNCTIONS
###------------------------------------------------------------------------------------------------------------------
"""


# Old function to compute start of run and end of run boundaries
def extract_boundaries(data, animal, session, dist, height=None):
    # animals lick in the extremities, so they spend more time there, so probability of them being there is more important than the probability of being in the middle of the apparatus. we compute the two average positions of these resting points. We defined a run as the trajectory of the animal between the resting points. So we have to find these resting points. In later stages of the experiments the start/end of runs is defined based on the speed of the animals.
    # function params : data is X position array for the session that we analyse, height = parameter to define a limit to the probability of detecting a place as significantly more probable than another.
    # We use a KDE (Kernel Density Estimate) to find said places. See testhitopeak.ipynb 2nd method for histogram. histogram is coded in next cell between """ """, but does not work on linux
    kde = stats.gaussian_kde(data)
    # compute KDE = get the position probability curve and compute peaks of the curve
    peak_pos, peak_height = [], []
    nb_samples = 120  # played a bit with the values, this works (number of data bins, we chose 1 per cm, also tested 10 bins per cm)
    samples = np.linspace(0, 120, nb_samples)
    probs = kde.evaluate(samples)
    maxima_index = find_peaks(probs, height)
    peak_pos = maxima_index[0]
    peak_height = maxima_index[1]["peak_heights"]
    # print("values", peak_pos, peak_height)
    # if there is more than two peaks (e.g. an animal decides to stay in the middle of the treadmill), only keep the two biggest peaks (should be the extremities) and remove the extra peak/s if there is one or more
    peak_posLeft, peak_heightLeft, peak_posRight, peak_heightRight = [], [], [], []
    for i, j in zip(peak_pos, peak_height):
        if i < dist/2:
            peak_posLeft.append(i)
            peak_heightLeft.append(j)
        if i > dist/2:
            peak_posRight.append(i)
            peak_heightRight.append(j)
    leftBoundaryPeak = peak_posLeft[np.argmax(peak_heightLeft)]
    rightBoundaryPeak = peak_posRight[np.argmax(peak_heightRight)]
    # print("computing bounds", animal, leftBoundaryPeak, rightBoundaryPeak)
    return leftBoundaryPeak, rightBoundaryPeak, kde


# convert scale, convert i = 0 to 120 --> 60 to-60 which correspnds to the speed to the right (0 to 60) and to the left (0 to -60)
def convert_scale(number):
    old_min = 0
    old_max = 120
    new_max = -60
    new_min = 60
    return int(((number - old_min) / (old_max - old_min)) * (new_max - new_min) + new_min)


# compute mask to separate runs and stays based on speed
def filterspeed(animal, session, dataPos, dataSpeed, dataTime, threshold, dist):
    # dissociate runs from non runs, we want a cut off based on animal, speed. How to define this speed? If we plot the speed of the animal in function of the X position in the apparatus, so we can see that there is some blobs of speeds close to 0 and near the extremities of the treadmill, these are the ones that we want to define as non running speeds. With this function we want to compute the area of these points of data (higher density, this technique might not work when animals are not properly trained) in order to differentiate them.
    middle = dist/2
    xmin, xmax = 0, 120  # specify the x and y range of the window that we want to analyse
    ymin, ymax = -60, 60
    position = np.array(dataPos, dtype=float)  # data needs to be transformed to float perform the KDE
    speed = np.array(dataSpeed, dtype=float)
    time = np.array(dataTime, dtype=float)
    X, Y = np.mgrid[xmin:xmax:120j, ymin:ymax:120j]  # create 2D grid to compute KDE
    positions = np.vstack([X.ravel(), Y.ravel()])
    values = np.vstack([position, speed])
    kernel = stats.gaussian_kde(values)  # compute KDE, this gives us an estimation of the point density
    Z = np.reshape(kernel(positions).T, X.shape)
    # Using the KDE that we just computed, we select the zones that have a density above a certain threshold (after testing 0.0001 seems to work well), which roughly corresponds to the set of data that we want to extract.
    # We loop through the 2D array, if the datapoint is > threshold we get the line (speed limit) and the row (X position in cm). This gives us the speed limit for each part of the
    # treadmill, so basically a zone delimited with speed limits (speed limits can be different in different points of the zone).
    i, j = [], []  # i is the set of speeds (lines) for which we will perform operations, j is the set of positions (rows) for each speed for which we will perform operations
    for line in range(0, len(np.rot90(Z))):
        if len(np.where(np.rot90(Z)[line] > threshold)[0]) > 1:
            i.append(convert_scale(line))
            j.append(np.where(np.rot90(Z)[line] > threshold)[0])
    # create a mask using the zone computed before and combine them. We have two zones (left and right), so we perform the steps on each side, first part is on the left.
    rawMask = np.array([])
    # pos is the array of positions for which the speed of the animal is under the speed limit. "11 [ 7  8  9 10 11]" for instance here the speed limit is 11 cm/s, and is attained between 7 and 11cm on the treadmill.
    # "10 [4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 105, 106, 107, 108, 109, 110]" when we decrease the speed limit, here 10, we here have 2 zones, one between 4 and 13cm and another between 105 and 110cm.
    # we continue through these speed values (roughly from 11 to 0, then from 0 to -10 in this example)
    for line, pos in zip(i, j):
        if pos[pos < middle].size:
            low = pos[pos < middle][0]  # first value of the array, explained above (7)
            high = pos[pos < middle][-1]  # last value of the array (11)
            a = np.ma.masked_less(position, high)  # take everything left of the rightmost point, if the value is ok == True
            b = np.ma.masked_greater(position, low)  # take everything right of the leftmost point
            c = np.ma.masked_less(speed, line+0.5)  # take everything below the high point
            d = np.ma.masked_greater(speed, line-0.5)  # take everything above the low point
            mask = np.logical_and(a.mask, b.mask)  # first combination for all the rows (=Xposition), keep the intersection of mask a AND b, so keep all between the leftmost and rightmost points
            mask2 = np.logical_and(c.mask, d.mask)  # second combination for all the lines (=speed), keep the intersection of mask c AND d, so keep all between speed+0.5:speed-0.5
            combiLeft = np.logical_and(mask, mask2)  # combine the first and the second mask, so we only keep the intersection of the two masks, intersection is TRUE, the rest is FALSE
            if not rawMask.size:  # must do that for the first iteration so it's not empty
                rawMask = combiLeft
            else:
                rawMask = np.logical_xor(combiLeft, rawMask)  # merge the newly computed mask with the previously computed masks. We use XOR so that the TRUE values of the new mask are added to the complete mask. Same step left and right just add to the existing full mask wether the new mask is on the left or the right.
        # same as above for the right part
        if pos[pos > middle].size:
            low = pos[pos > middle][0]
            high = pos[pos > middle][-1]
            a = np.ma.masked_less(position, high)
            b = np.ma.masked_greater(position, low)
            c = np.ma.masked_less(speed, line + 0.5)
            d = np.ma.masked_greater(speed, line - 0.5)
            mask = np.logical_and(a.mask, b.mask)
            mask2 = np.logical_and(c.mask, d.mask)
            combiRight = np.logical_and(mask, mask2)
            if not rawMask.size:
                rawMask = combiRight
            else:
                rawMask = np.logical_xor(combiRight, rawMask)
    return ~rawMask


# Mask smoothing, what it does is if we have a small part of running in either sides between parts of not running, we say that this is not running and modify the mask. So we have to set up all possible cases and generate an appropriate response, but in all cases encountered these problems were only in the waiting times and not in running.
def removeSplits_Mask(inputMask, inputPos, animal, session, dist):
    correctedMask = [list(val) for key, val in groupby(inputMask[animal, session], lambda x: x == True)]
    splitPos = []
    middle = (dist)/2
    count = [0, 0, 0, 0, 0, 0]
    start, end = 0, 0
    for elem in correctedMask:
        start = end
        end = start + len(elem)
        splitPos.append(inputPos[animal, session][start:end])
    for m, p in zip(correctedMask, splitPos):
        if p[0] < middle and p[-1] < middle:
            if m[0] == False:
                pass
                # print("in L")
            if m[0] == True:
                # print("bug")
                correctedMask[count[5]] = [False for val in m]
                count[0] += 1
            count[5] += 1
            # print(m, p, "all left")
        elif p[0] > middle and p[-1] > middle:
            if m[0] == False:
                pass
                # print("in R")
            if m[0] == True:
                # print("bug")
                correctedMask[count[5]] = [False for val in m]
                count[1] += 1
            count[5] += 1
            # print(m, p, "all right")
        elif p[0] < middle and p[-1] > middle:
            if m[0] == True:
                pass
                # print("runLR")
            if m[0] == False:
                # print("bug")
                correctedMask[count[5]] = [True for val in m]
                count[2] += 1
            count[5] += 1
            # print(m, p, "left right")
        elif p[0] > middle and p[-1] < middle:
            if m[0] == True:
                pass
                # print("runRL")
            if m[0] == False:
                # print("bug")
                correctedMask[count[5]] = [True for val in m]
                count[3] += 1
            count[5] += 1
            # print(m, p, "right left")
        else:  # print(m, p, "bbb")
            count[4] += 1
            count[5] += 1
    # print(count)
    return np.concatenate(correctedMask)


# separate runs/stays * left/right + other variables into dicts
def extract_runSpeedBin(dataPos, dataSpeed, dataTime, dataLickR, dataLickL, openR, openL, mask, animal, session, blocks, boundary, treadmillspeed, rewardProbaBlock):
    runs = {}
    stays = {}
    runs[animal, session] = {}
    stays[animal, session] = {}
    position, speed, time, running_Xs, idle_Xs, goodSpeed, badSpeed, goodTime, badTime = ({bin: [] for bin in range(0, len(blocks))} for _ in range(9))
    speedRunToRight, speedRunToLeft, XtrackRunToRight, XtrackRunToLeft, timeRunToRight, timeRunToLeft, timeStayInRight, timeStayInLeft, XtrackStayInRight, XtrackStayInLeft, TtrackStayInRight, TtrackStayInLeft, instantSpeedRight, instantSpeedLeft, maxSpeedRight, maxSpeedLeft, whenmaxSpeedRight, whenmaxSpeedLeft, wheremaxSpeedRight, wheremaxSpeedLeft, lick_arrivalRight, lick_drinkingRight, lick_waitRight, lick_arrivalLeft, lick_drinkingLeft, lick_waitLeft = ({bin: [] for bin in range(0, len(blocks))} for _ in range(26))
    rewardedLeft, rewardedRight = ({bin: [] for bin in range(0, len(blocks))} for _ in range(2))

    for i in range(0, len(blocks)):
        position[i] = np.array(dataPos[animal, session][i], dtype=float)
        speed[i] = np.array(dataSpeed[animal, session][i], dtype=float)
        time[i] = np.array(dataTime[animal, session][i], dtype=float)

        running_Xs[i] = [val[0] if val[1] == False else None for val in [[i, j] for i, j in zip(position[i], mask[animal, session][i])]]
        idle_Xs[i] = [val[0] if val[1] == True else None for val in [[i, j] for i, j in zip(position[i], mask[animal, session][i])]]
        goodSpeed[i] = [val[0] if val[1] == False else None for val in [[i, j] for i, j in zip(speed[i], mask[animal, session][i])]]
        badSpeed[i] = [val[0] if val[1] == True else None for val in [[i, j] for i, j in zip(speed[i], mask[animal, session][i])]]
        goodTime[i] = [val[0] if val[1] == False else None for val in [[i, j] for i, j in zip(time[i], mask[animal, session][i])]]
        badTime[i] = [val[0] if val[1] == True else None for val in [[i, j] for i, j in zip(time[i], mask[animal, session][i])]]

        stays[animal, session][i] = [[e[0], e[1], e[2], e[3], e[4]] if [e[0], e[1], e[2]] != [None, None, None] else 0 for e in [[i, j, k, l, m] for i, j, k, l, m in zip(running_Xs[i], goodSpeed[i], goodTime[i], dataLickR[animal, session][i], dataLickL[animal, session][i])]]
        runs[animal, session][i] = [[e[0], e[1], e[2], e[3], e[4]] if [e[0], e[1], e[2]] != [None, None, None] else 0 for e in [[i, j, k, l, m] for i, j, k, l, m in zip(idle_Xs[i], badSpeed[i], badTime[i], openL[i], openR[i])]]

        for run in split_a_list_at_zeros(runs[animal, session][i]):
            # calculate distance run as the distance between first and last value
            distanceRun = abs(run[0][0]-run[-1][0])
            # calculate time as sum of time interval between frames
            totaltimeRun = []
            xTrackRun = []
            instantSpeed = []
            maxSpeed = []
            valveL, valveR = [], []
            for item in run:
                xTrackRun.append(item[0])
                instantSpeed.append(abs(item[1]))  # if no abs() here > messes with maxspeed, if-TMspeed > not clean platform + TM changes direction with rat
                totaltimeRun.append(item[2])
                valveL.append(item[3])
                valveR.append(item[4])
            if np.sum(np.diff(totaltimeRun)) != 0:
                speedRun = distanceRun/np.sum(np.diff(totaltimeRun)) - treadmillspeed[i]
                maxSpeed = max(instantSpeed) - treadmillspeed[i]
                wheremaxSpeed = xTrackRun[np.argmax(instantSpeed)] - xTrackRun[0] if xTrackRun[0] < xTrackRun[np.argmax(instantSpeed)] else xTrackRun[0] - xTrackRun[np.argmax(instantSpeed)]
                whenmaxSpeed = np.sum(np.diff(totaltimeRun[0:np.argmax(instantSpeed)]))  # totaltimeRun[np.argmax(instantSpeed)]-
                # check if the subsplit starts on the left or the right -> determine if the animal is running left or right
                if run[0][0] < ((boundary[0]+boundary[1])/2):
                    # check if the subsplit is ending on the other side -> determine if this is a run
                    if run[-1][0] > ((boundary[0]+boundary[1])/2):
                        speedRunToRight[i].append(speedRun)
                        XtrackRunToRight[i].append(xTrackRun)
                        timeRunToRight[i].append(totaltimeRun)
                        instantSpeedRight[i].append(instantSpeed)
                        maxSpeedRight[i].append(maxSpeed)
                        wheremaxSpeedRight[i].append(wheremaxSpeed)
                        whenmaxSpeedRight[i].append(whenmaxSpeed)
                        if np.any(split_a_list_at_zeros(valveR)):  # if at least one != 0
                            rewardedRight[i].append(1 if split_a_list_at_zeros(valveR)[0][0] <= rewardProbaBlock[i] else 0)
                        else:
                            rewardedRight[i].append(10)

                # same thing for the runs that go to the other side
                elif run[0][0] > ((boundary[0]+boundary[1]) / 2):
                    if run[-1][0] < ((boundary[0]+boundary[1]) / 2):
                        speedRunToLeft[i].append(speedRun)
                        XtrackRunToLeft[i].append(xTrackRun)
                        timeRunToLeft[i].append(totaltimeRun)
                        instantSpeedLeft[i].append(instantSpeed)
                        maxSpeedLeft[i].append(maxSpeed)
                        wheremaxSpeedLeft[i].append(wheremaxSpeed)
                        whenmaxSpeedLeft[i].append(whenmaxSpeed)
                        if np.any(split_a_list_at_zeros(valveL)):  # if at least one != 0
                            rewardedLeft[i].append(1 if split_a_list_at_zeros(valveL)[0][0] <= rewardProbaBlock[i] else 0)
                        else:
                            rewardedLeft[i].append(10)

        for stay in split_a_list_at_zeros(stays[animal, session][i]):
            tInZone = []
            xTrackStay = []
            lickR = []
            lickL = []
            for item in stay:
                xTrackStay.append(item[0])
                tInZone.append(item[2])
                lickR.append(item[3])
                lickL.append(item[4])
            totaltimeStay = np.sum(np.diff(tInZone))
            # first identify if the subsplit created is on the left or right by comparing to the middle
            if stay[0][0] > ((boundary[0]+boundary[1]) / 2):
                # if hasLick == True:
                if not all(v == 0 for v in lickR):
                    pre = []
                    drink = []
                    post = []
                    for t, l in zip(tInZone[0:np.min(np.nonzero(lickR))], lickR[0:np.min(np.nonzero(lickR))]):
                        pre.append(t)
                    for t, l in zip(tInZone[np.min(np.nonzero(lickR)):np.max(np.nonzero(lickR))], lickR[np.min(np.nonzero(lickR)):np.max(np.nonzero(lickR))]):
                        drink.append(t)
                    for t, l in zip(tInZone[np.max(np.nonzero(lickR)):-1], lickR[np.max(np.nonzero(lickR)):-1]):
                        post.append(t)
                    # drink <- dig in that later on to have more info on lick (lick rate, number of licks, etc.)
                    lick_arrivalRight[i].append(np.sum(np.diff(pre)))
                    lick_drinkingRight[i].append(np.sum(np.diff(drink)))
                    lick_waitRight[i].append(np.sum(np.diff(post)))
                timeStayInRight[i].append(totaltimeStay)
                XtrackStayInRight[i].append(xTrackStay)
                TtrackStayInRight[i].append(tInZone)
            elif stay[0][0] < ((boundary[0] + boundary[1]) / 2):
                # if hasLick == True:
                if not all(v == 0 for v in lickL):
                    preL = []
                    drinkL = []
                    postL = []
                    for t, l in zip(tInZone[0:np.min(np.nonzero(lickL))], lickR[0:np.min(np.nonzero(lickL))]):
                        preL.append(t)
                    for t, l in zip(tInZone[np.min(np.nonzero(lickL)):np.max(np.nonzero(lickL))], lickL[np.min(np.nonzero(lickL)):np.max(np.nonzero(lickL))]):
                        drinkL.append(t)
                    for t, l in zip(tInZone[np.max(np.nonzero(lickL)):-1], lickL[np.max(np.nonzero(lickL)):-1]):
                        postL.append(t)
                    lick_arrivalLeft[i].append(np.sum(np.diff(preL)))
                    lick_drinkingLeft[i].append(np.sum(np.diff(drinkL)))
                    lick_waitLeft[i].append(np.sum(np.diff(postL)))
                timeStayInLeft[i].append(totaltimeStay)
                XtrackStayInLeft[i].append(xTrackStay)
                TtrackStayInLeft[i].append(tInZone)
    return speedRunToRight, speedRunToLeft, XtrackRunToRight, XtrackRunToLeft, timeRunToRight, timeRunToLeft, timeStayInRight, timeStayInLeft, XtrackStayInRight, XtrackStayInLeft, TtrackStayInRight, TtrackStayInLeft, instantSpeedRight, instantSpeedLeft, maxSpeedRight, maxSpeedLeft, whenmaxSpeedRight, whenmaxSpeedLeft, wheremaxSpeedRight, wheremaxSpeedLeft, lick_arrivalRight, lick_drinkingRight, lick_waitRight, lick_arrivalLeft, lick_drinkingLeft, lick_waitLeft, rewardedRight, rewardedLeft


# due to the way blocks are computed, some runs may have started in block[n] and ended in block [n+1], this function appends the end of the run to the previous block. See reCutBins.
def fixSplittedRunsMask(animal, session, input_Binmask, blocks):
    output_Binmask = copy.deepcopy(input_Binmask)
    for i in range(1, len(blocks)):  # print(input_Binmask[i-1][-1], input_Binmask[i][0])
        if not all(v == False for v in output_Binmask[i]):  # if animal did not do any run (only one stay along the whole block) don't do the operation
            if output_Binmask[i-1][-1] == True and output_Binmask[i][0] == True:  # print(i, "case1")
                # print(session, i-1, i)  # uncomment to see whether/which bins have had fixes.
                while output_Binmask[i][0] == True:
                    output_Binmask[i-1] = np.append(output_Binmask[i-1], output_Binmask[i][0])
                    output_Binmask[i] = np.delete(output_Binmask[i], 0)
            if output_Binmask[i-1][-1] == False and output_Binmask[i][0] == False:  # print(i, "case2")
                # print(session, i-1, i)
                while output_Binmask[i][0] == False:
                    output_Binmask[i-1] = np.append(output_Binmask[i-1], output_Binmask[i][0])
                    output_Binmask[i] = np.delete(output_Binmask[i], 0)
    return output_Binmask


# following fixSplittedRunsMask, this function re cuts the bins of a variable at the same Length as the fixed binMask we just computed.
def reCutBins(data_to_cut, data_template):
    output = {}
    start_of_bin = 0
    for i, bin in enumerate(data_template):
        end_of_bin = start_of_bin + len(data_template[i])
        output[i] = data_to_cut[start_of_bin: end_of_bin]
        start_of_bin = end_of_bin
    return output


# function to stitch together all the bins of a variable to form the full session variable.
def stitch(input):
    dataSession = copy.deepcopy(input)
    for i, data in enumerate(input):
        dataSession[i] = list(itertools.chain.from_iterable(list(data.values())))
    return dataSession


#replace first 0s in animal position (animal not found / cam init) 
# if animal not found == camera edit, so replace with the first ok position
def fix_start_session(pos, edit):
    fixed = np.array(copy.deepcopy(pos))
    _edit = np.array(copy.deepcopy(edit))
    first_zero = next((i for i, x in enumerate(_edit) if not x), None)
    fixed[:first_zero] = pos[first_zero]
    _edit[:first_zero] = 0
    return fixed.flatten(), _edit.flatten()

# linear interpolation of the position when the camera did not find the animal
def fixcamglitch(time, pos, edit):
    last_good_pos = 0
    fixed = np.array(copy.deepcopy(pos))
    _ = [_ for _ in range(0, len(time))]
    _list = [[p, e, __] if e == 0 else 0 for p, e, __ in zip(pos, edit, _)]

    for i in range(1, len(_list)-1):
        if isinstance((_list[i-1]), list) and isinstance((_list[i]), int) and isinstance((_list[i+1]), list):
            _list[i] = [_list[i-1][0] + (_list[i+1][0] - _list[i-1][0])/2, 0, i]

    for _ in split_a_list_at_zeros(_list):
        if len(_) > 1:
            next_good_pos = _[0][2]
            try:
                patch = np.linspace(_list[last_good_pos][0], _list[next_good_pos][0], next_good_pos - last_good_pos + 1)
            except TypeError:
                print("TypeError, happens when restarting session in Labview whitout stopping VI, \
                cam still has last session position (so non zero and not caught by fix_start_session).\
                    Only a few cases")
            for i in range(last_good_pos, next_good_pos+1):
                fixed[i] = patch[i-last_good_pos]
            last_good_pos = _[-1][2]
    return fixed.flatten()


# DATA PROCESSING FUNCTION
def processData(root, ID, sessionIN, index, buggedSessions, redoCompute=False, redoFig=False, printFigs=False, redoMask=False):
    index = index
    animal = ID

    # initialise all Var dicts
    params, rat_markers, water = {}, {}, {}
    extractTime, extractPositionX, extractPositionY, extractLickLeft, extractLickRight, framebuffer, solenoid_ON_Left, solenoid_ON_Right, cameraEdit = ({} for _ in range(9))
    rawTime, rawPositionX, rawPositionY, rawLickLeftX, rawLickRightX, rawLickLeftY, rawLickRightY, smoothMask, rawSpeed = ({} for _ in range(9))
    binPositionX, binPositionY, binTime, binLickLeftX, binLickRightX, binSolenoid_ON_Left, binSolenoid_ON_Right = ({} for _ in range(7))
    leftBoundaryPeak, rightBoundaryPeak, kde = {}, {}, {}
    smoothMask, rawMask, binSpeed, binMask = {}, {}, {}, {}
    running_Xs, idle_Xs, goodSpeed, badSpeed = {}, {}, {}, {}
    speedRunToRight,    speedRunToLeft,    XtrackRunToRight,    XtrackRunToLeft,    timeRunToRight,    timeRunToLeft,    timeStayInRight,    timeStayInLeft,    XtrackStayInRight,    XtrackStayInLeft,    TtrackStayInRight,    TtrackStayInLeft,    instantSpeedRight,    instantSpeedLeft,    maxSpeedRight,    maxSpeedLeft,    whenmaxSpeedRight,    whenmaxSpeedLeft,    wheremaxSpeedRight,    wheremaxSpeedLeft,    lick_arrivalRight,    lick_drinkingRight,    lick_waitRight,    lick_arrivalLeft,    lick_drinkingLeft,    lick_waitLeft = ({} for _ in range(26))
    speedRunToRightBin, speedRunToLeftBin, XtrackRunToRightBin, XtrackRunToLeftBin, timeRunToRightBin, timeRunToLeftBin, timeStayInRightBin, timeStayInLeftBin, XtrackStayInRightBin, XtrackStayInLeftBin, TtrackStayInRightBin, TtrackStayInLeftBin, instantSpeedRightBin, instantSpeedLeftBin, maxSpeedRightBin, maxSpeedLeftBin, whenmaxSpeedRightBin, whenmaxSpeedLeftBin, wheremaxSpeedRightBin, wheremaxSpeedLeftBin, lick_arrivalRightBin, lick_drinkingRightBin, lick_waitRightBin, lick_arrivalLeftBin, lick_drinkingLeftBin, lick_waitLeftBin = ({} for _ in range(26))
    nb_runs_to_rightBin, nb_runs_to_leftBin, nb_runsBin, total_trials = {}, {}, {}, {}
    nb_rewardBlockLeft, nb_rewardBlockRight, nbWaterLeft, nbWaterRight, totalWater, totalDistance = ({} for _ in range(6))
    rewardedRight, rewardedLeft, rewardedRightBin, rewardedLeftBin = {}, {}, {}, {}
    lickBug, notfixed, F00lostTRACKlick, buggedRatSessions, boundariesBug, runstaysepbug = buggedSessions


    palette = [(0.55, 0.0, 0.0),  (0.8, 0.36, 0.36),   (1.0, 0.27, 0.0),  (.5, .5, .5), (0.0, 0.39, 0.0),    (0.13, 0.55, 0.13),   (0.2, 0.8, 0.2), (.5, .5, .5)]  # we use RGB [0-1] not [0-255]. See www.colorhexa.com for conversion #old#palette = ['darkred', 'indianred', 'orangered', 'darkgreen', 'forestgreen', 'limegreen']
    if fnmatch.fnmatch(animal, 'RatF*'):
        rat_markers[animal] = [palette[index], "$\u2640$"]
    elif fnmatch.fnmatch(animal, 'RatM*'):
        rat_markers[animal] = [palette[index], "$\u2642$"]
    elif fnmatch.fnmatch(animal, 'Rat00*'):
        rat_markers[animal] = [palette[index], "$\u2426$"]
    else:
        print("error, this is not a rat you got here")

    if sessionIN != []:
        sessionList = sessionIN
    else:
        sessionList = []#sorted([os.path.basename(expPath) for expPath in glob.glob(root+os.sep+animal+os.sep+"Experiments"+os.sep+"Rat*")])

    time.sleep(0.1*(index+1))
    for sessionindex, session in enumerate(sessionList):
        figPath = root + os.sep + animal + os.sep + "Experiments" + os.sep + session + os.sep + "Figures" + os.sep + "recapFIG%s.png" %session
        if redoCompute == True:

            # extract/compute parameters from behav_params and create a parameter dictionnary for each rat and each session
            # change of behav_param format 07/2020 -> labview ok 27/07/2020 before nOk #format behavparam ? #catchup manual up to 27/07
            params[animal, session] = {"sessionDuration": read_params(root, animal, session, "sessionDuration"),
                                       "acqPer": read_params(root, animal, session, "acqPer"),
                                       "waterLeft": round((read_params(root, animal, session, "waterLeft", valueType=float) - read_params(root, animal, session, "cupWeight", valueType=float))/10*1000, 2),
                                       "waterRight": round((read_params(root, animal, session, "waterRight", valueType=float) - read_params(root, animal, session, "cupWeight", valueType=float))/10*1000, 2),
                                       "treadmillDist": read_params(root, animal, session, "treadmillSize"),
                                       "weight": read_params(root, animal, session, "ratWeight"),
                                       "lastWeightadlib": read_params(root, animal, session, "ratWeightadlib"),
                                       "lastDayadlib": read_params(root, animal, session, "lastDayadlib"),
                                       "lickthresholdLeft": read_params(root, animal, session, "lickthresholdLeft"),  # added in Labview 2021/07/06. Now uses the custom lickthreshold for each side. Useful when lickdata baseline drifts and value is directly changed in LV. Only one session might be bugged, so this parameter is session specific. Before, the default value (300) was used and modified manually during the analysis.
                                       "lickthresholdRight": read_params(root, animal, session, "lickthresholdRight"),
                                       "realEnd": str(read_params(root, animal, session, "ClockStop")),
                                       "brainstatus": read_params(root, animal, session, "brainstatus", valueType="other")}

            # initialize boundaries to be computed later using the KDE function
            params[animal, session]["boundaries"] = []

            # compute number of days elapsed between experiment day and removal of the water bottle
            lastDayadlib = str(datetime.datetime.strptime(str(read_params(root, animal, session, "lastDayadlib")), "%Y%m%d").date())
            stringmatch = re.search(r'\d{4}_\d{2}_\d{2}_\d{2}_\d{2}_\d{2}', session)
            experimentDay = str(datetime.datetime.strptime(stringmatch.group(), '%Y_%m_%d_%H_%M_%S'))
            daysSinceadlib = datetime.date(int(experimentDay[0:4]), int(experimentDay[5:7]), int(experimentDay[8:10])) - datetime.date(int(lastDayadlib[0:4]), int(lastDayadlib[5:7]), int(lastDayadlib[8:10]))
            params[animal, session]["daysSinceadLib"] = daysSinceadlib.days

            # compute IRL elapsed session time
            if params[animal, session]['realEnd'] != 'None':
                startExpe = datetime.time(int(experimentDay[11:13]), int(experimentDay[14:16]), int(experimentDay[17:19]))
                endExpe = datetime.time(hour=int(params[animal, session]['realEnd'][0:2]), minute=int(params[animal, session]['realEnd'][2:4]), second=int(params[animal, session]['realEnd'][4:6]))
                params[animal, session]["realSessionDuration"] = datetime.datetime.combine(datetime.date(1, 1, 1), endExpe) - datetime.datetime.combine(datetime.date(1, 1, 1), startExpe)
            else:
                params[animal, session]["realSessionDuration"] = None

            # determine block duration set based on the block timing defined in labview. 1 block in labview is comprised of a ON period and a OFF period. Max 12 blocks in LabView (12 On + 12 Off)*repeat.
            blocklist = []  # raw blocks from LabView -> 1 block (ON+OFF) + etc
            for blockN in range(1, 13):  # 13? or more ? Max 12 blocks, coded in LabView...
                # add block if  block >0 seconds then get data from file.
                # Data from behav_params as follows: Block N°: // ON block Duration // OFF block duration // Repeat block // % reward ON // % reward OFF // Treadmill speed.
                if read_params(root, animal, session, "Block " + str(blockN), dataindex=-6, valueType=str) != 0:
                    blocklist.append([read_params(root, animal, session, "Block " + str(blockN), dataindex=-6, valueType=str), read_params(root, animal, session, "Block " + str(blockN), dataindex=-5, valueType=str),
                                      read_params(root, animal, session, "Block " + str(blockN), dataindex=-4, valueType=str), read_params(root, animal, session, "Block " + str(blockN), dataindex=-3, valueType=str),
                                      read_params(root, animal, session, "Block " + str(blockN), dataindex=-2, valueType=str), read_params(root, animal, session, "Block " + str(blockN), dataindex=-1, valueType=str), blockN])
            # create an array [start_block, end_block] for each block using the values we have just read -> 1 block ON + 1 bloc OFF + etc.
            timecount, blockON_start, blockON_end, blockOFF_start, blockOFF_end = 0, 0, 0, 0, 0
            blocks = []  # blocks that we are going to use in the data processing. 1 block ON + 1 bloc OFF + etc.
            rewardP_ON = []  # probability of getting the reward in each ON phase
            rewardP_OFF = []  # same for OFF
            treadmillSpeed = []  # treadmill speed for each block (ON + OFF blocks not differenciated for now)
            rewardProbaBlock = []
            for block in blocklist:
                for repeat in range(0, block[2]):  # in essence blocks = [a, b], [b, c], [c, d], ...
                    blockON_start = timecount
                    timecount += block[0]
                    blockON_end = timecount
                    blockOFF_start = timecount
                    timecount += block[1]
                    blockOFF_end = timecount
                    blocks.append([blockON_start, blockON_end])
                    if blockOFF_start - blockOFF_end != 0:
                        blocks.append([blockOFF_start, blockOFF_end])
                    rewardP_ON.append(block[3])
                    rewardP_OFF.append(block[4])
                    rewardProbaBlock.extend(block[3:5])
                    treadmillSpeed.append(block[5])
                    treadmillSpeed.append(block[5])
            params[animal, session]["blocks"], params[animal, session]["rewardP_ON"], params[animal, session]["rewardP_OFF"], params[animal, session]["treadmillSpeed"], params[animal, session]['rewardProbaBlock'] = blocks, rewardP_ON, rewardP_OFF, treadmillSpeed, rewardProbaBlock
            
            # Extract data for each .position file generated from LabView
            # Data loaded : time array, position of the animal X and Y axis, Licks to the left and to the right, and frame number
            extractTime[animal, session]      = read_csv_pandas((root+os.sep+animal+os.sep+"Experiments"+os.sep + session + os.sep+session+".position"), Col=[3])  # old format = 5
            extractPositionX[animal, session] = read_csv_pandas((root+os.sep+animal+os.sep+"Experiments"+os.sep + session + os.sep+session+".position"), Col=[4])  # old format = 6
            extractPositionY[animal, session] = read_csv_pandas((root+os.sep+animal+os.sep+"Experiments"+os.sep + session + os.sep+session+".position"), Col=[5])
            extractLickLeft[animal, session]  = read_csv_pandas((root+os.sep+animal+os.sep+"Experiments"+os.sep + session + os.sep+session+".position"), Col=[6])
            extractLickRight[animal, session] = read_csv_pandas((root+os.sep+animal+os.sep+"Experiments"+os.sep + session + os.sep+session+".position"), Col=[7])
            solenoid_ON_Left[animal, session] = read_csv_pandas((root+os.sep+animal+os.sep+"Experiments"+os.sep + session + os.sep+session+".position"), Col=[8])
            solenoid_ON_Right[animal, session]= read_csv_pandas((root+os.sep+animal+os.sep+"Experiments"+os.sep + session + os.sep+session+".position"), Col=[9])
            framebuffer[animal, session]      = read_csv_pandas((root+os.sep+animal+os.sep+"Experiments"+os.sep + session + os.sep+session+".position"), Col=[10])
            cameraEdit[animal, session]       = read_csv_pandas((root+os.sep+animal+os.sep+"Experiments"+os.sep + session + os.sep+session+".position"), Col=[11])

            # Cut leftover data at the end of the session (e.g. session is 1800s long, data goes up to 1820s because session has not been stopped properly/stopped manually, so we remove the extra 20s)
            rawTime[animal, session]          =      extractTime[animal, session][extractTime[animal, session] <= params[animal, session]["sessionDuration"]]
            rawPositionX[animal, session]     = extractPositionX[animal, session][extractTime[animal, session] <= params[animal, session]["sessionDuration"]]
            rawPositionY[animal, session]     = extractPositionY[animal, session][extractTime[animal, session] <= params[animal, session]["sessionDuration"]]
            rawLickLeftX[animal, session]     =  extractLickLeft[animal, session][extractTime[animal, session] <= params[animal, session]["sessionDuration"]]
            rawLickLeftY[animal, session]     =  extractLickLeft[animal, session][extractTime[animal, session] <= params[animal, session]["sessionDuration"]]  # not needed, check
            rawLickRightX[animal, session]    = extractLickRight[animal, session][extractTime[animal, session] <= params[animal, session]["sessionDuration"]]
            rawLickRightY[animal, session]    = extractLickRight[animal, session][extractTime[animal, session] <= params[animal, session]["sessionDuration"]]  # not needed, check
            solenoid_ON_Left[animal, session] = solenoid_ON_Left[animal, session][extractTime[animal, session] <= params[animal, session]["sessionDuration"]]
            solenoid_ON_Right[animal, session]=solenoid_ON_Right[animal, session][extractTime[animal, session] <= params[animal, session]["sessionDuration"]]  # not needed, check
            cameraEdit[animal, session]       =       cameraEdit[animal, session][extractTime[animal, session] <= params[animal, session]["sessionDuration"]]
            
            # convert data from px to cm
            rawPositionX[animal, session], rawPositionY[animal, session] = datapx2cm(rawPositionX[animal, session]), datapx2cm(rawPositionY[animal, session])
            rawSpeed[animal, session] = compute_speed(rawPositionX[animal, session], rawTime[animal, session])
            smoothMask[animal, session] = np.array([True])

            # usually rat is not found in the first few frames, so we replace Xposition by the first nonzero value
            # this is detected as a camera edit, so we fix that as well
            rawPositionX[animal, session], cameraEdit[animal, session] = fix_start_session(rawPositionX[animal, session], cameraEdit[animal, session])
            rawPositionX[animal, session] = fixcamglitch(rawTime[animal, session], rawPositionX[animal, session], cameraEdit[animal, session])

            #######################################################################################
            # smoothing
            smoothPos, smoothSpeed = True, True
            sigmaPos, sigmaSpeed = 2, 2  # seems to work, less: not smoothed enough, more: too smoothed, not sure how to objectively compute an optimal value.
            if smoothPos == True:
                if smoothSpeed == True:
                    rawPositionX[animal, session] = smooth(rawPositionX[animal, session], sigmaPos)
                    rawSpeed[animal, session] = smooth(compute_speed(rawPositionX[animal, session], rawTime[animal, session]), sigmaSpeed)
                else:
                    rawPositionX[animal, session] = smooth(rawPositionX[animal, session], sigmaPos)
            ######################################################################################

            # Load lick data -- Licks == measure of conductance at the reward port. Conductance is ____ and when lick, increase of conductance so ___|_|___, we define it as a lick if it is above a threshold. But baseline value can randomly increase like this ___----, so baseline can be above threshold, so false detections. -> compute moving median to get the moving baseline (median, this way we eliminate the peaks in the calculation of the baseline) and then compare with threshold. __|_|__---|---|----
            window = 200
            if params[animal, session]["lickthresholdLeft"] == None:
                params[animal, session]["lickthresholdLeft"] = 300
            if params[animal, session]["lickthresholdRight"] == None:
                params[animal, session]["lickthresholdRight"] = 300
            rawLickLeftX[animal, session] = [k if i-j >= params[animal, session]["lickthresholdLeft"] else 0 for i, j, k in zip(rawLickLeftX[animal, session], movinmedian(rawLickLeftX[animal, session], window), rawPositionX[animal, session])]
            rawLickRightX[animal, session] = [k if i-j >= params[animal, session]["lickthresholdRight"] else 0 for i, j, k in zip(rawLickRightX[animal, session], movinmedian(rawLickRightX[animal, session], window), rawPositionX[animal, session])]

            # Specify if a session has lick data problems, so we don't discard the whole session (keep the run behavior, remove lick data)
            if all(v == 0 for v in rawLickLeftX[animal, session]):
                params[animal, session]["hasLick"] = False
            elif all(v == 0 for v in rawLickRightX[animal, session]):
                params[animal, session]["hasLick"] = False
            elif animal + " " + session in lickBug:
                params[animal, session]["hasLick"] = False
            else:
                params[animal, session]["hasLick"] = True

            # Water data. Drop size and volume rewarded. Compute drop size for each reward port. Determine if drops are equal, or which one is bigger. Assign properties (e.g. line width for plots) accordingly.
            limitWater_diff = 5
            watL = round(params[animal, session]["waterLeft"], 1)  # print(round(params[animal, session]["waterLeft"], 1), "µL/drop")
            watR = round(params[animal, session]["waterRight"], 1)  # print(round(params[animal, session]["waterRight"], 1), "µL/drop")
            if watL-(watL*limitWater_diff/100) <= watR <= watL+(watL*limitWater_diff/100):
                water[animal, session] = ["Same Reward Size", "Same Reward Size", 2, 2]  # print(session, "::", watL, watR, "     same L-R") #print(watL-(watL*limitWater_diff/100)) #print(watL+(watL*limitWater_diff/100))
            elif watL < watR:
                water[animal, session] = ["Small Reward", "Big Reward", 1, 5]  # print(session, "::", watL, watR, "     bigR")
            elif watL > watR:
                water[animal, session] = ["Big Reward", "Small Reward", 5, 1]  # print(session, "::", watL, watR, "     bigL")
            else:
                water[animal, session] = ["r", "r", 1, 1]

            # Compute boundaries
            border = 5  # define arbitrary border
            leftBoundaryPeak[animal, session], rightBoundaryPeak[animal, session], kde[animal, session] = extract_boundaries(rawPositionX[animal, session], animal, session, params[animal, session]['treadmillDist'], height=0.001)


            for s in boundariesBug:
                if session == s[0]:
                    params[animal, session]["boundaries"] = s[1]
                    break
                else:
                    params[animal, session]["boundaries"] = [rightBoundaryPeak[animal, session] - border, leftBoundaryPeak[animal, session] + border]
            
            # Compute or pickle run/stay mask
            maskpicklePath = root+os.sep+animal+os.sep+"Experiments"+os.sep+session+os.sep+"Analysis"+os.sep+"mask.p"
            if os.path.exists(maskpicklePath) and (not redoMask):
                binMask[animal, session] = get_from_pickle(root, animal, session, name="mask.p")
            else:
                if session in runstaysepbug:
                    septhreshold = 0.0004
                else:
                    septhreshold = 0.0002
                rawMask[animal, session] = filterspeed(animal, session, rawPositionX[animal, session], rawSpeed[animal, session], rawTime[animal, session], septhreshold, params[animal, session]["treadmillDist"])  # threshold 0.0004 seems to work ok for all TM distances. lower the thresh the bigger the wait blob zone taken, which caused problems in 60cm configuration.
                smoothMask[animal, session] = removeSplits_Mask(rawMask, rawPositionX, animal, session, params[animal, session]["treadmillDist"])
                binMask[animal, session] = fixSplittedRunsMask(animal, session, bin_session(animal, session, smoothMask, rawTime, blocks), blocks)
            smoothMask[animal, session] = stitch([binMask[animal, session]])[0]
            running_Xs[animal, session] = [val[0] if val[1] == True else None for val in [[i, j] for i, j in zip(rawPositionX[animal, session], smoothMask[animal, session])]]
            idle_Xs[animal, session] = [val[0] if val[1] == False else None for val in [[i, j] for i, j in zip(rawPositionX[animal, session], smoothMask[animal, session])]]
            goodSpeed[animal, session] = [val[0] if val[1] == True else None for val in [[i, j] for i, j in zip(rawSpeed[animal, session], smoothMask[animal, session])]]
            badSpeed[animal, session] = [val[0] if val[1] == False else None for val in [[i, j] for i, j in zip(rawSpeed[animal, session], smoothMask[animal, session])]]
            binSpeed[animal, session] = reCutBins(rawSpeed[animal, session], binMask[animal, session])
            binTime[animal, session] = reCutBins(rawTime[animal, session], binMask[animal, session])
            binPositionX[animal, session] = reCutBins(rawPositionX[animal, session], binMask[animal, session])
            binPositionY[animal, session] = reCutBins(rawPositionY[animal, session], binMask[animal, session])
            binLickLeftX[animal, session] = reCutBins(rawLickLeftX[animal, session], binMask[animal, session])
            binLickRightX[animal, session] = reCutBins(rawLickRightX[animal, session], binMask[animal, session])
            binSolenoid_ON_Left[animal, session] = reCutBins(solenoid_ON_Left[animal, session], binMask[animal, session])
            binSolenoid_ON_Right[animal, session] = reCutBins(solenoid_ON_Right[animal, session], binMask[animal, session])

            # Extract all variables.
            speedRunToRightBin[animal, session], speedRunToLeftBin[animal, session], XtrackRunToRightBin[animal, session], XtrackRunToLeftBin[animal, session], timeRunToRightBin[animal, session], timeRunToLeftBin[animal, session], timeStayInRightBin[animal, session], timeStayInLeftBin[animal, session], XtrackStayInRightBin[animal, session], XtrackStayInLeftBin[animal, session], TtrackStayInRightBin[animal, session], TtrackStayInLeftBin[animal, session], instantSpeedRightBin[animal, session], instantSpeedLeftBin[animal, session], maxSpeedRightBin[animal, session], maxSpeedLeftBin[animal, session], whenmaxSpeedRightBin[animal, session], whenmaxSpeedLeftBin[animal, session], wheremaxSpeedRightBin[animal, session], wheremaxSpeedLeftBin[animal, session], lick_arrivalRightBin[animal, session], lick_drinkingRightBin[animal, session], lick_waitRightBin[animal, session], lick_arrivalLeftBin[animal, session], lick_drinkingLeftBin[animal, session], lick_waitLeftBin[animal, session], rewardedRightBin[animal, session], rewardedLeftBin[animal, session] = extract_runSpeedBin(binPositionX, binSpeed, binTime, binLickRightX, binLickLeftX, binSolenoid_ON_Right[animal, session], binSolenoid_ON_Left[animal, session], binMask, animal, session, params[animal, session]['blocks'], params[animal, session]["boundaries"],  params[animal, session]["treadmillSpeed"], params[animal, session]['rewardProbaBlock'])
            speedRunToRight[animal, session],    speedRunToLeft[animal, session],    XtrackRunToRight[animal, session],    XtrackRunToLeft[animal, session],    timeRunToRight[animal, session],    timeRunToLeft[animal, session],    timeStayInRight[animal, session],    timeStayInLeft[animal, session],    XtrackStayInRight[animal, session],    XtrackStayInLeft[animal, session],    TtrackStayInRight[animal, session],    TtrackStayInLeft[animal, session],    instantSpeedRight[animal, session],    instantSpeedLeft[animal, session],    maxSpeedRight[animal, session],    maxSpeedLeft[animal, session],    whenmaxSpeedRight[animal, session],    whenmaxSpeedLeft[animal, session],    wheremaxSpeedRight[animal, session],    wheremaxSpeedLeft[animal, session],    lick_arrivalRight[animal, session],    lick_drinkingRight[animal, session],    lick_waitRight[animal, session],    lick_arrivalLeft[animal, session],    lick_drinkingLeft[animal, session],    lick_waitLeft[animal, session], rewardedRight[animal, session], rewardedLeft[animal, session] = stitch([speedRunToRightBin[animal, session], speedRunToLeftBin[animal, session], XtrackRunToRightBin[animal, session], XtrackRunToLeftBin[animal, session], timeRunToRightBin[animal, session], timeRunToLeftBin[animal, session], timeStayInRightBin[animal, session], timeStayInLeftBin[animal, session], XtrackStayInRightBin[animal, session], XtrackStayInLeftBin[animal, session], TtrackStayInRightBin[animal, session], TtrackStayInLeftBin[animal, session], instantSpeedRightBin[animal, session], instantSpeedLeftBin[animal, session], maxSpeedRightBin[animal, session], maxSpeedLeftBin[animal, session], whenmaxSpeedRightBin[animal, session], whenmaxSpeedLeftBin[animal, session], wheremaxSpeedRightBin[animal, session], wheremaxSpeedLeftBin[animal, session], lick_arrivalRightBin[animal, session], lick_drinkingRightBin[animal, session], lick_waitRightBin[animal, session], lick_arrivalLeftBin[animal, session], lick_drinkingLeftBin[animal, session], lick_waitLeftBin[animal, session], rewardedRightBin[animal, session], rewardedLeftBin[animal, session]])
            nb_runs_to_rightBin[animal, session], nb_runs_to_leftBin[animal, session], nb_runsBin[animal, session], total_trials[animal, session] = {}, {}, {}, 0
            for i in range(0, len(params[animal, session]['blocks'])):
                nb_runs_to_rightBin[animal, session][i] = len(speedRunToRightBin[animal, session][i])
                nb_runs_to_leftBin[animal, session][i] = len(speedRunToLeftBin[animal, session][i])
                nb_runsBin[animal, session][i] = len(speedRunToRightBin[animal, session][i]) + len(speedRunToLeftBin[animal, session][i])
                total_trials[animal, session] = total_trials[animal, session] + nb_runsBin[animal, session][i]

            nb_rewardBlockLeft[animal, session], nb_rewardBlockRight[animal, session], nbWaterLeft[animal, session], nbWaterRight[animal, session] = {}, {}, 0, 0
            for i in range(0, len(params[animal, session]['blocks'])):
                nb_rewardBlockLeft[animal, session][i] = sum([1 if t[0] <= params[animal, session]['rewardProbaBlock'][i] else 0 for t in split_a_list_at_zeros(binSolenoid_ON_Left[animal, session][i])])  # split a list because in data file we have %open written along valve opening time duration (same value multiple time), so we only take the first one, verify >threshold, ...
                nb_rewardBlockRight[animal, session][i] = sum([1 if t[0] <= params[animal, session]['rewardProbaBlock'][i] else 0 for t in split_a_list_at_zeros(binSolenoid_ON_Right[animal, session][i])])  # print(i+1, nb_rewardBlockLeft[animal, session][i], nb_rewardBlockRight[animal, session][i])
            nbWaterLeft[animal, session] = sum(nb_rewardBlockLeft[animal, session].values())
            nbWaterRight[animal, session] = sum(nb_rewardBlockRight[animal, session].values())
            totalWater[animal, session] = round((nbWaterLeft[animal, session] * params[animal, session]["waterLeft"] + nbWaterRight[animal, session] * params[animal, session]["waterRight"])/1000, 2), 'mL'  # totalWater[animal, session] = nbWaterLeft[animal, session] * params[animal, session]["waterLeft"], "+", nbWaterRight[animal, session] * params[animal, session]["waterRight"]

            # compute total X distance moved during the session for each rat. maybe compute XY.
            totalDistance[animal, session] = sum(abs(np.diff(rawPositionX[animal, session])))/100

            # sequences
            changes = np.argwhere(np.diff(smoothMask[animal, session])).squeeze()
            full = []
            full.append(smoothMask[animal, session][:changes[0]+1])
            for i in range(0, len(changes)-1):
                full.append(smoothMask[animal, session][changes[i]+1:changes[i+1]+1])
            full.append(smoothMask[animal, session][changes[-1]+1:])
            fulltime = recut(rawTime[animal, session], full)
            openings = recut(solenoid_ON_Left[animal, session] + solenoid_ON_Right[animal, session], full)
            positions = recut(rawPositionX[animal, session], full)
            d = {}
            for item, (j, t, o, p) in enumerate(zip(full, fulltime, openings, positions)):
                proba = split_a_list_at_zeros(o)[0][0] if np.any(split_a_list_at_zeros(o)) else 100
                #     #action start time        #run or stay       #get reward (1) or not (0)                                                        #action duration       #dist/time=avg speed if run 
                d[item] = t[0], "run" if j[0] == True else "stay", 1 if proba < params[animal, session]['rewardProbaBlock'][get_block(t[0])] else 0, t[-1] - t[0], (p[-1] - p[0])/(t[-1] - t[0]) if j[0] == True else "wait"


        if os.path.exists(figPath) and (not redoFig):
            if printFigs == True:
                display(Image(filename=figPath))
        else:
            if redoCompute == False:
                print(session, " Error, you need to recompute everything to generate Fig.")
            else:
                # Plot figure
                fig = plt.figure(constrained_layout=False, figsize=(32, 42))
                fig.suptitle(session, y=0.9, fontsize=24)
                gs = fig.add_gridspec(75, 75)
                ax00 = fig.add_subplot(gs[0:7, 0:4])
                ax00 = plot_peak(ax00, rawPositionX[animal, session], leftBoundaryPeak[animal, session], rightBoundaryPeak[animal, session], kde[animal, session], [0.05, 0, 0], [0, 120, 0], xyLabels=["Position (cm)", "%"])
                ax01 = fig.add_subplot(gs[0:7, 5:75])
                ax01 = plot_BASEtrajectoryV2(animal, session, rawTime[animal, session], running_Xs[animal, session], idle_Xs[animal, session], rawLickLeftX[animal, session], rawLickRightX[animal, session], params[animal, session]['rewardProbaBlock'], params[animal, session]['blocks'], barplotaxes=[0, params[animal, session]['sessionDuration'], 50, 90, 0, 22, 10],  xyLabels=["Time (min)", " ", "Position (cm)", "", "", "", 14, 12], title=[session, "", " ", "", 16], linewidth=[1.5])
                plt.plot([0, params[animal, session]['sessionDuration']], [params[animal, session]["boundaries"][0], params[animal, session]["boundaries"][0]], ":", color='k', alpha=0.5)
                plt.plot([0, params[animal, session]['sessionDuration']], [params[animal, session]["boundaries"][1], params[animal, session]["boundaries"][1]], ":", color='k', alpha=0.5)

                gs00 = gs[8:13, 0:75].subgridspec(2, 75)
                ax11 = fig.add_subplot(gs00[0, 5:75])
                ax12 = fig.add_subplot(gs00[1, 0:75])
                ax11.plot(rawTime[animal, session], goodSpeed[animal, session], color='dodgerblue')
                ax11.plot(rawTime[animal, session], badSpeed[animal, session], color='orange')
                ax11.set_xlabel('time (s)')
                ax11.set_ylabel('speed (cm/s)')
                ax11.set_xlim(0, 3600)
                ax11.set_ylim(-200, 200)
                ax11.spines['top'].set_color("none")
                ax11.spines['right'].set_color("none")
                ax11.spines['left'].set_color("none")
                ax11.spines['bottom'].set_color("none")
                ax12.scatter(rawPositionX[animal, session], goodSpeed[animal, session], color='dodgerblue', s=0.5)
                ax12.scatter(rawPositionX[animal, session], badSpeed[animal, session], color='orange', s=0.5)
                ax12.set_xlabel('position (cm)')
                ax12.set_ylabel('speed (cm/s)')
                ax12.set_xlim(0, 130)
                ax12.set_ylim(-150, 150)
                ax12.spines['top'].set_color("none")
                ax12.spines['right'].set_color("none")
                ax12.spines['left'].set_color("none")
                ax12.spines['bottom'].set_color("none")
                yline = [0, 120]
                xline = [0, 0]
                ax12.plot(yline, xline, ":", color='k')

                ax20 = fig.add_subplot(gs[17:22, 0:10])
                ax20 = plot_tracks(ax20, XtrackRunToRight[animal, session], timeRunToRight[animal, session], params[animal, session]["boundaries"], xylim=[-0.1, 2, 0, 120], color=['paleturquoise', 'tomato'],  xyLabels=["Time (s)", "X Position (cm)", 14], title=["Tracking run to Right",  16])
                ax21 = fig.add_subplot(gs[17:22, 15:25])
                ax21 = plot_tracks(ax21, XtrackRunToLeft[animal, session], timeRunToLeft[animal, session], params[animal, session]["boundaries"], xylim=[-0.1, 2, 0, 120], color=['darkcyan', 'darkred'], xyLabels=["Time (s)", "", 14], title=["Tracking run to Left", 16])
                ax20 = fig.add_subplot(gs[17:22, 30:40])
                ax20 = cumul_plot(ax20, speedRunToRight[animal, session], speedRunToLeft[animal, session], barplotaxes=[0, 120, 0, 1], maxminstepbin=[0, 120, 1], scatterplotaxes=[0, 0, 0, 0], color=['paleturquoise', 'darkcyan', 'tomato', 'darkred'], xyLabels=["Speed cm/s", "Cumulative Frequency Run Speed", 14, 12], title=["Cumulative Plot Good Run Speed", 16], linewidth=[1.5], legend=["To Right: " + water[animal, session][1] + " " + str(params[animal, session]["waterRight"]) + "µL/drop", "To Left:  " + water[animal, session][0] + " " + str(params[animal, session]["waterLeft"]) + "µL/drop", water[animal, session][2], water[animal, session][3]])
                ax21 = fig.add_subplot(gs[17:22, 45:55])
                ax21 = distribution_plot(ax21, speedRunToRight[animal, session], speedRunToLeft[animal, session], barplotaxes=[0, 0, 0, 0], maxminstepbin=[0, 120, 1], scatterplotaxes=[0.5, 2.5, 0, 120], color=['paleturquoise', 'darkcyan', 'tomato', 'darkred'], xyLabels=["Speed (cm/s)", "Direction of run", "To Right" + "\n" + water[animal, session][1], "To Left" + "\n" + water[animal, session][0], 14, 12], title=["Distribution of All Run Speed", 16], linewidth=[1.5], legend=["To Right: Good Runs ", "To Left: Good Runs", "To Right: Bad Runs", "To Left: Bad Runs"])

                gs23 = gs[15:22, 60:75].subgridspec(5, 2)
                ax231 = fig.add_subplot(gs23[0:2, 0:2])
                if len(framebuffer[animal, session]) != 0:
                    ax231.set_title("NbBug/TotFrames: %s/%s = %.2f" % (sum(np.diff(framebuffer[animal, session])-1), len(framebuffer[animal, session]), sum(np.diff(framebuffer[animal, session])-1)/len(framebuffer[animal, session])), fontsize=16)
                ax231.scatter(list(range(1, len(framebuffer[animal, session]))), [x-1 for x in np.diff(framebuffer[animal, session])], s=5)
                ax231.set_xlabel("frame index")
                ax231.set_ylabel("dFrame -1 (0 is ok)")
                ax232 = fig.add_subplot(gs23[3:5, 0:2])
                ax232.set_title(params[animal, session]["realSessionDuration"], fontsize=16)
                ax232.plot(np.diff(rawTime[animal, session]), label="data")
                ax232.plot(movinavg(np.diff(rawTime[animal, session]), 100), label="moving average")
                ax232.set_xlim(0, len(np.diff(rawTime[animal, session])))
                ax232.set_ylim(0, 0.1)
                ax232.set_xlabel("frame index")
                ax232.set_ylabel("time per frame (s)")

                ax30 = fig.add_subplot(gs[25:30, 0:10])
                ax30 = cumul_plot(ax30, maxSpeedRight[animal, session], maxSpeedLeft[animal, session], barplotaxes=[0, 200, 0, 1], maxminstepbin=[0, 200, 1], scatterplotaxes=[0.5, 2.5, 0, 100], color=['lightgreen', 'darkgreen', 'tomato', 'darkred'], xyLabels=["Speed cm/s", "Cumulative Frequency MAX Run Speed", 14, 12], title=["Cumulative Plot MAX Run Speed", 16], linewidth=[1.5], legend=["To Right: " + water[animal, session][1] + " " + str(params[animal, session]["waterRight"]) + "µL/drop", "To Left:  " + water[animal, session][0] + " " + str(params[animal, session]["waterLeft"]) + "µL/drop", water[animal, session][2], water[animal, session][3]])
                ax31 = fig.add_subplot(gs[25:30, 15:25])
                ax31 = distribution_plot(ax31, maxSpeedRight[animal, session], maxSpeedLeft[animal, session], barplotaxes=[0, 100, 0, 1], maxminstepbin=[0, 100, 1], scatterplotaxes=[0.5, 2.5, 0, 200], color=['lightgreen', 'darkgreen', 'tomato', 'darkred'], xyLabels=["Speed (cm/s)", "Direction of run", "To Right" + "\n" + water[animal, session][1], "To Left" + "\n" + water[animal, session][0], 14, 12], title=["Distribution of MAX Run Speed", 16], linewidth=[1.5], legend=["To Right: Good Runs", "To Left: Good Runs", "To Right: Bad Runs", "To Left: Bad Runs"])
                ax32 = fig.add_subplot(gs[25:30, 30:40])
                ax32 = plot_speed(ax32, instantSpeedRight[animal, session], timeRunToRight[animal, session], [0, 0], xylim=[-0.1, 4, 0, 200], xyLabels=["Time (s)", "X Speed (cm/s)", 14], title=["To Right" + "\n" + "To " + water[animal, session][1] + " " + str(params[animal, session]["waterRight"]) + "µL/drop", 12])
                ax33 = fig.add_subplot(gs[25:30, 45:55])
                ax33 = plot_speed(ax33, instantSpeedLeft[animal, session], timeRunToLeft[animal, session], [0, 0], xylim=[-0.1, 4, 0, 200], xyLabels=["Time (s)", "", 14], title=["To Left" + "\n" + "To " + water[animal, session][0] + " " + str(params[animal, session]["waterLeft"]) + "µL/drop", 12])
                ax34 = fig.add_subplot(gs[25:30, 60:70])
                ax34 = plot_speed(ax34, instantSpeedRight[animal, session] + instantSpeedLeft[animal, session], timeRunToRight[animal, session] + timeRunToLeft[animal, session], [0, 0], xylim=[-0.1, 4, 0, 200], xyLabels=["Time (s)", "", 14], title=["Speed" + "\n" + " To left and to right", 12])

                ax40 = fig.add_subplot(gs[35:40, 0:8])
                ax40 = cumul_plot(ax40, maxSpeedRight[animal, session], maxSpeedLeft[animal, session], barplotaxes=[0, 250, 0, 1], maxminstepbin=[0, 250, 1], scatterplotaxes=[0, 0, 0, 0], color=['lightgreen', 'darkgreen', 'tomato', 'darkred'], xyLabels=["Speed cm/s", "Cumulative Frequency MAX Run Speed", 14, 12], title=["CumulPlt MAXrunSpeed <TreadmillCorrected>", 16], linewidth=[1.5], legend=["To Right: " + water[animal, session][1] + " " + str(params[animal, session]["waterRight"]) + "µL/drop", "To Left:  " + water[animal, session][0] + " " + str(params[animal, session]["waterLeft"]) + "µL/drop", water[animal, session][2], water[animal, session][3]])
                ax41 = fig.add_subplot(gs[35:40, 12:23])
                ax41 = distribution_plot(ax41, maxSpeedRight[animal, session], maxSpeedLeft[animal, session], barplotaxes=[0, 0, 0, 0], maxminstepbin=[0, 0, 0], scatterplotaxes=[0.5, 2.5, 0, 250], color=['lightgreen', 'darkgreen', 'tomato', 'darkred'], xyLabels=["Speed (cm/s)", "Direction of run", "To Right" + "\n" + water[animal, session][1], "To Left" + "\n" + water[animal, session][0], 14, 12], title=["Distr. of MAXrunSpeed <TreadmillCorrected>", 16], linewidth=[1.5], legend=["To Right: Good Runs", "To Left: Good Runs", "To Right: Bad Runs", "To Left: Bad Runs"])
                ax42 = fig.add_subplot(gs[35:40, 26:34])  # where maxspeed
                ax42 = cumul_plot(ax42, wheremaxSpeedRight[animal, session], wheremaxSpeedLeft[animal, session], barplotaxes=[0, 120, 0, 1], maxminstepbin=[0, 120, 1], scatterplotaxes=[0, 0, 0, 0], color=['lightgreen', 'darkgreen', 'tomato', 'darkred'], xyLabels=["Position maxSpeed reached (cm)", "Cumulative Frequency MAX runSpeed Position", 14, 12], title=["CumulPlt MAXrunSpeed \nPosition from start of run", 16], linewidth=[1.5], legend=["To Right: " + water[animal, session][1] + " " + str(params[animal, session]["waterRight"]) + "µL/drop", "To Left:  " + water[animal, session][0] + " " + str(params[animal, session]["waterLeft"]) + "µL/drop", water[animal, session][2], water[animal, session][3]])
                ax43 = fig.add_subplot(gs[35:40, 38:49])
                ax43 = distribution_plot(ax43, wheremaxSpeedRight[animal, session], wheremaxSpeedLeft[animal, session], barplotaxes=[0, 0, 0, 0], maxminstepbin=[0, 0, 0], scatterplotaxes=[0.5, 2.5, 0, 120], color=['lightgreen', 'darkgreen', 'tomato', 'darkred'], xyLabels=["X Position (cm)", "Direction of run", "To Right" + "\n" + water[animal, session][1], "To Left" + "\n" + water[animal, session][0], 14, 12], title=["Distr. MAXrunSpeed \nPosition from start of run", 16], linewidth=[1.5], legend=["To Right: Good Runs", "To Left: Good Runs", "To Right: Bad Runs", "To Left: Bad Runs"])
                ax44 = fig.add_subplot(gs[35:40, 52:60])  # when maxspeed
                ax44 = cumul_plot(ax44, whenmaxSpeedRight[animal, session], whenmaxSpeedLeft[animal, session], barplotaxes=[0, 2.5, 0, 1], maxminstepbin=[0, 2.5, 0.04], scatterplotaxes=[0, 0, 0, 0], color=['lightgreen', 'darkgreen', 'tomato', 'darkred'], xyLabels=["Time MAX runSpeed reached (s)", "Cumulative Frequency", 14, 12], title=["CumulPlt Time of \nMAXrunSpeed from start of run", 16], linewidth=[1.5], legend=["To Right: " + water[animal, session][1] + " " + str(params[animal, session]["waterRight"]) + "µL/drop", "To Left:  " + water[animal, session][0] + " " + str(params[animal, session]["waterLeft"]) + "µL/drop", water[animal, session][2], water[animal, session][3]])
                ax45 = fig.add_subplot(gs[35:40, 64:75])
                ax45 = distribution_plot(ax45, whenmaxSpeedRight[animal, session], whenmaxSpeedLeft[animal, session], barplotaxes=[0, 0, 0, 0], maxminstepbin=[0, 0, 0], scatterplotaxes=[0.5, 2.5, 0, 2.5], color=['lightgreen', 'darkgreen', 'tomato', 'darkred'], xyLabels=["Time MAX runSpeed reached (s)", "Direction of run", "To Right" + "\n" + water[animal, session][1], "To Left" + "\n" + water[animal, session][0], 14, 12], title=["Distr. Time of MAXrunSpeed \nfrom start of run", 16], linewidth=[1.5], legend=["To Right: Good Runs", "To Left: Good Runs", "To Right: Bad Runs", "To Left: Bad Runs"])

                ax50 = fig.add_subplot(gs[45:50, 0:10])
                ax50 = plot_tracks(ax50, XtrackStayInRight[animal, session], TtrackStayInRight[animal, session], params[animal, session]["boundaries"], xylim=[-1, 10, params[animal, session]['treadmillDist']-40, params[animal, session]['treadmillDist']], color=['moccasin', 'tomato'], xyLabels=["Time (s)", "X Position (cm)", 14, 12], title=["Tracking in Right", 16])
                ax51 = fig.add_subplot(gs[45:50, 15:25])
                ax51 = plot_tracks(ax51, XtrackStayInLeft[animal, session], TtrackStayInLeft[animal, session], params[animal, session]["boundaries"], xylim=[-1, 10, 0, 40], color=['darkorange', 'darkred'], xyLabels=["Time (s)", "", 14, 12], title=["Tracking in Left", 16])
                ax52 = fig.add_subplot(gs[45:50, 30:40])
                ax52 = cumul_plot(ax52, timeStayInRight[animal, session], timeStayInLeft[animal, session], barplotaxes=[0, 15, 0, 1], maxminstepbin=[0, 15, 0.1], scatterplotaxes=[0, 0, 0, 0], color=['moccasin', 'darkorange', 'tomato', 'darkred'], xyLabels=["Time in zone (s)", "Cumulative Frequency Time In Zone", 14, 12], title=["Cumulative Plot Good Time In Zone", 16], linewidth=[1.5], legend=["In Right: " + water[animal, session][1] + " " + str(params[animal, session]["waterRight"]) + "µL/drop", "In Left:  " + water[animal, session][0] + " " + str(params[animal, session]["waterLeft"]) + "µL/drop", water[animal, session][2], water[animal, session][3]])
                ax53 = fig.add_subplot(gs[45:50, 45:60])
                ax53 = distribution_plot(ax53, timeStayInRight[animal, session], timeStayInLeft[animal, session], barplotaxes=[0, 0, 0, 0], maxminstepbin=[0, 30, 1], scatterplotaxes=[0.5, 2.5, 0, 30], color=['moccasin', 'darkorange', 'tomato', 'darkred'], xyLabels=["Time in zone (s)", "Zone", "In Right" + "\n" + water[animal, session][1], "In Left" + "\n" + water[animal, session][0], 14, 12], title=["Distribution of All Time In Zone", 16], linewidth=[1.5], legend=["To Right: Good Runs", "To Left: Good Runs", "To Right: Bad Runs", "To Left: Bad Runs"])

                ax60 = fig.add_subplot(gs[55:60, 0:8])
                ax60 = cumul_plot(ax60, lick_arrivalRight[animal, session], lick_arrivalLeft[animal, session], barplotaxes=[0, 2, 0, 1], maxminstepbin=[0, 2, 0.1], scatterplotaxes=[0.5, 2.5, 0, 100], color=['moccasin', 'darkorange', 'tomato', 'darkred'], xyLabels=["Time (s)", "Cumulative Frequency", 14, 12], title=["Cumulative Plot preDrink Time", 16], linewidth=[1.5], legend=["In Right: " + water[animal, session][1] + " " + str(params[animal, session]["waterRight"]) + "µL/drop", "In Left:  " + water[animal, session][0] + " " + str(params[animal, session]["waterLeft"]) + "µL/drop", water[animal, session][2], water[animal, session][3]])
                ax61 = fig.add_subplot(gs[55:60, 12:23])
                ax61 = distribution_plot(ax61, lick_arrivalRight[animal, session], lick_arrivalLeft[animal, session], barplotaxes=[0, 100, 0, 1], maxminstepbin=[0, 100, 1], scatterplotaxes=[0.5, 2.5, 0, 2], color=['moccasin', 'darkorange', 'tomato', 'darkred'], xyLabels=["Time (s)", "Zone", "In Right" + "\n" + water[animal, session][1], "In Left" + "\n" + water[animal, session][0], 14, 12], title=["Distribution preDrink Time", 16], linewidth=[1.5], legend=["In Right", "In Left", " ", " "])
                ax62 = fig.add_subplot(gs[55:60, 26:34])
                ax62 = cumul_plot(ax62, lick_drinkingRight[animal, session], lick_drinkingLeft[animal, session], barplotaxes=[0, 4, 0, 1], maxminstepbin=[0, 4, 0.1], scatterplotaxes=[0.5, 2.5, 0, 100], color=['moccasin', 'darkorange', 'tomato', 'darkred'], xyLabels=["Time (s)", "Cumulative Frequency", 14, 12], title=["Cumulative Plot Drink Time", 16], linewidth=[1.5], legend=["In Right: " + water[animal, session][1] + " " + str(params[animal, session]["waterRight"]) + "µL/drop", "In Left:  " + water[animal, session][0] + " " + str(params[animal, session]["waterLeft"]) + "µL/drop", water[animal, session][2], water[animal, session][3]])
                ax63 = fig.add_subplot(gs[55:60, 38:49])
                ax63 = distribution_plot(ax63, lick_drinkingRight[animal, session], lick_drinkingLeft[animal, session], barplotaxes=[0, 100, 0, 1], maxminstepbin=[0, 100, 1], scatterplotaxes=[0.5, 2.5, 0, 4], color=['moccasin', 'darkorange', 'tomato', 'darkred'], xyLabels=["Time (s)", "Zone", "In Right" + "\n" + water[animal, session][1], "In Left" + "\n" + water[animal, session][0], 14, 12], title=["Distribution of Drink Time", 16], linewidth=[1.5], legend=["In Right", "In Left", " ", " "])
                ax64 = fig.add_subplot(gs[55:60, 52:60])
                ax64 = cumul_plot(ax64, lick_waitRight[animal, session], lick_waitLeft[animal, session], barplotaxes=[0, 10, 0, 1], maxminstepbin=[0, 10, 0.1], scatterplotaxes=[0.5, 2.5, 0, 100], color=['moccasin', 'darkorange', 'tomato', 'darkred'], xyLabels=["Time (s)", "Cumulative Frequency", 14, 12], title=["Cumulative Plot postDrink Time", 16], linewidth=[1.5], legend=["In Right: " + water[animal, session][1] + " " + str(params[animal, session]["waterRight"]) + "µL/drop", "In Left:  " + water[animal, session][0] + " " + str(params[animal, session]["waterLeft"]) + "µL/drop", water[animal, session][2], water[animal, session][3]])
                ax65 = fig.add_subplot(gs[55:60, 64:75])
                ax65 = distribution_plot(ax65, lick_waitRight[animal, session], lick_waitLeft[animal, session], barplotaxes=[0, 100, 0, 1], maxminstepbin=[0, 100, 1], scatterplotaxes=[0.5, 2.5, 0, 10], color=['moccasin', 'darkorange', 'tomato', 'darkred'], xyLabels=["Time (s)", "Zone", "In Right" + "\n" + water[animal, session][1], "In Left" + "\n" + water[animal, session][0], 14, 12], title=["Distribution of postDrink Time", 16], linewidth=[1.5], legend=["In Right", "In Left", " ", " "])

                if len(params[animal, session]['blocks']) > 1:
                    stat = "Med. "
                    ax70 = fig.add_subplot(gs[63:70, 0:9])
                    ax70 = plot_figBin(ax70, [nb_runsBin[animal, session][i]/(int((blocks[i][1]-blocks[i][0])/60)) for i in range(0, len(blocks))], params[animal, session]['rewardProbaBlock'], params[animal, session]['blocks'], barplotaxes=[0, params[animal, session]['sessionDuration']/60, 0, 25], color=['k'], xyLabels=["Time (min)", "\u0023 runs / min", 14, 12], title=["", 16], stat=stat)
                    ax72 = fig.add_subplot(gs[63:70, 20:29])
                    ax72 = plot_figBin(ax72, [speedRunToLeftBin[animal, session][i] + speedRunToRightBin[animal, session][i] for i in range(0, len(blocks))], params[animal, session]['rewardProbaBlock'], params[animal, session]['blocks'], barplotaxes=[0, params[animal, session]['sessionDuration']/60, 0, 100], color=['dodgerblue'], xyLabels=["Time (min)", "Avg. run speed (cm/s)", 14, 12], title=["", 16], scatter=True, stat=stat)
                    ax74 = fig.add_subplot(gs[63:70, 40:49])
                    ax74 = plot_figBin(ax74, [maxSpeedRightBin[animal, session][i] + maxSpeedLeftBin[animal, session][i] for i in range(0, len(blocks))], params[animal, session]['rewardProbaBlock'], params[animal, session]['blocks'], barplotaxes=[0, params[animal, session]['sessionDuration']/60, 0, 150], color=['red'], xyLabels=["Time (min)", "Average max speed (cm/s)", 14, 12], title=["", 16], scatter=True, stat=stat)
                    ax76 = fig.add_subplot(gs[63:70, 60:69])
                    ax76 = plot_figBin(ax76, [timeStayInLeftBin[animal, session][i] + timeStayInRightBin[animal, session][i] for i in range(0, len(blocks))], params[animal, session]['rewardProbaBlock'], params[animal, session]['blocks'], barplotaxes=[0, params[animal, session]['sessionDuration']/60, 0, 25], color=['orange'], xyLabels=["Time (min)", "Avg. time in sides (s)", 14, 12], title=["", 16], scatter=True, stat=stat)

                    ax71 = fig.add_subplot(gs[63:70, 10:15])
                    ax71 = plot_figBinMean(ax71, [i/(int((params[animal, session]['blocks'][block][1]-params[animal, session]['blocks'][block][0])/60)) for block, i in enumerate(poolByReward([nb_runsBin[animal, session]], params[animal, session]["rewardP_OFF"][0], params[animal, session]['blocks'], params[animal, session]['rewardProbaBlock']))], [i/(int((params[animal, session]['blocks'][block][1]-params[animal, session]['blocks'][block][0])/60)) for block, i in enumerate(poolByReward([nb_runsBin[animal, session]], params[animal, session]["rewardP_ON"][0], params[animal, session]['blocks'], params[animal, session]['rewardProbaBlock']))], color=['k'], ylim=(0, 25))
                    ax73 = fig.add_subplot(gs[63:70, 30:35])
                    ax73 = plot_figBinMean(ax73, [np.mean(i) for i in poolByReward([speedRunToRightBin[animal, session], speedRunToLeftBin[animal, session]], params[animal, session]["rewardP_OFF"][0], params[animal, session]['blocks'], params[animal, session]['rewardProbaBlock'])], [np.mean(i) for i in poolByReward([speedRunToRightBin[animal, session], speedRunToLeftBin[animal, session]], params[animal, session]["rewardP_ON"][0], params[animal, session]['blocks'], params[animal, session]['rewardProbaBlock'])], color=['dodgerblue'], ylim=(0, 100))
                    ax75 = fig.add_subplot(gs[63:70, 50:55])
                    ax75 = plot_figBinMean(ax75, [np.mean(i) for i in poolByReward([maxSpeedRightBin[animal, session], maxSpeedLeftBin[animal, session]], params[animal, session]["rewardP_OFF"][0], params[animal, session]['blocks'], params[animal, session]['rewardProbaBlock'])], [np.mean(i) for i in poolByReward([maxSpeedRightBin[animal, session], maxSpeedLeftBin[animal, session]], params[animal, session]["rewardP_ON"][0], params[animal, session]['blocks'], params[animal, session]['rewardProbaBlock'])], color=['red'], ylim=(0, 150))
                    ax77 = fig.add_subplot(gs[63:70, 70:75])
                    ax77 = plot_figBinMean(ax77, [np.mean(i) for i in poolByReward([timeStayInRightBin[animal, session], timeStayInLeftBin[animal, session]], params[animal, session]["rewardP_OFF"][0], params[animal, session]['blocks'], params[animal, session]['rewardProbaBlock'])], [np.mean(i) for i in poolByReward([timeStayInRightBin[animal, session], timeStayInLeftBin[animal, session]], params[animal, session]["rewardP_ON"][0], params[animal, session]['blocks'], params[animal, session]['rewardProbaBlock'])], color=['orange'], ylim=(0, 25))

                ax80 = fig.add_subplot(gs[73:74, 0:60])
                ax80.spines['top'].set_color("none")
                ax80.spines['right'].set_color("none")
                ax80.spines['left'].set_color("none")
                ax80.spines['bottom'].set_color("none")
                ax80.tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False)
                ax80.tick_params(axis='y', which='both', left=False, right=False, labelleft=False)
                text = ("sessionDuration: {0} | acqPer: {1} | waterLeft: {2} | waterRight: {3} | treadmillDist: {4} | weight: {5} | lastWeightadlib: {6} | lastDayadlib: {7} | lickthresholdLeft: {8} | lickthresholdRight: {9} | realEnd: {10} | boundaries: {11} | daysSinceadLib: {12} \n realSessionDuration: {13} | blocks: {14} | \n rewardP_ON: {15} | rewardP_OFF: {16} | treadmillSpeed: {17} | rewardProbaBlock: {18} | hasLick: {19}").format(params[animal, session]['sessionDuration'], params[animal, session]['acqPer'], params[animal, session]['waterLeft'], params[animal, session]['waterRight'], params[animal, session]['treadmillDist'], params[animal, session]['weight'], params[animal, session]['lastWeightadlib'], params[animal, session]['lastDayadlib'], params[animal, session]['lickthresholdLeft'], params[animal, session]['lickthresholdRight'], params[animal, session]['realEnd'], params[animal, session]['boundaries'], params[animal, session]['daysSinceadLib'], params[animal, session]['realSessionDuration'], params[animal, session]['blocks'], params[animal, session]['rewardP_ON'], params[animal, session]['rewardP_OFF'], params[animal, session]['treadmillSpeed'], params[animal, session]['rewardProbaBlock'], params[animal, session]['hasLick'])
                ax80 = plt.text(0 ,0, str(text), wrap=True)

        ### -----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
        ### SAVE + PICKLE
        ### -----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

        if redoCompute == True:
            save_sessionplot_as_png(root, animal, session, 'recapFIG%s.png'%session, dpi='figure', transparent=False, background='w')
            save_as_pickle(root, params[animal, session],     animal, session, "params.p")
            save_as_pickle(root, binMask[animal, session], animal, session, "mask.p")
            save_as_pickle(root, nb_runsBin[animal, session], animal, session, "nbRuns.p")
            save_as_pickle(root, [totalDistance[animal, session], totalWater[animal, session], total_trials[animal, session]], animal, session, "misc.p")

            save_as_pickle(root, [speedRunToLeftBin[animal, session], speedRunToRightBin[animal, session]], animal, session, "avgSpeed.p")
            save_as_pickle(root, [[[np.sum(np.diff(j)) for j in timeRunToLeftBin[animal, session][i]]for i in range(0, len(params[animal, session]['blocks']))],
                                  [[np.sum(np.diff(j)) for j in timeRunToRightBin[animal, session][i]]for i in range(0, len(params[animal, session]['blocks']))]], animal, session, "timeRun.p")
            save_as_pickle(root, [maxSpeedLeftBin[animal, session], maxSpeedRightBin[animal, session]], animal, session, "maxSpeed.p")
            save_as_pickle(root, [timeStayInLeftBin[animal, session], timeStayInRightBin[animal, session]], animal, session, "timeinZone.p")
            save_as_pickle(root, [XtrackRunToLeftBin[animal, session], XtrackRunToRightBin[animal, session]], animal, session, "trackPos.p")
            save_as_pickle(root, [instantSpeedLeftBin[animal, session], instantSpeedRightBin[animal, session]], animal, session, "trackSpeed.p")
            save_as_pickle(root, [timeRunToLeftBin[animal, session], timeRunToRightBin[animal, session]], animal, session, "trackTime.p")
            save_as_pickle(root, [binLickLeftX[animal, session], binLickRightX[animal, session], binSolenoid_ON_Left[animal, session], binSolenoid_ON_Right[animal, session]], animal, session, "lick_valves.p")
            save_as_pickle(root, [rewardedRightBin[animal, session], rewardedLeftBin[animal, session]], animal, session, "rewarded.p")
            save_as_pickle(root, [TtrackStayInLeft[animal, session], TtrackStayInRight[animal, session]], animal, session, "trackTimeinZone.p")
            save_as_pickle(root, d, animal, session, "sequence.p")

            if printFigs == False:
                plt.close('all')

        ### -----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
        ### FLUSH
        ### -----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

            # Delete all data for this session
            params, rat_markers, water = {}, {}, {}
            extractTime, extractPositionX, extractPositionY, extractLickLeft, extractLickRight, framebuffer, solenoid_ON_Left, solenoid_ON_Right, cameraEdit = ({} for i in range(9)) 
            rawTime, rawPositionX, rawPositionY, rawLickLeftX, rawLickRightX, rawLickLeftY, rawLickRightY, smoothMask, rawSpeed = ({} for i in range(9)) 
            binPositionX, binPositionY, binTime, binLickLeftX, binLickRightX, binSolenoid_ON_Left, binSolenoid_ON_Right = ({} for i in range(7))
            leftBoundaryPeak, rightBoundaryPeak, kde = {}, {}, {}
            smoothMask, rawMask, binSpeed, binMask = {}, {}, {}, {}
            running_Xs, idle_Xs, goodSpeed, badSpeed = {}, {}, {}, {}
            speedRunToRight,    speedRunToLeft,    XtrackRunToRight,    XtrackRunToLeft,    timeRunToRight,    timeRunToLeft,    timeStayInRight,    timeStayInLeft,    XtrackStayInRight,    XtrackStayInLeft,    TtrackStayInRight,    TtrackStayInLeft,    instantSpeedRight,    instantSpeedLeft,    maxSpeedRight,    maxSpeedLeft,    whenmaxSpeedRight,    whenmaxSpeedLeft,    wheremaxSpeedRight,    wheremaxSpeedLeft,    lick_arrivalRight,    lick_drinkingRight,    lick_waitRight,    lick_arrivalLeft,    lick_drinkingLeft,    lick_waitLeft    = ({} for i in range(26))
            speedRunToRightBin, speedRunToLeftBin, XtrackRunToRightBin, XtrackRunToLeftBin, timeRunToRightBin, timeRunToLeftBin, timeStayInRightBin, timeStayInLeftBin, XtrackStayInRightBin, XtrackStayInLeftBin, TtrackStayInRightBin, TtrackStayInLeftBin, instantSpeedRightBin, instantSpeedLeftBin, maxSpeedRightBin, maxSpeedLeftBin, whenmaxSpeedRightBin, whenmaxSpeedLeftBin, wheremaxSpeedRightBin, wheremaxSpeedLeftBin, lick_arrivalRightBin, lick_drinkingRightBin, lick_waitRightBin, lick_arrivalLeftBin, lick_drinkingLeftBin, lick_waitLeftBin = ({} for i in range(26))
            nb_runs_to_rightBin, nb_runs_to_leftBin, nb_runsBin, total_trials = {}, {}, {}, {}

        clear_output(wait=True)


##########################################################################################################################################
# Median run computation
# Modified from: Averaging GPS segments competition 2019. https://doi.org/10.1016/j.patcog.2020.107730
#                T. Karasek, "SEGPUB.IPYNB", Github 2019. https://gist.github.com/t0mk/eb640963d7d64e14d69016e5a3e93fd6
##########################################################################################################################################

def median(lst): 
    sortedLst = sorted(lst)
    lstLen = len(lst)
    index = (lstLen - 1) // 2    
    return sortedLst[index] 
    
def zscore(l):
    if len(np.unique(l)) == 1:
        return np.full(len(l),0.)
    return (np.array(l)  - np.mean(l)) / np.std(l)
    
def disterr(x1,y1, x2, y2):        
    sd = np.array([x1[0]-x2[0],y1[0]-y2[0]])
    ed = np.array([x1[0]-x2[-1],y1[0]-y2[-1]])
    if np.linalg.norm(sd) > np.linalg.norm(ed):
        x2 = np.flip(x2, axis=0)
        y2 = np.flip(y2, axis=0)
        
    offs = np.linspace(0,1,10)
    xrs1, yrs1 = Traj((x1,y1)).getPoints(offs)
    xrs2, yrs2 = Traj((x2,y2)).getPoints(offs)
    return np.sum(np.linalg.norm([xrs1-xrs2, yrs1-yrs2],axis=0))

def rdp(points, epsilon):
    dmax = 0.0
    index = 0
    for i in range(1, len(points) - 1):
        d = point_line_distance(points[i], points[0], points[-1])
        if d > dmax:
            index = i
            dmax = d
    if dmax >= epsilon:
        results = rdp(points[:index+1], epsilon)[:-1] + rdp(points[index:], epsilon)
    else:
        results = [points[0], points[-1]]
    return results
    
def distance(a, b): 
    return  np.sqrt((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2)

def point_line_distance(point, start, end):
    if (start == end):
        return distance(point, start)
    else:
        n = abs((end[0] - start[0]) * (start[1] - point[1]) - (start[0] - point[0]) * (end[1] - start[1]))
        d = np.sqrt((end[0] - start[0]) ** 2 + (end[1] - start[1]) ** 2)
        return n / d

class OnlyOnePointError(Exception):
    pass

class SampleSet:
    def __init__(self, ll):
        # ll is list of tuples [x_array,y_array] for every trajectory in sample
        self.trajs = [Traj(l) for l in ll]
        self.xp = None
        self.yp = None
        self.d = None
        self.filtix = None
        self.lenoutix = None
        self.disoutix = None
        self.eps = None

    def getRawAvg(self):
        trajLen = median([len(t.xs) for t in self.trajs])
        offs = np.linspace(0,1,trajLen)
        xm = []
        ym = []
        for t in self.trajs:
            xs, ys = t.getPoints(offs)
            xm.append(xs)
            ym.append(ys)        
        xp, yp = np.median(xm, axis=0), np.median(ym, axis=0)
        #xp, yp = np.mean(xm, axis=0), np.mean(ym, axis=0)
        return xp, yp

    def endpoints(self):
        cs = np.array([[self.trajs[0].xs[0],self.trajs[0].xs[-1]], [self.trajs[0].ys[0],self.trajs[0].ys[-1]]])
        xs = np.hstack([t.xs[0] for t in self.trajs] + [t.xs[-1] for t in self.trajs])
        ys = np.hstack([t.ys[0] for t in self.trajs] + [t.ys[-1] for t in self.trajs])       
        clabs = []
        oldclabs = []
        for j in range(10):
            for i in range(len(xs)):
                ap = np.array([[xs[i]],[ys[i]]])
                dists = np.linalg.norm(ap - cs, axis=0)
                clabs.append(np.argmin(dists))
            #cx = np.array([np.mean(xs[np.where(np.array(clabs)==0)]), np.mean(xs[np.where(np.array(clabs)==1)])])
            #cy = np.array([np.mean(ys[np.where(np.array(clabs)==0)]), np.mean(ys[np.where(np.array(clabs)==1)])])
            if oldclabs == clabs: 
                break
            oldclabs = clabs
            clabs = []
        for i,l in enumerate(clabs[:len(clabs)//2]):
            if l == 1:
                oldT = self.trajs[i]                
                reversedTraj = (np.flip(oldT.xs, axis=0), np.flip(oldT.ys, axis=0))
                self.trajs[i] = Traj(reversedTraj)   

    def zlen(self):
        ls = np.array([t.cuts[-1] for t in self.trajs])
        return zscore(ls)
        
    def getFiltered(self, dismax, lenlim):
        xa, ya = self.getRawAvg()
        d = zscore(np.array([disterr(t.xs, t.ys, xa, ya) for t in self.trajs]))
        l = self.zlen()
        self.lenoutix = np.where((l<lenlim[0])|(l>lenlim[1]))[0]
        lenix = np.where((l>lenlim[0])&(l<lenlim[1]))[0]
        self.disoutix = np.where(d>dismax)[0]
        disix = np.where(d<dismax)[0]
        self.d = d
        self.l = l
        self.filtix = np.intersect1d(lenix,disix)

    def getAvg(self, dismax, lenlim, eps, stat='Med.'):  # median
        self.eps = eps
        self.endpoints()        
        self.getFiltered(dismax, lenlim)
        atleast = 4
        if len(self.filtix) <= atleast:            
            distrank = np.argsort(self.d)
            self.disoutix = distrank[atleast:]
            self.lenoutix = []
            self.filtix = distrank[:atleast]
        filtered = [self.trajs[i] for i in self.filtix]
        trajLen = median([len(t.xs) for t in filtered])
        offs = np.linspace(0,1,trajLen*10)
        xm = []
        ym = []
        for t in filtered:
            xs, ys = t.getPoints(offs)            
            xm.append(xs)
            ym.append(ys)
        if stat == "Med.":
            self.xp, self.yp = zip(*rdp(list(zip(np.median(xm, axis=0),np.median(ym, axis=0))), eps))
        elif stat == "Avg.":
            self.xp, self.yp = zip(*rdp(list(zip(np.mean(xm, axis=0),np.mean(ym, axis=0))), eps))
        #self.xp, self.yp = np.mean(xm, axis=0), np.mean(ym, axis=0)
        xp, yp = self.xp,self.yp
        return xp, yp
 
    def pax(self, ax):
        ax.set_xlim(0,2.5)
        ax.set_xticks([])
        ax.set_yticks([])        
        ax.set_ylim(0,130)
        for _, t in enumerate(self.trajs):    
            ax.plot(t.xs,t.ys, c="b", marker="o", markersize=2)
        for n, t in enumerate([self.trajs[i] for i in self.disoutix]):    
            ax.plot(t.xs,t.ys, c="g")
        for n, t in enumerate([self.trajs[i] for i in self.lenoutix]):    
            ax.plot(t.xs,t.ys, c="cyan")
        for n, t in enumerate([self.trajs[i] for i in np.intersect1d(self.lenoutix,self.disoutix)]):    
            ax.plot(t.xs,t.ys, c="magenta")
        if self.xp is not None:
            ax.plot(self.xp,self.yp, marker='D', color='r', linewidth=3)                

class Traj:
    def __init__(self,xsys):
        xs, ys = xsys
        a = np.array(xsys).T
        _, filtered = np.unique(a, return_index=True,axis=0)
        if len(filtered) < 2:
            raise OnlyOnePointError()
        self.xs = np.array(xs)[sorted(filtered)]
        self.ys = np.array(ys)[sorted(filtered)]
        self.xd = np.diff(xs)
        self.yd = np.diff(ys)
        self.dists = np.linalg.norm([self.xd, self.yd],axis=0)
        self.cuts = np.cumsum(self.dists)
        self.d = np.hstack([0,self.cuts])
        
    def getPoints(self, offsets):        
        offdists = offsets * self.cuts[-1]
        ix = np.searchsorted(self.cuts, offdists)        
        offdists -= self.d[ix]
        segoffs = offdists/self.dists[ix]
        x = self.xs[ix] + self.xd[ix]*segoffs
        y = self.ys[ix] + self.yd[ix]*segoffs
        return x,y     

def compute_median_trajectory(posdataRight, timedataRight, stat='Med.'):
    # eps, zmax, lenlim used in outlier detection. Here they are set so they don't exclude any outlier in the median computation. Outlying runs will be//are removed beforehand.
    eps = 0.001
    zmax = np.inf
    lenlim=(-np.inf, np.inf)
    data = list(zip([t - t[0] for t in timedataRight], posdataRight))

    ss = SampleSet(data)
    ss.getAvg(zmax, lenlim, eps, stat) # not supposed to do anything but has to be here to work ??????? Therefore, no touchy. 
    X, Y = ss.getAvg(zmax, lenlim, eps, stat)

    # Here median computation warps time (~Dynamic Time Warping) so interpolate to get back to 0.04s increments.
    interpTime = np.linspace(X[0], X[-1], int(X[-1]/0.04)+1) # create time from 0 to median arrival time, evenly spaced 0.04s
    interpPos = np.interp(interpTime, X, Y) # interpolate the position at interpTime
    return interpTime, interpPos

