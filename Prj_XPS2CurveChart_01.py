#!/usr/bin/env python
# coding: utf-8

# In[1]:


###############################################
#           H E A D E R   F I L E S
###############################################
import os
import fitz
import cv2
import numpy as np
from scipy import interpolate
import csv
import matplotlib.pyplot as plt
import logging

###############################################
#          F U N C T I O N   L I S T
###############################################
## @brief Description: get specified file in the folder dir
#  @param [in] folder    : folder dir
#  @param [in] extension : file extension
#  
#  @return file paths
#  @date 20220324  danielwu
def GetPaths(folder, extension=['.pdf', '.xps']):
    paths = []

    for response in os.walk(folder):       # response = (dirpath, dirname, filenames)
        if response[2] is None:
            continue
        
        for f in response[2]:
            # only append pdf/xps file in paths 
            if os.path.splitext(f)[1] in extension:  
                paths.append(os.path.join(response[0], f))
    
    return paths

## @brief Description: get image and output the black and white ROI image
#  @param [in] img_path : image path
#  
#  @return output image
#  @date 20230201  danielwu
def GetImage(img_path):
    img = cv2.imread(img_path)
    h, w, _ = img.shape
    
    # the ROI is manual input
    ROI_x1 = int(0.5275*w)
    ROI_y1 = int(0.2057*h)
    ROI_x2 = int(0.8252*w)
    ROI_y2 = int(0.8694*h)
    img_crop = img[ROI_y1:ROI_y2, ROI_x1:ROI_x2]
    
    img_gray = cv2.cvtColor(img_crop, cv2.COLOR_BGR2GRAY) # colorful image to gray image
    img_blur = cv2.GaussianBlur(img_gray, (5, 5), 0)      # gaussian blur # set (5, 5) or (7, 7)
    img_dilate = cv2.dilate(img_blur, cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))) # dilate # set (5, 5)
    img_thresh = cv2.adaptiveThreshold(img_dilate, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2) # binarization
    
    contours, _ = cv2.findContours(img_thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE) # find all contours
    
    if len(contours) > 0:
#         # [METHIOD 1] find the biggest contour (c) by the area
#         c = max(contours, key = cv2.contourArea)
        
        # [METHIOD 2] find the contour which is the closest target ROI
        c = min(contours, key = lambda x: abs(cv2.contourArea(x) - (.2953*w)*(.5891*h)))
        x, y, w, h = cv2.boundingRect(c)
        
    
    _, img_copy = cv2.threshold(img_gray, 200, 255, cv2.THRESH_BINARY) # output white and black image
    img_output = img_copy[y:y+h, x: x+w] # get ROI
    
    # debug: save ROI img after pre-processing (white and black image)
    if img_logger.isEnabledFor(logging.INFO):
        save_img_path = os.path.join(os.path.split(img_path)[0], f'out_{os.path.split(img_path)[1]}')
        cv2.imwrite(save_img_path, img_output)
    
    # debug: show img pre-processing
    if img_logger.isEnabledFor(logging.DEBUG):
        img_cnts = cv2.drawContours(img_crop.copy(), contours, -1, (0,255,0), 3) # draw all contours in green
        cv2.rectangle(img_crop, (x, y), (x+w, y+h), (0,0,255), 3) # draw the biggest/closest contour (cnt) in red
        
        cv2.namedWindow("img_blur", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("img_blur", 375, 971)
        cv2.imshow('img_blur', img_blur)

        cv2.namedWindow("img_dilate", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("img_dilate", 375, 971)
        cv2.imshow('img_dilate', img_dilate)

        cv2.namedWindow("img_thresh", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("img_thresh", 375, 971)
        cv2.imshow('img_thresh', img_thresh)

        cv2.namedWindow("img_cnts", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("img_cnts", 375, 971)
        cv2.imshow('img_cnts', img_cnts)

        cv2.namedWindow("img_crop", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("img_crop", 375, 971)
        cv2.imshow('img_crop', img_crop)

        cv2.namedWindow("img_output", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("img_output", 375, 971)
        cv2.imshow('img_output', img_output)

        cv2.waitKey(0)
        cv2.destroyAllWindows()
    
    return img_output

## @brief Description: find frequence
#  @param [in] page  : current page in the pdf/xps file
#  
#  @return frequence
#  @date 20230220  danielwu
def FindHz(page):
    if page.search_for('Averaged Responses for4 kHz'):
        return '4kHz'
    if page.search_for('Averaged Responses for2 kHz'):
        return '2kHz'
    if page.search_for('Averaged Responses for1 kHz'):
        return '1kHz'
    if page.search_for('Averaged Responses for500 Hz'):
        return '500Hz'

## @brief Description: find vertical unit
#  @param [in] page  : current page in the pdf/xps file
#  
#  @return unit
#  @date 20230220  danielwu
def FindVerticalUnit(page):
    # try and error to get the rect (290, 210, 320, 290)
    # to get vertical axis unit length
    return float(page.get_textbox((290, 210, 320, 290)))

## @brief Description: get image and output the ROI image which is used calculate unit
#  @param [in] img_path : image path
#  
#  @return output image
#  @date 20230220  danielwu
def GetImageCalUnit(img_path):
    img = cv2.imread(img_path)
    h, w, _ = img.shape
    
    # the VERTICAL ROI is manual input
    
    ROI_x1 = int(0.518*w)
    ROI_y1 = int(0.200*h)
    ROI_x2 = int(0.527*w)
    ROI_y2 = int(0.434*h)
    
    img_crop = img[ROI_y1:ROI_y2, ROI_x1:ROI_x2]
    
    img_gray = cv2.cvtColor(img_crop, cv2.COLOR_BGR2GRAY)
    _, img_output = cv2.threshold(img_gray, 200, 255, cv2.THRESH_BINARY)
    
    # debug: save ROI unit img
    if img_logger.isEnabledFor(logging.DEBUG):
        save_path = os.path.join(os.path.split(img_path)[0], f'unit_{os.path.split(img_path)[1]}')
        cv2.imwrite(save_path, img_output)
    
    return img_output

## @brief Description: get horizontal baselines
#  @param [in] img   : image
#  
#  @return baselines
#  @date 20230206  danielwu
def GetBaselines(img):
    pixels = cv2.split(img)
    w = len(pixels[0][0])
    h = len(pixels[0])
    
    # pixels is tuple (size is 1, h, w), pixels2d is np array (size is h, w)
    pixels2d = np.zeros((w, h), dtype = 'int')
    for x in range(w):
        for y in range(h):
            pixels2d[x][y] = pixels[0][y][x]
    
    # record horizontal baseline idx (there usually is 970 and 971 (or 971 and 972))
    # check pixel is black or not
    # check range is:
    # 1. 40% of the height and end at 60% of the height
    # 2. idx:2 to idx:49 of width
    flag = False
    baselines = []
    for j in range(int(h*.4), int(h*.6)):
        for i in range(2, 50):
            if pixels2d[i][j] == 0:
                flag = True
            else:
                flag = False
                break
        if flag == True:
            baselines.append(j)
    
    return baselines

## @brief Description: get vertical segments
#  @param [in]  img      : image
#  @param [in]  segments : vertival segments
#  @param [in]  case     : 'VERTICAL' | 'GROUP_VERTICAL'
#  @param [out] segments : vertival segments dict
#  
#  @return segments
#  @date 20230210  danielwu
def GetSegments(img, segments, case):
    start_idx = None
    h, w = img.shape
    
    if case == 'VERTICAL':
        for i in range(w):
            for j in range(h):
                if img[j][i] == 0:
                    if start_idx is None:
                        start_idx = j
                    elif j == h-1 or img[j+1][i] != 0:
                        segments[(start_idx, j)] = j - start_idx
                        start_idx = None
    elif case == 'GROUP_VERTICAL':
         for i in range(w):
            group = []
            start_idx = None
            for j in range(h):
                if img[j][i] == 0:
                    if start_idx is None:
                        start_idx = j
                    elif j == h-1 or img[j+1][i] != 0:
                        if start_idx != 0 and j != h:
                            group.append([start_idx, j])
                        start_idx = None
                elif img[j][i] != 0 and start_idx != None:
                    if j - start_idx > 1:
                        group.append([start_idx, j])
                    start_idx = None
            segments.append(group)

## @brief Description: get length of vertical unit
#  @param [in] img   : image
#  @param [in] case  : 'VERTICAL' | 'GROUP_VERTICAL'
#  
#  @return longest length
#  @date 20230215  danielwu
def GetLength(img, case):
    segments = {}
    candidate_start_idx = []
    candidate_end_idx = []
    
    # get segments and sort the segments (large length to small length)
    GetSegments(img, segments, case)
    segments = sorted(segments.items(), key=lambda x: x[1], reverse=True)
    
    # get the longest length, and the start and end point of the longest length
    longest_length = segments[0][1]
    start_idx = segments[0][0][0]
    end_idx = segments[0][0][1]
    
    # get the max offset
    candidates_start_idx = [segment for segment in segments[1:] if segment[0][0] == start_idx]
    candidates_end_idx = [segment for segment in segments[1:] if segment[0][1] == end_idx]
    
    offset = max(candidates_start_idx[i][1] for i in range(len(candidates_start_idx))
                 for j in range(len(candidates_end_idx))
                 if candidates_start_idx[i][1] == candidates_start_idx[j][1])
    
    return longest_length - offset

## @brief Description: segments post processing
#  @param [in]  group_segments : vertical segments
#  @param [in]  baselines      : baselines
#  @param [out] group_segments : vertical segments after post processing
#  
#  @return None
#  @date 20230222  danielwu
def SegmentsPostproc(group_segments, baselines):
    new_segments = []
    baselines = [baselines[0], baselines[-1]]
    text_logger.debug('Start to Segments Post Processing')
    text_logger.debug(f'baselines: {baselines}')
    text_logger.debug('=================================')
    
    for i in range(len(group_segments)):
        if i < 2:
            text_logger.debug(f'[{i}]: {group_segments[i]}')
            
            # segment only has one segment
            if len(group_segments[i]) == 1:
                # segment is equal baseline
                if group_segments[i][0][0] == baselines[0] and group_segments[i][0][1] == baselines[1]:
                    if group_segments[i+1][0][0] < baselines[0]:
                        Pother = baselines[1]
                        group_segments[i] = baselines[0]
                    elif group_segments[i+1][0][1] >= baselines[1]:
                        Pother = baselines[0]
                        group_segments[i] = baselines[1]
                else:
                    if group_segments[i+1][0][0] <= group_segments[i][0][0]:
                        if group_segments[i+1][0][1] > group_segments[i][0][1]:
                            Pother = group_segments[i][0][1]
                            group_segments[i] = group_segments[i][0][0]
                        Pother = group_segments[i][0][0]
                        group_segments[i] = group_segments[i][0][1]
                    elif group_segments[i+1][0][1] >= group_segments[i][0][1]:
                        Pother = group_segments[i][0][1]
                        group_segments[i] = group_segments[i][0][0]
            # segments include baselines and another segment
            elif len(group_segments[i]) > 1:
                if group_segments[i][0][1] < baselines[0]:
                    Pother = group_segments[i][0][1]
                    group_segments[i] = group_segments[i][0][0]
                else:
                    Pother = group_segments[i][1][0]
                    group_segments[i] = group_segments[i][1][1]
            # segments is none
            else:
                continue
        else:
            skip = False
            Pprevious = group_segments[i-1]
            text_logger.debug(f'[in][{i-1}]: {Pprevious}')
            text_logger.debug(f'[in][{i}]: {group_segments[i]}')
            text_logger.debug('---------------------------------')
            
            if baselines in group_segments[i]:
                # find baselines index in the group segments
                pointer = group_segments[i].index(baselines)
                # move the baselines to the back of group segments
                if group_segments[i][-1] != group_segments[i][pointer]:
                    test_flag = 1
                    group_segments[i].remove(baselines)
                    group_segments[i].append(baselines)
            
            for segment in group_segments[i]:
                if skip:
                    continue
                
                Pstart = segment[0]
                Pend = segment[1]
                
                text_logger.debug(f'  [Pprevious]: {Pprevious}')
                text_logger.debug(f'  [Pother]   : {Pother}')
                text_logger.debug(f'  [Pstart]   : {Pstart}')
                text_logger.debug(f'  [Pend]     : {Pend}')
                                
                if Pstart <= Pprevious and Pprevious <= Pend:
                    if Pstart < baselines[0]:
                        group_segments[i] = Pstart
                        Pother = Pend
                    elif Pstart == baselines[0]:
                        group_segments[i] = Pend
                        Pother = Pstart
                    elif Pstart > baselines[1]:
                        group_segments[i] = Pend
                        Pother = Pstart
                    skip = True
                elif Pstart <= Pother and Pother <= Pend:
                    if Pstart > baselines[1] and Pend > baselines[1]:
                        group_segments[i] = Pend
                        Pother = Pstart
                    elif Pstart < baselines[0] and Pend > baselines[1]:
                        group_segments[i] = Pstart
                        Pother = Pend
                    elif Pstart == baselines[0] and Pend > baselines[1]:
                        if Pstart <= Pprevious:
                            group_segments[i] = Pstart
                            Pother = Pend
                        else:
                            group_segments[i] = Pend
                            Pother = Pstart
                    elif Pstart < baselines[0] and Pend < baselines[1]:
                        group_segments[i] = Pstart
                        Pother = Pend
                    elif Pstart < baselines[0] and Pend == baselines[1]:
                        if Pstart < Pprevious:
                            group_segments[i] = Pend
                            Pother = Pstart
                        else:
                            group_segments[i] = Pstart
                            Pother = Pend
                    elif Pstart == baselines[0] and Pend == baselines[1]:
                        if Pstart < Pprevious:
                            group_segments[i] = Pstart
                            Pother = Pend
                        else:
                            group_segments[i] = Pend
                            Pother = Pstart
                    skip = True
                elif Pprevious < Pstart and Pstart < Pother:
#                     # avoid baseline
#                     if Pstart - Pprevious > 4 or Pother - Pstart > 4:
#                         continue
                    group_segments[i] = Pstart
                    Pother = Pend
                    skip = True
                elif Pother < Pstart and Pstart < Pprevious:
#                     # avoid baseline
#                     if Pstart - Pother > 4 or Pprevious - Pstart > 4:
#                         continue
                    group_segments[i] = Pend
                    Pother = Pstart
                    skip = True
                else:
                    text_logger.debug('- - - - - - - - - - - - - - - - -')
                    continue
                text_logger.debug(f'[{i}]: {group_segments[i]}')
                text_logger.debug(f'[Pother]: {Pother}')
                text_logger.debug('- - - - - - - - - - - - - - - - -')
                
        text_logger.debug(f'[out][{i}]: {group_segments[i]} ({Pother})')
        text_logger.debug('= = = = = = = = = = = = = = = = =')
        
    # remove empty points
    while [] in group_segments:
        group_segments.remove([])

## @brief Description: save valid image
#  @param [in] img       : image
#  @param [in] save_path : image save path
#  @param [in] pixels    : black pixel coordinates
#  
#  @return None
#  @date 20230217  danielwu
def ValImage(img, save_path, pixels):
    img_gray2colorful = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    mask = np.full(img_gray2colorful.shape, 128).astype(np.uint8)
    color = (0, 0, 255) # BGR
#     thickness = 1 # METHOD 1: draw line
#     cnts = []     # METHOD 1: draw line

    for x, y in enumerate(pixels):
        cnt = (x+1, y)
        cv2.rectangle(mask, cnt, cnt, color, -1)                   # METHOD 2: draw dot
#         cnts.append(cnt)                                           # METHOD 1: draw line

#     cnt_arr = np.array(cnts, np.int32)                             # METHOD 1: draw line
#     mask = cv2.polylines(mask, [cnt_arr], False, color, thickness) # METHOD 1: draw line

    img_val = cv2.addWeighted(img_gray2colorful, 1., mask, .5, 0)
    cv2.imwrite(save_path, img_val)
    
    # debug: show mask and img_val
    if img_logger.isEnabledFor(logging.DEBUG):
        cv2.namedWindow("mask", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("mask", 375, 971)
        cv2.imshow('mask', mask)

        cv2.namedWindow("img_val", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("img_val", 375, 971)
        cv2.imshow('img_val', img_val)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        
## @brief Description: pixel coordinates to real value
#  @param [in] segments : black pixel coordinates
#  @param [in] scaleX   : scale of x axis
#  @param [in] scaleY   : scale of y axis
#  @param [in] baseline : baseline (y coordinate)
#  
#  @return x: x is integer which range is -200 to 600, y: y corresponding to the integer x
#  @date 20230224  danielwu
def Relative2RealValue(segments, scaleX, scaleY, baseline):
    xpoints = []
    ypoints = []
    
    # relative value to real value
    for i, value in enumerate(segments):
        xpoints.append(scaleX * i - 200)            # x = scaleX * i - 200
        ypoints.append(scaleY * (baseline - value)) # y = scaleY * (baseline - value[0])
    
    # use interpolate function to cal the y corresponding to the integer x
    f = interpolate.interp1d(xpoints, ypoints) # define interpolate function
    x = list(range(round(xpoints[0]), round(xpoints[-1]))) # set x coordinate (it maybe is -200:600)
    y = f(x) # calculate interpolate y value
    
    return x, y

## @brief Description: save real value and output the csv file
#  @param [in] file_path : pdf/xps file path
#  
#  @return None
#  @date 20230224  danielwu
def File2CSV(file_path):
    dirname, basename = os.path.split(file_path)
    prefix_basename = basename.split('.')[0]
    save_dir_main = os.path.join(dirname, prefix_basename)
    
    doc = fitz.open(file_path)
    
    for page in doc:
        pix = page.get_pixmap(dpi=300)
        freq = FindHz(page)
        text_logger.debug(f'####### {basename} {freq} #######')
        
        save_dir_hz = os.path.join(save_dir_main, freq)
        if not os.path.isdir(save_dir_hz):
            os.makedirs(save_dir_hz)
        
        # debug: save the image whitch converted by pdf/xps page
        if img_logger.isEnabledFor(logging.INFO):
            pix.save(os.path.join(save_dir_hz, f'{prefix_basename}_{freq}.png'))
        
        # save the ROI image
        img_path = os.path.join(save_dir_hz, f'{prefix_basename}_{freq}.png')
        img_output = GetImage(img_path) # get ROI image and save it
        
        # get the baselines and pixels
        baselines = GetBaselines(img_output)  # get baselines
        baseline = np.mean(baselines) # calculate mean baseline
        group_segments = []
        GetSegments(img_output, group_segments, 'GROUP_VERTICAL') # get the pixels
        SegmentsPostproc(group_segments, baselines) # pixels post processing
        
        # get the vertical and horizontal axis unit
        factor = FindVerticalUnit(page)
        img_caluni = GetImageCalUnit(img_path)
        scaleY_length = GetLength(img_caluni, 'VERTICAL') + 1
        scaleY = factor / scaleY_length
        # length = 600-(-200), number of spacing = group_segments"-1",
        # "-1" is two points have only one spacing
        scaleX = 800 / (len(group_segments)-1)
        
        # debug: save the val image
        if img_logger.isEnabledFor(logging.DEBUG):
            save_path = os.path.join(save_dir_hz, f'val_{prefix_basename}_{freq}.png')
            ValImage(img_output, save_path, group_segments)
        
        # get the x coordinates and y coordinates, which x is integer -200 to 600
        x_coors, y_coors = Relative2RealValue(group_segments, scaleX, scaleY, baseline)
        
        # save pixels coors to csv
        save_path = os.path.join(save_dir_hz, f'out_{prefix_basename}_{freq}.csv')
        with open(save_path, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile) # make csv file writer
            writer.writerows(zip(x_coors, y_coors)) # write table in the csv file
        
        # save curve line draw by x coors and y coors
        save_path = os.path.join(save_dir_hz, f'plot_{prefix_basename}_{freq}.png')
        PlotCurve(x_coors, y_coors, freq, save_path)
    
    doc.close()
    os.rename(file_path, os.path.join(save_dir_main, basename))

## @brief Description: plot the curve line and save it
#  @param [in] x : the x coordinates
#  @param [in] y : the y coordinates
#  @param [in] save_path : save path of plot image
#  
#  @return None
#  @date 20230224  danielwu
def PlotCurve(x, y, freq, save_path):
    font1 = {'family':'serif', 'color':'blue', 'size':16}
    font2 = {'family':'serif', 'color':'black', 'size':12}
    
    plt.title(f'Average Responses for {freq}', **font1)
    plt.xlabel('ms', loc='right', **font2)
    plt.ylabel('\u03BCV', loc='top', **font2)
    plt.xticks(np.linspace(-200,600,9), **font2)
    plt.yticks(np.linspace(-10,10,9), ['', '', '-5.0', '-2.5', '65dB\nHL', '2.5', '5.0', '', ''], **font2)
    plt.xlim([-201, 601])
    # plt.ylim([-15, 15])
    plt.grid(axis='y', color='gray', linewidth='.2')
    
    plt.plot(x, y, color='red', linewidth='1.')
    plt.savefig(save_path, dpi=300)
    
#     plt.show()
    plt.close()


# In[2]:


###############################################
#             D A T A   T Y P E S
###############################################

###############################################
#              C O N S T A N T S
###############################################

###############################################
#        G L O B A L   V A R I A B L E
###############################################
# set root path
ROOT = 'C:\\Users\\danielwu\\Desktop\\ee\\xps_ALL\\xps_NH'


# In[3]:


###############################################
#                   M A I N
###############################################
if __name__ == '__main__':
    # set logging config
    # Level: NOTSET, DEBUG, INFO, WARNING, ERROR, CRITICAL
    logging_file_path = os.path.join(ROOT, 'logging.log')
    logging.basicConfig(level=logging.INFO, filename=logging_file_path, filemode='w')
    
    img_logger = logging.getLogger('img')
    img_logger.setLevel(logging.INFO)
    
    text_logger = logging.getLogger('text')
    text_logger.setLevel(logging.DEBUG)
    
    # get pdf/xps files
    files = GetPaths(ROOT)
    
    # transform pdf/xps file to csv file
    for file in files:
        try:
            File2CSV(file)
        except:
            print(f'{file} has some error, please remove it, or modify parameter of GaussianBlur!')
            text_logger.error(f'{file} has some error!')
    
    # close logging
    logging.shutdown()
    
    print('done')

