# containes all methods needed for saving frames from videos, using them and for augmentation
# used by siamese_dataset

import itertools
import math
import os
import random
from functools import partial
from pathlib import Path

import cv2
import h5py
import matplotlib.pyplot as plt
import numpy as np
from scipy import ndimage


def save_frames(video_paths, result_path):
    count = 0
    for video_path in video_paths:
        vidcap = cv2.VideoCapture(Path(video_path).as_posix())
        success,image = vidcap.read()
        while success:
            print(count)
            cv2.imwrite(f"{result_path}/frame{count}.jpg", image)
            success,image = vidcap.read()
            count += 1

def edge_detection(image, number_points):
    edges = cv2.Canny(image,100,200)
    indices = np.where(edges != [0])
    coordinates = list(zip(indices[1], indices[0])) # x,y

    if len(coordinates) < number_points:
        a = [range(0,256)]*2
        every_possible_point = list(itertools.product(*a))
        coordinates = coordinates + random.sample(every_possible_point, number_points-len(coordinates))

    if len(coordinates) > number_points:
        coordinates = random.sample(coordinates, number_points)

    return coordinates

# TODO enable not squaring for being the original in crop_with_overlap
def get_image_from_list(list_with_paths, idx, number_points=7000, only_points_on_edges=False):
    addr = list_with_paths[idx]
    addr_full = os.path.abspath(addr)

    orig = cv2.imread(addr_full)
    orig = cv2.cvtColor(orig, cv2.COLOR_BGR2RGB)
    orig = make_image_squared(orig)
    orig = cv2.resize(orig, (256, 256), interpolation=cv2.INTER_CUBIC)


    if only_points_on_edges: orig_points = edge_detection(orig, number_points)
    else:
        a = [range(0,256)]*2
        orig_points = list(itertools.product(*a)) 
        orig_points = random.sample(orig_points, number_points)

    return orig, orig_points

def make_image_squared(image):
    new_half_lenght = int(min(len(image[:,0,0]), len(image[0,:,0])) /2)
    center_1, center_2 = int(len(image[:,0,0]) /2), int(len(image[0,:,0]) /2)
    sp_1, sp_2 = center_1-new_half_lenght, center_2-new_half_lenght
    image = image[sp_1:sp_1+new_half_lenght*2, sp_2:sp_2+new_half_lenght*2]

    return image

def create_augmented_image(image, points, list_with_possible_changes):
    picked_changes = random.sample(list_with_possible_changes,random.randint(1,len(list_with_possible_changes)))
    new_img = image.copy()
    new_img_points = points
    for changes in picked_changes:
        new_img, new_img_points = changes(new_img, new_img_points)
    
    return new_img, new_img_points

def channel_flipping(value_0, value_1, value_2, image, point):
    image = image.copy()
    image[:,:,0], image[:,:,1], image[:,:,2] = image[:,:,value_0].copy(),image[:,:,value_1].copy(),image[:,:,value_2].copy()
    return image, point

def brightness_change(image, point):
    image = image.copy()
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    # value = random.randint(-50,50)
    value = random.randint(*random.choice([(-50, -10), (10, 50)]))
    v = hsv[:,:,2]

    v = v.astype(np.int)
    v += value
    v = np.clip(v, 0, 255)
    v = v.astype(np.uint8)

    hsv[:,:,2] = v

    image = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    return image, point

def get_original(image, point):
    return image, point

# rotates counter clockwise
def rotating(degree, image,points):
    image = image.copy()
    dim = image.shape[0]
    image = ndimage.rotate(image, degree, reshape=False)
    points = [_get_point_after_rotation(x,y,dim,degree) for (x,y) in points]
    return image, points

# rotates counter clockwise
def _get_point_after_rotation(x,y, height, degree):
    if degree > 0: degree = 360 - degree
    else: degree = degree*(-1)

    radiant = math.radians(degree)
    x_rel = x-(height/2)
    y_rel = y-(height/2)
    x_new = x_rel*math.cos(radiant) - (y_rel*math.sin(radiant)) + (height/2)
    y_new = (x_rel*math.sin(radiant)) + (y_rel*math.cos(radiant))  + (height/2)

    return (int(x_new),int(y_new))

def crop_with_overlap(image, points, return_points_for_comparison=False):
    image = image.copy()
    length_original = min(len(image[:,0,0]), len(image[0,:,0]))

    image_1, points_1, image_1_start_1, image_1_start_2, image_1_end_1, image_1_end_2 = _crop(image, points, return_points_for_comparison)
    length_1 = min(len(image_1[:,0,0]), len(image_1[0,:,0]))
    
    image_2_start_1 = random.randint(image_1_start_1,image_1_start_1+int(length_1*0.9))
    image_2_start_2 = random.randint(image_1_start_2,image_1_start_2+int(length_1*0.9))
    remaining_length = min(length_original - image_2_start_1, length_original - image_2_start_2)
    lenght_2 = random.randint(min(image_1_end_1-image_2_start_1,image_1_end_2-image_2_start_2), remaining_length)
    image_2 = image[image_2_start_1:image_2_start_1+lenght_2, image_2_start_2:image_2_start_2+lenght_2,:]

    # overlapping_area from (image_2_start_1, image_2_start_2) to (image_1_end_1, image_1_end_2)
    image_2_start_1_in_image_1 = image_2_start_1 -image_1_start_1
    image_2_start_2_in_image_1 = image_2_start_2 -image_1_start_2
    points_1 = _find_cutout(points_1, image_2_start_1_in_image_1, image_2_start_2_in_image_1, image_1_end_1, image_1_end_2)# [(x,y) for (x,y) in points_1 if image_2_start_2 <= x and x <= image_1_end_2 and image_2_start_1 <= y and y <= image_1_end_1]
    points_2 = [(x+image_1_start_2-image_2_start_2,y+image_1_start_1-image_2_start_1) for (x,y) in points_1] # p + s1 - s2

    image_1_len = image_1.shape[0]
    image_2_len = image_2.shape[0]
    image_1 = cv2.resize(image_1, (256, 256), interpolation=cv2.INTER_CUBIC)
    image_2 = cv2.resize(image_2, (256, 256), interpolation=cv2.INTER_CUBIC)
    points_1 = [_scale_point(x,y,image_1_len,256) for (x,y) in points_1]
    points_2 = [_scale_point(x,y,image_2_len,256) for (x,y) in points_2]

    return image_1, points_1, image_2, points_2

def _find_cutout(points_1, image_2_start_1, image_2_start_2, image_1_end_1, image_1_end_2):
    result = []
    for point in points_1:
        x,y = point
        if x >= image_2_start_2 and y >= image_2_start_1:
            if x <= image_1_end_2 and y <= image_1_end_1:
                result.append(point)
    return result

def _crop(image, points, return_points_for_comparison=False):
    image = image.copy()
    length = min(len(image[:,0,0]), len(image[0,:,0]))

    new_length = random.randint(int(length*0.1),length-int(length*0.1))

    start_1 = random.randint(0,length-new_length) # prob. top left
    start_2 = random.randint(0,length-new_length) 

    end_1 = start_1 + new_length # prob. bottom right
    end_2 = start_2 + new_length

    image = image[start_1:end_1, start_2:end_2,:]

    if return_points_for_comparison:
        points_new = [(x-start_2,y-start_1) if start_2 <= x <= end_2 and start_1 <= y <= end_1 else (None,None) for (x,y) in points]
    else:
        points_new = [(x-start_2,y-start_1) for (x,y) in points if start_2 <= x <= end_2 and start_1 <= y <= end_1]

    return image, points_new, start_1, start_2, end_1, end_2

def crop_out_of_the_other(image, points, return_points_for_comparison=False):
    image = image.copy()
    image, points_new, start_1, start_2, end_1, end_2 = _crop(image, points, return_points_for_comparison)
    oldimg = image.copy()
    old_length = oldimg.shape[0]
    image = cv2.resize(image, (256, 256), interpolation=cv2.INTER_CUBIC)
 
    points_new = [_scale_point(x,y,old_length,256) if x is not None and y is not None else (x,y) for (x,y) in points_new]
        
    return image, points_new

def _scale_point(old_1, old_2, old_length, new_image_length):
    return int(old_1 * (float(new_image_length)/old_length)), int(old_2 * (float(new_image_length)/old_length))


def test(frame_path):

    img_list = list((Path(frame_path)).glob('*.jpg'))
    img_list.sort()

    # process_pipline = [brightness_change] # , cropping]
    # permuations = list(itertools.permutations([0, 1, 2]))
    # process_pipline.extend(partial(channel_flipping,*permuation) for permuation in permuations)

    # preprocessing for all images
    for i in range(len(img_list)):
        
        orig, orig_points = get_image_from_list(img_list, i, 2, True)

        image, points = rotating(90, orig, orig_points)
        f, axarr = plt.subplots(2)
        axarr[0].imshow(orig)
        axarr[0].scatter(*zip(*orig_points))
        axarr[1].imshow(image)
        axarr[1].scatter(*zip(*points))

        image_180, points_180 = rotating(180, orig, orig_points)
        f, axarr = plt.subplots(2)
        axarr[0].imshow(orig)
        axarr[0].scatter(*zip(*orig_points))
        axarr[1].imshow(image_180)
        axarr[1].scatter(*zip(*points_180))

        image_180, points_180 = rotating(-90, orig, orig_points)
        f, axarr = plt.subplots(2)
        axarr[0].imshow(orig)
        axarr[0].scatter(*zip(*orig_points))
        axarr[1].imshow(image_180)
        axarr[1].scatter(*zip(*points_180))

        # im_1, p_1, im_2, p_2 = crop_with_overlap(orig, orig_points)
        # f, axarr = plt.subplots(3)
        # axarr[0].imshow(orig)
        # axarr[0].scatter(*zip(*orig_points))
        # axarr[1].imshow(im_1)
        # print(p_1)
        # print(p_2)
        # if p_1 != []:
        #     print('not empty')
        #     axarr[1].scatter(*zip(*p_1))
        # axarr[2].imshow(im_2)
        # if p_2 != []:
        #     print('not empty')
        #     axarr[2].scatter(*zip(*p_2))
        # axarr[3].imshow(im_1)
        # if p_2 != []:
        #     print('not empty')
        #     axarr[3].scatter(*zip(*points_uncut))
        plt.show()


if __name__ == "__main__":

    # test('frames')

    from argparse import ArgumentParser
    parser = ArgumentParser(
        description="Convert a directory of videos to frames.")
    parser.add_argument('--videos')
    parser.add_argument('--output')
    args = parser.parse_args()
    
    save_frames(Path(args.videos).glob('*.*'), args.output)
