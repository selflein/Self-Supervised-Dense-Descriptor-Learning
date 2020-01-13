# opens a window with the original and some augmented image with colored points for comparison

import random
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np

import dataset_helper


def comparing(left, points_left, right, points_right, number_points):
    nrows, ncols = len(left), 2
    figsize = [6, 8]

    fig, ax = plt.subplots(nrows=nrows, ncols=ncols, figsize=figsize)
    colors = [(random.randint(0,255), random.randint(0,255), random.randint(0,255)) for i in range(number_points)]

    for i, axi in enumerate(ax.flat):
        if i%2 != 0:
            image = right[int(i/2)]
            points = points_right[int(i/2)]
            for p in range(len(points)):
                x,y = points[p]
                if x is not None and y is not None:
                    cv2.circle(image, points[p], 5, colors[p], 2)
            axi.imshow(image)

        else:
            image = left[int(i/2)]
            points = points_left[int(i/2)]
            for p in range(len(points)):
                cv2.circle(image, points[p], 5, colors[p], 2)
            axi.imshow(image)

    plt.tight_layout(True)
    plt.show()

def create_shown_images(frame_path, number_points):
    img_list = list((Path(frame_path)).glob('*.jpg'))
    left, points_left, right, points_right = [], [], [], []

    orig_image, orig_points = dataset_helper.get_image_from_list(img_list, 7, number_points, True)

    image_1, points_1 = dataset_helper.brightness_change(orig_image, orig_points)
    left.append(orig_image)
    points_left.append(orig_points)
    right.append(image_1)
    points_right.append(points_1)

    image_2, points_2 = dataset_helper.channel_flipping(2, 1, 0, orig_image, orig_points)
    left.append(orig_image)
    points_left.append(orig_points)
    right.append(image_2)
    points_right.append(points_2)

    image_3, points_3 = dataset_helper.crop_out_of_the_other(orig_image, orig_points, True)
    left.append(orig_image)
    points_left.append(orig_points)
    right.append(image_3)
    points_right.append(points_3)

    image_4, points_4, image_5, points_5 = dataset_helper.crop_with_overlap(orig_image, orig_points)
    left.append(image_4)
    points_left.append(points_4)
    right.append(image_5)
    points_right.append(points_5)

    image_6, points_6 = dataset_helper.rotating(90, orig_image, orig_points)
    left.append(orig_image)
    points_left.append(orig_points)
    right.append(image_6)
    points_right.append(points_6)

    return left, points_left, right, points_right, number_points

    
    

if __name__ == "__main__":
    dataset_helper.save_frames('video', 'frames')
    comparing(*create_shown_images('frames',10))
