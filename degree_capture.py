import numpy as np
import math
from video_processing import frame_capture
import os
import time


def degree_pixel(image_shape, width=55/360, height=113/180):
    # return height, width, dh, dw in pixel
    return math.ceil(image_shape[0]/2 * height), math.ceil(image_shape[1] * width), \
           math.ceil(1/180 * image_shape[0]/2), math.ceil(image_shape[1] * 5/360)

def one_eye_box_cropping(arr, height, width, dh, dw, big_dict, frame_count, i, right=False):
    arr_shape = [arr.shape[1], arr.shape[2]]
    for a in range(0, arr_shape[0], dh):
        for b in range(0, arr_shape[1], dw):
            if right:
                a_, b_ = a + arr_shape[0], b + arr_shape[1]
            else:
                a_, b_ = a, b
            if (a_,b_) in big_dict:  
                ab = big_dict[(a_, b_)]
            else:
                ab = np.zeros(shape=(frame_count, 1))
            current_sum_x, current_sum_y = 0, 0
            if a < height//2 and b < width //2:
                current_sum_x += np.sum(arr[0, a - height // 2 :, b - width // 2 :])
                current_sum_y += np.sum(arr[1, a - height // 2 :, b - width // 2 :])
            elif a < height // 2 and b > arr_shape[1] - math.ceil(width/2):
                current_sum_x += np.sum(arr[0, a - height // 2 :, : b + math.ceil(width/2) - arr_shape[1]])
                current_sum_y += np.sum(arr[1, a - height // 2 :, : b + math.ceil(width/2) - arr_shape[1]])
            elif a > arr_shape[0] - math.ceil(height/2) and b < width//2:
                current_sum_x += np.sum(arr[0, :a + math.ceil(height/2) - arr_shape[0], b - width//2:])
                current_sum_y += np.sum(arr[1, :a + math.ceil(height/2) - arr_shape[0], b - width//2:])
            elif a > arr_shape[0] - math.ceil(height/2) and b > arr_shape[1] - math.ceil(width/2):
                current_sum_x += np.sum(arr[0, :a + math.ceil(height/2) - arr_shape[0], : b + math.ceil(width/2) - arr_shape[1]])
                current_sum_y += np.sum(arr[1, :a + math.ceil(height/2) - arr_shape[0], : b + math.ceil(width/2) - arr_shape[1]])
            elif a < height // 2:
                current_sum_x += np.sum(arr[0, a - height // 2 :, b - width // 2 : b + math.ceil(width/2)])
                current_sum_y += np.sum(arr[1, a - height // 2 :, b - width // 2 : b + math.ceil(width/2)])
            elif a > arr_shape[0] - math.ceil(height/2):
                current_sum_x += np.sum(arr[0, :a + math.ceil(height/2) - arr_shape[0], b - width // 2 : b + math.ceil(width/2)])
                current_sum_y += np.sum(arr[1, :a + math.ceil(height/2) - arr_shape[0], b - width // 2 : b + math.ceil(width/2)])
            elif b < width // 2:
                current_sum_x += np.sum(arr[0, a - height //2 : a + math.ceil(height/2), b - width // 2 :])
                current_sum_y += np.sum(arr[1, a - height //2 : a + math.ceil(height/2), b - width // 2 :])
            elif b > arr_shape[1] - math.ceil(width/2):
                current_sum_x += np.sum(arr[0, a - height //2 : a + math.ceil(height/2), : b + math.ceil(width/2) - arr_shape[1]])
                current_sum_y += np.sum(arr[1, a - height //2 : a + math.ceil(height/2), : b + math.ceil(width/2) - arr_shape[1]])
            current_sum_x += np.sum(arr[0, max(a - height //2, 0) : min(a + math.ceil(height/2), arr_shape[0]), max(b - width // 2, 0) : min(arr_shape[1], b + math.ceil(width/2))])
            current_sum_y += np.sum(arr[1, max(a - height //2, 0) : min(a + math.ceil(height/2), arr_shape[0]), max(b - width // 2, 0) : min(arr_shape[1], b + math.ceil(width/2))])
            m1, m2 = current_sum_x/(height * width), current_sum_y / (height * width)
            ab[i][0] = math.sqrt(m1 **2 + m2**2)
            big_dict[(a_, b_)] = ab


def individual_avg(video_name, path, save_path, frame_count):
    if not os.path.exists(save_path + video_name + "/"):
        os.makedirs(save_path + video_name + "/")
    arr = np.load(path + "10" + ".npy")
    arr_shape = [arr.shape[1], arr.shape[2]] 
    height, width, dh, dw = degree_pixel(arr_shape)

    big_dict = {}
    for i in range(1, frame_count): 
        arr = np.load(path + str(i) + ".npy")
        one_eye_box_cropping(arr=arr[:, :arr.shape[1]//2, :], height=height,width=width, dh=dh, dw=dw, big_dict=big_dict,frame_count=frame_count, i=i)
        one_eye_box_cropping(arr=arr[:, -arr.shape[1]//2:, :], height=height,width=width, dh=dh, dw=dw, big_dict=big_dict,frame_count=frame_count, i=i, right=True)
        print(i)
    
    for (a, b) in big_dict:
        arr = big_dict[(a,b)]
        a, b = a / (arr_shape[0] / 2) * 180, b / (arr_shape[1]) * 360
        np.save(save_path + video_name + "/centering_at_row_degree_%.1f"%a + "_col_degree_%.1f"%b, arr)
        print("saved for ", a, b)

    print("done for ", video_name)
 
if __name__ == "__main__":
    video_names = ['sharks', 'ship', 'skyhouse']
    for video_name in video_names:
        video_path = "/usr/project/xtmp/ct214/daml/vr_sickness/pytorch-spynet/original_optical_flow/"+ video_name + "/"
        of_count = len([name for name in os.listdir(video_path) if os.path.isfile(os.path.join(video_path, name))])
        individual_avg(path=video_path, video_name=video_name, save_path="./original_boxed_of_center/", frame_count=of_count) 
