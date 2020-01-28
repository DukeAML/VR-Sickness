import numpy as np
import math
from video_processing import frame_capture
import os


def degree_pixel(image_shape, width=55/360, height=113/360):
    return math.ceil(image_shape[0]/2 * height), math.ceil(image_shape[1] * width), \
           math.ceil(55/360 * image_shape[0]/2), math.ceil(image_shape[1] * 25/360)

def main():
    print("main started")
    for root, dirs, files in os.walk("/usr/project/xtmp/ct214/daml/vr_sickness/pytorch-spynet/6-numpy-holder/"):
        for dir in dirs:
            if dir == "sum_of_norms_without_filter" or dir == "average_of_norms_without_filter":
                continue
            else:
                path = os.path.join("/usr/project/xtmp/ct214/daml/vr_sickness/pytorch-spynet/6-numpy-holder/", dir)
                save_path = "/usr/project/xtmp/ct214/daml/vr_sickness/pytorch-spynet/7-boxed-of/"
                if not os.path.exists(save_path + dir + "/"):
                    os.makedirs(save_path + dir + "/")
                arr = np.load(path + "/10" + ".npy")
                arr_shape = [arr.shape[1], arr.shape[2]]
                height, width, dh, dw = degree_pixel(arr_shape)
                print(height, width, dh, dw)
                print(arr_shape)
                for a in range(0, arr_shape[0] - height, dh):
                    for b in range(0, arr_shape[1] - width, dw):
                        ab = np.zeros(shape=(1493, 2, height, width))
                        for i in range(5, 1498):
                            arr = np.load(path + "/" + str(i) + ".npy")
                            ab[i-5] = arr[:, a:a+height, b:b+width]
                            if i%100 == 0:
                                print(i)
                        np.save(save_path + dir + "/starting_at_row_" + str(a) + "_col_" + str(b), ab)
                        print("done saving box for ", str(a),str(b))
                print("done for video")

def individual(video_dir, video_name):
    # count = frame_capture(video_dir, video_name)
    # save_numpy("./"+ video_name + "_frames/", video_name, "./temp/", count)
    # path = "./temp/"
    path = "/usr/project/xtmp/ct214/daml/vr_sickness/pytorch-spynet/6-numpy-holder/gardens/"
    save_path = "/usr/project/xtmp/ct214/daml/vr_sickness/pytorch-spynet/individual_boxed/"

    if not os.path.exists(save_path + video_name + "/"):
        os.makedirs(save_path + video_name + "scaled_down/")
    arr = np.load(path + "10" + ".npy")
    arr_shape = [arr.shape[1], arr.shape[2]]
    height, width, dh, dw = degree_pixel(arr_shape)
    for a in range(0, arr_shape[0] - height, dh):
        for b in range(0, arr_shape[1] - width, dw):
            ab = np.zeros(shape=(1797-5, 2, height, width))
            for i in range(5, 1797-5):
                arr = np.load(path + "/" + str(i) + ".npy")
                ab[i - 5] = arr[:, a:a + height, b:b + width]
                if i % 100 == 0:
                    print(i)
            np.save(save_path + video_name + "/starting_at_row_" + str(a) + "_col_" + str(b), ab)
            print("done saving box for ", str(a), str(b))
    print("done for ", video_name)


def calc_majority_avg(arr):
    count_p, count_n = 0, 0
    sum_p, sum_n = 0, 0
    assert(len(arr.shape) == 2)
    for i in range(arr.shape[0]):
        for j in range(arr.shape[1]):
            if arr[i,j] > 0:
                count_p += 1
                sum_p += arr[i, j]
            else:
                count_n += 1
                sum_n += arr[i, j]
    if count_n > count_p:
        return sum_n / count_n
    else:
        return sum_p / count_p


def main_average():
    print("main average started")
    for root, dirs, files in os.walk("/usr/project/xtmp/ct214/daml/vr_sickness/pytorch-spynet/6-numpy-holder/"):
        for dir in dirs:
            if dir == "sum_of_norms_without_filter" or dir == "average_of_norms_without_filter":
                continue
            else:
                path = os.path.join("/usr/project/xtmp/ct214/daml/vr_sickness/pytorch-spynet/6-numpy-holder/", dir)
                save_path = "/usr/project/xtmp/ct214/daml/vr_sickness/pytorch-spynet/8-boxed-of-average-majority/"
                if not os.path.exists(save_path + dir + "/"):
                    os.makedirs(save_path + dir + "/")
                arr = np.load(path + "/10" + ".npy")
                arr_shape = [arr.shape[1], arr.shape[2]]
                height, width, dh, dw = degree_pixel(arr_shape)
                for a in range(0, arr_shape[0] - height, dh):
                    for b in range(0, arr_shape[1] - width, dw):
                        ab = np.zeros(shape=(1493, 2, 1))
                        for i in range(5, 1498):
                            arr = np.load(path + "/" + str(i) + ".npy")
                            box_1 = arr[0, a:a + height, b:b + width]
                            box_2 = arr[1, a:a + height, b:b + width]
                            ab[i - 5][0] = calc_majority_avg(box_1)
                            ab[i - 5][1] = calc_majority_avg(box_2)
                            if i%100 == 0:
                                print(i)
                        np.save(save_path + dir + "/starting_at_row_" + str(a) + "_col_" + str(b), ab)
                        print("done saving box for ", str(a),str(b))
                print("done for video")

def individual_avg(video_name):
    path = "/usr/project/xtmp/ct214/daml/vr_sickness/pytorch-spynet/individual/gardens/"
    #path = "/usr/project/xtmp/ct214/daml/vr_sickness/pytorch-spynet/6-numpy-holder/gardens/"
    save_path = "/usr/project/xtmp/ct214/daml/vr_sickness/pytorch-spynet/individual_boxed_avg/"
    #
    # if not os.path.exists(save_path + video_name + "scaled_down/"):
    #     os.makedirs(save_path + video_name + "scaled_down/")
    if not os.path.exists(save_path + video_name + "/"):
        os.makedirs(save_path + video_name + "/")
    arr = np.load(path + "10" + ".npy")
    arr_shape = [arr.shape[1], arr.shape[2]]
    height, width, dh, dw = degree_pixel(arr_shape)
    count = 0

    for a in range(0, arr_shape[0] - height, dh):
        for b in range(0, arr_shape[1] - width, dw):
            count += 1

    vid_num = 0
    for a in range(0, arr_shape[0] - height, dh):
        for b in range(0, arr_shape[1] - width, dw):
            # 1502 1797
            ab = np.zeros(shape=(1797-10, 2, 1))
            for i in range(5, 1797-5):
                arr = np.load(path + "/" + str(i) + ".npy")
                box_1 = arr[0, a:a + height, b:b + width]
                box_2 = arr[1, a:a + height, b:b + width]
                ab[i - 5][0] = calc_majority_avg(box_1)
                ab[i - 5][1] = calc_majority_avg(box_2)
            #np.save(save_path + video_name + "scaled_down/starting_at_row_" + str(a) + "_col_" + str(b), ab)
            np.save(save_path + video_name + "/starting_at_row_" + str(a) + "_col_" + str(b), ab)
            print("done saving box for ", str(a), str(b))
            vid_num += 1
            print("done ", vid_num/count)

    print("done for ", video_name)

if __name__ == "__main__":
    print("before main")
#    main_average()
    individual_avg("gardens")
#   individual("/usr/project/xtmp/ct214/daml/vr_sickness/pytorch-spynet/original_videos/gardens.mp4", "gardens")
