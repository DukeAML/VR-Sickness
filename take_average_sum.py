import numpy as np
import os


target_dir = "/usr/project/xtmp/ct214/daml/vr_sickness/pytorch-spynet/6-numpy-holder/"

for root, dirs, files in os.walk("/usr/project/xtmp/ct214/daml/vr_sickness/pytorch-spynet/6-numpy-holder/"):
    for dir in dirs:
        if dir == "sum_of_norms_without_filter" or dir == "average_of_norms_without_filter":
            continue
        else:
            path = os.path.join("/usr/project/xtmp/ct214/daml/vr_sickness/pytorch-spynet/6-numpy-holder/", dir)
            save_path = "/usr/project/xtmp/ct214/daml/vr_sickness/pytorch-spynet/6-numpy-holder/" \
                        "average_of_norms_without_filter/"
            sum_save_path = "/usr/project/xtmp/ct214/daml/vr_sickness/pytorch-spynet/6-numpy-holder/" \
                        "sum_of_norms_without_filter/"

            video_avg = []
            video_sum = []
            for i in range(5, 1498):
                arr = np.load(path + "/" + str(i) + ".npy")
                temp = [[np.linalg.norm([arr[0,a,j], arr[1,a,j]]) for j in range(arr[0].shape[1])] for a in range(arr[0].shape[0])]
                avg = np.average(temp)
                sum = np.sum(temp)
                video_avg.append(avg)
                video_sum.append(sum)
                if i%100 == 0:
                    print(i)
            np.save(save_path + dir, video_avg)
            np.save(sum_save_path + dir, video_sum)
            print("done saving average for ", dir)
            print("done saving sum for ", dir)


    # for file in files:
    #     path = os.path.join(root, file)
    #     if "sum_of_norms_without_filter" in root:
    #         continue
    #     video = path.split("/")[-2]
    #     arr =


