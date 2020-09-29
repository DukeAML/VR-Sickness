# Program To Read video
# and Extract Frames
import cv2
import os

# Function to extract frames
def frame_capture(path, video):
    # Path to video file
    print("Start capturing frames for ", video)
    vidObj = cv2.VideoCapture(path)

    # Used as counter variable
    count = 0
    if not os.path.exists("original-frames/" + video+"_frames"):
        os.mkdir("original-frames/" + video+"_frames")
    else:
        print("did frame capture before")
        return len(os.listdir("original-frames/" + video+"_frames"))
    # checks whether frames were extracted
    success = 1

    while success:
        # vidObj object calls read
        # function extract frames
        success, image = vidObj.read()

        # Saves the frames with frame-count
        cv2.imwrite("original-frames/" + video + "_frames/" + video+"%05d.jpeg" % count, image)

        count += 1
        if count % 100 == 0:
            print("finished capturing frame "+ str(count) + " video " + video)

    return count

if __name__ == '__main__':
    # Calling the function
    for root, dir, files in os.walk("/usr/project/xtmp/ct214/daml/vr_sickness/pytorch-spynet/original_videos/"):
        for file in files:
            path = os.path.join(root, file)
            name = file[:-4]
            print(path, name)
            frame_capture(path, video=name)