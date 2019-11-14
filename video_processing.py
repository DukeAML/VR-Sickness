# Program To Read video
# and Extract Frames
import cv2
import os

# Function to extract frames
def frame_capture(path, video):
    # Path to video file
    vidObj = cv2.VideoCapture(path)

    # Used as counter variable
    count = 0
    if not os.path.exists(video+"_frames"):
        os.mkdir(video+"_frames")
    # checks whether frames were extracted
    success = 1

    while success:
        # vidObj object calls read
        # function extract frames
        success, image = vidObj.read()

        # Saves the frames with frame-count
        cv2.imwrite(video + "_frames/frame%d.jpg" % count, image)

        count += 1
        if count % 100 == 0:
            print("finished frame "+ str(count) + " video " + video)

    return count

# Driver Code
if __name__ == '__main__':
    # Calling the function
    frame_capture("./test.mp4")