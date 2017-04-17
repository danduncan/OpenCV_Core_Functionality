# Note: Multithreaded classes before were borrowed from:
# Video file playback:
# http://www.pyimagesearch.com/2017/02/06/faster-video-file-fps-with-cv2-videocapture-and-opencv/
# Webcam:
# http://www.pyimagesearch.com/2015/12/21/increasing-webcam-fps-with-python-and-opencv/
#
# The imutils.video.* library also contains these classes and others
# imutils can be installed with "pip install imutils"

import cv2 as cv
import numpy as np
from imutils.video import FPS
import imutils
from threading import Thread
from Queue import Queue
import time

# Create class for mulithreaded video streaming
class FileVideoStream:
    def __init__(self, path, queueSize=256):
        # initialize the file video stream along with the boolean
        # used to indicate if the thread should be stopped or not
        self.stream = cv.VideoCapture(path)
        self.stopped = False

        # initialize the queue used to store frames read from
        # the video file
        self.Q = Queue(maxsize=queueSize)

    def start(self):
        # start a thread to read frames from the file video stream
        t = Thread(target=self.update, args=())
        t.setDaemon(True) # This causes thread to kill itself automatically
        #t.daemon = True # This causes thread to kill itself automatically
        t.start()
        return self

    def update(self):
        # keep looping infinitely
        while True:
            # if the thread indicator variable is set, stop the
            # thread
            if self.stopped:
                return

            # otherwise, ensure the queue has room in it
            if not self.Q.full():
                # read the next frame from the file
                (grabbed, frame) = self.stream.read()

                # if the `grabbed` boolean is `False`, then we have
                # reached the end of the video file
                if not grabbed:
                    self.stop()
                    return

                # add the frame to the queue
                self.Q.put(frame)

    def read(self):
        # return next frame in the queue
        if self.Q.qsize() > 0:
            return self.Q.get()
        else:
            return False

    def more(self):
        # return True if there are still frames in the queue
        # Or if new frames will be loaded into the queue
        return (self.stopped == False) or (self.Q.qsize() > 0)
        #return self.Q.qsize() > 0

    def stop(self):
        # indicate that the thread should be stopped
        self.stopped = True
        self.stream.release()


# Create class for multithreaded webcam streaming
class WebcamVideoStream:
    def __init__(self, src=0):
        # initialize the video camera stream and read the first frame
        # from the stream
        self.stream = cv.VideoCapture(src)
        (self.grabbed, self.frame) = self.stream.read()

        # initialize the variable used to indicate if the thread should
        # be stopped
        self.stopped = False

        # Initialize thread
        self.t = Thread(target=self.update, args=())
        self.t.setDaemon(True)

    def start(self):
        # start the thread to read frames from the video stream
        self.t.start()
        return self

    def update(self):
        # keep looping infinitely until the thread is stopped
        while True:
            # if the thread indicator variable is set, stop the thread
            if self.stopped:
                return

            # otherwise, read the next frame from the stream
            (self.grabbed, self.frame) = self.stream.read()

    def read(self):
        # return the frame most recently read
        return self.frame

    def stop(self):
        # indicate that the thread should be stopped
        self.stopped = True


# Select modules to run
module0 = False  # Video file playback
module1 = False  # Video file playback w/ FPS tracking
module2 = False  # Multithreaded video file playback
module3 = False  # Webcam playback
module4 = True   # Multithreaded webcam playback

inputFile = "input/dachie.mov"

# Video playback in single-threaded mode
# Slow playback is likely due to decoding bottleneck
if module0 == True:
    # iPhone video resolution is 1920 x 1080
    # 3x downsize = 640 x 360
    cap = cv.VideoCapture(inputFile)

    # Create a window of fixed size
    #cv.namedWindow('frame',cv.WINDOW_NORMAL)
    #cv.resizeWindow('frame',640,360)

    while(cap.isOpened()):
        ret, frame = cap.read()

        #gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

        # Resize frame to fit on screen
        # Original frame was 1920 x 1080 (iPhone 6+)
        # Note that np.shape returns (height,width)
        # But opencv takes arguments as (width,height)
        # For zooming, use interpolation = linear or cubic
        frame_small = cv.resize(frame, (360, 640), interpolation=cv.INTER_AREA)

        cv.imshow('frame',frame_small)
        if cv.waitKey(1) & 0xFF == ord('q'):
            print("Q pressed. Quitting.")
            break

    #print(frame.shape)
    cap.release()
    cv.destroyAllWindows()
    cv.waitKey(1)


# Module 1: Single-Threaded with FPS counting and text added
# Achieves framerate of 7.59 fps
if module1 == True:
    stream = cv.VideoCapture(inputFile)
    fps = FPS().start() # Object that counts the frame rate

    # Loop over frames from video file stream
    while True:
        # Grab next frame
        grabbed, frame = stream.read()

        # if the frame not grabbed, we have reached end of stream
        if not grabbed:
            break

        # resize the frame and convert it to grayscale (while still
        # retaining 3 channels)
        frame = imutils.resize(frame, width=450)
        frame = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
        frame = np.dstack([frame, frame, frame])

        # Display text on frame (for benchmarking vs threaded method)
        cv.putText(frame, "Slow Method", (10, 30),
                    cv.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        # Show the frame and update the FPS counter
        cv.imshow("Frame", frame)
        fps.update()

        if cv.waitKey(1) & 0xFF == ord('q'):
            print("Q pressed. Quitting.")
            break

    fps.stop()
    print("Elapsed time: {:.2f}".format(fps.elapsed()))
    print("Approx. FPS: {:.2f}".format(fps.fps()))

    stream.release()
    cv.destroyAllWindows()
    cv.waitKey(1)


# Module 2: Multi-threaded video streaming
# Improves to > 12 fps if a large enough buffer is used
# Bottleneck for speed is reading and decoding frames
# May get further improvement by having many concurrent threads read frames
if module2 == True:
    fvs = FileVideoStream(inputFile,queueSize=512).start()
    time.sleep(3.0)  # Wait for fvs to initialize
    fps = FPS().start()  # start the FPS timer

    # loop over frames from the video file stream
    while True:

        if fvs.more() == False:
            print("No more frames")
            break

        # grab the frame from the threaded video file stream, resize
        # it, and convert it to grayscale (while still retaining 3
        # channels)
        frame = fvs.read()
        if frame is False:
            #print("Display exceeded frame read speed")
            continue

        # If new frame was loaded, display it
        frame = imutils.resize(frame, width=450)
        frame = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
        frame = np.dstack([frame, frame, frame])

        # display the size of the queue on the frame
        cv.putText(frame, "Queue Size: {}".format(fvs.Q.qsize()),
                    (10, 30), cv.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        # show the frame and update the FPS counter
        cv.imshow("Frame", frame)
        fps.update()
        if cv.waitKey(1) & 0xFF == ord('q'):
            print("Q pressed. Quitting.")
            fvs.stop()
            break

    # Stop the timer and display FPS information
    fps.stop()
    print("Elapsed time: {:.2f}".format(fps.elapsed()))
    print("Approx. FPS: {:.2f}".format(fps.fps()))

    # do a bit of cleanup
    cv.destroyAllWindows()
    cv.waitKey(1)
    #fvs.stop()


# Single-threaded webcam playback
if module3 == True:
    # Argument 0 accesses webcam
    cap = cv.VideoCapture(0)
    fps = FPS().start()  # start the FPS timer

    while(cap.isOpened()):
        ret, frame = cap.read()

        # Check if error occurred reading frame
        if ret == False:
            # Error occurred
            continue

        # Resize frame to fit on screen
        # Webcame frame.shape = (720,1280,3)
        # Note that np.shape returns (height,width)
        # But opencv takes arguments as (width,height)
        # For zooming, use interpolation = linear or cubic
        #frame = cv.resize(frame, (360, 640), interpolation=cv.INTER_AREA)

        cv.imshow('frame',frame)
        fps.update()
        if cv.waitKey(1) & 0xFF == ord('q'):
            print("Q pressed. Quitting.")
            break

    # Stop the timer and display FPS information
    fps.stop()
    print("Elapsed time: {:.2f}".format(fps.elapsed()))
    print("Approx. FPS: {:.2f}".format(fps.fps()))

    #print(frame.shape)
    cap.release()
    cv.destroyAllWindows()
    cv.waitKey(1)


# Multi-threaded webcam playback
# Achieves framerate of 14 fps
if module4 == True:
    wvs = WebcamVideoStream().start()
    fps = FPS().start()

    while True:
        # Grab most recent frame
        frame = wvs.read()

        cv.imshow('frame', frame)
        fps.update()
        if cv.waitKey(1) & 0xFF == ord('q'):
            print("Q pressed. Quitting.")
            break

    # Display FPS information
    fps.stop()
    print("Elapsed time: {:.2f}".format(fps.elapsed()))
    print("Approx. FPS: {:.2f}".format(fps.fps()))

    # do a bit of cleanup
    cv.destroyAllWindows()
    cv.waitKey(1)
    wvs.stop()

# Finish
print("All done!")

