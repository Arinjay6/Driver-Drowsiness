





import datetime as dt
import time
from pygame import mixer
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from EAR_calculator import *
from imutils import face_utils 
from imutils.video import VideoStream
import matplotlib.pyplot as plt
import matplotlib.animation as animate
from matplotlib import style 
import imutils 
import dlib
import time 
import argparse 
import cv2 
import os 
import csv
import numpy as np
import pandas as pd
from datetime import datetime
style.use('fivethirtyeight')

# Creating the dataset (stores captured frames from the video streams)
def assure_path_exists(path):
    dir = os.path.dirname(path)
    if not os.path.exists(dir):
        os.makedirs(dir)

# all eye and mouth aspect ratio with time (several lists)
ear_list = []
total_ear = []
mar_list = []
total_mar = []
ts = []
total_ts = []

# Construct the argument parser and parse the arguments 
ap = argparse.ArgumentParser() #creates an instance of ArgumentParser class from argparse module (It handles command line arguments)
ap.add_argument("-p", "--shape_predictor", required=True, help="shape_predictor_68_face_landmarks.dat") # here p is a short form and shape is the long form of the argument 
# p uses command line arguments with a single dash
ap.add_argument("-r", "--picamera", type=int, default=-1, help="whether raspberry pi camera shall be used or not")

args = vars(ap.parse_args()) # parses the command line arguments and tore them in dict args

# Declare a constant which will work as the threshold for EAR value, below which it will be regarded as a blink 
EAR_THRESHOLD = 0.3
# Declare another constant to hold the consecutive number of frames to consider for a blink 
CONSECUTIVE_FRAMES = 20 
# Another constant which will work as a threshold for MAR value
MAR_THRESHOLD = 14

# Initialize two counters 
BLINK_COUNT = 0 
FRAME_COUNT = 0 
count_sleep = 0  # Initialize count of sleepy events

# Now, initialize the dlib's face detector model as 'detector' and the landmark predictor model as 'predictor'
print("[INFO] Loading the predictor...")
detector = dlib.get_frontal_face_detector() 
predictor = dlib.shape_predictor(args["shape_predictor"])  #By combining these parts, dlib.shape_predictor(args["shape_predictor"]) loads the shape predictor model using the path provided as the value for the --shape_predictor command-line argument.

# Grab the indexes of the facial landmarks for the left and right eye respectively 
(lstart, lend) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
(rstart, rend) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]
(mstart, mend) = face_utils.FACIAL_LANDMARKS_IDXS["mouth"]

# Now start the video stream (VideoStram (class) of imutils.video and allow the camera to warm-up
print("[INFO] Loading Camera...")
vs = VideoStream(usePiCamera=args["picamera"] > 0).start()
time.sleep(2) 

assure_path_exists("dataset/")
count_yawn = 0  # Initialize count of yawn events

# Now, loop over all the frames and detect the faces
while True: 
    # Extract a frame 
    frame = vs.read()
    cv2.putText(frame, "PRESS 'q' TO EXIT", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 3) # font, font scale, bgr, thickness
    # Resize the frame 
    frame = imutils.resize(frame, width=500)
    # Convert the frame 
    # to grayscale 
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # Detect faces in gray scale and unsapmling
    rects = detector(frame, 1)

    # Now loop over all the face detections and apply the predictor 
    for (i, rect) in enumerate(rects): 
        #  predict the facial landmarks for the current face region defined by rect and stores them as coordinate (x,y)
        shape = predictor(gray, rect)
        # Convert it to a (68, 2) size numpy array 
        shape = face_utils.shape_to_np(shape)

        # Draw a rectangle over the detected face 
        (x, y, w, h) = face_utils.rect_to_bb(rect) 
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)    
        # Put a number 
        cv2.putText(frame, "Driver", (x - 10, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        leftEye = shape[lstart:lend]
        rightEye = shape[rstart:rend] 
        mouth = shape[mstart:mend]
        # Compute the EAR for both the eyes 
        leftEAR = eye_aspect_ratio(leftEye)
        rightEAR = eye_aspect_ratio(rightEye)

        # Take the average of both the EAR
        EAR = (leftEAR + rightEAR) / 2.0
        # Live data write in csv
        ear_list.append(EAR)
        ts.append(dt.datetime.now().strftime('%H:%M:%S.%f'))

        # Compute the convex hull for both the eyes and then visualize it
        leftEyeHull = cv2.convexHull(leftEye)
        rightEyeHull = cv2.convexHull(rightEye)
        # Draw the contours 
        cv2.drawContours(frame, [leftEyeHull], -1, (0, 255, 0), 1)
        cv2.drawContours(frame, [rightEyeHull], -1, (0, 255, 0), 1)
        cv2.drawContours(frame, [mouth], -1, (0, 255, 0), 1)

        MAR = mouth_aspect_ratio(mouth)
        mar_list.append(MAR / 10)

        # Check if EAR < EAR_THRESHOLD, if so then it indicates that a blink is taking place 
        # Thus, count the number of frames for which the eye remains closed 
        if EAR < EAR_THRESHOLD: 
            FRAME_COUNT += 1

            cv2.drawContours(frame, [leftEyeHull], -1, (0, 0, 255), 1)
            cv2.drawContours(frame, [rightEyeHull], -1, (0, 0, 255), 1)

            if FRAME_COUNT >= CONSECUTIVE_FRAMES and count_sleep == 0: 
                count_sleep += 1
                # Add the frame to the dataset as a proof of drowsy driving
                cv2.imwrite("dataset/frame_sleep%d_%s.jpg" % (count_sleep, dt.datetime.now().strftime('%Y%m%d_%H%M%S')), frame)
        else: 
            if FRAME_COUNT >= CONSECUTIVE_FRAMES:
                mixer.init()  
                mixer.music.load(r"C:\Users\DELL\Downloads\project6semester\sound_files\warning.mp3")
                mixer.music.play()
                while mixer.music.get_busy():  # wait for music to finish playing
                    time.sleep(1)
            FRAME_COUNT = 0

        # Check if the person is yawning
        if MAR > MAR_THRESHOLD:
            count_yawn += 1
            cv2.drawContours(frame, [mouth], -1, (0, 0, 255), 1) 
            # Add the frame to the dataset as a proof of drowsy driving
            cv2.imwrite("dataset/frame_yawn%d_%s.jpg" % (count_yawn, dt.datetime.now().strftime('%Y%m%d_%H%M%S')), frame)
            mixer.init()
            mixer.music.load(r"C:\Users\DELL\Downloads\project6semester\sound_files\alarm.mp3")
            mixer.music.play()
            while mixer.music.get_busy():  # wait for music to finish playing
                time.sleep(1)
            mixer.init()
            mixer.music.load(r"C:\Users\DELL\Downloads\project6semester\sound_files\warning_yawn.mp3")
            mixer.music.play()
            while mixer.music.get_busy():  # wait for music to finish playing
                time.sleep(1)
    
    # Total data collection for plotting
    total_ear.extend(ear_list)
    total_mar.extend(mar_list)
    total_ts.extend(ts)

    # Display the frame 
    cv2.imshow("Output", frame)
    key = cv2.waitKey(1) & 0xFF 

    if key == ord('q'):
        break

# Calculate the duration of driving in minutes
driving_duration = len(total_ts) / (60 * len(rects))

# Determine the driving skill assessment based on the count of sleepy events
driving_skill = ""
if count_sleep > 0:
    driving_skill = "Bad"
elif count_yawn == 0:
    driving_skill = "Excellent"
elif count_yawn <= 3:
    driving_skill = "Good"
elif count_yawn <= 6:
    driving_skill = "Average"
else:
    driving_skill = "Poor"

# Generate the report
report = f"Driving Report\n"
report += f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
report += f"Total Driving Time (minutes): {driving_duration}\n"
report += f"Number of Yawning Events: {count_yawn}\n"
report += f"Number of Sleepy Events: {count_sleep}\n"
report += f"Driving Skill Assessment: {driving_skill}"

# driving_report.csv
# Save the report to a CSV file
report_filename = "driving_report.csv"
with open(report_filename, mode='a', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(["Driving Report"])
    writer.writerow(["Date", datetime.now().strftime('%Y-%m-%d %H:%M:%S')])
    # writer.writerow(["Total Driving Time (minutes)", driving_duration])
    writer.writerow(["Number of Yawning Events", count_yawn])
    # writer.writerow(["Number of Sleepy Events", count_sleep])
    writer.writerow(["Driving Skill Assessment", driving_skill])

print(f"Report saved to {report_filename}")

# Plot the EAR and MAR values over time
plt.plot(total_ts, total_ear, label='EAR')
plt.plot(total_ts, total_mar, label='MAR')
plt.xlabel('Time')
plt.ylabel('Ratio')
plt.title('EAR and MAR Calculation over Time')
plt.legend()
plt.show()

cv2.destroyAllWindows()
vs.stop()
