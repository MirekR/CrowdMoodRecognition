import image_emotion_gender_demo
import sys
import os, time

import cv2
import time
import numpy as np
from utils.inference import load_image
import matplotlib.pyplot as plt

dir_path = os.path.dirname(os.path.realpath(__file__))
pipe_path = dir_path + "/../term_sig/end"
print(dir_path)
if not os.path.exists(pipe_path):
    os.mkfifo(pipe_path)

cv2.namedWindow('window_frame')
video_capture = cv2.VideoCapture(0)

total_results = {0:[],1:[],2:[],3:[],4:[],5:[],6:[]}
i = 0
maxScale = 2

pipe_fd = os.open(pipe_path, os.O_RDONLY | os.O_NONBLOCK)
with os.fdopen(pipe_fd) as pipe:
	while True:
		print("==========================================")
		bgr_image = video_capture.read()[1]
		# bgr_image = load_image('/Users/mirekrousal/workspace/CrowMoodRecognition/pics/test.jpg', grayscale=False)
		results = image_emotion_gender_demo.generateResults(bgr_image, i)
		print(results)

		if results:
			i = i + 1
			maxScale = len(results) if len(results) > maxScale else maxScale
			x = {}
			for r in results:
				cnt = x.get(r[1], 0) + 1
				x[r[1]] = cnt

			for r in range(0,7):
				total_results.get(r).append(x.get(r,0))
			print(total_results)

		message = pipe.read()
		if message:
			print("Received: '%s'" % message)
			break

		time.sleep(.5)
		

print("We have attempted this many samples " + str(i))
ind = np.arange(i)    # the x locations for the groups
width = 0.35       # the width of the bars: can also be len(x) sequence

p0=plt.bar(ind, total_results[0], width, color='#00FFFF') #Angry
bottom_total=total_results[0];
p1=plt.bar(ind, total_results[1], width,bottom=bottom_total, color='#f4330a') #Disgust
bottom_total=[x + y for x, y in zip(bottom_total, total_results[1])]
p2=plt.bar(ind, total_results[2], width,bottom=bottom_total, color='#c037d8') #Fear
bottom_total=[x + y for x, y in zip(bottom_total, total_results[2])]
p3=plt.bar(ind, total_results[3], width,bottom=bottom_total, color='#54E730') #Happy
bottom_total=[x + y for x, y in zip(bottom_total, total_results[3])]
p4=plt.bar(ind, total_results[4], width,bottom=bottom_total, color='#405cee') #Sad
bottom_total=[x + y for x, y in zip(bottom_total, total_results[4])]
p5=plt.bar(ind, total_results[5], width,bottom=bottom_total, color='#ea84db') #Surprise
bottom_total=[x + y for x, y in zip(bottom_total, total_results[5])]
p6=plt.bar(ind, total_results[6], width,bottom=bottom_total, color='#89897f') #Neutral


plt.ylabel('Count')
plt.title('Time stamp')
plt.xticks(ind)
plt.yticks(np.arange(0, maxScale, 1))
plt.legend((p0[0],p1[0], p2[0],p3[0],p4[0],p5[0],p6[0]), ('Angry', 'Disgust','Fear','Happy','Sad','Surprise','Neutral'))

plt.show()