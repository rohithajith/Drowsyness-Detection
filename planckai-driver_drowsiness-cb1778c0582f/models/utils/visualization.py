import cv2
import numpy as np
from matplotlib import pyplot as plt

l_eye_start, l_eye_end = (36, 42)
r_eye_start, r_eye_end = (42, 48)
EYE_AR_THRESH = 0.25


ax1 = plt.subplot(221)
ax1.axis('off')
ax3 = plt.subplot(222)
ax3.axis('off')
ax2 = plt.subplot(212)
ax2.axhline(y=EYE_AR_THRESH, color='r', linestyle='-')
ax2.yaxis.set_ticks(np.arange(0, 0.6, 0.1))


def draw_landmarks_on_face(frame, shape, landmark_mapping=None):
    for i, (x, y) in enumerate(shape):
        if l_eye_start <= i < r_eye_end:
            cv2.circle(frame, (x, y), 2, (0, 255, 0), -1)
        else:
            cv2.circle(frame, (x, y), 2, (0, 0, 255), -1)
    return frame


def display_output(frame, im1, ears, ax2, EYE_AR_THRESH):
    im1.set_data(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    ax2.clear()
    ax2.axhline(y=EYE_AR_THRESH, color='r', linestyle='-')
    ax2.plot(ears)
    ax2.yaxis.set_ticks(np.arange(0, 0.6, 0.1))
    return
