"""
This code is for detecting drowsiness.

How it works?
- Step1: Capture image from webcam
- Step2: Detect Face
- Step3: Detect eye
- Step4: Calculate EAR
- Step5: analyze last 2sec eye state (open or close)
- Step6: if eye close for more than 2sec, raise alarm
- step7: if alarm gets triggered an email is sent to the service provider.
"""

from utils.dlib_face_detector import *
from utils.ear_calculations import eye_aspect_ratio
import playsound
from threading import Thread
from utils.visualization import *
import smtplib

def sound_alarm(path):
    # play an alarm sound
    playsound.playsound(path)


face_detector = dlib.get_frontal_face_detector()
shape_predictor_path = '../models/dlib_shape_predictor/shape_predictor_68_face_landmarks.dat'
shape_predictor = dlib.shape_predictor(shape_predictor_path)

# define two constants, one for the eye aspect ratio to indicate
# blink and then a second constant for the number of consecutive
# frames the eye must be below the threshold for to set off the
# alarm
EYE_AR_THRESH = 0.25
EYE_AR_CONSEC_FRAMES = 5

COUNTER = 0
ALARM_ON = False
l_eye_start, l_eye_end = (36, 42)
r_eye_start, r_eye_end = (42, 48)

cap = cv2.VideoCapture(0)
ears = []

ret, frame = cap.read()

mng = plt.get_current_fig_manager()
mng.full_screen_toggle()
im1 = ax1.imshow(frame)
eye_im = ax3.imshow(np.zeros((5, 20)))
plt.ion()

while True:
    # Capture frame-by-frame
    ret, raw_frame = cap.read()
    frame_h, frame_w = raw_frame.shape[:2]

    frame = raw_frame.copy()

    # Detect face & Get landmarks
    rects = get_face_bounding_boxes_dlib_hog(frame, face_detector)
    rects = convert_json_face_locations_into_dlib_rectangle(rects)

    if len(rects) > 0:
        shape = shape_predictor(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), rects[0])
        shape = shape_to_np(shape)

        # Use Eyes landmarks to calculate EAR
        l_eye = shape[l_eye_start:l_eye_end]
        r_eye = shape[r_eye_start:r_eye_end]
        l_ear = eye_aspect_ratio(l_eye)
        r_ear = eye_aspect_ratio(r_eye)
        ear = (l_ear + r_ear) / 2
        ears.append(ear)

        eye_box_x1 = max(min(shape[18:30][:, 0]), 0)
        eye_box_x2 = min(max(shape[18:30][:, 0]), frame_w)
        eye_box_y1 = max(min(shape[18:30][:, 1]), 0)
        eye_box_y2 = min(max(shape[18:30][:, 1]), frame_h)

        # use last 2sec image output to tell whether person is drowsy or not - raise alarm
        if ear < EYE_AR_THRESH:
            COUNTER += 1

            if COUNTER >= EYE_AR_CONSEC_FRAMES:

                if not ALARM_ON:
                    ALARM_ON = True
                    #send email with message
                    sender_email = "2210316501@gitam.in"
                    receiver_email = "rohithajith123@gmail.com"
                    senders_password = "rohith_official"
                    message = "Driver Drowzy !"
                    server = smtplib.SMTP('smtp.gmail.com', 587)
                    server.starttls()
                    server.login(sender_email, senders_password)
                    server.sendmail(sender_email, receiver_email, message)
                    #trigger alarm/ voice note
                    t = Thread(target=sound_alarm, args=("../utils/alarmSound.mp3",))
                    t.deamon = True
                    t.start()

                cv2.putText(frame, "DROWSINESS ALERT!", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        else:
            COUNTER = 0
            ALARM_ON = False

        # plot output frame & EAR-time chart
        frame = draw_landmarks_on_face(frame, shape)

        eye_image = raw_frame[eye_box_y1:eye_box_y2, eye_box_x1:eye_box_x2]

        if eye_image is not None:
            eye_im.set_data(cv2.cvtColor(eye_image, cv2.COLOR_BGR2RGB))

    # Display the resulting frame
    ears = ears[-20:]
    display_output(frame, im1, ears, ax2, EYE_AR_THRESH)
    plt.pause(0.1)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()
plt.show()
