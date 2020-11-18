import dlib
import cv2
import numpy as np


def get_face_bounding_boxes_dlib_hog(frame, hog_face_detector):
    rects = hog_face_detector(frame, 1)
    face_locations = []
    for i, d in enumerate(rects):
        x = d.left()
        y = d.top()
        w = d.right() - d.left()
        h = d.bottom() - d.top()
        face_locations.append({'x': x, 'y': y, 'w': w, 'h': h})

    return face_locations


def get_face_bounding_boxes_dlib_cnn(frame, cnn_face_detector):
    rects = cnn_face_detector(frame, 1)
    face_locations = []
    for i, d in enumerate(rects):
        x = d.rect.left()
        y = d.rect.top()
        w = d.rect.right() - d.rect.left()
        h = d.rect.bottom() - d.rect.top()
        face_locations.append({'x': x, 'y': y, 'w': w, 'h': h})

    return face_locations


def add_face_padding(face_location, img_w, img_h):
    ad = 0.5
    x1 = face_location['x']
    y1 = face_location['y']
    x2 = face_location['x'] + face_location['w']
    y2 = face_location['y'] + face_location['h']
    w = x2 - x1
    h = y2 - y1
    xw1 = max(int(x1 - ad * w), 0)
    yw1 = max(int(y1 - ad * h), 0)
    xw2 = min(int(x2 + ad * w), img_w - 1)
    yw2 = min(int(y2 + ad * h), img_h - 1)

    modified_rect = (xw1, yw1, xw2, yw2)

    return modified_rect


def draw_bounding_boxes_dlib(frame, face_locations, box_color=(0, 155, 255), box_thickness=4):
    for i, face_location in enumerate(face_locations):
        x1 = face_location['x']
        y1 = face_location['y']
        x2 = face_location['x'] + face_location['w']
        y2 = face_location['y'] + face_location['h']

        # outer color
        cv2.rectangle(frame,
                      (x1, y1),
                      (x2, y2),
                      box_color,
                      box_thickness)
        # inner color
        cv2.rectangle(frame,
                      (x1, y1),
                      (x2, y2),
                      (255, 255, 255),
                      1)

    return frame


def convert_json_face_locations_into_dlib_rectangle(face_locations):
    rects = []
    for face_location in face_locations:
        x1 = face_location['x']
        y1 = face_location['y']
        x2 = face_location['x'] + face_location['w']
        y2 = face_location['y'] + face_location['h']

        # (top, right, bottom, left)
        rects.append(dlib.rectangle(x1, y1, x2, y2))
    return rects


def shape_to_np(shape, dtype="int"):
    # initialize the list of (x, y)-coordinates
    coords = np.zeros((68, 2), dtype=dtype)
    # loop over the 68 facial landmarks and convert them
    # to a 2-tuple of (x, y)-coordinates
    for i in range(0, 68):
        coords[i] = (shape.part(i).x, shape.part(i).y)
    # return the list of (x, y)-coordinates
    return coords
