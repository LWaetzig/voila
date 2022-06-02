import os
import cv2
import logging

from src.DetectorAPI import Detector

def detect_faces(input_path : str , threshold : float) -> None:
    """function to detect face in image

    Args:
        input_path (str): path to image to detect face
        threshold (float): value to decide wheater face fits Face Template or not
    """

    # if path to image doesn´t exists print out warning
    if not os.path.isfile(input_path):
        logging.error(f"invalid input file. Check if there are other file types besides images in the directory")

    # declare path to trained face model
    FACE_MODEL_PATH = "face_model/face.pb"

    input_path = "data/" if input_path == None else input_path
    threshold = 0.2 if threshold == None else threshold

    # call Detector Class
    detector = Detector(FACE_MODEL_PATH , name="detection")

    # get image name from input path and read in image as
    image_name = input_path.split("/")[-1]
    image = cv2.imread(input_path)
    
    # use detector class to detect face in image and store return in variable
    faces = detector.detect_objects(image , threshold)

    if len(faces) == 0 : pass

    # set output path
    output_image = f"data/detected_{image_name}"
    # draw rectangle around roi
    draw_rectangle(image,output_image,faces)


def draw_rectangle(image , output_image : str , boxes : dict) -> None:
    """function to draw red rectangle around roi in image

    Args:
        image (array): image with face to blur
        boxes (dict): dict with coordinates where face was detected
        output_image (str): path where image with the detected face should be saved to
    """

    for box in boxes:
        x1 , y1 = box["x1"] , box["y1"]
        x2 , y2 = box["x2"] , box["y2"]

        # draw red rectangle
        rectangle = cv2.rectangle(image , (x1 , y1) , (x2 , y2) , (0,0,255) , 2)

        cv2.imwrite(output_image,rectangle)