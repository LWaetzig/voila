import logging
import os

import cv2

from src.DetectorAPI import Detector


def face_blurring(input_path : str , threshold : float):
    """_summary_

    Args:
        input_path (str): _description_
        threshold (float): _description_
    """

    input_path = "data/" if input_path == None else input_path

    detector = Detector("face_model/face.pb", name="detection")

    if not os.path.isfile(input_path):
        logging.error(f"invalid input file. Check if there are other file types besides image files in the folder")
    image_name = input_path.split("/")[-1]
    image = cv2.imread(input_path)
    faces = detector.detect_objects(image,threshold)

    if len(faces) == 0 : pass
    
    output_image = f"data/blurred_{image_name}"
    image = blur_snippet(image, faces)
    cv2.imwrite(output_image, image)


def blur_snippet(image, boxes : dict):
    """_summary_

    Args:
        image (_type_): _description_
        boxes (dict): _description_
    """
    for box in boxes:
        # unpack each box
        x1 , y1 = box["x1"] , box["y1"]
        x2 , y2 = box["x2"] , box["y2"]

        # crop the image due to the current box
        sub = image[y1:y2 , x1:x2]

        # apply Blur in cropped area
        blur = cv2.blur(sub, (25,25))

        # paste blurred snippet
        image[y1:y2 , x1:x2] = blur

    return image
