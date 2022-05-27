import os
import cv2
import logging

from IPython.display import Image
from src.DetectorAPI import Detector

def detect_faces(input_path : str , threshold : float):
    """_summary_

    Args:
        input_path (str): _description_
        threshold (float): _description_
    """
    FACE_MODEL_PATH = "face_model/face.pb"

    input_path = "data/" if input_path == None else input_path
    threshold = 0.2 if threshold == None else threshold



    detector = Detector(FACE_MODEL_PATH , name="detection")

    if not os.path.isfile(input_path):
        logging.error(f"invalid input file. Check if there are other file types besides images in the directory")
    
    image_name = input_path.split("/")[-1]
    image = cv2.imread(input_path)
    faces = detector.detect_objects(image , threshold)

    if len(faces) == 0 : pass

    output_image = f"data/detected_{image_name}"
    draw_rectangle(image,output_image,faces)
    
    Image(filename="data/detected_image.jpg")

def draw_rectangle(image , output_image : str , boxes : dict):
    """_summary_

    Args:
        image (np.array): _description_
        output_image (str): _description_
        faces (dict): _description_
    """

    for box in boxes:
        x1 , y1 = box["x1"] , box["y1"]
        x2 , y2 = box["x2"] , box["y2"]

        # draw red rectangle
        rectangle = cv2.rectangle(image , (x1 , y1) , (x2 , y2) , (0,0,255) , 2)

        cv2.imwrite(output_image,rectangle)