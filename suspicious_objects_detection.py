import torch
import numpy as np
import cv2 #ok
from time import time
from ultralytics import YOLO

from supervision.draw.color import ColorPalette
from supervision import Detections
from supervision import BoxAnnotator

import os
import cv2
import numpy as np
import time
import datetime
from email.message import EmailMessage #ok
import ssl
import smtplib
from pygame import mixer #ok
from pathlib import Path
import arrow



# Import kivy UX components
from kivy.app import App #ok
from kivy.uix.widget import Widget
from kivy.properties import ObjectProperty
from kivy.uix.floatlayout import FloatLayout
from kivy.uix.label import Label
from kivy.uix.image import Image
from kivy.uix.button import Button
from kivy.uix.textinput import TextInput


# Import other kivy stuff
from kivy.clock import Clock
from kivy.graphics.texture import Texture
from kivy.logger import Logger


    

# sound_path = r"C:\Users\sas\Desktop\camera_yolo\SusCam\mixkit-police-siren-1641.wav"
sound_path = r"C:\Users\User\Desktop\SusCam\mixkit-police-siren-1641.wav"

mixer.init()
mixer.music.load(sound_path)

email_sender = 'kaloanas07@gmail.com'
email_password = 'ufmfgkixttmtekbo' #2nd capture verification 


subject = "New suspicious object was detected"
body = """Please, check this video attachment for suspicious object. 
"""
SECONDS_TO_RECORD_AFTER_DETECTION = 5
fourcc = cv2.VideoWriter_fourcc(*"mp4v")
frame_size = (1280, 720)

class MyLayout(Widget):
    save_dir = ObjectProperty(None)
    mail = ObjectProperty(None)
    ip_address = ObjectProperty(None)
    web_cam = ObjectProperty(None)
    
    
    obj_det = None
    
    def process_data(self):
        self.save_dir = f"{self.save_dir.text}"
        # self.ip_address = int(self.ip_address.text) #this is for webcam
        self.ip_address = self.ip_address.text
        self.mail = self.mail.text
        
        self.obj_det = ObjectDetection(self.mail, self.save_dir)
        
        criticalTime = arrow.now().shift(hours=+5).shift(days=-7) # here specify time to remove

        for item in Path(self.save_dir).glob('*'):
            if item.is_file():
                print (str(item.absolute()))
                itemTime = arrow.get(item.stat().st_mtime)
                if itemTime < criticalTime:
                    os.remove(item)
    
        # Setup video capture device
        self.capture = cv2.VideoCapture(self.ip_address)
        # self.capture = cv2.VideoCapture(self.ip_address)
        assert self.capture.isOpened()
        self.capture.set(3, 1280)
        self.capture.set(4, 720)
        
        Clock.schedule_interval(self.update, 1.0/33.0)

    # Run continuously to get webcam feed
    def update(self, *args):

        # Read frame from opencv
        ret, frame = self.capture.read()
        assert ret
        
        results = self.obj_det.predict(frame)
        frame = self.obj_det.plot_bboxes(results, frame)

        buf = cv2.flip(frame, 0).tostring()
        img_texture = Texture.create(size=(frame.shape[1], frame.shape[0]), colorfmt='bgr')
        img_texture.blit_buffer(buf, colorfmt='bgr', bufferfmt='ubyte')
        self.web_cam.texture = img_texture


class CamApp(App):
    
    def build(self):
        return MyLayout()


class ObjectDetection:
    def __init__(self, user_email, user_save_dir):
        self.user_email = user_email
        self.user_save_dir = user_save_dir
        
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print("Using Device: ", self.device)
        
        self.model = self.load_model()
        
        self.CLASS_NAMES_DICT = self.model.model.names
    
        self.box_annotator = BoxAnnotator(color=ColorPalette.default(), thickness=3, text_thickness=3, text_scale=1.5)
        
        self.detection = False
        self.out = None
        self.timer_started = False
        self.detection_stopped_time = None
        self.new_video_name = None
    

    def load_model(self):
       
        model = YOLO("yolov8m.pt")  # load a pretrained YOLOv8n model
        model.fuse()
    
        return model


    def predict(self, frame):
       
        results = self.model(frame)
        
        return results
    

    def plot_bboxes(self, results, frame):
        
        xyxys = []
        confidences = []
        class_ids = []
        
        # Extract detections for person class
        for result in results[0]:
            class_id = result.boxes.cls.cpu().numpy().astype(int)
            
            if result.boxes.conf.cpu().numpy() > 0.8:
                
                xyxys.append(result.boxes.xyxy.cpu().numpy())
                confidences.append(result.boxes.conf.cpu().numpy())
                class_ids.append(result.boxes.cls.cpu().numpy().astype(int))
            
        
        # Setup detections for visualization
        detections = Detections(
                    xyxy=results[0].boxes.xyxy.cpu().numpy(),
                    confidence=results[0].boxes.conf.cpu().numpy(),
                    class_id=results[0].boxes.cls.cpu().numpy().astype(int),
                    )
        
    
        # Format custom labels
        self.labels = [f"{self.CLASS_NAMES_DICT[class_id]} {confidence:0.2f}"
        for _, confidence, class_id, tracker_id
        in detections]
        
        if confidences:
            if self.detection:
                self.timer_started = False
            else:
                mixer.music.play()
                self.detection = True
                current_time = datetime.datetime.now().strftime("%d-%m-%Y-%H-%M-%S")
                self.new_video_name = f"{current_time}.mp4"
                print("Beginning recording")
                self.out = cv2.VideoWriter(os.path.join(self.user_save_dir, self.new_video_name), fourcc, 20, frame_size)
        elif self.detection:
            if self.timer_started:
                if time.time() - self.detection_stopped_time >= SECONDS_TO_RECORD_AFTER_DETECTION:
                    self.detection = False
                    self.timer_started = False
                    self.out.release()
                    em = EmailMessage()
                    em['From'] = email_sender
                    em['To'] = self.user_email
                    em['subject'] = subject
                    em.set_content(body)
                
                    VIDEO_PATH = f"{self.user_save_dir}\\{self.new_video_name}"
                    with open(VIDEO_PATH, "rb") as f:
                        file_data = f.read()
                        file_name = f.name
                        em.add_attachment(file_data, maintype="application", subtype="mp4", filename=file_name)

                    context = ssl.create_default_context()

                    with smtplib.SMTP_SSL('smtp.gmail.com', 465, context=context) as smtp:
                        smtp.login(email_sender, email_password) 
                        smtp.sendmail(email_sender, self.user_email, em.as_string())
            else:
                self.timer_started = True
                self.detection_stopped_time = time.time()

        if self.detection:
            print("Recording")
            self.out.write(frame)

        
        # Annotate and display frame
        frame = self.box_annotator.annotate(scene=frame, detections=detections, labels=self.labels)
        
        return frame
                  
        
if __name__ == "__main__":
    CamApp().run()