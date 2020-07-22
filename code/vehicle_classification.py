from imageai.Detection import ObjectDetection
import os
import cv2
import shutil
#import numpy as np
#import glob
import Helmet_detection_YOLOV3

def directory_path():
    os.chdir(r"C:\ProjectCollege")
    execution_path = os.getcwd()
    return execution_path

def vehicle_classifier(video):
    shutil.rmtree('Images')
    shutil.rmtree(r'C:\ProjectCollege\flaskblog\static\result')
    print("Image folder is removed")
    os.mkdir('Images')
    os.mkdir('Images\Frames')
    path = 'Images\Frames'
    os.mkdir(r"C:\ProjectCollege\Images\Evaluated_Frames")
    os.mkdir('Images\WithHelmet')
    os.mkdir('Images\WithoutHelmet')
    os.mkdir('Images\Result')
    os.mkdir(r'C:\ProjectCollege\flaskblog\static\result')
    execution_path = directory_path()
    vidcap = cv2.VideoCapture(video)
    count = 1
    e = 1
    c = 0
    f = 1
    success = True
    #fps = int(vidcap.get(cv2.CAP_PROP_FPS))
    #print("frames per second:", fps)
    print("Reading the frames ")
    try :
        while (vidcap.isOpened()):
            success,imag = vidcap.read()
            if success == False :
                break
            if count%23.5 == 0:
                c=c+1
                print(c)
                cv2.imwrite(os.path.join(path,'frame%d.jpg'%c),imag)
            count+=1
        print("total no of key frames:", c)
        detector = ObjectDetection()
        detector.setModelTypeAsRetinaNet()
        detector.setModelPath( os.path.join(execution_path , "resnet.h5"))
        detector.loadModel()
        custom = detector.CustomObjects(person = True)
        while c!=0:
            imgfile = cv2.imread(r"C:\ProjectCollege\Images\Frames\frame%d.jpg"%f)
            path = r"C:\ProjectCollege\Images\Frames"
            path1 = r"C:\ProjectCollege\Images\Evaluated_Frames"
            detections = detector.detectCustomObjectsFromImage(custom_objects = custom,extract_detected_objects=True, input_image= os.path.join(path, "frame%d.jpg"%f),output_image_path=os.path.join(path1,"frame%d.jpg"%f))
            print(detections)
            """if detections!=([], []):
                path3 = r"C:\ProjectCollege\Images\Evaluated_Frames\frame%d.jpg-objects"%f
                cv2.imwrite(os.path.join(path3,'frame.jpg'),imgfile)"""
            print("objects in frame%d"%f)
            img_dir = "C:\ProjectCollege\Images\Evaluated_Frames\\frame%d.jpg-objects"%f # Enter Directory of all images 
            print(img_dir)
            Helmet_detection_YOLOV3.a(r'C:\ProjectCollege\Images\Evaluated_Frames\frame%d.jpg-objects/*.jpg'%f,f)
            
            f = f + 1
            c = c - 1
                
    except Exception as e:
            print(str(e)) 
    
