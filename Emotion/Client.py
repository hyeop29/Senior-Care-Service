import time
import socket
import cv2
import numpy
import time
import datetime
import base64
import sys
import queue
import threading
import random
import cv2
import numpy as np
from keras.models import load_model
from statistics import mode
from utils.datasets import get_labels
from utils.inference import detect_faces
from utils.inference import draw_text
from utils.inference import draw_bounding_box
from utils.inference import apply_offsets
from utils.inference import load_detection_model
from utils.preprocessor import preprocess_input
from datetime import datetime
import Jetson.GPIO as GPIO
import time
import math
import logging



class ClientSocket:
    def __init__(self, ip, port):


        self.TCP_SERVER_IP = ip
        self.TCP_SERVER_PORT = port
        self.connectCount = 0
        self.connectServer()
        self.detectflag = False
        self.capture = cv2.VideoCapture(0)
        self.capture.set(cv2.CAP_PROP_FRAME_WIDTH, 480)
        self.capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 315)
        self.initArr()

    def initArr(self) :
        self.x = []
        self.angry = []
        self.happy = []
        self.sad = []
        self.neutral = []
       
    def run(self):
        global threads
        self.distancethread=threading.Thread(target=self.distance)
        self.distancethread.daemon=True
        self.distancethread.start()
        threads.append(self.distancethread)

        self.sendImagethread=threading.Thread(target=self.sendImage)
        self.sendImagethread.daemon=True
        self.sendImagethread.start()
        threads.append(self.sendImagethread)
       
        self.detectthread = threading.Thread(target=self.detectemotion)
        self.detectthread.daemon = True
        self.detectthread.start()
        threads.append(self.detectthread)
     
    def distance(self):
        logging.info('run distance thread')
        GPIO.setmode(GPIO.BOARD)
        GPIO.setwarnings(False)

        TRIG = 23
        ECHO = 24
        LED = 11
        GPIO.setup(LED, GPIO.OUT, initial=GPIO.LOW)
        GPIO.setup(TRIG, GPIO.OUT)
        GPIO.setup(ECHO, GPIO.IN)
        start, stop = 0, 0

        GPIO.output(TRIG, False)

        #time.sleep(2)
       
        try:
            while True:
                GPIO.output(TRIG,True)
                #time.sleep(0.00001)
                GPIO.output(TRIG, False)

                while GPIO.input(ECHO) == 0:
                    start = time.time()
                while GPIO.input(ECHO) == 1:
                    stop = time.time()
                check_time = stop - start
               
                lock2.acquire()
                distance = check_time * 34300/2
               

                logging.info('Distance : %f',distance)

                if 0 < distance < 60 :
                    GPIO.output(LED, True)
                    self.detectflag = True
                    time.sleep(0.5)

                else :
                    GPIO.output(LED, False)
                    self.detectflag = False
                lock2.release()
                time.sleep(0.5)
       
        finally :
            GPIO.cleanup()


    def connectServer(self):
        try:
            self.sock = socket.socket()
            self.sock.connect((self.TCP_SERVER_IP, self.TCP_SERVER_PORT))
            logging.info('Client socket is connected with Server socket [ TCP_SERVER_IP: %s, TCP_SERVER_PORT : %s]',self.TCP_SERVER_IP, str(self.TCP_SERVER_PORT))
            self.connectCount = 0
        except Exception as e:
            logging.error('%d', e)
            self.connectCount += 1
            if self.connectCount == 10:
                logging.error('Connect fail %d times. exit program', self.connectCount)
                sys.exit()
            logging.error('%d times try to connect with server',self.connectCount)
            self.connectServer()

    def detectemotion(self):
        logging.info('run detectemotion thread')
        global emotion_model_path, emotion_labels, frame_window, emotion_offsets, face_cascade, emotion_classifier, emotion_target_size ,emotion_text
        global emotion_window, lock2

        while self.capture.isOpened() :
            if self.detectflag :
                logging.info('detect start')
                lock2.acquire()

                ret, bgr_image = self.capture.read()
                gray_image = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2GRAY)
                rgb_image = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2RGB)

                faces = face_cascade.detectMultiScale(gray_image, scaleFactor=1.1, minNeighbors=5,
                        minSize=(30, 30), flags=cv2.CASCADE_SCALE_IMAGE)

                for face_coordinates in faces:
                    start = time.time()
                    logging.info('face detect')
                    x1, x2, y1, y2 = apply_offsets(face_coordinates, emotion_offsets)
                    gray_face = gray_image[y1:y2, x1:x2]
                    try:
                        gray_face = cv2.resize(gray_face, (emotion_target_size))
                    except:
                        continue

                    gray_face = preprocess_input(gray_face, True)
                    gray_face = np.expand_dims(gray_face, 0)
                    gray_face = np.expand_dims(gray_face, -1)
                    emotion_prediction = emotion_classifier.predict(gray_face)
                    emotion_probability = np.max(emotion_prediction)
                    emotion_label_arg = np.argmax(emotion_prediction)
                    emotion_text = emotion_labels[emotion_label_arg]
                    end = time.time()
                    # print("얼굴 인식부터 감정 분류 시간 :")
                    # print(f"{end - start: .5f} sec")
               
                    if emotion_text == 'angry':
                        color = emotion_probability * np.asarray((255, 0, 0))
                    elif emotion_text == 'sad' or 'fear' or 'disgust':
                        emotion_text = 'sad'
                        color = emotion_probability * np.asarray((0, 0, 255))
                    elif emotion_text == 'happy' or 'surprise':
                        emotion_text = 'happy'
                        color = emotion_probability * np.asarray((255, 255, 0))
                    elif emotion_text == 'neutral':
                        color = emotion_probability * np.asarray((0, 255, 255))
                    else:
                        logging.error("invalid emotion : %s", emotion_text)

                    emotion_window.append(emotion_text)

                    if len(emotion_window) > frame_window:
                        emotion_window.pop(0)
                    try:
                        emotion_mode = mode(emotion_window)
                    except:
                        continue


                    color = color.astype(int)
                    color = color.tolist()

                    draw_bounding_box(face_coordinates, rgb_image, color)
                    draw_text(face_coordinates, rgb_image, emotion_mode, color, 0, -45, 1, 1)
               
                    bgr_image = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2BGR)
               
               
                    if emotion_text == 'angry' :
                        self.angry.append(bgr_image)
                    elif emotion_text == 'sad' :
                        self.sad.append(bgr_image)
                    elif emotion_text == 'happy' :
                        self.happy.append(bgr_image)
                    elif emotion_text == 'neutral' :
                        self.neutral.append(bgr_image)
               
                lock2.release()
                time.sleep(0.5)
            else :
                time.sleep(0.5)
   
    def sumArr(self) :
        cnt= len(self.angry) + len(self.sad) + len(self.happy) + len(self.neutral)
        logging.info("cnt : %d", cnt)
        return cnt

    def maxEmotionImg(self) :
        self.x = [len(self.angry), len(self.sad), len(self.happy), len(self.neutral)]
        m = self.x.index(max(self.x))

        if m == 0 :
            emotion_text = 'angry'
            img = self.angry[random.randint(0, self.x[0]-1)]
        elif m == 1 :
            emotion_text = 'sad'
            img = self.sad[random.randint(0, self.x[1]-1)]
        elif m == 2 :
            emotion_text = 'happy'
            img = self.sad[random.randint(0, self.x[2]-1)]
        else :
            emotion_text = 'neutral'
            img = self.neutral[random.randint(0, self.x[3]-1)]
       
        return img



    def sendImage(self):
        global seniorname, birth
        logging.info("run sendImage thread")
        try:
            while(True):
                    if(self.sumArr() >= 10):

                        lock2.acquire()
                        bgr_image = self.maxEmotionImg()
                        resize_frame = cv2.resize(bgr_image, dsize=(480, 315), interpolation=cv2.INTER_AREA)
                        now = datetime.now()
                        #stime = datetime.datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S.%f') + " " + seniorname + " " + str(birth) + " " + emotion_text
                        stime = now.strftime('%Y-%m-%d %H:%M:%S') + " " + seniorname + " " + str(birth) + " " + emotion_text
                        stime = stime.strip()
                        encode_param=[int(cv2.IMWRITE_JPEG_QUALITY),90]
                        result, imgencode = cv2.imencode('.jpg', resize_frame, encode_param)
                        data = numpy.array(imgencode)
                        stringData = base64.b64encode(data)
                        length = str(len(stringData))
                        self.sock.sendall(length.encode('utf-8').ljust(64))
                        self.sock.send(stringData)
                        self.sock.send(stime.encode('utf-8').ljust(64))
                        logging.info("send data | emotion : %s", emotion_text)
                        self.initArr()
                        lock2.release()
                       
                        time.sleep(0.095)
                    else :
                        time.sleep(2)
        except Exception as e: # 예외 발생 시 소켓을 죽였다가 다시 열어서 연결되기를 기다린다.
            logging.error(e)
            self.receiveThread=threading.Thread(target=self.sendImage)
            self.receiveThread.daemon=True
            self.receiveThread.start()
            threads.append(self.receiveThread)
            self.socketClose()
            cv2.destroyAllWindows()






if __name__ == "__main__":
    logging.basicConfig(level= 'INFO')
   
    imagequeue = queue.Queue()
    emotion_model_path = './models/emotion_model.hdf5'
    emotion_labels = get_labels('fer2013')
    frame_window = 10
    emotion_offsets = (20, 40)
    face_cascade = cv2.CascadeClassifier('./models/haarcascade_frontalface_default.xml')
    emotion_classifier = load_model(emotion_model_path)
    emotion_classifier._make_predict_function()
    emotion_target_size = emotion_classifier.input_shape[1:3]
    emotion_text = ""
    emotion_window = []
    threads = []

    distance = 0
    seniorname = 
    birth = 

    # cv2.namedWindow('window_frame')
    # capture = cv2.VideoCapture(0)
    # capture.set(cv2.CAP_PROP_FRAME_WIDTH, 480)
    # capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 315)

    TCP_IP = 
    TCP_PORT = 
    lock2 = threading.Lock()
    client = ClientSocket(TCP_IP, TCP_PORT)
    client.run()

    for t in threads:
        t.join()
    logging.info("Exiting Main Thread")
