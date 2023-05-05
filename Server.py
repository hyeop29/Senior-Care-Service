import socket 
import cv2
import numpy
import datetime
import base64
import time
import threading
import collections
import queue
import pymysql  
import boto3
import re
from datetime import datetime
import logging as log

class ServerSocket:

    def __init__(self, socket, lock):
        super().__init__()
        
        self.sock = socket
        self.lock = lock

    def run(self):
        self.receiveThread = threading.Thread(target=self.receiveImages)
        self.receiveThread.daemon = True
        self.receiveThread.start()


    def socketClose(self):
        self.sock.close()


    def receiveImages(self):
        global store_queue #공유변수

        log.info("receive 스레드 시작")
        try:
            while True:
                
                log.info('데이터 기다리는 중')

                length = self.recvall(self.sock, 64)
                length1 = length.decode('utf-8')
                stringData = self.recvall(self.sock, int(length1)) 
                log.info("노인 이미지 받기 완료")
                stime = self.recvall(self.sock, 64)
                log.info('노인 정보 받기 완료')

                now = datetime.now()
                recievetime = now.strftime('%Y-%m-%d %H:%M:%S.%f')
                #print('송신 시각 : ' + str(recievetime))
                person_info = stime.decode('utf-8')
                data = numpy.frombuffer(base64.b64decode(stringData), numpy.uint8)
                decimg = cv2.imdecode(data, 1) 

                #공유변수에 값 저장
                #print('데이터 저장중...')
                self.lock.acquire()
                store_queue.put([person_info, decimg])
                self.lock.release()
                log.info('공유 변수에 데이터 저장 완료')

                # print("===========================")
                # print(store_queue.queue)
                time.sleep(0.01)

        except Exception as e:
            print(e)
            self.socketClose()
            #cv2.destroyAllWindows()
            delete_socket(self) #소켓리스트에서 제거

    def recvall(self, sock, count):
        buf = b''
        while count:
            newbuf = sock.recv(count)
            if not newbuf: return None
            buf += newbuf
            count -= len(newbuf)
        return buf


#클라이언트 소켓 관리
def create_socket(socket, lock):
    global socket_list

    S = ServerSocket(socket, lock)
    socket_list.append(S)
    index = socket_list.index(S)
    socket_list[index].deamon = True
    socket_list[index].run()

def delete_socket(socketClass):
    global socket_list
    socket_list.remove(socketClass)


#DB 전송 스레드
def send(lock, conn, s3):
    log.info('DB 전송 스레드 시작')
    global store_queue
    while True:
        try:
            if(store_queue.qsize()>0):

                log.info('DB 전송 시작')
                lock.acquire()
                temp, decimg =store_queue.get()
                print(temp)
                date_y, date_t, name, birth, emotions = temp.split()

                #RDS DB에 노인의 정보 저장
                cur = conn.cursor()
                str = "INSERT INTO senior (name, birth,emotions,re_date) VALUES('"+name+"','"+birth+"','"+emotions+"','"+date_y+" "+date_t +"')"
                cur.execute(str)
                conn.commit()
                lock.release()
                log.info('DB 전송 완료')

                #로컬에 이미지 일단 저장
                date_y = re.sub("-","",date_y)
                date_t = re.sub(":","",date_t)
                path='E:/rnqhstlr/temp/Image/'+birth+date_y+date_t+'.jpg' 
                cv2.imwrite(path, decimg)
                log.info('로컬 저장소에 이미지 저장 완료')

                #AWS S3에 사진 저장
                filepath = 'E:/rnqhstlr/temp/Image/'+birth+date_y+date_t+'.jpg'  #로컬에서의 파일명
                bucket = 'imagestorge'  #s3의 파일명
                accesss_key = birth+date_y+date_t+'.jpg'  #s3에 저장되는 파일명
                s3.upload_file(filepath, bucket, accesss_key)
                log.info('S3 저장소에 전송 완료')

            time.sleep(0.1)

        except Exception as e:
            print(e)
            conn.close
    
def s3_connection():
    try:
        s3 = boto3.client (
            service_name = 
            region_name =
            aws_access_key_id = 
            aws_secret_access_key = 
        )
    except Exception as e:
        print(e)
    else:
        log.info('S3 연결 success')
        return s3

if __name__ == "__main__":
    
    log.basicConfig(level = 'INFO')
    
    #공유 변수
    socket_list = []
    LOCK=threading.Lock()
    store_queue = queue.Queue()

    #RDS 연결 세팅
    ht = 
    database = 
    port = 
    username = 
    password = 
    conn = pymysql.connect(
        host=ht, user=username, passwd=password, db=database,
        port=port, use_unicode=True, charset='utf8')  
    log.info('RDS DB 연결 success')

    #S3 연결 세팅
    s3 = s3_connection()

    #AWS(RDS,S3) 전송 스레드 시작
    send_trd = threading.Thread(target=send, args=(LOCK,conn, s3)).start()

    #server 세팅
    ip = 
    port = 
    s = socket.socket(socket.AF_INET,socket.SOCK_STREAM) # 소켓 객체를 생성
    s.bind((ip, port))
    s.listen(10) # 연결 수신 대기 상태(리스닝 수(동시 접속) 설정)


    while True:
        log.info('클라이언트 연결 대기')
        conn, addr = s.accept()

        log.info("클라이언트 연결 완료 addr = %s", addr)
        create_socket(conn, LOCK) #접속한 클라이언트 생성


