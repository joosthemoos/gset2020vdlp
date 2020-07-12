import cv2
import socket
from tkinter import *
from tkinter import simpledialog
import tkinter.messagebox as box
from threading import Thread
from PIL import Image, ImageTk
import PIL.ImageGrab
import numpy as np
import copy
import math
from numpy import fromstring, uint8, array
import pyaudio
import pickle, os
from googletrans import Translator
from gtts import gTTS
from playsound import playsound
import speech_recognition as sr
from mss import mss
import webbrowser
from Crypto.Cipher import AES




s = socket.socket()
port = 4096

host = socket.gethostname()

remote_host = socket.gethostbyname(socket.gethostname())

captionLength = 500
pollLength = 500

captionOn = False

if not os.path.isfile('remote_hosts.dat'):
    remote_hosts = [['Host1', '192.168.2.2'],
                    ['Host2', '192.168.2.3'],
                    ['Host3', '192.168.1.167']]
else:
    file = open('remote_hosts.dat', 'rb')
    remote_hosts = pickle.load(file)
    file.close()


class WebcamVideoStream:
    def __init__(self, src):

        self.stream = cv2.VideoCapture(src)
        (self.grabbed, self.frame) = self.stream.read()
        self.stopped = False
        self.screenSharing = False

    def start(self):
        Thread(target=self.update, args=()).start()
        return self

    def update(self):
        while True:
            if self.stopped:
                return
            (self.grabbed, self.frame) = self.stream.read()
            if self.screenSharing == False:
                self.img_str = cv2.imencode('.jpg', self.frame)[1].tobytes()
            else:
                cap = array(PIL.ImageGrab.grab())
                # cap2 = cv2.resize(cap, dsize=(640, 480), interpolation=cv2.INTER_CUBIC)
                cap2 = cv2.resize(cap, dsize=(640, 480), interpolation=cv2.INTER_AREA)
                cap3 = cv2.cvtColor(cap2, cv2.COLOR_BGR2RGBA)

                self.img_str = cv2.imencode('.jpg', cap3)[1].tobytes()

    def get_jpeg(self):
        return self.img_str

    def get_preview(self):
        preview=cv2.flip(preview,1)
        if vs.screenSharing == False:
            preview = cv2.resize(self.frame, (160, 120))
        else:
            cap = array(PIL.ImageGrab.grab())
            # preview = cv2.resize(cap, dsize=(160, 120), interpolation=cv2.INTER_CUBIC)
            cap2 = cv2.resize(cap, dsize=(160, 120), interpolation=cv2.INTER_AREA)
            preview = cv2.cvtColor(cap2, cv2.COLOR_BGR2RGBA)
            

        if Gui.pollDetection == True:
            font=cv2.FONT_HERSHEY_SIMPLEX
            cv2.putText(preview, 'Vote by placing your fingers in the box', (0, 50), font, 2, (0, 0, 255), 3, cv2.LINE_AA)
            cv2.rectangle(preview, (100,25), (150,75), (255, 0, 0), 2)
        cv2preview = cv2.cvtColor(preview, cv2.COLOR_BGR2RGBA)
        imgPrev = Image.fromarray(cv2preview)
        self.tkPrev = ImageTk.PhotoImage(imgPrev)
        return self.tkPrev

    def stop(self):
        self.stopped = True
        return self

vs = WebcamVideoStream(src=0).start()


class SendStream:
    def __init__(self, host, port):
        self.addr = host
        self.port = port
        self.callInit = False
        self.connStatus = False
        self.disconnect = False
        self.audio_buff = b''
        self.stopped = False
        self.caption = ''
        self.captionSecondLan = ''
        self.transText = ''
        self.questionAudioThread = Thread(target=self.questionAudio,args=())
        self.optionOneAudioThread = Thread(target=self.optionOneAudio,args=())
        self.optionTwoAudioThread = Thread(target=self.optionTwoAudio,args=())
        self.optionThreeAudioThread = Thread(target=self.optionThreeAudio,args=())

    def getConn(self):
        self.sock = socket.socket()
        self.sock.settimeout(2.0)
        self.connStatus = True
        err_val = False
        try:
            self.sock.connect((self.addr, self.port))
        except socket.error as exc:
            err_val = True
            self.connStatus = False
            box.showerror('TCP Transceiver - Socket Connect Error', exc)
        return err_val

    def sendData(self):
        global StatusStr
        trans1 = Translator()
        display = ''
        emptyPollBytes = bytes("".ljust(500, ' '), 'utf-8')
        while self.stopped == False:
            if self.connStatus == False:
                if self.callInit:
                    StatusStr = 'TCP Transceiver --- Connecting to ' + remote_host + ' port ' + str(port)
                    send_err = send.getConn()
                    self.callInit = False
                    if send_err: Gui.callBtn['text'] = 'Call'
                    if captionOn == False:
                        sendTextThread.start()
                        globals()['captionOn'] = True
            else:
                if self.disconnect:
                    self.callInit = False
                    self.close()
                    self.disconnect = False
                else:
                    tempCaption = self.caption

                    img_str = vs.get_jpeg()
                    self.audio_buff = sendStream.read(chunk)

                    if tempCaption != '':
                        if Gui.languageType == "Spanish":
                            self.captionSecondLan = trans1.translate(self.caption, scr='en', dest='es').text
                            Gui.addCaption1(self.captionSecondLan)
                            self.captionSecondLan = ''
                        elif Gui.languageType == 'English':
                            Gui.addCaption1(self.caption)

                    indexPoll = tempCaption.find("poll")
                    indexGset = tempCaption.find("create")
                    indexDone = tempCaption.find("done")
                    indexDunn = tempCaption.find("dunn")
                    indexClear = tempCaption.find("clear")

                    if indexPoll != -1 and indexGset != -1:
                        Gui.poll = True
                        Gui.pollState = 1
                    if Gui.poll:
                        if Gui.pollState == 1:
                            self.questionAudioThread.start()
                            Gui.pollState = 2
                        elif Gui.pollState ==2:
                            if indexClear != -1:
                                Gui.question = ''
                            if indexPoll == -1 and indexGset == -1:
                                Gui.question += tempCaption
                            if indexDone != -1 or indexDunn != -1 or len(Gui.question) > 200:
                                self.optionOneAudioThread.start()
                                Gui.question = Gui.question.replace("clear","")
                                Gui.question = Gui.question.replace("dunn","")
                                Gui.question = Gui.question.replace("done","") + "?"
                                Gui.pollState = 3
                        elif Gui.pollState == 3:
                            # if indexDone == -1:
                            if indexClear != -1:
                                Gui.option1 = ""
                            Gui.option1 += tempCaption
                            if indexDone != -1 or indexDunn != -1:
                                self.optionTwoAudioThread.start()
                                Gui.option1 = Gui.option1.replace("clear","")
                                Gui.option1 = Gui.option1.replace("dunn","")
                                Gui.option1 = Gui.option1.replace("done","")
                                Gui.pollState = 4
                        elif Gui.pollState == 4:
                            if indexClear != -1:
                                Gui.option2 = ""
                            Gui.option2 += tempCaption
                            if indexDone != -1 or indexDunn != -1:
                                self.optionThreeAudioThread.start()
                                Gui.option2 = Gui.option2.replace("clear","")
                                Gui.option2 = Gui.option2.replace("dunn","")
                                Gui.option2 = Gui.option2.replace("done","")
                                Gui.pollState = 5
                        elif Gui.pollState == 5:
                            if indexClear != -1:
                                Gui.option3 = ''
                            Gui.option3 += tempCaption
                            if indexDone != -1 or indexDunn != -1:
                                Gui.option3 = Gui.option3.replace("clear","")
                                Gui.option3 = Gui.option3.replace("dunn","")
                                Gui.option3 = Gui.option3.replace("done","")
                                Gui.pollState = 6
                        elif Gui.pollState == 6:
                            font = cv2.FONT_HERSHEY_SIMPLEX
                            display = Gui.question + '\nOption 1 : ' + Gui.option1 + '\nOption 2 : ' + Gui.option2 + '\nOption 3: ' + Gui.option3
                            Gui.pollDisplayCount += 1
                            if Gui.pollDisplayCount > 500:
                                Gui.pollState = 7

                        elif Gui.pollState == 7:
                            Gui.pollState = 0
                            Gui.poll = False
                            Gui.pollDisplayCount = 0
                            Gui.question = ''
                            Gui.option1 = ''
                            Gui.option2 = ''
                            Gui.option3 = ''

                    captionBytes = bytes(tempCaption.ljust(500, ' '), 'utf-8')
                    # secondCaptionBytes = bytes(self.captionSecondLan.ljust(250, ' '), 'utf-8')
                    if Gui.pollState == 6:
                        pollBytes = bytes(display.ljust(500, ' '), 'utf-8')
                    else:
                        pollBytes = emptyPollBytes

                    frame_size = (4 + len(img_str) + len(self.audio_buff) + captionLength + pollLength).to_bytes(4, byteorder='big')

                    send_frame = frame_size + img_str + self.audio_buff + captionBytes + pollBytes
                    send_frame_encryption = Gui.encryptionObject.encrypt(send_frame)

                    if self.caption == tempCaption:
                        self.caption = ''
                        self.captionSecondLan = ''

                    try:
                        self.sock.sendall(send_frame_encryption)

                    except socket.error as exc:
                        # print(exc)
                        # self.close()
                        # box.showinfo('TCP Transceiver',' Call Terminated')
                        # Gui.callBtn['text'] = 'Call'
                        continue

        return
    def questionAudio(self):
        tts = gTTS(text="Please say your question and use the keyword done to finish and the keyword clear to reset question", lang='en')
        tts.save("speech.mp3")
        playsound("speech.mp3")
    def optionOneAudio(self):
        tts = gTTS(text="Please say option one and use the keyword done to finish and the keyword clear to reset option one", lang='en')
        os.remove("speech.mp3")
        tts.save("speech.mp3")
        playsound("speech.mp3")
    def optionTwoAudio(self):
        tts = gTTS(text="Please say option two and use the keyword done to finish and the keyword clear to reset option two", lang='en')
        os.remove("speech.mp3")
        tts.save("speech.mp3")
        playsound("speech.mp3")
    def optionThreeAudio(self):
        tts = gTTS(text="Please say option three and use the keyword done to finish and the keyword clear to reset option three", lang='en')
        os.remove("speech.mp3")
        tts.save("speech.mp3")
        playsound("speech.mp3")

    def sendCaption(self):
        r = sr.Recognizer()
        trans = Translator()
        mic = sr.Microphone()

        while True:
            with mic as source:
                r.adjust_for_ambient_noise(source, duration=0.5)
                audio = r.listen(source)
                try:
                    if Gui.languageType == 'English':
                        self.transText = r.recognize_google(audio).lower()
                        self.caption = self.transText
                    elif Gui.languageType == 'Spanish':
                        real_audio = r.recognize_google(audio, language ="es-ES")
                        string = str(real_audio)
                        self.transText = trans.translate(string, scr='es', dest='en').text
                        self.caption = self.transText
                except:
                    self.transText = ''
                    self.captionSecondLan = self.transText
                    self.caption = self.transText

    def stop(self):
        self.stopped = True
        return

    def close(self):
        self.sock.close()
        self.connStatus = False
        return

send = SendStream(host=remote_host, port=port)
sendThread = Thread(target=send.sendData, args=())
sendTextThread = Thread(target=send.sendCaption, args=())


def handPoll(frame, timeElapsed):
    import cv2
    import numpy as np
    import copy
    import math

    # parameters
    cap_region_x_begin = 0.6  # start point/total width
    cap_region_y_end = 0.7  # start point/total width
    blurValue = 41  # GaussianBlur parameter
    bgSubThreshold = 90 #original 50
    areaHand = 0
    areaHull = 0
    areaRatio = 0
    backgroundSubtractor= None
    isBgCaptured = 0 


    #detects and removes background
    def removeBG(frame):
        #removes
        foreground_mask = backgroundSubtractor.apply(frame, learningRate=0) #static background model (first frame)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (4, 4)) #mask is comprised of dots of dimensions (x,y)
        #mask = cv2.morphologyEx(foreground_mask, cv2.MORPH_OPEN, kernel) #erodes then dilates
        #kernel = np.ones((3, 3), np.uint8)
        foreground_mask = cv2.dilate(foreground_mask, kernel, iterations=2) #erode "subtracts", dilate "adds"
        mask = cv2.bitwise_and(frame, frame, mask=foreground_mask)
        return mask


    def calculateFingers(handOutline, drawing):  # -> finished bool, cnt: finger count
        #  convexity defect
        hull = cv2.convexHull(handOutline)
        if (len(hull)>3):
            areaHull = cv2.contourArea(hull)
        else:
            areaHull=0
        print("length", len(hull), "areahull", areaHull)  # delete later
        hull = cv2.convexHull(handOutline, returnPoints=False)


        if len(hull) > 3:
            # defects saves into an array  [ start point, end point, farthest point, approximate distance to farthest point ]
            defects = cv2.convexityDefects(handOutline, hull) # identifies defects
            if type(defects) != type(None):  # avoid crashing
                defect = 0
                for i in range(defects.shape[0]):  # calculate the angle
                    s, e, f, d = defects[i][0]
                    start = tuple(handOutline[s][0])
                    end = tuple(handOutline[e][0])
                    far = tuple(handOutline[f][0])
                    # pythagorean theorem
                    a = math.sqrt((end[0] - start[0]) ** 2 + (end[1] - start[1]) ** 2)
                    b = math.sqrt((far[0] - start[0]) ** 2 + (far[1] - start[1]) ** 2)
                    c = math.sqrt((end[0] - far[0]) ** 2 + (end[1] - far[1]) ** 2)
                    # cosine theorem
                    angle = math.acos((b ** 2 + c ** 2 - a ** 2) / (2 * b * c))
                    if angle <= math.pi / 2:  # angle less than 90 degree, treat as fingers
                        defect += 1
                        # draws a blue circle where the farthest (defect) is, fills it in
                        cv2.circle(drawing, far, 8, [211, 84, 0], -1)
                return True, defect, areaHull
        return False, 0, areaHull



    if timeElapsed < 500: # CHANGE number later -> for GUI
        threshold = 100
        #frame = cv2.bilateralFilter(frame, 5, 50, 100)  # smoothing filter
        frame = cv2.flip(frame, 1)  # flip the frame horizontally

        if isBgCaptured==1:
            img = removeBG(frame)
            img = img[100:300, 400:600]  # clip the ROI [y1:y2,x1,x2]
            #cv2.imshow('mask', img)

            # convert the image into binary image
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            blur = cv2.GaussianBlur(gray, (blurValue, blurValue), 0)
            ret, thresh = cv2.threshold(blur, threshold, 255, cv2.THRESH_BINARY)

            # get the contours
            thresh1 = copy.deepcopy(thresh)
            _, contours, hierarchy = cv2.findContours(thresh1, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            length = len(contours)
            maxArea = -1
            if length > 0:
                for i in range(length):  # find the biggest contour (according to area)
                    temp = contours[i]
                    area = cv2.contourArea(temp)
                    if area > maxArea:
                        maxArea = area
                        hand = i

                areaHand = maxArea
                handOutline = contours[hand]
                handHull = cv2.convexHull(handOutline)
                areaHull = cv2.contourArea(handOutline)
                drawing = np.zeros(img.shape, np.uint8) # black bg canvas w/ same dimensions as image
                cv2.drawContours(drawing, [handOutline], 0, (0, 255, 0), 2) # precision green outline around hand
                cv2.drawContours(drawing, [handHull], 0, (0, 0, 255), 3) # approx geometric outline around hand

                isFinishCal, cnt, areaHull = calculateFingers(handOutline, drawing)
                if (areaHand!=0):
                    areaRatio = ((areaHull - areaHand) / areaHand) * 100

                if isFinishCal is True:
                    if cnt==0:
                        if areaRatio >15: # change value later
                           # print("one finger", arearatio)
                            return 1
                        else:
                            #print("zero", areaRatio)
                            return 0
                    if cnt==1:
                        #print("two finger", areaRatio)
                        return 2
                    if cnt==2:
                        #print("three finger", areaRatio)
                        return 3
                    if cnt>2:
                        return 0
                        #print("too many finger", areaRatio)

        #give buffer time to load
        if timeElapsed == 3:
            #history, threshold
            globals()['isBgCaptured'] = 1
            globals()['backgroundSubtractor'] = cv2.createBackgroundSubtractorMOG2(0, bgSubThreshold)
    return 0
class RecvStream:
    def __init__(self, sock, host, port):
        self.sock = sock
        self.conn = socket.socket()
        sock.bind((host, port))
        self.img = None
        self.imgtk = None
        self.image_ready = False
        self.audio = None
        sock.listen(1)
        self.connStatus = False
        self.disconnect = False
        self.stopped = False

    def checkConn(self):
        addr = ''
        self.sock.setblocking(0)
        try:
            self.conn, addr = self.sock.accept()
            self.conn.setblocking(1)
            if (addr == ''):
                self.connStatus = False
            else:
                self.connStatus = True
        except:
            pass
        self.sock.setblocking(1)
        return addr

    def recvData(self):
        global StatusStr
        transText = Translator()
        while self.stopped == False:
            if self.connStatus == False:
                StatusStr = 'TCP Transceiver --- Listening on ' + host + ' port ' + str(port)
                addr = self.checkConn()
                if self.connStatus:
                    StatusStr = 'TCP Transceiver --- Connection from ' + addr[0] + ' port ' + str(port)
                    if send.connStatus == False:
                        send.addr = addr[0]
                        send.callInit = True
                        Gui.callBtn['text'] = 'Disconnect'
            else:
                if self.disconnect:
                    self.close()
                    self.disconnect = False
                else:
                    self.conn.settimeout(1.0)
                    d = b''
                    frame_buff = b''
                    image_size = 0
                    recv_size = 4096
                    d = self.conn.recv(recv_size)
                    d = Gui.unencryptionObject.decrypt(d)
                    if len(d) != 0:
                        self.connStatus = True
                        frame_buff += d
                        frame_size = int.from_bytes(frame_buff[0:4], byteorder='big')
                    else:
                        self.connStatus = False

                    if self.connStatus == True:
                        while len(frame_buff) < frame_size:
                            d = self.conn.recv(recv_size)
                            d = Gui.unencryptionObject.decrypt(d)
                            if len(d) != 0:
                                frame_buff += d
                                diff = frame_size - len(frame_buff)
                                if diff < recv_size: recv_size = diff
                            else:
                                self.connStatus = False
                                break

                    if self.connStatus == True:
                        img_size = frame_size - 2 * chunk - 4 - captionLength - pollLength

                        jpeg = fromstring(frame_buff[4:img_size + 4], uint8)
                        image = cv2.imdecode(jpeg, -1)

                        cv2image = cv2.cvtColor(image, cv2.COLOR_BGR2RGBA)

                        if Gui.pollState == 6:
                            Gui.pollReceiveCount+=1
                            num = handPoll(image, Gui.pollReceiveCount)
                            if num != 0:
                                if num ==1:
                                    print('1')
                                elif num ==2:
                                    print("2")
                                elif num ==3:
                                    print("3")
                                Gui.pollState = 7
                                Gui.pollReceiveCount=0
                        


                        offset = frame_size - 2 * chunk - captionLength - pollLength
                        self.audio = frame_buff[offset:frame_size - captionLength - pollLength]
                        playStream.write(self.audio)

                        printCaption = frame_buff[frame_size - captionLength - pollLength:frame_size - pollLength]
                        printCaption = printCaption.decode('utf-8')
                        printCaption = printCaption.strip()

                        pollContent = frame_buff[frame_size - pollLength:frame_size]
                        pollContent = pollContent.decode('utf-8')
                        pollContent = pollContent.strip()

                        if pollContent != "":
                            font = cv2.FONT_HERSHEY_SIMPLEX
                            y0, dy = 300,30
                            for i, line in enumerate(pollContent.split('\n')):
                                y = y0 + i*dy
                                cv2image = cv2.putText(cv2image,line,(10,y), font, 0.5, (0,255,0), 1, cv2.LINE_AA)
                            Gui.pollDetection = True
                        else:
                            Gui.pollDetection = False

                        self.img = Image.fromarray(cv2image)
                        self.image_ready = True

                        if printCaption != "":
                            if Gui.languageType == 'Spanish':
                                printSecondCaption = transText.translate(printCaption, scr='en', dest='es').text
                                Gui.addCaption2(printSecondCaption)
                            elif Gui.languageType == 'English':
                                Gui.addCaption2(printCaption)

                    if self.connStatus == False:
                        self.StatusStr = 'TCP Transceiver --- Call Disconnected'

        return

    def get_image(self):
        self.imgtk = ImageTk.PhotoImage(self.img)
        return self.imgtk

    def stop(self):
        self.stopped = True

    def close(self):
        self.conn.close()
        self.connStatus = False
        return

recv = RecvStream(sock=s, host=host, port=port)
recvThread = Thread(target=recv.recvData, args=())

pa = pyaudio.PyAudio()
chunk = 1024
format = pyaudio.paInt16
channels = 1
rate = 8000

playStream = pa.open(format=format,
                     channels=channels,
                     rate=rate,
                     input=False,
                     output=True,
                     frames_per_buffer=chunk)

sendStream = pa.open(format=format,
                     channels=channels,
                     rate=rate,
                     input=True,
                     input_device_index = 1,
                     output=False,
                     frames_per_buffer=chunk)

def callback(event):

    if Gui.searchBool == False and Gui.search != "":
        temp = Gui.search.strip()
        temp = temp.replace(" ","+")
        temp = 'https://www.google.com/search?q=' + temp
        webbrowser.open(temp, new=1)
        Gui.searchBool = True
    else:
        Gui.searchBool = False

    Gui.search = ''

class GUI():
    global remote_host

    def __init__(self):
        self.xcvr = Tk()
        self.xcvr.minsize(700, 500)
        self.xcvr.resizable(0, 0)
        self.xcvr.title('Video Chat')

        self.callBtn = Button(self.xcvr)
        self.callBtn.configure(text='Call', width=10, command=self.call)
        self.callBtn.grid(row=0, column=0, sticky=NW)

        self.Label = Label(self.xcvr, text='Remote Hosts')
        self.Label.grid(row=1, column=0, sticky=NW)

        self.host_label_str = StringVar()
        self.host_label = Label(self.xcvr, width=10, bg='yellow', textvariable=self.host_label_str)
        self.host_label.grid(row=2, column=0, sticky=NW)

        self.hosts_listbox = Listbox(self.xcvr, height=8, width=11)
        for i in range(len(remote_hosts)):
            self.hosts_listbox.insert(i, remote_hosts[i][0])
        self.hosts_listbox.select_set(0)
        self.hosts_listbox.bind('<<ListboxSelect>>', self.onselect)
        self.hosts_listbox.grid(row=3, column=0, sticky=NW)

        self.selRemBtn = Button(self.xcvr)
        self.selRemBtn.configure(text='Select', width=10, command=self.selRem)
        self.selRemBtn.grid(row=4, column=0, sticky=NW)

        self.host_str = StringVar()
        self.host_entry = Entry(self.xcvr, width=11, textvariable=self.host_str)
        self.host_str.set(remote_hosts[0][0])
        self.host_entry.grid(row=11, column=0, sticky=NW)
        remote_host = remote_hosts[0][1]

        self.addr_str = StringVar()
        self.addr_entry = Entry(self.xcvr, width=11, textvariable=self.addr_str)
        self.addr_str.set(remote_hosts[0][1])
        self.addr_entry.grid(row=12, column=0, sticky=NW)

        self.editRemBtn = Button(self.xcvr)
        self.editRemBtn.configure(text='Edit', width=10, command=self.editRem)
        self.editRemBtn.grid(row=14, column=0, sticky=NW)

        self.addRemBtn = Button(self.xcvr)
        self.addRemBtn.configure(text='Add', width=10, command=self.addRem)
        self.addRemBtn.grid(row=15, column=0, sticky=NW)

        self.delRemBtn = Button(self.xcvr)
        self.delRemBtn.configure(text='Delete', width=10, command=self.delRem)
        self.delRemBtn.grid(row=16, column=0, sticky=NW)

        self.exitBtn = Button(self.xcvr)
        self.exitBtn.configure(text='Exit', width=10, command=self.exit)
        self.exitBtn.grid(row=25, column=0, sticky=NW)
        self.xcvr.wm_protocol('WM_DELETE_WINDOW', self.exit)

        self.screenShareBtn = Button(self.xcvr)
        self.screenShareBtn.configure(text='Screen On', width=10, command=self.screenShare)
        self.screenShareBtn.grid(row=27, column=0, sticky=NW)

        self.meetingidBtn = Button(self.xcvr)
        self.meetingidBtn.configure(text='Meeting ID', width=10, command=self.meetingid)
        self.meetingidBtn.grid(row=29, column=0, sticky=NW)
        self.meetingid = '1234567891234567'

        self.idLabel1 = Label(self.xcvr, text='123456789')
        self.idLabel1.grid(row=31, column=0, sticky=NW)

        self.idLabel2 = Label(self.xcvr, text='1234567')
        self.idLabel2.grid(row=32, column=0, sticky=NW)

        self.languageTypeBtn = Button(self.xcvr)
        self.languageTypeBtn.configure(text='Spanish', width=10, command=self.translate)
        self.languageTypeBtn.grid(row=20, column=0, sticky=NW)
        self.languageType = 'English'

        self.scrn = Canvas(self.xcvr, width=640, height=600, background="#cceeff")
        self.scrn.grid(row=0, column=1, rowspan=40)
        self.canvas_image = self.scrn.create_image(2, 2, anchor=NW, image=None)
        self.prev_image = self.scrn.create_image(482, 362, anchor=NW, image=None)

        self.T = Text(self.xcvr, height=5, width = 77, background = "#ffcccc", insertbackground= 'yellow')
        self.T.place(x=101,y=517)
        self.S = Scrollbar(self.xcvr)
        self.S.place(x=84,y=517)
        self.T.config(yscrollcommand=self.S.set)
        self.S.config(command=self.T.yview)
        self.search = ''
        self.searchBool = False
        self.StatusStr = ''
        self.T.bind("<ButtonRelease-1>", callback)

        self.username = "you"
        self.getUsername()

        self.poll = False
        self.pollState = 0
        self.question = ''
        self.option1 = ''
        self.option2 = ''
        self.option3 = ''
        self.pollDisplayCount = 0
        self.pollReceiveCount = 0
        self.pollDetection = False

        self.encryptionObject = AES.new("1234567891234567".encode("utf-8"), AES.MODE_CFB,
                                        'This is an IV456'.encode("utf-8"))
        self.unencryptionObject = AES.new("1234567891234567".encode("utf-8"), AES.MODE_CFB,
                                          'This is an IV456'.encode("utf-8"))


    def meetingid(self):
        id = simpledialog.askstring(title="Meeting ID",prompt="Enter meeting ID")
        if id != None:
            self.meetingid = id.ljust(16, " ")
            self.idLabel1['text'] = self.meetingid[0:9]
            self.idLabel2['text'] = self.meetingid[9:16]
        self.encryptionObject = AES.new(self.meetingid.encode("utf-8"), AES.MODE_CFB,
                                        'This is an IV456'.encode("utf-8"))
        self.unencryptionObject = AES.new(self.meetingid.encode("utf-8"), AES.MODE_CFB,
                                          'This is an IV456'.encode("utf-8"))
    def screenShare(self):
        if vs.screenSharing == False:
            vs.screenSharing = True
            self.screenShareBtn['text'] = 'Screen Off'
        else:
            vs.screenSharing = False
            self.screenShareBtn['text'] = 'Screen On'
    def translate(self):
        if self.languageType == 'English':
            self.languageType = 'Spanish'
            self.languageTypeBtn['text'] = 'English'
        elif self.languageType == 'Spanish':
            self.languageType = 'English'
            self.languageTypeBtn['text'] = 'Spanish'

    def getUsername(self):
        self.username = simpledialog.askstring(title='Name', prompt="What is your name ?")
        self.username = self.username.upper()

    def call(self):
        if self.callBtn['text'] == 'Call':
            self.callBtn['text'] = 'Disconnect'
            send.addr = remote_host
            send.callInit = True
        elif self.callBtn['text'] == 'Disconnect':
            if send.connStatus:
                send.disconnect = True
                while send.disconnect: pass
            if recv.connStatus:
                recv.disconnect = True
                while recv.disconnect: pass
            self.callBtn['text'] = 'Call'

    def addCaption1(self, caption):
        self.T.insert(END, self.username + ": " + caption + '\n')

    def addCaption2(self, caption):
        self.T.insert(END, "Person 2: " + caption + '\n')

    def onselect(self, evt):
        global remote_hosts

        i = self.hosts_listbox.curselection()[0]
        self.host_str.set(remote_hosts[i][0])
        self.addr_str.set(remote_hosts[i][1])

    def selRem(self):
        global remote_host

        i = self.hosts_listbox.curselection()[0]
        self.host_label_str.set(self.hosts_listbox.get(i))
        self.host_str.set(remote_hosts[i][0])
        self.addr_str.set(remote_hosts[i][1])
        remote_host = remote_hosts[i][1]

    def editRem(self):
        global remote_hosts

        i = self.hosts_listbox.curselection()[0]
        tempHost = self.host_entry.get()
        tempAddr = self.addr_entry.get()
        self.hosts_listbox.delete(i)
        self.hosts_listbox.insert(i, tempHost)
        remote_hosts[i][0] = tempHost
        remote_hosts[i][1] = tempAddr

        file = open('remote_hosts.dat', 'wb')
        pickle.dump(remote_hosts, file)
        file.close()

    def addRem(self):
        global remote_hosts

        tempHost = self.host_entry.get()
        tempAddr = self.addr_entry.get()
        err = False
        for i in range(len(remote_hosts)):
            if tempHost == remote_hosts[i][0] or tempAddr == remote_hosts[i][1]:
                err = True
                break
        if err:
            box.showerror('Add Remote Host Error', 'Host name or address already in use')
        else:
            remote_hosts.append([tempHost, tempAddr])
            self.hosts_listbox.insert(END, tempHost)
            file = open('remote_hosts.dat', 'wb')
            pickle.dump(remote_hosts, file)
            file.close()

    def delRem(self):
        global remote_hosts

        i = self.hosts_listbox.curselection()[0]
        self.hosts_listbox.delete(i)
        del remote_hosts[i:i + 1]
        self.hosts_listbox.select_set(0)

        file = open('remote_hosts.dat', 'wb')
        pickle.dump(remote_hosts, file)
        file.close()

    def exit(self):
        if send.connStatus:
            send.disconnect = True
            while send.disconnect: pass

        send.stop()
        while sendThread.is_alive(): pass
        if sendStream.is_active():
            sendStream.stop_stream()
            sendStream.close()

        recv.stop()
        while recvThread.is_alive(): pass
        recv.close()
        recv.sock.close()
        if playStream.is_active():
            playStream.stop_stream()
            playStream.close()

        pa.terminate()

        vs.stop()
        vs.stream.release()

        self.xcvr.destroy()

    def run(self):
        self.scrn.after(0, self.updateGUI())
        self.xcvr.mainloop()

    def updateGUI(self):

        self.xcvr.title(StatusStr)

        if recv.connStatus:
            if recv.image_ready:
                imgtk = recv.get_image()
                self.scrn.itemconfig(self.canvas_image, image=imgtk)
        else:
            self.scrn.itemconfig(self.canvas_image, image=b'')

        if send.connStatus:
            self.scrn.itemconfig(self.prev_image, image=vs.get_preview())
        else:
            self.scrn.itemconfig(self.prev_image, image=b'')

        self.xcvr.update_idletasks()
        self.scrn.after(20, self.updateGUI)


        try:
            if self.T.selection_own_get().__str__() == ".!text" and self.searchBool==False:
                self.search = self.T.selection_get()
        except:
            pass





if __name__ == '__main__':
    recvThread.start()
    sendThread.start()
    Gui = GUI()
    Gui.run()
