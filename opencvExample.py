from tkinter import *
import cv2
import PIL.Image, PIL.ImageTk
import time
import yaml
import os
import util
import datetime
import numpy as np

from custom_lk import CustomLK
from face_detector import FaceDetector

class App:
    def __init__(self,window,window_title,video_source=0):
        self.window = window
        self.window.title(window_title)
        self.video_source = video_source
        self.window.configure(background='white')
        
        #Get media directory
        self.curdir = os.path.dirname(__file__)
        self.yaml_file = os.path.join(self.curdir, 'config.yaml')
        with open(self.yaml_file, "r") as f:
            self.config = yaml.load(f)
        self.media_dir = os.path.join(self.curdir, self.config['media'])
        
        #Get output directory
        self.output_dir = os.path.join(self.curdir, self.config['output'])
        print(self.output_dir )
        
        #Reference to output file
        self.outputfilePath = os.path.join(self.output_dir,'outputFile.csv')
        
        #Show logo
        self.logoImgPath = os.path.join(self.media_dir, 'logo.png')
        self.logoWidget = Label(window, compound='top', borderwidth = 0,bg='white')
        self.logoWidget.logo = PhotoImage(file=self.logoImgPath)
        self.logoWidget['image']=self.logoWidget.logo
        self.logoWidget.pack()

        #open video source (i.e. webcam)
        self.vid = MyVideoCapture(self.video_source)
        
         # extract background subtraction image from bg vid
        self.bg_file = os.path.join(self.curdir, self.config['bg_img'])
        _, _, self.bg_frame = util.load_video(self.bg_file)
    
        self.custom_lk = CustomLK()
        self.haar_cascade = FaceDetector(curdir=self.curdir, type='face')
        
        #create canvas that can fit about the video source size
        self.canvas = Canvas(window, width = 853, height = 480)
        self.canvas.pack()
        
        #Play
        self.playBtnPath = os.path.join(self.media_dir, 'playButton.png')
        self.playBtnImg = PhotoImage(file=self.playBtnPath)
        self.playbtn = Button(window,text="snap", width = 125, command = lambda: self.playCallback(),border=0,bg='white')
        self.playbtn.config(image=self.playBtnImg)
        self.playbtn.image = self.playBtnImg
        self.playbtn.pack(anchor=CENTER, expand = True)
        
        #Establish self reference to warning image and ok/submit button
        self.warnImgPath = os.path.join(self.media_dir,'exclam.png')
        self.warnImg = PhotoImage(file=self.warnImgPath)
        
        self.okBtnImgPath =  os.path.join(self.media_dir,'okButton.png')
        self.okBtnImg = PhotoImage(file=self.okBtnImgPath)
        
        self.submitBtnImgPath = os.path.join(self.media_dir,'submitBut.png')
        self.submitBtnImg = PhotoImage(file=self.submitBtnImgPath) 
        
        #init string variable for entry
        self.strVar = StringVar()
        
        self.firstIter = True
        self.startERVA = False
        self.history = None
        self.flag = False
        
        #After it is called once, the update method will be automatically called
        self.delay = 15 # ms
        self.update()
        
        self.window.mainloop()
        
    def snapshot(self):
        #Great a frame from video source
        ret,frame = self.vid.get_frame()
        
        if ret:
            cv2.imwrite("Frame-"+time.strftime("%d-%m-%Y-%H-%M-%S") + ".jpg", cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
            
    def playCallback(self):
        self.startERVA =  not self.startERVA
        print(self.startERVA)
    
    def update(self):
        #Great a frame from video source    
        ret,frame = self.vid.get_frame()
        
        if ret :
           
            self.curFrame = frame
            
            
            if self.firstIter:
                self.prevFrame = self.curFrame
                _, self.mask = util.background_subtraction(self.curFrame, self.bg_frame,
                                               thresh=0.25)
                self.firstIter = False
            else:
                #give current and previous frames to chris
                if self.startERVA:
                    img, _, event, self.history = util.process_frame(self.prevFrame,self.curFrame,
                                                            self.mask, self.custom_lk,
                                                            self.haar_cascade, self.history)
                    if(~self.flag & event):
                        self.history = None
                        self.flag = True
                        self.popupmsg('2')
                  
                else:
                    img = self.curFrame[40: 680, 70: 1210]

                #show frame
                outImg = cv2.resize(img, dsize = (853,480), interpolation = cv2.INTER_CUBIC)
                outImg = PIL.Image.fromarray(outImg)
                self.photo = PIL.ImageTk.PhotoImage(image = outImg)
                self.canvas.create_image(0,0,image=self.photo,anchor=NW)
                
                #update previous frame 
                self.prevFrame = self.curFrame
            
            

            
            
        else:
            self.vid = MyVideoCapture(self.video_source)
        self.window.after(self.delay,self.update)
        
    
    def savePatientStatus(self,entry):
        # Append to file!
        fid = open(self.outputfilePath, 'a')
        timeStamp = datetime.datetime.now()
        fid.write(str(timeStamp) +',' + entry + '\n')
        fid.close()
        self.flag = False
        
    def entryCallback(self):
        print(self.strVar.get())
        
    def popupForm(self,patient):
        popup = Toplevel()
        popup.geometry("400x175")
        popup.configure(background='white')
        popup.wm_title("ERVA - Critical Action")
   
        #Title of popup
        labelTitle = Label(popup, bg='white',text='CRITICAL ACTION REQUIRED',font=("Helvetica", 14))
        labelTitle.grid(row=0,column =1, columnspan=2,sticky=W)
        
        #Show warning image
        labelImg = Label(popup, image = self.warnImg, bg='white')
        labelImg.grid(row=0, column=0, rowspan = 3)
        
        #Entry message
        entryMsg = 'Update patient ' + patient +"'s status:"
        entryMsgLabel = Label(popup, bg='white',text= entryMsg)
        entryMsgLabel.grid(row=1,column=1)
        
        #initialize strVar
        self.strVar.set('')
        #The actual text entry field
        e1 = Entry(popup,width=40, textvariable = self.strVar, validate="focusout", validatecommand=self.entryCallback)
        e1.grid(row=2, column=1)
        
        B1 = Button(popup, text="Okay", command = lambda:[ popup.destroy() or self.savePatientStatus( patient + ',' + self.strVar.get()) ], borderwidth = 0, bg='white',width=125)
        B1.config(image = self.okBtnImg)
        B1.image = self.okBtnImg
        B1.grid(row=3,column=1, columnspan = 2,sticky=E)
    
    def popupmsg(self,patient):
        popup = Toplevel()
        popup.geometry("400x175")
        popup.configure(background='white')
        popup.wm_title("ERVA - Critical Action")
        
        #Title of popup
        labelTitle = Label(popup, bg='white',text='CRITICAL ACTION REQUIRED',font=("Helvetica", 14))
        labelTitle.grid(row=0,column =1, columnspan=2)
        
        #Show warning image
        labelImg = Label(popup, image = self.warnImg, bg='white')
        labelImg.grid(row=0, column=0, rowspan = 3)
        
        #Show popup command
        critMsg = '\nCheck status of patient: ' +patient +'\n\n'
        labelMsg = Label(popup, bg='white',text=critMsg)
        labelMsg.grid(row=1,column=1,columnspan=2)
        
        #Ok button
        B1 = Button(popup, text="Okay", command = lambda: [popup.destroy() or self.popupForm(patient)], borderwidth = 0, bg='white',width=125)
        B1.config(image = self.okBtnImg)
        B1.image = self.okBtnImg
        B1.grid(row=2,column=2, sticky=E)
        
    
    
#########################################################

class MyVideoCapture:
    def __init__(self,video_source=0):
        #Open vidoe source
        self.vid = cv2.VideoCapture(video_source)
       
        self.frame_counter = 0
        if not self.vid.isOpened():
            raise ValueError("Unable to open video source ", video_source)
        # get video source width and height
        
        self.width = self.vid.get(cv2.CAP_PROP_FRAME_WIDTH)
        self.height = self.vid.get(cv2.CAP_PROP_FRAME_HEIGHT)
        
        
    def get_frame(self):
        if self.vid.isOpened():
            ret,frame=self.vid.read()
            if ret:
                
                return (ret,cv2.cvtColor(frame,cv2.COLOR_BGR2RGB))
            else:
                
                return (ret,None)
        else:
            return (ret,None)
    #Release video source when object is destroyed
    
    def __del__(self):
        if self.vid.isOpened():
            self.vid.release()
            

#########################################################




        
