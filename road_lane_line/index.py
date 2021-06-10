import tkinter as tk
from tkinter import *

import cv2
from PIL import Image, ImageTk
import os
import numpy as np
from numpy.lib.polynomial import roots


global last_frame
last_frame = np.zeros((480,640,3), dtype=np.uint8)

global last_frame1
last_frame1 = np.zeros((480,640,3), dtype=np.uint8)

global cap
global cap1

cap = cv2.VideoCapture("challenge.mp4")
cap1 = cv2.VideoCapture("challenge1.mp4")

def show_video():
    if not cap.isOpened():
        print("cant open the video 1")
    flag1, frame1 = cap.read()
    frame1 = cv2.resize(frame1,(800,400))
    if flag1 is None:
        print("Major error")
    elif flag1:
        global last_frame1
        last_frame1 = frame1.copy()
        pic = cv2.cvtColor(last_frame1,cv2.COLOR_BGR2RGB)
        img = Image.fromarray(pic)
        imgtk = ImageTk.PhotoImage(image=img)
        lmain.imgtk = imgtk
        lmain.configure(image=imgtk)
        lmain.after(10, show_video)

def show_video1():
    if not cap1.isOpened():                             
        print("cant open the camera2")
    flag2, frame2 = cap1.read()
    frame2 = cv2.resize(frame2,(400,500))
    if flag2 is None:
        print ("Major error2!")
    elif flag2:
        global last_frame2
        last_frame2 = frame2.copy()
        pic2 = cv2.cvtColor(last_frame2, cv2.COLOR_BGR2RGB)
        img2 = Image.fromarray(pic2)
        img2tk = ImageTk.PhotoImage(image=img2)
        lmain2.img2tk = img2tk
        lmain2.configure(image=img2tk)
        lmain2.after(10, show_video1)


if __name__ == "__main__":
    root = tk.Tk()
    lmain = tk.Label(master=root)
    lmain2 = tk.Label(master=root)
    lmain.pack(side=TOP)
    lmain2.pack(side = BOTTOM)
    root.title("Lane Line"
    )
    root.geometry("1000x800+100+10") 
    exitbutton = Button(root, text='Quit',fg="red",command=   root.destroy).pack(side = BOTTOM,)
    show_video()
    show_video1()
    # show_vid2()
    root.mainloop()                                  
    cap.release()