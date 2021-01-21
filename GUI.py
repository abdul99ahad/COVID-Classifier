from tkinter import * 
import os
import PIL.Image
from PIL import ImageTk
from tkinter import filedialog 
from predict_model import *
from process_image_for_gui import *
window = Tk(className='Covid Prediction Test')
window.geometry("550x600")
window.configure(background='#232323')
title_label = Label(window, 
		 text="COVID Classifier",
		 fg = "white",
		 font = ("Segoe UI", 26),
         bg = "#232323").grid(row=0,rowspan=2,column=4,columnspan=4)




def label_scores(Y_Score):
    global label1_score,label2_score,label3_score
    label1_score["text"] = str(round((Y_Score[0][0]*100),3))+'%'
    label2_score["text"] = str(round((Y_Score[0][1]*100),3))+'%'
    label3_score["text"] = str(round((Y_Score[0][2]*100),3))+'%'

Y_Score = []

def browseFiles():
    global Y_Score
    Y_Score = [] 
    filename = filedialog.askopenfilename(initialdir = "/", 
                                          title = "Select a File", 
                                          filetypes = (('all files', '.*'),
                                          ('JPG files', '*.jpg'),
                                          ('PNG files', '*.png'))) 
    print(filename)
    pre_image = processing_image(filename)
    Y_Score = extractFeaturesTest(pre_image)
    label_scores(Y_Score)
    img = PIL.Image.open(filename)
    img = img.resize((256,256),PIL.Image.ANTIALIAS)
    xray_img = ImageTk.PhotoImage(img)
    xray_image = Label(window,image = xray_img)
    xray_image.image=xray_img
    xray_image.grid(row=9, column=4)
    
label1= Label(window,text='Covid',font = ("Segoe UI", 20),bg='#232323',fg='white')
label2= Label(window,text="Normal",font = ("Segoe UI", 20),bg='#232323',fg='white')
label3= Label(window,text="Pneumonia",font = ("Segoe UI", 20),bg='#232323',fg='white')
label1_score= Label(window,text="0%",font = ("Segoe UI", 16),bg='#232323',fg='white')
label2_score= Label(window,text="0%",font = ("Segoe UI", 16),bg='#232323',fg='white')
label3_score= Label(window,text="0%",font = ("Segoe UI", 16),bg='#232323',fg='white')
upload_img = Button(window,borderwidth=1,relief="groove",text="Upload Image",font = ("Segoe UI", 14),bg='#232323',fg='white',command = browseFiles)

    
label1.grid(row=3,column=3)
label2.grid(row=4,column=3)
label3.grid(row=5,column=3)
label1_score.grid(row=3,column=10,padx=10)
label2_score.grid(row=4,column=10,padx=10)
label3_score.grid(row=5,column=10,ipadx=10)
upload_img.grid(row=6,column=4,columnspan=4)

window.mainloop()