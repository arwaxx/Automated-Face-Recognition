from Models_as_Classes import CNN, ANNHOG
import tkinter as tk
from tkinter import ttk
from tkinter import filedialog as fd
import cv2

cnn = CNN()
ann = ANNHOG()
# create the root window
root = tk.Tk()
root.title('Tkinter Open File Dialog')
root.resizable(False, False)
root.geometry('500x500')


# input in function io refer to path
def io(input):
    face_cascade = cv2.CascadeClassifier(
        'D:/Projects/anaconda/envs/tensorflow/Lib/site-packages/cv2/data/haarcascade_frontalface_default.xml')
    img = cv2.imread(input)

    cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    face_cascade.detectMultiScale(img, scaleFactor=1.1, minNeighbors=4, minSize=(20, 20))

    cv2.putText(img,"CNN :"+ str(cnn.recognize(input)), (0, 40), cv2.FONT_HERSHEY_TRIPLEX, .6, (0, 0, 255))
    cv2.putText(img,"ANN :"+ str(ann.recognize(input)), (0, 60), cv2.FONT_HERSHEY_TRIPLEX, .6, (0, 0, 255))

    cv2.imshow('image', cv2.resize(img, (555, 555)))
    cv2.waitKey(0)


def select_file():
    filetypes = (
        ('Photos', '*.jpg'),
        ('Photoss', '*.png'),
        ('All files', '.')
    )

    filename = fd.askopenfilename(
        title='Choose Image',
        initialdir='C:/Users/Ahmed Wanas/Downloads/ezyZip/dataset/dataset/',
        filetypes=filetypes)

    io(filename)


# open button
open_button = ttk.Button(
    root,
    text='Open a File',
    command=select_file
)

open_button.pack(expand=True)

# run the application
root.mainloop()
