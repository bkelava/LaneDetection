from tkinter import *
from tkinter import ttk
from tkinter import filedialog
from PIL import ImageTk, Image
from DetectionManager import *
import cv2



class Root(Tk):
    def __init__(self):
        super(Root, self).__init__()
        self.minsize(200, 200)
        self.labelFrame = Label(self, text = "Open image or video").pack()
        #self.labelFrame.grid(column=0, row=1, padx=20, pady=20)
        self.button()


    def button(self):
        self.button = Button(self.labelFrame, text = "Browse", command = self.fileDialog).pack()
        #self.button.grid(column = 1, row =1)


    def fileDialog(self):
        self.fileName = filedialog.askopenfilename(initialdir = "/", title="Select", filetype=(("video/image", "*.jpg *.jpeg *.mp4 *.wav"), ("All files", "*.*")))

        if self.fileName:
            self.readFile(self.fileName)
        else:
            print("NOTHING UPLOADED!")

    def readFile(self, path):
        self.detectionManager = DetectionManager()
        if "jpg" in path or "jpeg" in path:
            self.detectionManager.detectFromImage(path)
        elif "mp4" in path or "wav" in path:
            self.detectionManager.detectFromVideo(path)
        else:
            print("SOMETHING WENT WRONG!")

if __name__ == '__main__':
    root = Root()
    root.mainloop()



