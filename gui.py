from tkinter import *
from PIL import ImageTk,Image
from tkinter import filedialog
import main

imagePath = "Char-dataset/cover/0.25x.png"
# imagePath = "C:/Users/RosCheat/Pictures/241-2416641_madara-uchiha-by-aikawaiichan-aikawaiichan-madara.jpg"

def checkForImage():
    filetypes = (
        ('Jpg Files', '*.jpg'),
        ('Png Files', '*.png')
    )
    filepath = filedialog.askopenfilename(filetypes=filetypes)
    file = open(filepath,'r')
    imagePath = file.name
    img = ImageTk.PhotoImage(Image.open(imagePath))

    imageLabel.config(image=img)
    imageLabel.image = img
    output , pre_idex , = main.outPutResult(imagePath)
    imageRightLabel.config(text=output,font=("Calibri",300,"bold"))

    bottomLabel.config(text="Prediction index : " + pre_idex, font=("Calibri",15,"bold"))
    # bottomLabel.config(text="Prediction index : " + key, font=("Calibri", 10, "bold"))

root = Tk()

r_width = root.winfo_screenwidth()
r_height = root.winfo_screenheight()

root.geometry(f"{r_width}x{r_height}")
root.title("SIFT Descriptor for Printed Khmer Characters")
root.resizable(0,0)

titleLabel = Label(root,text="CHARACTER PREDICTION",font=("Calibri",20,'bold'),pady=int(r_height*0.05),background="#ddd").pack(fill=X,side=TOP)

global leftFrame
leftFrame = Frame(root,background="#ddd")
leftFrame.place(relx=0.00, rely=0.15, relwidth=0.49,relheight=0.50)

img = ImageTk.PhotoImage(Image.open(imagePath))

global imageLabel
imageLabel = Label(leftFrame,image=img)
imageLabel.place(relx=0.01, rely=0.01, relwidth=0.98,relheight=0.88)

rightFrame = Frame(root,background="#ddd")
rightFrame.place(relx=0.51, rely=0.15, relwidth=0.49,relheight=0.50)

global imageRightLabel
imageRightLabel = Label(rightFrame)
imageRightLabel.place(relx=0.01, rely=0.01, relwidth=0.98,relheight=0.88)

outPutButton = Button(root,text="Find File And OUTPUT",font=("Calibri",16,"bold"),bg="#3944bc",fg="#ffffff",activebackground="#1338be",activeforeground="#ffffff",command=checkForImage)
outPutButton.place(relx=0.40, rely=0.755, relwidth=0.20,relheight=0.05)

bottomFrame = Frame(root,background="#ddd")
bottomFrame.place(relx=0.0, rely=0.81, relwidth=1.0,relheight=0.16)

global bottomLabel
bottomLabel = Label(bottomFrame)
bottomLabel.place(relx=0.01, rely=0.01, relwidth=0.98,relheight=0.88)

root.mainloop()