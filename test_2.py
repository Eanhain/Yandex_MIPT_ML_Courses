from Tkinter import *
from PIL import Image, ImageTk
import numpy as np


b1 = "up"
xold, yold = None, None
list = []

def int():
   sum = np.sum(np.array(list)[1:, 1])
   print "SUM:", sum
   print list,"FINAL LIST"

def clear():
   print "clear"
   drawing_area.delete("all")
   drawing_area.create_image(10, 10, image=tkimage, anchor=NW)
   list[:]=[]
   print list


def calcLHS():
 print "THIS IS THE LIST",list
 drawing_area.delete ("recs")
 a = round(np.array(list)[:,0][0],2)  # left boundary of area
 b = round(np.array(list)[:,0][-1],2)  # right boundary of area
 n = w.get()  # number of intervals
 dx = round(((b - a) / n),2) # width of the intervals
 LHSarea = 0

 for i in range(1, n + 1):
     x0 = (i - 1) * dx
     x1 =  i * dx
     LHSi = dx * np.array(list)[:,1][x0]
     drawing_area.create_rectangle(x0 * 9 + 48 +a*9 , -9.7 * np.array(list)[:, 1][x0] + 506 , x1 * 9 + 48 + a*9, 506, tags="recs")
     LHSarea = LHSarea + LHSi

 print ("Final LHS Area=", LHSarea)
 drawing_area.delete("text")
 drawing_area.create_text(600, 100, text=("a=",a), font=("Arial", 25), anchor=W, tags="text")
 drawing_area.create_text(600, 130, text=("b=",b), font=("Arial", 25), anchor=W, tags="text")
 drawing_area.create_text(600, 160, text=("dx=",dx), font=("Arial", 25), anchor=W, tags="text")
 drawing_area.create_text(600, 190, text=("area=", round(LHSarea,2)), font=("Arial", 25), anchor=W, tags="text")


def calcRHS():
    drawing_area.delete("recs")
    a = round(np.array(list)[:, 0][0], 2)  # left boundary of area
    b = round(np.array(list)[:, 0][-1], 2)  # right boundary of area
    n = w.get()  # number of intervals
    dx = round(((b - a) / n),2)  # width of the intervals

    RHSarea = 0

    for i in range(1, n + 1):
        x0 = (i - 1) * dx
        x1 = i * dx
        RHSi = dx * np.array(list)[:,1][x1]
        drawing_area.create_rectangle(x0 * 9 + 48 + a * 9, -9.7 * np.array(list)[:, 1][x1] + 506, (x1) * 9 + 48 + a * 9,
                                      506, tags="recs")
        RHSarea = RHSarea + RHSi

    print ("RHS Area=", RHSarea)
    drawing_area.delete("text")
    drawing_area.create_text(600, 100, text=("a=", a), font=("Arial", 25), anchor=W, tags="text")
    drawing_area.create_text(600, 130, text=("b=", b), font=("Arial", 25), anchor=W, tags="text")
    drawing_area.create_text(600, 160, text=("dx=", dx), font=("Arial", 25), anchor=W, tags="text")
    drawing_area.create_text(600, 190, text=("area=", round(RHSarea, 2)), font=("Arial", 25), anchor=W, tags="text")

def calcTRAP():
    drawing_area.delete("recs")
    a = round(np.array(list)[:, 0][0], 2)  # left boundary of area
    b = round(np.array(list)[:, 0][-1], 2)  # right boundary of area
    n = w.get()  # number of intervals
    dx = round(((b - a) / n),2)  # width of the intervals
    TRAParea = 0
    for i in range(1, n + 1):
        x0 = (i - 1) * dx
        x1 = i * dx
        TRAPi = dx * (np.array(list)[:,1][x0] + np.array(list)[:,1][x1]) / 2.
        drawing_area.create_polygon(x0 * 9 + 48 + a * 9, 506, x0 * 9 + 48 + a * 9,  -9.7 * np.array(list)[:, 1][x0] + 506, x1 * 9 + 48 + a * 9, -9.7 * np.array(list)[:, 1][x1] + 506, x1 * 9 + 48 + a * 9, 506, fill='', outline="black",tags="recs")
        TRAParea = TRAParea + TRAPi

    print ("TRAP Area=", TRAParea)
    drawing_area.delete("text")
    drawing_area.create_text(600, 100, text=("a=", a), font=("Arial", 25), anchor=W, tags="text")
    drawing_area.create_text(600, 130, text=("b=", b), font=("Arial", 25), anchor=W, tags="text")
    drawing_area.create_text(600, 160, text=("dx=", dx), font=("Arial", 25), anchor=W, tags="text")
    drawing_area.create_text(600, 190, text=("area=", round(TRAParea, 2)), font=("Arial", 25), anchor=W, tags="text")


def calcMID():
  drawing_area.delete("recs")
  a = np.array(list)[:, 0][0]  # left boundary of area
  b = np.array(list)[:, 0][-1]  # right boundary of area
  n = w.get()  # number of intervals
  dx = (b - a) / n  # width of the intervals
  MIDarea = 0

  for i in range(1, n + 1):
      x0 = (i - 1) * dx
      x1 = i * dx

      print "X1:", x1
      MIDi = dx * np.array(list)[:, 1][(x0 + x1) / 2]

      drawing_area.create_rectangle(x0 * 9 + 49 + a * 9, -9.7 * (np.array(list)[:, 1][((x0 + x1) / 2)]) + 506, x1 * 9 + 49 + a * 9, 506, tag="recs")
      MIDarea = MIDarea + MIDi
  print ("Mid Area=", MIDarea)
  drawing_area.delete("text")
  drawing_area.create_text(600, 100, text=("a=", a), font=("Arial", 25), anchor=W, tags="text")
  drawing_area.create_text(600, 130, text=("b=", b), font=("Arial", 25), anchor=W, tags="text")
  drawing_area.create_text(600, 160, text=("dx=", dx), font=("Arial", 25), anchor=W, tags="text")
  drawing_area.create_text(600, 190, text=("area=", round(MIDarea, 2)), font=("Arial", 25), anchor=W, tags="text")


def calcSIM():
    SIMarea=0
    drawing_area.delete("recs")
    a = round(np.array(list)[:, 0][0], 2)  # left boundary of area
    b = round(np.array(list)[:, 0][-1], 2)  # right boundary of area
    n = w.get()  # number of intervals
    dx = round(((b - a) / n),2)  # width of the intervals

    for i in range(1, n + 1):
        x0 = (i - 1) * dx
        x1 = i * dx
        MIDi = dx * np.array(list)[:, 1][(x0 + x1) / 2]
        TRAPi = dx * (np.array(list)[:,1][x0] + np.array(list)[:,1][x1]) / 2.
        SIMi= (2*MIDi+TRAPi)/3
        SIMarea = SIMarea +SIMi

    print ("Sim Area=", SIMarea)
    drawing_area.delete("text")
    drawing_area.create_text(600, 100, text=("a=", a), font=("Arial", 25), anchor=W, tags="text")
    drawing_area.create_text(600, 130, text=("b=", b), font=("Arial", 25), anchor=W, tags="text")
    drawing_area.create_text(600, 160, text=("dx=", dx), font=("Arial", 25), anchor=W, tags="text")
    drawing_area.create_text(600, 190, text=("area=", round(SIMarea, 2)), font=("Arial", 25), anchor=W, tags="text")


def b1down(event):
  global b1
  b1 = "down"


def b1up(event):
  global b1, xold, yold
  b1 = "up"
  xold = None
  yold = None


def motion(event):
  if b1 == "down":
      global xold, yold
      if xold is not None and yold is not None:
          event.widget.create_line(xold,yold,event.x,event.y,smooth=TRUE , fill="red")
          xold = (xold - 48) / 9.
          yold = -1 * (yold - 506) / 9.7

          if(xold>=0):
              if(len(list)==0):
                  list.append([xold, yold])
              else:
                  if  (str(list[-1]).startswith('[%d' % xold)):
                      list[-1]=[xold, yold]
                  else:
                        list.append([xold, yold])

      xold = event.x
      yold = event.y

#layout

root = Tk()

b2 = Button(root, text="Clear", command=clear)
b2.pack()

v = IntVar()

Radiobutton(root, text="Left Hand Sum", variable=v, value=1, command=calcLHS).pack(anchor=W)
Radiobutton(root, text="Right Hand Sum", variable=v, value=2, command=calcRHS).pack(anchor=W)
Radiobutton(root, text="Midpoint", variable=v, value=3, command=calcMID).pack(anchor=W)
Radiobutton(root, text="Trapezoid", variable=v, value=4, command=calcTRAP).pack(anchor=W)
Radiobutton(root, text="Simpson's Rule", variable=v, value=5, command=calcSIM).pack(anchor=W)

w = Scale(root, from_=2, to=20, orient=HORIZONTAL, length=200, sliderlength=15)
w.pack(anchor=CENTER)
w.set(5)

drawing_area = Canvas(root, width=800, height=800)
drawing_area.pack()
drawing_area.bind("<Motion>", motion)
drawing_area.bind("<ButtonPress-1>", b1down)
drawing_area.bind("<ButtonRelease-1>", b1up)

im = Image.open('Graph_axis.png')
tkimage = ImageTk.PhotoImage(im)
drawing_area.create_image(10, 10, image=tkimage, anchor=NW)

root.mainloop()