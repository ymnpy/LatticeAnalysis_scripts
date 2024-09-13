import matplotlib.pyplot as plt
import seaborn as sns
import random
import time
import numpy as np
import pandas as pd
%matplotlib inline
sns.set_theme(style="ticks",font_scale=1.5)

"""PLOTS 2D REPRESENTATION OF THE GENERATED LATTICE STRUCTURES"""

def get_image(model,t,c="saddlebrown"):
    """ OLD BUT GOLD ? NO """
    hor_x=-hor_x
    ver_y=ver_y
    linewidth=thickness*2
    # offset=0
    
    plt.figure(figsize=(12,8))
    for j in range(cells):
        for i in range(cells):
            plt.plot((p1[0]+hor_x*i,p3[0]+hor_x*i),(p1[1]+ver_y*j,p3[1]+ver_y*j),color=color,linewidth=linewidth)
            plt.plot((p3[0]+hor_x*i,p2[0]+hor_x*i),(p3[1]+ver_y*j,p2[1]+ver_y*j),color=color,linewidth=linewidth)
            plt.plot((p2[0]+hor_x*i,p4[0]+hor_x*i),(p2[1]+ver_y*j,p4[1]+ver_y*j),color=color,linewidth=linewidth)
            plt.plot((p4[0]+hor_x*i,p1[0]+hor_x*i),(p4[1]+ver_y*j,p1[1]+ver_y*j),color=color,linewidth=linewidth)
    plt.plot((p1[0]-12,p2[0]+hor_x*i+12),(0,0),color=color,linewidth=linewidth)
    plt.plot((p1[0]-12,p2[0]+hor_x*i+12),(p3[1]+ver_y*j,p3[1]+ver_y*j),color=color,linewidth=linewidth)
    plt.title(f"Model Name: {model} | Thickness = {thickness} mm")
    plt.xlabel("Width (mm)")
    plt.ylabel("Height (mm)")
    plt.savefig("2D-"+model+".png")
    
    w=p2[0]-p1[0]
    h=p3[1]-p4[1]
    sym=0
    cellx=6
    celly=8
    
    plt.figure(figsize=(12,8))
    for y in range(celly):
        for x in range(cellx):
            if x==0:
                sym=0
            elif x%2==0:
                sym+=2*p3[0]
            else:
                sym+=2*(p2[0]-p3[0])
            
            plt.plot((p1[0]+w*x,p3[0]+sym),(p1[1]+h*y,p3[1]+h*y),color=c,linewidth=t*2)
            plt.plot((p3[0]+sym,p2[0]+w*x),(p3[1]+h*y,p2[1]+h*y),color=c,linewidth=t*2)
            plt.plot((p2[0]+w*x,p4[0]+sym),(p2[1]+h*y,p4[1]+h*y),color=c,linewidth=t*2)
            plt.plot((p4[0]+sym,p1[0]+w*x),(p4[1]+h*y,p1[1]+h*y),color=c,linewidth=t*2)

    plt.plot((p1[0]-12,cellx*w+12),(p1[1],p2[1]),color=c,linewidth=t*2)
    plt.plot((p1[0]-12,cellx*w+12),(p1[1]+celly*h+p4[1],p2[1]+celly*h+p4[1]),color=c,linewidth=t*2)
    
    plt.title(f"Model Name: {model} | Thickness = {thickness} mm")
    plt.xlabel("Width (mm)")
    plt.ylabel("Height (mm)")
    plt.savefig("2D-"+model+".png")
    
    return None

df=pd.read_excel("data.xlsx")
rows,columns=df.shape

for r in range(rows):
    model=df["MODEL_NO"][r]
    
    p1=[int(i) for i in df["P1"][r].strip("[]").split(",")]
    p2=[int(i) for i in df["P2"][r].strip("[]").split(",")]
    p3=[int(i) for i in df["P3"][r].strip("[]").split(",")]
    p4=[int(i) for i in df["P4"][r].strip("[]").split(",")]
    
    hor_x=float(df["hor_x"][r])
    ver_y=float(df["ver_y"][r])
    thickness=float(df["THICKNESS"][r])
    
    get_image(model,thickness)
