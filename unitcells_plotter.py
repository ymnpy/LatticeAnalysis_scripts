import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from math import tan,atan,degrees,radians
sns.set_theme(style="ticks",font_scale=1.5)

"""PLOTS 2D REPRESENTATION OF THE GENERATED UNIT-CELLS"""

def find_intersection(p1,p2,p3,p4):
    px=((p1[0]*p2[1]-p1[1]*p2[0])*(p3[0]-p4[0])-(p1[0]-p2[0])*(p3[0]*p4[1]-p3[1]*p4[0])) \
        /((p1[0]-p2[0])*(p3[1]-p4[1])-(p1[1]-p2[1])*(p3[0]-p4[0]))
    py=((p1[0]*p2[1]-p1[1]*p2[0])*(p3[1]-p4[1])-(p1[1]-p2[1])*(p3[0]*p4[1]-p3[1]*p4[0])) \
        /((p1[0]-p2[0])*(p3[1]-p4[1])-(p1[1]-p2[1])*(p3[0]-p4[0]))
    return [px,py]

def plot_line(n1,n2,t,c="saddlebrown"):
    plt.plot((n1[0],n2[0]),(n1[1],n2[1]),color=c,linewidth=t)

t1_top,t1_bot=[0,20],[0,0]
t2_top,t2_bot=[20,20],[20,0]

storage={"model_name":[],"svf_approx":[],"svf_exact":[]}

with open("data.log","r") as fin:
    for i,line in enumerate(fin.readlines()):
        if i==0:
            cols=[i.rstrip() for i in line.split(",")]
            
        elif "time" in line:
            pass
        else:
            model_name=line.split()[0]
            p1=[int(i.strip("(),'")) for i in line.split()[1:3]]
            p2=[int(i.strip("(),'")) for i in line.split()[3:5]]
            p3=[int(i.strip("(),'")) for i in line.split()[5:7]]
            p4=[int(i.strip("(),'")) for i in line.split()[7:9]]
            
            plt.figure(figsize=(8,6))
            
            plt.axhline(y=0,linestyle="-.",color="black",linewidth=0.5)
            plt.axhline(y=20,linestyle="-.",color="black",linewidth=0.5)
            plt.axvline(x=0,linestyle="-.",color="black",linewidth=0.5)
            plt.axvline(x=20,linestyle="-.",color="black",linewidth=0.5)
            
            t=float(line.split()[15])
            model_name=line.split()[0]
            
            plot_line(p1,p3,t)
            plot_line(p3,p2,t)
            plot_line(p2,p4,t)
            plot_line(p4,p1,t)
            
            offset=p3[1]-p4[1]
            
            p1_top,p1_bot=[p1[0],p1[1]+offset],[p1[0],p1[1]-offset]
            p2_top,p2_bot=[p2[0],p2[1]+offset],[p2[0],p2[1]-offset]
            p3_top,p3_bot=[p3[0],p3[1]+offset],[p3[0],p3[1]-offset]
            p4_top,p4_bot=[p4[0],p4[1]+offset],[p4[0],p4[1]-offset]
            
            pp=find_intersection(p1_top,p3_top,t1_top,t2_top)
            plot_line(p1_top,pp,t)
            l5=((pp[0]-p1_top[0])**2+(pp[1]-p1_top[1])**2)**0.5
            
            pp=find_intersection(p1_top,p4_top,t1_top,t2_top)
            plot_line(p1_top,pp,t)
            l6=((pp[0]-p1_top[0])**2+(pp[1]-p1_top[1])**2)**0.5
            
            pp=find_intersection(p2_top,p3_top,t1_top,t2_top)
            plot_line(p2_top,pp,t)
            l7=((pp[0]-p2_top[0])**2+(pp[1]-p2_top[1])**2)**0.5
            
            pp=find_intersection(p2_top,p4_top,t1_top,t2_top)
            plot_line(p2_top,pp,t)
            l8=((pp[0]-p2_top[0])**2+(pp[1]-p2_top[1])**2)**0.5

            pp=find_intersection(p1_bot,p3_bot,t1_bot,t2_bot)
            plot_line(p4,pp,t)
            l9=((pp[0]-p4[0])**2+(pp[1]-p4[1])**2)**0.5
            
            pp=find_intersection(p2_bot,p3_bot,t1_bot,t2_bot)
            plot_line(p4,pp,t)
            l10=((pp[0]-p4[0])**2+(pp[1]-p4[1])**2)**0.5
            
            
            l1=((p3[0]-p1[0])**2+(p3[1]-p1[1])**2)**0.5
            l2=((p2[0]-p3[0])**2+(p2[1]-p3[1])**2)**0.5
            l3=((p4[0]-p2[0])**2+(p4[1]-p2[1])**2)**0.5
            l4=((p1[0]-p4[0])**2+(p1[1]-p4[1])**2)**0.5
            
            
            svf_exact=(l1+l2+l3+l4+l5+l6+l7+l8+l9+l10)*t/400
            svf_approx=(l1+l2+l3+l4)*t/400
            storage["svf_approx"].append(svf_approx)
            storage["svf_exact"].append(svf_exact)
            storage["model_name"].append(model_name)
            
            plt.savefig(f"{model_name}_cell.png")
            plt.show()

pd.DataFrame(storage).to_excel("yo.xlsx")