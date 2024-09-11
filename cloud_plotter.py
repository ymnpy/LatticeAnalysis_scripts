import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from math import atan,degrees
import random
%matplotlib inline
sns.set_theme(style="ticks",font_scale=3.6)
plt.rcParams["font.family"] = "Times New Roman"

"""this one takes the prepared .xlsx file
and plot the design cloud"""

def get_points(df,deneme=False):
    
    points={"p1x":[],"p1y":[],\
            "p2x":[],"p2y":[],\
            "p3x":[],"p3y":[],\
            "p4x":[],"p4y":[],\
            "t":[]}
        
    if deneme==False:
        """cloud with the actual results"""
        rows,columns=df.shape
        
        for r in range(rows):
            r1x=float(df["P1"].str.split(",")[r][0].strip("["))
            r2x=float(df["P2"].str.split(",")[r][0].strip("["))
            r3x=float(df["P3"].str.split(",")[r][0].strip("["))
            r4x=float(df["P4"].str.split(",")[r][0].strip("["))
				
            r1y=float(df["P1"].str.split(",")[r][1].strip("]"))
            r2y=float(df["P2"].str.split(",")[r][1].strip("]"))
            r3y=float(df["P3"].str.split(",")[r][1].strip("]"))
            r4y=float(df["P4"].str.split(",")[r][1].strip("]"))
            
            df["L1"][r]=((r3x-r1x)**2+(r3y-r1y)**2)**(1/2)
            df["L2"][r]=((r3x-r2x)**2+(r3y-r2x)**2)**(1/2)
            df["L3"][r]=((r4x-r2x)**2+(r4y-r2y)**2)**(1/2)
            df["L4"][r]=((r1x-r4x)**2+(r1y-r4x)**2)**(1/2)
            
            t=df["THICKNESS"][r]
            
            points["p1x"].append(r1x)
            points["p1y"].append(r1y)
            points["p2x"].append(r2x)
            points["p2y"].append(r2y)
            points["p3x"].append(r3x)
            points["p3y"].append(r3y)
            points["p4x"].append(r4x)
            points["p4y"].append(r4y)
            points["t"].append(t)
    
    else:
        """cloud without the results, just for a showcase"""
        storage=[]
        count=0
        model_no=100    
        thickness_list=[0.8,1.2,1.6]
        
        while count<model_no:
            r1x,r1y=0,0
            r2x,r2y=20,0
            r3x,r3y=random.randint(0,20),20
            r4x,r4y=r3x,random.randint(0,20)
            thickness=random.choice(thickness_list)
            
            a13=(r3y-r1y)/(r3x-r1x+0.001)
            a23=-(r3y-r2y)/(r3x-r2x+0.001)
            a14=(r4y-r1y)/(r4x-r1x+0.001)
            a24=-(r4y-r2y)/(r4x-r2x+0.001)
            
            #lengths of a strut
            l1=((r3x-r1x)**2+(r3y-r1y)**2)**(0.5)
            l2=((r3x-r2x)**2+(r3y-r2y)**2)**(0.5)
            l3=((r4x-r2x)**2+(r4y-r2y)**2)**(0.5)
            l4=((r1x-r4x)**2+(r1y-r4y)**2)**(0.5)
            
            height=(r3y-r4y)*8+r4y
            width=r2x*8
            solid_vol=height*width
            lattice_vol=(l1+l2+l3+l4)*thickness*8**2 # 8x8 cell var
            svf=lattice_vol/solid_vol
            
            if not (degrees(atan(a13))<85 and degrees(atan(a23))<85 \
            and degrees(atan(a14))>5 and degrees(atan(a24))>5 \
            and degrees(atan(a13))>degrees(atan(a14))+5 \
            and degrees(atan(a23))>degrees(atan(a24))+5 \
            and a13>a14 and a23>a24 \
            and [r1x,r1y,r2x,r2y,r3x,r3y,r4x,r4y] not in storage \
            and 0.15<=svf<=0.45):
                continue
            
            count+=1
            
            points["p1x"].append(r1x)
            points["p1y"].append(r1y)
            points["p2x"].append(r2x)
            points["p2y"].append(r2y)
            points["p3x"].append(r3x)
            points["p3y"].append(r3y)
            points["p4x"].append(r4x)
            points["p4y"].append(r4y)
    
    return points

def plot_scattercloud(df):
    """ THIS ONE IS OLD """
    """L1 represents p1-p3 truss length
       L2 represents p1-p4 truss length
       THETA2 represents p1-p4-p2 angle"""
    
    df["SLEND1"]=df["L1"]/df["THICKNESS"]
    df["SLEND2"]=df["L3"]/df["THICKNESS"]
    # c=sns.color_palette("cubehelix", as_cmap=True)
    plt.figure(figsize=(24,8))
    plt.tight_layout()
    g=sns.scatterplot(data=df,x="SLEND2",y="SLEND1",hue="THETA2", size="THETA1",
                      palette="coolwarm",legend="auto",sizes=(100,300))

    # g.legend(ncol=2)
    sns.move_legend(g, "upper left", bbox_to_anchor=(1, 1))
    plt.savefig("scatter_cloud.png")
    
    return None


def plot_linecloud(points,dah_number=5):
    colors=sns.color_palette("pastel")
    
    df_points=pd.DataFrame(points)
    rows,columns=df_points.shape
    print(df_points.columns)
    color="royalblue"
    linewidth=1
    linestyle="-"
    
    plt.figure(figsize=(12,9))
    
    #pointler ve boundaryleri
    for r in range(rows):
        # color="saddlebrown"
        plt.scatter(df_points["p1x"][r],df_points["p1y"][r],color="gray",s=400,edgecolors="black",linewidths=0.4,zorder=20,marker="^")
        plt.scatter(df_points["p2x"][r],df_points["p2y"][r],color="gray",s=400,edgecolors="black",linewidths=0.4,zorder=20,marker="^")
        plt.scatter(df_points["p3x"][r],df_points["p3y"][r],color="crimson",s=150,edgecolors="black",linewidths=0.4,zorder=20)
        plt.scatter(df_points["p4x"][r],df_points["p4y"][r],color="orange",s=150,edgecolors="black",linewidths=0.4,zorder=20)
    
    for r in range(dah_number):
        if r==0 or r==4:
            t=df_points["t"][r]*2
            # linewidth=0.5
            # linewidth=linewidth-0.1*r
            # plt.plot([df_points["p1x"][r],df_points["p3x"][r]],[df_points["p1y"][r],df_points["p3y"][r]],color=colors[r],linewidth=linewidth,linestyle=linestyle)
            plt.plot([df_points["p1x"][r],df_points["p3x"][r]],[df_points["p1y"][r],df_points["p3y"][r]],color=color,linewidth=t,linestyle=linestyle)
            plt.plot([df_points["p3x"][r],df_points["p2x"][r]],[df_points["p3y"][r],df_points["p2y"][r]],color=color,linewidth=t,linestyle=linestyle)
            plt.plot([df_points["p2x"][r],df_points["p4x"][r]],[df_points["p2y"][r],df_points["p4y"][r]],color=color,linewidth=t,linestyle=linestyle)
            plt.plot([df_points["p4x"][r],df_points["p1x"][r]],[df_points["p4y"][r],df_points["p1y"][r]],color=color,linewidth=t,linestyle=linestyle)
            # plt.grid()
            plt.xlabel("px",labelpad=6)
            plt.ylabel("py",labelpad=10)
            plt.xticks([0,5,10,15,20])
            plt.yticks([0,5,10,15,20])
        
    # plt.axis("off")
    plt.tight_layout(h_pad=-0.2)
    plt.savefig("line_cloud.png")
    plt.show()
    return None

def plot_pointspace(points):
    rows,columns=df.shape
    sns.set_theme(style="ticks",font_scale=2.0)
    # df_points=pd.DataFrame(points)
    # rows,columns=df_points.shape
    # plt.figure(figsize=(12,8))
    sns.jointplot(data=points,x="p3x",y="p4y",hue="t",s=200,palette="coolwarm",height=12)
    plt.xlabel("X variable (mm)")
    plt.ylabel("Y variable (mm)")
    # plt.title(f"Number of models/points = {rows}",fontsize=24)
    plt.savefig("scatter_cloud.png")
    
    
path=r"data.xlsx"
df=pd.read_excel(path)
# df=df.head(10)
points=get_points(df,deneme=False)
# plot_scattercloud(df)
# plot_pointspace(points)
plot_linecloud(points)

