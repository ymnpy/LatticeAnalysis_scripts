import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import numpy as np
from paretoset import paretoset
import os
from scipy.ndimage.filters import gaussian_filter
from sklearn.preprocessing import StandardScaler
sns.set_theme(style="ticks",font_scale=2.8)
%matplotlib auto
plt.rcParams["font.family"] = "Times New Roman"

""" Perform various Exploratory Data Analysis and visualization """

def standardize(df_in):
    standizer=StandardScaler()
    for col in df_in.columns:
        print(col)
        try: df_in[col]=standizer.fit_transform(df_in[[col]])
        except: pass
    
    return df_in

def get_describe(df):
    df.describe().to_excel("describe.xlsx")
    
def get_boxplot(df,cols,color="crimson"):
    df=df[cols]
    plt.figure(figsize=(18,12))
    for i in range(len(cols)):
        plt.subplot(2,4,i+1)
        sns.boxplot(data=df,y=df[cols[i]],color=color,medianprops=dict(color="orange", alpha=1, linewidth=4))
        plt.subplots_adjust(left=0.1,
                    bottom=0.1, 
                    right=0.9, 
                    top=0.9, 
                    wspace=0.6, 
                    hspace=0.2)
        if cols[i]=="SW useful (kJ.mm$^{3}$/kg)" and color=="steelblue":
            plt.yticks([0,1.0,2.0,3.0])
        elif cols[i]=="SW useful (kJ.mm$^{3}$/kg)" and color=="green":
            plt.yticks([0,1.0,2.0,3.0])
        else:
            plt.gca().yaxis.set_major_locator(MaxNLocator(nbins=5))
            
        if "_" in cols[i]:
            if cols[i].startswith("SW"):
                plt.ylabel("SW useful (kJ.mm$^{3}$/kg)")
            else:
                plt.ylabel(f"{cols[i].split('_')[0]} {cols[i].split('_')[1]}")
        else:
            plt.ylabel(f"{cols[i]}")
    
    # pylab.gca().yaxis.set_major_formatter
        
    # plt.tight_layout()
    print(df.shape[0])
    plt.savefig(f"plot_box_{df.shape[0]}.png")

def get_histplot(df,cols):
    df=df[cols]
    df.hist(figsize=(18,12))
    plt.savefig("plot_hist.png")
    
def get_matrix(df,cols,annot=False):
    df=df[cols]
    corr=df.corr()
    matrix=np.triu(corr)
    plt.figure(figsize=(16,12),constrained_layout=True)
    sns.heatmap(df.corr(),annot=annot,linewidth=0,
                fmt=".2f",mask=matrix,annot_kws={"fontsize":24},
                vmin=-0.75,vmax=0.75)
    # plt.tight_layout()
    # plt.xticks(rotation=60)
    # plt.yticks(rotation=75)
    # plt.tight_layout()
    plt.savefig(os.getcwd()+"\plot_matrix.png")
    
def get_regplot(df,cols,xx):
    df=df[cols]
    plt.figure(figsize=(18,12))
    for i in range(1,len(cols)):
        plt.subplot(1,len(xx),i)
        sns.regplot(data=df, x=xx, y=cols[i],ci=0,line_kws=dict(color="r"))
        plt.subplots_adjust(left=0.1,
                    bottom=0.1, 
                    right=0.9, 
                    top=0.9, 
                    wspace=0.5, 
                    hspace=0.2)
    plt.savefig("plot_reg.png")

def get_joint(df,cols):
    df=df[cols]
    # plt.figure(figsize=(18,12))
    g=sns.jointplot(data = df, x = cols[-1], y = cols[-2], hue=cols[0], sizes=cols[1], kind ="scatter")  
    g.plot_joint(sns.kdeplot, color="r", zorder=0, levels=6)
    g.fig.set_size_inches((16, 12))


def get_scatterplot_sns(df, xs, ys, hue, size):
    fig, axes = plt.subplots(len(xs), len(ys), figsize=(5 * len(ys), 5), sharey=True,sharex=True)
    
    for j, x in enumerate (xs):
        for i, y in enumerate(ys):
            print(i,j)
            g = sns.scatterplot(ax=axes[i*(j+1)], data=df, x=x, y=y, hue=hue, size=size, sizes=(20, 200))
            axes[i*(j+1)].set_xlabel(x)
            axes[i*(j+1)].set_ylabel(y)
            
            # Remove the individual legends
            g.legend_.remove()

    # Create a colorbar for the hue
    norm = plt.Normalize(df[hue].min(), df[hue].max())
    cmap = sns.cubehelix_palette(light=1, as_cmap=True)
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])

    # Position the colorbar to the right of all subplots
    cax = fig.add_axes([0.92, 0.15, 0.02, 0.7])
    fig.colorbar(sm, cax=cax)
    
    # Add a size legend based on the first subplot's size mapping
    handles, labels = axes[0].get_legend_handles_labels()
    size_legend = fig.legend(handles[1:], labels[1:], title=size, loc='center right', bbox_to_anchor=(1, 0.5), frameon=False)

    plt.tight_layout(rect=[0, 0, 0.9, 1])
    plt.show()



def get_scatterplot2(data,xs,y,hue,s=300,marker='o',palette="coolwarm"):
    fig, axes = plt.subplots(nrows=1, ncols=len(xs), figsize=(32, 7),sharex=False,sharey=True)
    
    # plt.subplots_adjust(left=0.1,
    #                 bottom=0.1, 
    #                 right=0.9, 
    #                 top=0.9, 
    #                 wspace=0.4, 
    #                 hspace=0.4)
                    
    for i, ax in enumerate(axes.flat):
        plt.subplots_adjust(left=0.08, bottom=0.2, right=0.95, top=0.95, wspace=0.4, hspace=0.4)
        
        points = ax.scatter(df[xs[i]], df[y], c=df[hue], s=s, cmap=palette, 
                            edgecolors="black", linewidths=0.2, marker=marker, label="sth")
        
        ax.set_xlabel(xs[i])  # Set x label for each subplot
        ax.set_ylabel(y)  # Set y label for each subplot
        
        if xs[i]=="t (mm)":
            ax.set_xticks([0.8,1.6,2.4,3.2])
        elif xs[i]=="py":
            # plt.xticks(np.arange(round(min(df[x])),round(max(df[x])+4),4))
            ax.set_xticks([0,5,10,15,20])
            # plt.xticks([4,6,8,10,12,16])
            # plt.xticks(np.linspace(round(min(df[x])),round(max(df[x])),4))
        elif xs[i]=="px":
            ax.set_xticks([0,5,10,15,20])
        
        else:
            # plt.xticks(np.arange(round(min(df[x])),round(max(df[x])),6))
            plt.gca().xaxis.set_major_locator(plt.MultipleLocator(2))
     
    cbar=fig.colorbar(points, ax=axes.ravel().tolist())
    cbar.set_label(hue,labelpad=10)
    
    plt.show()
    plt.savefig(f"scatter_plot_{y[0:3]}.png")
    
    
def get_scatterplot(df,x,ys,hue,palette="coolwarm",s=300,marker='o'):
    fig, axes = plt.subplots(nrows=1, ncols=len(ys), figsize=(32, 7),sharex=True,sharey=False)
    # fig, ax = plt.subplots(1, 3, figsize=(32, 9),tight_layout=False)
    # fig, ax = plt.subplots(1, 3, figsize=(27, 9), sharex=True, layout='constrained')
    # fig, ax = plt.subplots(1, 3, figsize=(27, 9), sharex=True)

    for i, ax in enumerate(axes.flat):
        plt.subplots_adjust(left=0.08, bottom=0.2, right=0.95, top=0.95, wspace=0.4, hspace=0.4)
        
        points = ax.scatter(df[x], df[ys[i]], c=df[hue], s=s, cmap=palette, 
                            edgecolors="black", linewidths=0.2, marker=marker, label="sth")
        
        ax.set_xlabel(x)  # Set x label for each subplot
        ax.set_ylabel(ys[i],labelpad=8)  # Set y label for each subplot
        
        # if x=="t (mm)":
        #     plt.xticks([0.8,1.6,2.4,3.2])
        # elif x=="py":
        #     # plt.xticks(np.arange(round(min(df[x])),round(max(df[x])+4),4))
        #     plt.gca().xaxis.set_major_locator(plt.MultipleLocator(4))
        #     # plt.xticks([4,6,8,10,12,16])
        #     # plt.xticks(np.linspace(round(min(df[x])),round(max(df[x])),4))
        # elif x=="px":
        #     plt.gca().xaxis.set_major_locator(plt.MultipleLocator(4))
        
        # else:
        #     # plt.xticks(np.arange(round(min(df[x])),round(max(df[x])),6))
        #     plt.gca().xaxis.set_major_locator(plt.MultipleLocator(4))
        
        if x=="t (mm)":
            ax.set_xticks([0.8,1.6,2.4,3.2])
            # plt.gca().xaxis.set_major_locator(plt.MultipleLocator(3))
        elif x=="py":
            # plt.xticks(np.arange(round(min(df[x])),round(max(df[x])+4),4))
            # ax.set_xticks([0,5,10,15,20])
            ax.set_xticks([2,4,6,8,10,12])
            # plt.xticks(np.linspace(round(min(df[x])),round(max(df[x])),4))
        elif x=="px":
            ax.set_xticks([4,6,8,10,12,14,16])
        
        else:
            pass
            # plt.xticks(np.arange(round(min(df[x])),round(max(df[x])),6))
            # plt.gca().xaxis.set_major_locator(plt.MultipleLocator(3))
            
        if ys[i].startswith("Pla"):
            ax.set_yticks([0,25,50,75,100,125])
        elif ys[i].startswith("W"):
            ax.set_yticks([0,12,24,36,48,60])
        else:
            ax.set_yticks([20,25,30,35,40,45])
        # if i+1==len(ys):
        #     # # plt.legend(title='SR_avg',loc="upper left",bbox_to_anchor=(1.1,1.0),ncols=1,
        #     # #                         fancybox=True,shadow=True)
        #     # norm = plt.Normalize(df[hue].min(), df[hue].max())
        #     # sm = plt.cm.ScalarMappable(cmap="RdBu", norm=norm)
        #     # sm.set_array([])
        #     # # Remove the legend and add a colorbar
        #     # ax.get_legend().remove()
        #     # ax.figure.colorbar(sm)
        #     import matplotlib.ticker as tkr
        #     cbar=plt.colorbar(points,ax=ax.ravel().tolist(),format=tkr.FormatStrFormatter('%.1f'))
        #     cbar.set_label(label=f"${hue}$",rotation=90,labelpad=20,color="black")
        #     # cbar.set_label(label="$SR_{avg}$",rotation=90,labelpad=25,color="gray")
        # else:
        #     plt.legend('',frameon=False)
    
    
    # cbar = ax.cax.colorbar(points)
    # cbar = grid.cbar_axes[0].colorbar(points)
    
    cbar=fig.colorbar(points, ax=axes.ravel().tolist())
    cbar.set_label(hue,labelpad=10)
    if hue=="t (mm)":
        cbar.set_ticks([0.8,1.6,2.4,3.2])
        cbar.set_ticklabels([0.8,1.6,2.4,3.2])
        # cbar.set_ticks([1.0,1.6,2.2,3.0])
        # cbar.set_ticklabels([0.8,1.6,2.4,3.2])
    else:
        cbar.set_ticks([6,12,18,24])
        cbar.set_ticks([6,12,18,24])
    # # plt.subplots_adjust(left=0.1, right=0.95, top=0.95, bottom=0.1, wspace=0.3, hspace=0.4)
    # plt.show()
    sns.despine()
    # plt.tight_layout()
    plt.savefig(f"scatter_plot_{x}2.png")

def get_paretoplot(df,cols,sense,hue):
    """
    columns (list) -> objectives
    sense (list) -> min,max etc. respectively
    df (dataframe)
    """
    # x="px"
    # y="py"
    
    plt.figure(figsize=(12,9))
    points=plt.scatter(df[cols[0]],df[cols[1]],c=df[hue],s=100,cmap="viridis",edgecolors="black",linewidths=0.4)
    cbar=plt.colorbar(points,pad=0.05)
    cbar.set_label(label=hue,rotation=90,labelpad=10)
    
    mask=paretoset(df[cols],sense=sense)
    df_pareto=df[mask]
    sns.scatterplot(df_pareto,x=cols[0],y=cols[1],s=200,zorder=20,color="crimson",marker='o',label="Pareto Set")
    sns.regplot(df_pareto,x=cols[0],y=cols[1],order=2,ci=None,
                scatter=False, truncate=False,
                line_kws={'color':'crimson','lw':4},label="Pareto Fit")
    
    
    plt.legend(shadow=True)
    plt.xticks([0,20,40,60,80,100,120])
    plt.yticks([0,10,20,30,40,50,60])

    plt.xlabel(cols[0],labelpad=10)
    plt.ylabel(cols[1],labelpad=10)
        
    plt_name="_vs_".join([str(i) for i in cols])
    plt.savefig(f"Pareto-{plt_name}.png")   
    
    return None


def remove_outliers(df,columns) -> 'dataframe':
    """columns -> list
       df -> data frame
       
       takes columns, loop them through remove outliers based on
       upper and lower quartile values, a proven method
    """
       
    for col in columns:
        Q1=np.percentile(df[col],25,method='midpoint')
        Q3=np.percentile(df[col],75,method='midpoint')
        IQR=Q3-Q1
        upper=Q3+1.5*IQR
        lower=Q1-1.5*IQR
        df=df[(df[col]>=lower) & (df[col]<=upper)]
        
    return df

"""-------------------------------------------------------------------------------------------                          
Index(['P1-X', 'P1-Y', 'P2-X', 'P2-Y', 'P3-X', 'P3-Y', 'P4-X', 'P4-Y', 'L1',
       'L2', 'L3', 'L4', 'HEIGHT', 'WIDTH', 'THICKNESS', 'THETA1', 'THETA2',
       'SVF', 'RO_EFF', 'MASS', 'E (GPa)', 'EA (J)', 'SEA (J/kg)',
       'Plateau Stress (MPa)', 'Yield Stress (MPa)', 'Yield Strain',
       'Densification Stress (MPa)', 'Densification Strain', 'n (%)'],
      dtype='object')
"""

"""PREP"""
df1=pd.read_excel(os.getcwd()+"\data.xlsx")
df2=pd.read_excel(os.getcwd()+"\Mechanical_props.xlsx")
df=pd.concat([df1,df2],axis=1)

df = df.loc[:,~df.columns.duplicated()].copy()

rows,cols=df.shape
p1x,p1y=[],[]
p2x,p2y=[],[]
p3x,p3y=[],[]
p4x,p4y=[],[]
for r in range(rows):
    r1x,r1y=float(df["P1"].str.split(",")[r][0].strip("[")),float(df["P1"].str.split(",")[r][1].strip("]"))
    r2x,r2y=float(df["P2"].str.split(",")[r][0].strip("[")),float(df["P2"].str.split(",")[r][1].strip("]"))
    r3x,r3y=float(df["P3"].str.split(",")[r][0].strip("[")),float(df["P3"].str.split(",")[r][1].strip("]"))
    r4x,r4y=float(df["P4"].str.split(",")[r][0].strip("[")),float(df["P4"].str.split(",")[r][1].strip("]"))
    
    p1x.append(r1x), p1y.append(r1y)
    p2x.append(r2x), p2y.append(r2y)
    p3x.append(r3x), p3y.append(r3y)
    p4x.append(r4x), p4y.append(r4y)

df.insert(1,"P4-Y",p4y), df.insert(1,"P4-X",p4x)
df.insert(1,"P3-Y",p3y), df.insert(1,"P3-X",p3x)
df.insert(1,"P2-Y",p2y), df.insert(1,"P2-X",p2x)
df.insert(1,"P1-Y",p1y), df.insert(1,"P1-X",p1x)

df.to_excel("THE_DATA_9000.xlsx")


"""
'MODEL_NO', 'P1-X', 'P1-Y', 'P2-X', 'P2-Y', 'P3-X', 'P3-Y', 'P4-X',
       'P4-Y', 'P1', 'P2', 'P3', 'P4', 'L1', 'L2', 'L3', 'L4', 'HEIGHT',
       'WIDTH', 'THICKNESS', 'THETA1', 'THETA2', 'SVF', 'RO_EFF', 'MASS',
       'E (GPa)', 'EA (J)', 'SEA (J/kg)', 'Plateau Stress (MPa)',
       'Yield Stress (MPa)', 'Yield Strain', 'Densification Stress (MPa)',
       'Densification Strain', 'n (%)'
"""

"""
Index(['P1-X', 'P1-Y', 'P2-X', 'P2-Y', 'P3-X', 'P3-Y', 'P4-X', 'P4-Y', 'L1',
       'L2', 'L3', 'L4', 'HEIGHT', 'WIDTH', 'THICKNESS', 'THETA1', 'THETA2',
       'SVF', 'RO_EFF', 'MASS (grams)', 'E (MPa)', 'EA (J)', 'W_useful (J)',
       'SW_useful (J.mm3/kg)', 'Plateau Stress (MPa)', 'Yield Stress (MPa)',
       'Yield Strain', 'Dens. Stress (MPa)', 'Densification Strain', 'n (%)',
       'SR_avg'],
"""    

try:
    drops=['MODEL_NO',"Model_name",'P1', 'P2', 'P3', 'P4']
    df.drop(drops,inplace=True,axis=1)
except:
    drops=['MODEL_NO','P1', 'P2', 'P3', 'P4']
    df.drop(drops,inplace=True,axis=1)


df["SR_avg"]=(df["L1"]+df["L2"]+df["L3"]+df["L4"])/(4*df["THICKNESS"])

df=df.astype(float)
df["w_useful"]=df["w_useful"]/1e3
# df['E (MPa)']=df['E (MPa)']/1e3

df.rename(columns = {'P4-X':'px', 'P4-Y':'py','THICKNESS':'t (mm)',
                      'E (MPa)':'E (MPa)',
                      'Densification Stress (MPa)':'Dens. Stress (MPa)',
                      'Densification Strain':'Dens. Strain (mm/mm)', 
                      'MASS':'MASS (grams)',
                      'Yield Strain':'Yield Strain (mm/mm)',
                      'W_useful':'W useful (J)', 'SR_avg':'SR avg',
                      'w_useful':'SW useful (J.mm$^{3}$/kg)'}, inplace = True)


# # df["X/Y"]=df["P3-X"]/df["P4-Y"]
odbs=['MASS (grams)', 'E (MPa)', 'EA (J)', 'W useful (J)',
'SW useful (J.mm$^{3}$/kg)', 'Plateau Stress (MPa)', 'Yield Stress (MPa)',
'Dens. Stress (MPa)', 'Yield Strain (mm/mm)', 'Dens. Strain (mm/mm)', 'n (%)',
'SR avg']

# cols_mini=['E (GPa)', 'EA (J)', 'W useful (J)',
# 'SW useful (J.mm$^{3}$/kg)', 'Plateau Stress (MPa)', 'Yield Stress (MPa)', 
#   'Dens. Stress (MPa)', 'n (%)']

# cols_mini=['E (GPa)', 'EA (J)', 'W useful (J)',
# 'SW useful (J.mm$^{3}$/kg)', 'Plateau Stress (MPa)', 'Yield Stress (MPa)', 
# 'Yield Strain (mm/mm)', 'Dens. Stress (MPa)', 'Dens. Strain (mm/mm)', 'n (%)']

# print(df.shape)
# plt.legend(ncol=2)
# get_boxplot(df, cols_mini,color="steelblue")

df["px'"]=(abs(10-df['px']))
# df["p vector"]=((10-df['px'])**2+df['py']**2)**0.5
cols_mini=["px'","px",'py','t (mm)']+odbs
# df["px'"]=(abs(10-df['px'])**2+df['py']**2)**0.5
get_matrix(df,cols_mini,annot=True)
# odbs_outliers=['E (GPa)', 'EA (J)', 'W useful (J)']
# df=remove_outliers(df, odbs_outliers)
# # df_in=df.copy()
# df_standard=standardize(df_in)


""" on the road stuff"""
# plt.figure(figsize=(18,12))
# for i,col in enumerate(cols_mini):
#     plt.subplot(2,4,i+1)
#     sns.kdeplot(df[col], color="steelblue", shade=True)
#     plt.ylabel(col)
#     plt.yticks([])
#     plt.xlabel("")
#     # plt.gca().xaxis.set_major_locator(plt.MultipleLocator(2.5))
#     plt.subplots_adjust(left=0.1,
#                 bottom=0.1, 
#                 right=0.9, 
#                 top=0.9, 
#                 wspace=0.25, 
#                 hspace=0.2)
    
#     # plt.gca().xaxis.set_major_locator(MaxNLocator(nbins=3))
#     if i==0: plt.gca().xaxis.set_major_locator(plt.MultipleLocator(1))
#     elif i==1: plt.gca().xaxis.set_major_locator(plt.MultipleLocator(60))
#     elif i==2: plt.gca().xaxis.set_major_locator(plt.MultipleLocator(15))
#     elif i==3: plt.gca().xaxis.set_major_locator(plt.MultipleLocator(1000))
#     elif i==4: plt.gca().xaxis.set_major_locator(plt.MultipleLocator(30))
#     elif i==5: plt.gca().xaxis.set_major_locator(plt.MultipleLocator(30))
#     elif i==6: plt.gca().xaxis.set_major_locator(plt.MultipleLocator(60))
#     elif i==7: plt.gca().xaxis.set_major_locator(plt.MultipleLocator(10))
# plt.savefig(os.getcwd()+"\\before_scale.png")

# plt.figure(figsize=(18,12))
# for i,col in enumerate(cols_mini):
#     plt.subplot(2,4,i+1)
#     sns.kdeplot(df_standard[col], color="green", shade=True)
#     plt.ylabel(col)
#     plt.yticks([])
#     plt.xlabel("")
#     plt.gca().xaxis.set_major_locator(plt.MultipleLocator(2))
#     plt.subplots_adjust(left=0.1,
#                 bottom=0.1, 
#                 right=0.9, 
#                 top=0.9, 
#                 wspace=0.25, 
#                 hspace=0.2)
# plt.savefig(os.getcwd()+"\\after_scale.png")
    
# sns.set_theme(style="ticks",font_scale=3)
# outliers=['Plateau Stress (MPa)','W useful (J)','n (%)']
# df=df[outliers]

# plt.subplots_adjust(wspace=0.3)
# plt.subplot(121)
# sns.histplot(data=df, x='Plateau Stress (MPa)',color="blue",
#               kde=True,line_kws={'linewidth': 2})
# plt.xticks(([20,40,60,80]))

# plt.subplot(122)
# sns.histplot(data=df, x='W useful (J)',color="green",
#               kde=True,line_kws={'linewidth': 2})
# plt.xticks(([10,20,30,40]))

# cols_mini=['t (mm)','px','py','SR avg','E (MPa)', 'EA (J)', 'W useful (J)',
# 'SW useful (J.mm$^{3}$/kg)', 'Plateau Stress (MPa)', 'Yield Stress (MPa)', 
# 'Dens. Stress (MPa)', 'n (%)']


# get_boxplot(df, cols_mini,color="green")

# 

# cols=["px","py","t (mm)","Yield Stress (MPa)",'SW useful (J.mm$^{3}$/kg)']
# get_matrix(df,cols)
# get_paretoplot(df, ["Plateau Stress (MPa)",'W useful (J)','MASS (grams)'], ['min','max','min'],'MASS (grams)')

# plt.scatter(data=df,x="Plateau Stress (MPa)",y='W useful (J)',c='MASS (grams)',cmap="viridis")
# get_scatterplot(df, ["px","px","py"], ["SW useful (J.mm$^{3}$/kg)","SW useful (J.mm$^{3}$/kg)","SW useful (J.mm$^{3}$/kg)"],"SR avg")
# get_boxplot(df, cols_mini,color="steelblue")
# print(df.shape)
# df_before10=df[df['px']<10]
# df_after10=df[df['px']>10]

# plt.scatter(df_before10['px'],df_before10['n (%)'],color="steelblue")
# plt.scatter(df_after10['px'],df_after10['n (%)'],color="crimson")
# sns.scatterplot(df,x="px'",y="n (%)",hue="t (mm)",size="t (mm)",
                # palette="rainbow",sizes=(40,400))

# plt.legend(ncols=2)
# df=df[(df["Plateau Stress (MPa)"]<500) & (df["W_useful"]<250) & (df['Densification Stress (MPa)']<410)] #outlier removal
# get_scatterplot(df,"X/Y",['Plateau Stress (MPa)','W_useful','n (%)'],"SR_avg")
# get_matrix(df, cols_mini)

# sns.scatterplot(df,x="EA (J)",y="Plateau Stress (MPa)",hue="py", size="n (%)",palette="icefire", 
#                 alpha=0.8, sizes=(20,400),legend="brief")

# sns.scatterplot(df,x="EA (J)",y="py",hue="n (%)", size="Plateau Stress (MPa)",palette="icefire", 
#                 alpha=0.8, sizes=(20,400),legend="brief")

# sns.scatterplot(df,x="EA (J)",y="n (%)",hue="py", size="Plateau Stress (MPa)",palette="icefire", 
#                 alpha=0.8, sizes=(20,400),legend="brief")

# sns.scatterplot(df,x="EA (J)",y="Plateau Stress (MPa)",hue="py", size="n (%)",palette="icefire", 
#                 alpha=0.8, sizes=(20,300),legend="brief")
# plt.xlabel("EA (J)",labelpad=20)
# plt.ylabel("Plateau Stress (MPa)",labelpad=10)
# plt.legend(frameon=False,ncol=2,markerscale=3,loc="lower right", bbox_to_anchor=(1.08,0.01))
# sns.despine()

# get_scatterplot(df, "t (mm)", ["Plateau Stress (MPa)","W useful (J)",'n (%)'], "SR avg")
# get_scatterplot(df, "px", ["Plateau Stress (MPa)","W useful (J)",'n (%)'], "t (mm)",marker='p')
# get_scatterplot(df, "py", ["Plateau Stress (MPa)","W useful (J)",'n (%)'], "t (mm)",marker='s')
# get_scatterplot(df, "py", ["Plateau Stress (MPa)","W useful (J)",'n (%)'], "t (mm)",marker='p')
# get_scatterplot2(df, ["px","py","t (mm)"],'Dens. Strain',"SR avg")
# get_scatterplot2(df, ["px","py","t (mm)"],'Yield Strain',"SR avg")
# plt.subplot(211)
# sns.regplot(df,x="Plateau Stress (MPa)",y="py")
# plt.subplot(212)
# sns.regplot(df,x="Plateau Stress (MPa)",y="py")

# get_matrix(df, odbs+["px","py","t (mm)"],annot=True)