import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from scipy.interpolate import make_interp_spline
import os
import re
%matplotlib inline
sns.set_theme(style="ticks",font_scale=3)
colors=sns.color_palette()
path=os.getcwd()

"""The main script for reading the raw data and performan calculations on it,
it also extracts the force-displacement and stress-strain plots"""

def datalog_read():
    parsed=[]
    with open("data.log","r") as fin:
        for i,line in enumerate(fin.readlines()):
            
            if i==0:
                cols=[i.rstrip() for i in line.split(",")]
                
            elif "time" in line:
                pass
            
            else:
                model_name=line.split()[0]
                p1=[float(i.strip("(),'")) for i in line.split()[1:3]]
                p2=[float(i.strip("(),'")) for i in line.split()[3:5]]
                p3=[float(i.strip("(),'")) for i in line.split()[5:7]]
                p4=[float(i.strip("(),'")) for i in line.split()[7:9]]
                line_list=[i.rstrip() for i in line.split(" ")[9::]]
                line_list.insert(0,p4)
                line_list.insert(0,p3)
                line_list.insert(0,p2)
                line_list.insert(0,p1)
                line_list.insert(0,model_name)
                parsed.append(pd.DataFrame(line_list,index=cols).transpose())
    
    df=pd.concat(parsed,ignore_index=True)
    df.to_excel("data.xlsx",index=False)
    return None

def sorted_alphanumeric(data):
    convert = lambda text: int(text) if text.isdigit() else text.lower()
    alphanum_key = lambda key: [ convert(c) for c in re.split('([0-9]+)', key) ] 
    return sorted(data, key=alphanum_key)

def get_props(strains,stresses):
    """ilk 51 data içinde max stress alıp
    burası linear bölgedir diyerek E çekiyorum"""
    idx_yield=stresses.index(max(stresses[:51]))
    yield_stress=stresses[idx_yield]
    yield_strain=strains[idx_yield]
    E=(yield_stress/yield_strain)
    
    
    """Densification strain based on energy absorption graph and
        Work or energy absorption calculation"""
    W=0
    max_stress=1e-6
    ns=[]
    for i in range(1,len(strains),1):
        #pay = straine göre değişecek, curve altında alanı hesapla
        W+=(stresses[i]+stresses[i-1])*(strains[i]-strains[i-1])/2
        
        #payda = 1 * max stress
        if stresses[i]>max_stress:
            max_stress=stresses[i]
            
        #ENERGY ABSORPTION EFF - NEW
        n=100*W/(1 * max_stress)
        ns.append(n)
    
    n=max(ns)
    # n_plot(ns,strains[1::])
    
    #densification stress nasıl hesaplanır, efficiency düştüğü yer
    idx_n=ns.index(n)
    dens_stress=stresses[idx_n]
    dens_strain=strains[idx_n]
    idx_dens=stresses.index(dens_stress)
    
    """Getting Plateau stress"""
    ll=[]
    while idx_yield<=idx_dens:
        ll.append(stresses[idx_yield])
        idx_yield+=1
    plateau_stress=np.mean(ll)

    
    """Useful work calculation, this is until densification strain"""
    k=1
    W_useful=0
    while strains[k]<=dens_strain:
        W_useful+=(stresses[k]+stresses[k-1])*(strains[k]-strains[k-1])/2
        W_useful_ll.append(W)
        s_ll.append(stresses[k])
        e_ll.append(strains[k])
        n_ll.append(W/plateau_stress)
        if k==len(strains)-1:
            break
        else:
            k+=1
    
    """ SPECIFIC USEFUL ENERGY ABSORPTION calculation"""
    w_useful=W_useful/ro_eff
        
    #ENERGY ABSORPTION EFF - OLD
    # n=W/plateau_stress
    
    return n,W,W_useful,w_useful,E,yield_stress,yield_strain,dens_stress,dens_strain,plateau_stress


""" BURAYI HALLET INSALLAH """
def n_plot(ns,strains):
    """daha önce listeye almistim her W ve Stress stepini onlari plot ediyorum,
    ancak her n sayida bir plot etmeyi secebiliyorum, daha düz bir plot umuduyla"""
    
    #FIGURE 10 burasi
    # n=1
    
    # plt.figure(figsize=(12,8))
    # s2_ll=[s_ll[i] for i in range(len(s_ll)) if i%n==0]
    # W2_ll=[W_ll[i] for i in range(len(W_ll)) if i%n==0]
    # plt.plot(s2_ll,W2_ll)
    # plt.ylabel("Absorbed Energy per Unit Volume (J/$mm^3$)")
    # plt.xlabel("Stress (MPa)")
    # plt.title(f"Model: {model_name}")
    # plt.savefig(f"{model_name}-Energy_vol.png")
    
    
    #FIGURE 11 burasi
    plt.figure(figsize=(12,12))
    plt.subplot(2,1,2)
    plt.plot(strains,ns,linewidth=3,color="crimson")
    plt.ylabel("Absorbed Energy\nEfficiency (%)")
    plt.xlabel("Strain (mm/mm)")
    # plt.title(f"Model: {model_name}")
    # plt.savefig(f"{model_name}-Eff_strain.png")
    
    return None
    

def all_plot():
    cm = plt.get_cmap('tab20')
    fig = plt.figure(figsize=(16,12))
    ax = fig.add_subplot(111)
    ax.set_prop_cycle('color', [cm(1.*i/20) for i in range(20)])
    
    for i,file in enumerate(files):
        if file.endswith(".curve"):
            forces=[]
            disps=[]
            stresses=[]
            strains=[]
            with open(path+"\\"+file,"r") as fin:
                for line in fin.readlines():
                    data=line.split(",")
                    disps.append(float(data[0]))
                    forces.append(float(data[1].rstrip()))
                    strains.append(float(data[2].rstrip()))
                    stresses.append(float(data[3].rstrip()))
            
            spline=make_interp_spline(disps,forces)
            disps=np.linspace(min(disps),max(disps),200)
            forces=spline(disps)
            
            plt.ylabel("Force (N)")
            plt.xlabel("Displacement (mm)")
            ax.plot(disps[:],forces[:],linewidth=1,
                      label=f"Model: {file.rstrip('.curve')}")
            title="Model: "+file.rstrip('.curve')
            # plt.legend(fontsize=12,shadow=True,fancybox=True,
            #            loc='center left', bbox_to_anchor=(1.01, 1))
            
            
    plt.tight_layout()
    plt.savefig("ALL_CURVES3.png")
    plt.show()
    return None

"""---------------------------------------MAIN STUFF-----------------------------------------------"""

datalog_read()

storage={"Model_name":[],"E (MPa)":[],"EA (J)":[],"W_useful":[], "w_useful":[],"SVF":[],
         "Plateau Stress (MPa)":[],"Yield Stress (MPa)":[],"Yield Strain":[],
         "Densification Stress (MPa)":[],"Densification Strain":[],"MASS":[],"n (%)":[]}

files=sorted_alphanumeric(os.listdir())
plot=True
density=2.7e-6

df=pd.read_excel("data.xlsx")
k=0

for file in files:
    #W list, get_props dolduruyor burayı
    W_useful_ll=[0]
    s_ll=[0]
    e_ll=[0]
    n_ll=[0]
    
    if file.endswith(".curve"):
        forces=[]
        disps=[]
        stresses=[]
        strains=[]
        svf=df["SVF"][k]
        mass=df["MASS"][k]
        ro_eff=df["RO_EFF"][k]*10e3
        model_name=file.rstrip('.curve')
        print(file)
        
        k+=1
        with open(path+"\\"+file,"r") as fin:
            for line in fin.readlines():
                data=line.split(",")
                disps.append(float(data[0]))
                forces.append(float(data[1].rstrip())/1e3)
                strains.append(float(data[2].rstrip()))
                stresses.append(float(data[3].rstrip()))
                
        #TAKING MECH PROP.
        n,W,W_useful,w_useful,E,yield_stress,yield_strain,dens_stress,dens_strain,plateau_stress=get_props(strains,stresses)
        
        #SMOOTH
        # spline=make_interp_spline(disps,forces)
        # disps=np.linspace(min(disps),max(disps),200)
        # forces=spline(disps)
        
        # spline2=make_interp_spline(strains,stresses)
        # strains=np.linspace(min(strains),max(strains),200)
        # stresses=spline2(strains)
        
        ####################################
        ############# P L O T ##############
        ####################################
        if plot:
            storage["Model_name"].append(file.rstrip('.curve'))
            storage["E (MPa)"].append(E)
            storage["EA (J)"].append(W)
            storage["W_useful"].append(W_useful)
            storage["w_useful"].append(w_useful)
            storage["SVF"].append(svf)
            storage["Plateau Stress (MPa)"].append(plateau_stress)
            storage["Yield Stress (MPa)"].append(yield_stress)
            storage["Yield Strain"].append(yield_strain)
            storage["Densification Stress (MPa)"].append(dens_stress)
            storage["Densification Strain"].append(dens_strain)
            storage["MASS"].append(mass)
            storage["n (%)"].append(n)
            
            
            # plt.figure(figsize=(16,12))
            
            # plt.subplot(2,1,1)
            # plt.ylabel("Force [N]")
            # plt.xlabel("Displacement [mm]")
            # plt.plot(disps[:101],forces[:101],linewidth=2,linestyle="--", 
            #          marker='o', markersize=8,markeredgecolor="black",markeredgewidth=1,
            #          label=f"Fmax = {max(forces)/1e3:0.1f} kN",color="blue")
            # plt.plot(disps[100:250:4],forces[100:250:4],linewidth=2,linestyle="--", 
            #          marker='o', markersize=8,markeredgecolor="black",markeredgewidth=1,
            #          label=f"Fmax = {max(forces)/1e3:0.1f} kN",color="blue")
            # title="Model: "+model_name
            # plt.legend(fontsize=16,title=title,shadow=True,fancybox=True)
            
            plt.figure(figsize=(12,9))
            plt.ylabel("Force (kN)")
            plt.xlabel("Displacement (mm)")
            plt.plot(disps[:101],forces[:101],linewidth=3,color="crimson")
            plt.plot(disps[100::4],forces[100::4],linewidth=3,color="crimson")
            #densification ve plateau noktasi koyuyorum
            # plt.scatter(yield_strain,yield_stress,s=200,color="green",
            #             marker='x',zorder=20,linewidth=2,edgecolors="gray")
            # plt.scatter(dens_strain,dens_stress,s=200,color="red",
            #             marker='x',zorder=20,linewidth=2,edgecolors="gray")
            # title="Model: "+model_name
            # plt.legend(fontsize=16,title=title,shadow=True,fancybox=True)
            plt.tight_layout()
            plt.savefig(path+"\\"+f"force_{file.rstrip('.curve')}.png")
            
            if k>10:
                break
            
    # df_out=pd.DataFrame(storage)
    # df_out.to_excel("Mechanical_props.xlsx",index=False)


# all_plot()  
