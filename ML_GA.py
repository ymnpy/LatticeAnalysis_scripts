import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import os
import random
import math
from tqdm import tqdm
from sklearn.model_selection import train_test_split,cross_val_score,KFold
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import learning_curve
from sklearn.metrics import r2_score,mean_squared_error,mean_absolute_percentage_error
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn import tree
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import matplotlib.ticker as ticker
import time
import warnings
from xgboost import XGBRegressor,plot_importance
warnings.simplefilter("ignore")
warnings.simplefilter(action='ignore', category=FutureWarning)
sns.set_theme(style="ticks",font_scale=3)
plt.rcParams["font.family"] = "Times New Roman"
os.chdir(r'C:\Users\User\Desktop\Okul\YL\yl_döküman\00_AUXETIC-METAMATERIALS\40_the1000\01_the1400')
np.random.seed(42)

"""THE MAIN SCRIPT FOR MACHINE LEARNING STUFF, including:
    -Three models training -> NN, RF, XGB
    -Hyperparameter optimization
    -Scaling by standardization
    -Outlier detection and removal by IQR method
    AND GENETIC ALGORITHM OPTIMIZATION
"""

def plot_learning_curve(model, model_name, cols, X, y):
    # sns.set_theme(style="ticks", font_scale=2.4)
    
    fig, axes = plt.subplots(nrows=5, ncols=10, figsize=(15, 20))
    plt.subplots_adjust(left=0.12, bottom=0.08, right=0.95, top=0.92, wspace=0.4, hspace=0.3)
    
    handles = []
    labels = []
    
    for i, (ax, col) in enumerate(zip(axes.flat, cols)):
        train_sizes, train_scores, test_scores = learning_curve(model, X, y[col], 
                                                                train_sizes=[0.2,0.4,0.6,0.8,1], cv=10, n_jobs=-1)
        line_train, = ax.plot(train_sizes, np.mean(train_scores, axis=1), 'o-', linewidth=3, label="Training Score")
        line_test, = ax.plot(train_sizes, np.mean(test_scores, axis=1), 's-', linewidth=3, label="Test Score")
        
        col_name=col.split("(")[0]
        
        # Collect handles and labels for the legend
        if i == 0:
            handles.extend([line_train, line_test])
            labels.extend(["Training Score", "Test Score"])
        
        # Set y-ticks for all axes
        ax.set_yticks([0.8, 0.88, 0.95, 1.02])
        
        # Add "R2" only to the leftmost subplots
        if i % 2 == 0:
            ax.set_ylabel(f"$R^{2}$",labelpad=10)
        else:
            ax.set_ylabel(f" ")
        
        if i == 6 or i == 7:  # indices for the 4th and 8th subplots
            ax.set_xlabel(f"Number of samples", labelpad=10)
            
        # Only set x-ticks for the bottom row
        if i < len(cols) - 2:
            ax.set_xticklabels([])
        
        # Only set y-ticks for the leftmost column
        if i % 2 != 0:
            ax.set_yticklabels([])
        
        ax.text(1.06, 0.5, col_name, va='center', ha='center', rotation=90,transform=ax.transAxes)
        
    # Create a single legend for all subplots
    fig.legend(handles[:2], labels[:2], loc='upper center', bbox_to_anchor=(0.5, 0.99), ncol=2,
               shadow=True, fancybox=True)

    plt.savefig("learningcurve.png")
    plt.show()

    return None

def remove_outliers(df,columns):
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

def normalize(df_in):
    normalizer=MinMaxScaler()
    for col in df_in.columns:
        try: df_in[col]=normalizer.fit_transform(df_in[[col]])
        except: pass
    
    return df_in

def standardize(df_in):
    standizer=StandardScaler()
    for col in df_in.columns:
        try: df_in[col]=standizer.fit_transform(df_in[[col]])
        except: pass
    
    return df_in

"""
def xgb_hpopt(model):
    df_out=df[odbs]
    df["vector"]=(((df_out.loc[:,odbs])**2).sum(axis=1))**0.5
    
    X=df[inp]
    y=df["vector"]
    X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2)
    
    parameters = {
              'n_estimators': [100,200,400,1000],
              'max_depth': [4,8,16, None],
              'min_child_weight':[1,2,4,8],
              'learning_rate': [1e-4, 1e-3, 1e-2,0.1], #so called `eta` value
              }
    
    xgb_grid = GridSearchCV(model,
                        parameters,
                        cv = 5, #KFold number
                        n_jobs = 4, #processors
                        verbose=False)

    xgb_grid.fit(X_train,y_train)
    print(xgb_grid.best_params_)
    return xgb_grid.best_estimator_,xgb_grid.best_params_
"""

def xgb_hpopt(model,X_valid,y_valid):
    
    # kf = KFold(n_splits=5, shuffle=True)
    parameters = {
              'n_estimators': [100,200,400,1000],
              'max_depth': [4,8,16, None],
              'min_child_weight':[1,2,4,8],
              'learning_rate': [1e-4, 1e-3, 1e-2,0.1], #so called `eta` value
              }
    
    xgb_grid = GridSearchCV(model,
                        parameters,
                        cv = 10, #KFold number
                        n_jobs = -1, #processors
                        verbose=False)

    xgb_grid.fit(X_valid,y_valid)
    print(xgb_grid.best_params_)
    return xgb_grid.best_estimator_,xgb_grid.best_params_
        

def rf_hpopt(model,X_valid,y_valid):
    
    parameters = {
              'n_estimators': [100,200,400,1000], 
              'max_features': ['sqrt', 'log2', None], 
              'max_depth': [4, 8, 16, None], 
              'min_samples_split': [2, 4, 8, 16]
              }
    
    xgb_grid = GridSearchCV(model,
                        parameters,
                        cv = 10, #KFold number
                        n_jobs = -1, #processors
                        verbose=False)

    xgb_grid.fit(X_valid,y_valid)
    print(xgb_grid.best_params_)
    return xgb_grid.best_estimator_,xgb_grid.best_params_


def mlp_hpopt(model,X_valid,y_valid):
    
    parameters = {
        'hidden_layer_sizes': [(6,12,6),(3,6,6,3)],
        'activation': ['tanh', 'relu'],
        # 'solver': ['adam','sgd'],
        'alpha': [0.0001,0.001],
        'max_iter':[400],
        'learning_rate_init': [1e-3, 1e-2, 0.1],
        'learning_rate': ['constant','adaptive'],
    }
    
    mlp_grid = GridSearchCV(model,
                        parameters,
                        cv = 10, #KFold number
                        n_jobs = -1, #processors
                        verbose=False)

    mlp_grid.fit(X_valid,y_valid)
    print(mlp_grid.best_params_)
    return mlp_grid.best_estimator_,mlp_grid.best_params_


# def run_model(model,model_name,output):
#     sns.set_theme(style="ticks",font_scale=6)
#     # if model_name=="XGB":
#     #     print("Hyperparameter optimization starts...")
#     #     model=xgb_hp_optimization(model, X, y)
#     X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2)
#     model.fit(X_train,y_train)
#     y_pred=model.predict(X_test)
#     # r2=r2_score(y_test,y_pred)
#     # rmse=mean_squared_error(y_test,y_pred,squared=False)
#     # mape=mean_absolute_percentage_error(y_test,y_pred)
    
#     # kf = KFold(n_splits=5)
#     # scores_r2=cross_val_score(model, X, y, scoring="r2",cv=kf)
#     # scores_mape=cross_val_score(model, X, y, scoring="neg_mean_absolute_percentage_error",cv=kf)
#     # scores_rmse=cross_val_score(model, X, y, scoring="neg_root_mean_squared_error",cv=kf)
#     # r2=np.mean(scores_r2)
#     # mape=np.mean(scores_mape)*-1
#     # rmse=np.mean(scores_rmse)*-1
    
#     # ss[output]=(r2,mape,rmse)
    
#     plot_learning_curve(model)

def run_model(model,model_name,output,X_train,y_train,X_test,y_test):
    %matplotlib auto
    plt.rcParams["font.family"] = "Times New Roman"
    sns.set_theme(style="ticks",font_scale=6)
    model.fit(X_train,y_train)
    y_pred=model.predict(X_test)
    
    r2=r2_score(y_test, y_pred)
    mape=mean_absolute_percentage_error(y_test, y_pred)
    rmse=mean_squared_error(y_test, y_pred)

    ss[output]=(r2,mape,rmse)
    
    plot=True
    if plot:
        plt.figure(figsize=(16,12),layout="constrained")
        # plt.suptitle(f"Algorithm: {model_name} | Output: {output}\nR2: {r2:0.2f} | RMSE: {rmse:0.2f} | MAPE: {mape:0.2f}\n",fontsize=22)
     
        #scatter
        if model_name=="MLP": color="crimson"
        if model_name=="XGB": color="steelblue"
        if model_name=="RF": color="green" 
        else: color="orange"
        
        
        # plt.scatter(y_test,y_pred,s=200,edgecolors=color,linewidths=2,facecolors='none')
        plt.scatter(y_test,y_pred,s=400,color=color,edgecolors="black",linewidths=1)
        # plt.xlabel("Simulation Values")
        # plt.ylabel("Prediction Values")
        # plt.ylim((min(y_pred)-0.5,max(y_pred)+1))
        # plt.xlim((min(y_pred)-0.5,max(y_pred)+1))
        # plt.gca().xaxis.set_major_locator(plt.MultipleLocator(1))
        # plt.gca().yaxis.set_major_locator(plt.MultipleLocator(1))
        plt.axline((0,0), slope=1,linestyle="-",color="gray",linewidth=3,zorder=-20)
        
        #10-range
        # plt.axline((0,0), slope=0.9/1,linestyle="dashed",linewidth=0.5,color="green",label="+- 10% line")
        # plt.axline((0,0), slope=1/0.9,linestyle="dashed",linewidth=0.5,color="green")
        # #30-range
        # plt.axline((0,0), slope=0.7/1,linestyle="-.",linewidth=0.5,color="orange",label="+- 30% line")
        # plt.axline((0,0), slope=1/0.7,linestyle="-.",linewidth=0.5,color="orange")
        # plt.legend()
        plt.savefig(f"{model_name}_{output[0:3]}_new.png")
        
        # plt.figure(figsize=(18,12),layout="constrained")
        # y_test2=y_test.reset_index(drop=True,inplace=False)
        # plt.plot(y_test2,linewidth=2,marker="^",label="Simulation Values",markersize=10)
        # plt.plot(y_pred,linewidth=2,marker="s",label="Prediction Values",markersize=10)
        # plt.legend(title=f"Algorithm: {model_name}\n Output: {output}\nR2: {r2:0.2f}\n MAPE: {mape:0.2f}\n",
        #            loc='best',fontsize=22)
        # plt.xlabel("Number of Samples")
        # plt.ylabel("Values")
        # plt.savefig(f"{model_name}_{output[0:3]}_line.png")
        
    if model_name=="XGB":
        #"weight" is the number of times a feature appears in a tree
        fig, ax = plt.subplots(1,1,figsize=(18,12))
        ax=plot_importance(model,importance_type='gain',
                           ax=ax,grid=False,color="steelblue",
                           show_values=False,title="") 
        ax.xaxis.grid(True)
        plt.ylabel("Features", labelpad=27)
        plt.xlabel("F score", labelpad=22)
        plt.tight_layout()
        plt.savefig(f"{model_name}_{output[0:3]}_features.png")
        
        feature_important = model.get_booster().get_score(importance_type='weight')
        keys = list(feature_important.keys())
        values = list(feature_important.values())
        
        with open("{model_name}_{output[0:8]}_features.txt","a") as fout:
            pass
        with open("{model_name}_{output[0:8]}_features.txt","a") as fout:
            for k,v in zip(keys,values):
                fout.write(f"{k}   :   {v}")
     
    elif model_name=="RF":
        tree.plot_tree(model.estimators_[0], feature_names=X_test.columns, 
                       filled=True, rounded=True)
        plt.savefig("decision_tree.png")
        
    return model

df=pd.read_excel('THE_DATA_9000.xlsx')
df.dropna(inplace=True)
df["SR_avg"]=(df["L1"]+df["L2"]+df["L3"]+df["L4"])/(4*df["THICKNESS"])

df.rename(columns = {'P4-X':'px', 'P4-Y':'py','THICKNESS':'t',
                     'Densification Stress (MPa)':'Dens. Stress (MPa)',
                     'Densification Strain':'Dens. Strain',
                     'W_useful':'W useful (J)', 'SR_avg':'SR avg',
                     'MASS':'MASS (grams)',
                     'w_useful':'SW useful (J.mm3/kg)'}, inplace = True)

# df['pp']=abs(10-df['px'])


# odbs=['MASS (grams)', 'E (MPa)', 'EA (J)', 'W useful (J)',
# 'SW useful (J.mm3/kg)', 'Plateau Stress (MPa)', 'Yield Stress (MPa)',
# 'Yield Strain', 'Dens. Stress (MPa)', 'Dens. Strain', 'n (%)']

odbs=['E (MPa)', 'EA (J)', 'W useful (J)',
'SW useful (J.mm3/kg)', 'Plateau Stress (MPa)', 'Yield Stress (MPa)',
'Dens. Stress (MPa)', 'n (%)']

# odbs=['n (%)']

# odbs=['n (%)']

odbs_outliers=['E (MPa)', 'EA (J)', 'W useful (J)']

inp=['px','py','t']

print(f"Whole data -> {df.shape}")
df=remove_outliers(df, odbs_outliers)
# df=standardize(df)
df_standard=df.copy()
# df_standard=normalize(df_standard)

df["vector"]=(((df_standard.loc[:,odbs])**2).sum(axis=1))**0.5

# # df=standardize(df)
odbs=odbs+["vector"]
X=df[inp]
y=df[odbs]

X_train,X_test,y_train,y_test=train_test_split(X,y, test_size=0.2)

print(f"After outlier removal -> {X.shape}")
print(f"Training data -> {X_train.shape}")
print(f"Test data -> {X_test.shape}")
# print(f"Validation data -> {X_val.shape}")


model=XGBRegressor()
best_xgb,params_xgb=xgb_hpopt(model,X_train,y_train["vector"])

best_rf=RandomForestRegressor(max_depth=4)
best_rf,params_rf=rf_hpopt(model,X_train,y_train["vector"])

model=MLPRegressor()
best_mlp,params_mlp=mlp_hpopt(model,X_train,y_train["vector"])    


models={"XGB":best_xgb, "RF": best_rf, "MLP": best_mlp}
ss={}

for model_name,m in models.items():
    start=time.time()
    for i,odb in enumerate(odbs):
        if odb!="vector":
            y_train_filtered = y_train[odb]
            y_test_filtered = y_test[odb]
            trained=run_model(m,model_name,odb,X_train,y_train_filtered,X_test,y_test_filtered)
        
        if model_name == "XGB":
            trained_xgb = trained
#     # model=SVR()
#     # model=KNeighborsRegressor()

    finish=time.time()
    with open(f"{model_name}_scores.result","w") as fin:
        pass
    for k,v in ss.items():
        with open(f"{model_name}_scores.result","a") as fin:
            fin.write(f"{k}   :    {v}\n")
            # fin.write(f"{finish}-{start} seconds")

# odbs=['MASS (grams)', 'E (MPa)', 'EA (J)', 'W useful (J)',
# 'SW useful (J.mm3/kg)', 'Plateau Stress (MPa)', 'Yield Stress (MPa)',
# 'Yield Strain', 'Dens. Stress (MPa)', 'n (%)']

# plot_learning_curve(trained_xgb,"XGB",odbs,X,y)
    

"""-----------------GENETIC ALGORITHM SECTION, TAKEN FROM THE WEB --------------------------"""
def generate_population(size, x1_boundaries, x2_boundaries, x3_boundaries):
    lower_x1_boundary, upper_x1_boundary = x1_boundaries
    lower_x2_boundary, upper_x2_boundary = x2_boundaries
    lower_x3_boundary, upper_x3_boundary = x3_boundaries

    population = []
    for i in range(size):
        individual = {
            "x1": random.uniform(lower_x1_boundary, upper_x1_boundary),
            "x2": random.uniform(lower_x2_boundary, upper_x2_boundary),
            "x3": random.uniform(lower_x3_boundary, upper_x3_boundary),
        }
        population.append(individual)

    return population


def sort_population_by_fitness(population):
    return sorted(population, key=apply_function)


def apply_function(individual):
    x1 = individual["x1"]
    x2 = individual["x2"]
    x3 = individual["x3"]
    return trained_xgb.predict(np.array([[x1,x2,x3]]))


def crossover(individual_a, individual_b):
    x1a = individual_a["x1"]
    x2a = individual_a["x2"]
    x3a = individual_a["x3"]
    
    x1b = individual_b["x1"]
    x2b = individual_b["x2"]
    x3b = individual_b["x3"]

    return {"x1": (x1a + x1b) / 2, "x2": (x2a + x2b) / 2, "x3": (x3a + x3b) / 2}


def mutate(individual,x1_boundaries,x2_boundaries,x3_boundaries,mutation_rate):
    next_x1 = individual["x1"] + random.uniform(-1, 1)*mutation_rate
    next_x2 = individual["x2"] + random.uniform(-1, 1)*mutation_rate
    next_x3 = individual["x3"] + random.uniform(-1, 1)*(mutation_rate/2)
    
    lower_boundary_x1, upper_boundary_x1 = x1_boundaries
    lower_boundary_x2, upper_boundary_x2 = x2_boundaries
    lower_boundary_x3, upper_boundary_x3 = x3_boundaries

    # Guarantee we keep inside boundaries
    next_x1 = min(max(next_x1, lower_boundary_x1), upper_boundary_x1)
    next_x2 = min(max(next_x2, lower_boundary_x2), upper_boundary_x2)
    next_x3 = min(max(next_x3, lower_boundary_x3), upper_boundary_x3)

    return {"x1": next_x1, "x2": next_x2, "x3": next_x3}


def choice_by_roulette(sorted_population, fitness_sum):
    offset = 0
    normalized_fitness_sum = fitness_sum

    lowest_fitness = apply_function(sorted_population[0])
    if lowest_fitness < 0:
        offset = -lowest_fitness
        normalized_fitness_sum += offset * len(sorted_population)

    draw = random.uniform(0, 1)

    accumulated = 0
    for individual in sorted_population:
        fitness = apply_function(individual) + offset
        probability = fitness / normalized_fitness_sum
        accumulated += probability

        if draw <= accumulated:
            # print(fitness_sum,individual)
            return individual
        
       
def make_next_generation(previous_population,x1_boundaries,x2_boundaries,x3_boundaries,mutation_rate):
    next_generation = []
    sorted_by_fitness_population = sort_population_by_fitness(previous_population)
    population_size = len(previous_population)
    fitness_sum = sum(apply_function(individual) for individual in population)

    for i in range(population_size):
        first_choice = choice_by_roulette(sorted_by_fitness_population, fitness_sum)
        second_choice = choice_by_roulette(sorted_by_fitness_population, fitness_sum)

        individual = crossover(first_choice, second_choice)
        individual = mutate(individual,x1_boundaries,x2_boundaries,x3_boundaries,mutation_rate)
        next_generation.append(individual)
        
        print(next_generation)
    return next_generation


#variable boundaries
x1_boundaries=(0,20)
x2_boundaries=(0,20)
x3_boundaries=(0.5,10)

population = generate_population(size=12, 
                                 x1_boundaries=x1_boundaries, 
                                 x2_boundaries=x2_boundaries,
                                 x3_boundaries=x3_boundaries)

generation_scores=[0]
generation_thicknesses=[]
xs,ys=[],[]
i=1

while True:
    # print(f"🧬 GENERATION {i}")
    
    individual_thicknesses=[]    
    individual_scores=[]
    for individual in population:
        # print(individual, apply_function(individual))
        individual_thicknesses.append(individual['x3'])
        individual_scores.append(apply_function(individual))
        
        # sns.jointplot(x = individual['x1'], y = individual['x2'] ,kind = "reg", dropna = True) 
        xs.append(individual['x1'])
        ys.append(individual['x2'])
    
    population = make_next_generation(population,x1_boundaries,x2_boundaries,x3_boundaries,mutation_rate=2)
    generation_scores.append(np.mean(individual_scores))
    generation_thicknesses.append(np.mean(individual_thicknesses))
    
    try:
        ds=generation_scores[i]-generation_scores[i-1]
        # print(generation_scores[i],generation_scores[i-1])
        print(ds)
        if abs(ds)<1e-6:
            break
    except:
        pass
    
    i+=1

sns.set_theme(style="ticks",font_scale=4)
fig=plt.figure(figsize=(10,8))
sns.kdeplot(x=xs,y=ys,shade = True, cmap = "Blues")
plt.xlabel("px",color="black",labelpad=15)
plt.ylabel("py",color="black",labelpad=15)
plt.xlim(x1_boundaries)
plt.ylim(x2_boundaries)
plt.tight_layout()
plt.savefig("GA_kde.png")
# plt.set(xlim=(min(xs), max(xs)), ylim=(min(ys), max(ys)))
# plt.set_title("Coordinates")
# cb.set_label('counts')

plt.figure(figsize=(10,8))
# plt.subplot(211)
plt.plot(np.arange(len(generation_scores)),generation_scores,
         color="steelblue",linewidth=4)
plt.xlabel("Number of Generations",color="black",labelpad=15)
plt.ylabel("XGB Fitness Score (%n)",color="black",labelpad=15)
plt.tight_layout()
# plt.savefig("GA_xy.png")

# plt.subplot(212)
# plt.plot(np.arange(len(generation_scores)-1),generation_thicknesses,color="green",linewidth=2)
# plt.xlabel("Number of Generations",color="gray")
# plt.ylabel("Thickness",color="gray")

# plt.suptitle(f"Number of Generations: {len(generation_scores)-1}")
plt.show()

best_individual = sort_population_by_fitness(population)[-1]
print("\n🔬 FINAL RESULT")
print(len(generation_scores)-1)
print(best_individual, apply_function(best_individual),len(generation_scores)-1)