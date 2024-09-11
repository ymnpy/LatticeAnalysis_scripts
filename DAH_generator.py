# -*- coding: utf-8 -*-
from part import *
from material import *
from section import *
from assembly import *
from step import *
from interaction import *
from load import *
from mesh import *
from optimization import *
from job import *
from sketch import *
from visualization import *
from connectorBehavior import *
from odbAccess import *
from abaqusConstants import *
from math import atan,degrees
import random
import time
import numpy as np
import os

"""THE MAIN SCRIPT FOR ABAQUS COMMANDS,
   generating, solving, storing FEA results of lattice structures in a loop
"""

def find_intersection(p1,p2,p3,p4):
    px=((p1[0]*p2[1]-p1[1]*p2[0])*(p3[0]-p4[0])-(p1[0]-p2[0])*(p3[0]*p4[1]-p3[1]*p4[0])) \
        /((p1[0]-p2[0])*(p3[1]-p4[1])-(p1[1]-p2[1])*(p3[0]-p4[0])+1e-6)
    py=((p1[0]*p2[1]-p1[1]*p2[0])*(p3[1]-p4[1])-(p1[1]-p2[1])*(p3[0]*p4[1]-p3[1]*p4[0])) \
        /((p1[0]-p2[0])*(p3[1]-p4[1])-(p1[1]-p2[1])*(p3[0]-p4[0])+1e-6)
    return [px,py]

def get_prevdata():
    storage=[]
    count=0
    path=r'C:/temp/data.log'
    with open(path,"r") as fin:
        for line in fin.readlines():
            try:
                count+=1
                a=line.split('(')[4].split(')')[0]
                x,y=float(a.split(',')[0]),float(a.split(',')[1])
                storage.append((x,y))
            except:
                pass
    return storage,count-1


"""----------------------------- INPUTS -----------------------------"""
now=time.time()
storage,count=get_prevdata() #residual aldigim yer
cells_x,cells_y=8,6
depth_length=4
print(storage)

offset=2
count=0
no_models=1400

mass_scale=1
step_time=0.1
density=2.7e-9
thickness_list=[0.8,1.2,1.6,2.4,3.2]

job_run=True
empty_pass=0
"""------------------------------------------------------------------"""

if len(storage)==0:
    with open("data.log","w") as fout:
        fout.write("MODEL_NO,P1,P2,P3,P4,L1,L2,L3,L4,HEIGHT,WIDTH,THICKNESS,THETA1,THETA2,SVF,RO_EFF,MASS\n")
    
while count <= no_models:
    r1x,r1y=0,0
    r2x,r2y=20,0
    r3x,r3y=round(random.random()*20,1),20
    r4x,r4y=r3x,round(random.random()*20,1)
    thickness=random.choice(thickness_list)
    
    model_name=str(count)+"_Al-"\
                +"X"+str(r3x).replace(".","")\
                +"Y"+str(r4y).replace(".","") \
                +"T"+(str(thickness).replace(".",""))
    
    #angle check for p4
    a13=(r3y-r1y)/(r3x-r1x+0.001)
    a23=-(r3y-r2y)/(r3x-r2x+0.001)
    a14=(r4y-r1y)/(r4x-r1x+0.001)
    a24=-(r4y-r2y)/(r4x-r2x+0.001)
    
    # lengths of a cell---------------------------------------------------------------
    l1=((r3x-r1x)**2+(r3y-r1y)**2)**(0.5)
    l2=((r3x-r2x)**2+(r3y-r2y)**2)**(0.5)
    l3=((r4x-r2x)**2+(r4y-r2y)**2)**(0.5)
    l4=((r1x-r4x)**2+(r1y-r4y)**2)**(0.5)
    lmin=min(l1,l2,l3,l4)
    mesh_auxetic=lmin/5
    
    p1,p2,p3,p4=(r1x,r1y),(r2x,r2y),(r3x,r3y),(r4x,r4y)
    
    off=p3[1]-p4[1]
    p1_top,p1_bot=[p1[0],p1[1]+off],[p1[0],p1[1]-off]
    p2_top,p2_bot=[p2[0],p2[1]+off],[p2[0],p2[1]-off]
    p3_top,p3_bot=[p3[0],p3[1]+off],[p3[0],p3[1]-off]
    p4_top,p4_bot=[p4[0],p4[1]+off],[p4[0],p4[1]-off]
    
    t1_top,t1_bot=[0,20],[0,0]
    t2_top,t2_bot=[20,20],[20,0]
    
    pp=find_intersection(p1_top,p3_top,t1_top,t2_top)
    l5=((pp[0]-p1_top[0])**2+(pp[1]-p1_top[1])**2)**0.5
    
    pp=find_intersection(p1_top,p4_top,t1_top,t2_top)
    l6=((pp[0]-p1_top[0])**2+(pp[1]-p1_top[1])**2)**0.5
    
    pp=find_intersection(p2_top,p3_top,t1_top,t2_top)
    l7=((pp[0]-p2_top[0])**2+(pp[1]-p2_top[1])**2)**0.5
    
    pp=find_intersection(p2_top,p4_top,t1_top,t2_top)
    l8=((pp[0]-p2_top[0])**2+(pp[1]-p2_top[1])**2)**0.5

    pp=find_intersection(p1_bot,p3_bot,t1_bot,t2_bot)
    l9=((pp[0]-p4[0])**2+(pp[1]-p4[1])**2)**0.5
    
    pp=find_intersection(p2_bot,p3_bot,t1_bot,t2_bot)
    l10=((pp[0]-p4[0])**2+(pp[1]-p4[1])**2)**0.5
    # ------------------------------------------------------------------------------------
    
    #solid volume frac / line density
    solid_vol=20*20
    lattice_vol=(l1+l2+l3+l4+l5+l6+l7+l8+l9+l10)*thickness
    svf=lattice_vol/solid_vol
    ro_effective=svf*density
    
    #thetas
    theta1=180-(degrees(atan(a13))+degrees(atan(a23)))
    theta2=180-(degrees(atan(a14))+degrees(atan(a24)))
    
    hor_x=p1[0]-p2[0]
    hor_y=p1[1]-p2[1]
    ver_x=p3[0]-p4[0]
    ver_y=p3[1]-p4[1]
    
    #for assembly
    height=(r3y-r4y)*cells_y+r4y
    width=r2x*cells_x
    crush_height=height*0.8 #making it a variable
    velocity=-crush_height/step_time
    
    #breaking after n amounts of repetitive failures
    if empty_pass>200: break

    #angle şartları, noktalar unique, svf şartı
    if degrees(atan(a13))>80 or degrees(atan(a23))>80:
        empty_pass=+1
        continue
    
    if degrees(atan(a14))<10 or degrees(atan(a24))<10:
        empty_pass=+1
        continue
    
    if not degrees(atan(a13))>=degrees(atan(a14))+10:
        empty_pass=+1
        continue
    
    if not degrees(atan(a23))>=degrees(atan(a24))+10:
        empty_pass=+1
        continue
    
    if svf>0.8:
        empty_pass=+1
        continue
     
    if (r4x,r4y) in storage:
        empty_pass=+1
        continue
    
    
    empty_pass=0 #unique parameters are found, zeroing it
    count+=1
    storage.append((r4x,r4y))
    
    #create model
    mdb.Model(modelType=STANDARD_EXPLICIT, name=model_name)
    
    #UNIT CELL
    mdb.models[model_name].ConstrainedSketch(name='__profile__', sheetSize=200.0)
    mdb.models[model_name].sketches['__profile__'].Spot(point=p1)
    mdb.models[model_name].sketches['__profile__'].Spot(point=p2)
    mdb.models[model_name].sketches['__profile__'].Spot(point=p3)
    mdb.models[model_name].sketches['__profile__'].Spot(point=p4)
    mdb.models[model_name].sketches['__profile__'].Line(point1=p1, point2=p3)
    mdb.models[model_name].sketches['__profile__'].Line(point1=p3, point2=p2)
    mdb.models[model_name].sketches['__profile__'].Line(point1=p2, point2=p4)
    mdb.models[model_name].sketches['__profile__'].Line(point1=p4, point2=p1)
    
    #CONSTRUCTION
    mdb.models[model_name].sketches['__profile__'].ConstructionLine(point1=(r2x, 
        0.0), point2=(r2x, 20.0))
    mdb.models[model_name].sketches['__profile__'].VerticalConstraint(addUndoState=
        False, entity=mdb.models[model_name].sketches['__profile__'].geometry[6])
    mdb.models[model_name].sketches['__profile__'].CoincidentConstraint(
        addUndoState=False, entity1=
        mdb.models[model_name].sketches['__profile__'].vertices[1], entity2=
        mdb.models[model_name].sketches['__profile__'].geometry[6])
    mdb.models[model_name].sketches['__profile__'].copyMirror(mirrorLine=
        mdb.models[model_name].sketches['__profile__'].geometry[6], objectList=(
        mdb.models[model_name].sketches['__profile__'].geometry[2], 
        mdb.models[model_name].sketches['__profile__'].geometry[3], 
        mdb.models[model_name].sketches['__profile__'].geometry[4], 
        mdb.models[model_name].sketches['__profile__'].geometry[5], 
        mdb.models[model_name].sketches['__profile__'].vertices[2], 
        mdb.models[model_name].sketches['__profile__'].vertices[3]))

    mdb.models[model_name].sketches['__profile__'].ConstructionLine(point1=(2*r2x, 
        0.0), point2=(2*r2x, 20))
    mdb.models[model_name].sketches['__profile__'].VerticalConstraint(addUndoState=
        False, entity=mdb.models[model_name].sketches['__profile__'].geometry[11])
    mdb.models[model_name].sketches['__profile__'].CoincidentConstraint(
        addUndoState=False, entity1=
        mdb.models[model_name].sketches['__profile__'].vertices[14], entity2=
        mdb.models[model_name].sketches['__profile__'].geometry[11])
    mdb.models[model_name].sketches['__profile__'].copyMirror(mirrorLine=
        mdb.models[model_name].sketches['__profile__'].geometry[11], objectList=(
        mdb.models[model_name].sketches['__profile__'].geometry[7], 
        mdb.models[model_name].sketches['__profile__'].geometry[8], 
        mdb.models[model_name].sketches['__profile__'].geometry[9], 
        mdb.models[model_name].sketches['__profile__'].geometry[10], 
        mdb.models[model_name].sketches['__profile__'].geometry[2], 
        mdb.models[model_name].sketches['__profile__'].geometry[3], 
        mdb.models[model_name].sketches['__profile__'].geometry[4], 
        mdb.models[model_name].sketches['__profile__'].geometry[5], 
        mdb.models[model_name].sketches['__profile__'].vertices[13], 
        mdb.models[model_name].sketches['__profile__'].vertices[0], 
        mdb.models[model_name].sketches['__profile__'].vertices[3]))

    mdb.models[model_name].sketches['__profile__'].ConstructionLine(point1=(4*r2x, 
        0.0), point2=(4*r2x, 20))
    mdb.models[model_name].sketches['__profile__'].VerticalConstraint(addUndoState=
        False, entity=mdb.models[model_name].sketches['__profile__'].geometry[20])
    mdb.models[model_name].sketches['__profile__'].CoincidentConstraint(
        addUndoState=False, entity1=
        mdb.models[model_name].sketches['__profile__'].vertices[22], entity2=
        mdb.models[model_name].sketches['__profile__'].geometry[20])
    mdb.models[model_name].sketches['__profile__'].copyMirror(mirrorLine=
        mdb.models[model_name].sketches['__profile__'].geometry[20], objectList=(
        mdb.models[model_name].sketches['__profile__'].geometry[12], 
        mdb.models[model_name].sketches['__profile__'].geometry[13], 
        mdb.models[model_name].sketches['__profile__'].geometry[14], 
        mdb.models[model_name].sketches['__profile__'].geometry[15], 
        mdb.models[model_name].sketches['__profile__'].geometry[16], 
        mdb.models[model_name].sketches['__profile__'].geometry[17], 
        mdb.models[model_name].sketches['__profile__'].geometry[18], 
        mdb.models[model_name].sketches['__profile__'].geometry[19], 
        mdb.models[model_name].sketches['__profile__'].geometry[7], 
        mdb.models[model_name].sketches['__profile__'].geometry[8], 
        mdb.models[model_name].sketches['__profile__'].geometry[9], 
        mdb.models[model_name].sketches['__profile__'].geometry[10], 
        mdb.models[model_name].sketches['__profile__'].geometry[2], 
        mdb.models[model_name].sketches['__profile__'].geometry[3], 
        mdb.models[model_name].sketches['__profile__'].geometry[4], 
        mdb.models[model_name].sketches['__profile__'].geometry[5], 
        mdb.models[model_name].sketches['__profile__'].vertices[23], 
        mdb.models[model_name].sketches['__profile__'].vertices[32], 
        mdb.models[model_name].sketches['__profile__'].vertices[13], 
        mdb.models[model_name].sketches['__profile__'].vertices[2], 
        mdb.models[model_name].sketches['__profile__'].vertices[3]))
    
    # PATTERN
    mdb.models[model_name].sketches['__profile__'].linearPattern(angle1=0.0, angle2=
        90.0, geomList=(mdb.models[model_name].sketches['__profile__'].geometry[2], 
        mdb.models[model_name].sketches['__profile__'].geometry[3], 
        mdb.models[model_name].sketches['__profile__'].geometry[4], 
        mdb.models[model_name].sketches['__profile__'].geometry[5], 
        mdb.models[model_name].sketches['__profile__'].geometry[6], 
        mdb.models[model_name].sketches['__profile__'].geometry[7], 
        mdb.models[model_name].sketches['__profile__'].geometry[8], 
        mdb.models[model_name].sketches['__profile__'].geometry[9], 
        mdb.models[model_name].sketches['__profile__'].geometry[10], 
        mdb.models[model_name].sketches['__profile__'].geometry[11], 
        mdb.models[model_name].sketches['__profile__'].geometry[12], 
        mdb.models[model_name].sketches['__profile__'].geometry[13], 
        mdb.models[model_name].sketches['__profile__'].geometry[14], 
        mdb.models[model_name].sketches['__profile__'].geometry[15], 
        mdb.models[model_name].sketches['__profile__'].geometry[16], 
        mdb.models[model_name].sketches['__profile__'].geometry[17], 
        mdb.models[model_name].sketches['__profile__'].geometry[18], 
        mdb.models[model_name].sketches['__profile__'].geometry[19], 
        mdb.models[model_name].sketches['__profile__'].geometry[20], 
        mdb.models[model_name].sketches['__profile__'].geometry[21], 
        mdb.models[model_name].sketches['__profile__'].geometry[22], 
        mdb.models[model_name].sketches['__profile__'].geometry[23], 
        mdb.models[model_name].sketches['__profile__'].geometry[24], 
        mdb.models[model_name].sketches['__profile__'].geometry[25], 
        mdb.models[model_name].sketches['__profile__'].geometry[26], 
        mdb.models[model_name].sketches['__profile__'].geometry[27], 
        mdb.models[model_name].sketches['__profile__'].geometry[28], 
        mdb.models[model_name].sketches['__profile__'].geometry[29], 
        mdb.models[model_name].sketches['__profile__'].geometry[30], 
        mdb.models[model_name].sketches['__profile__'].geometry[31], 
        mdb.models[model_name].sketches['__profile__'].geometry[32], 
        mdb.models[model_name].sketches['__profile__'].geometry[33], 
        mdb.models[model_name].sketches['__profile__'].geometry[34], 
        mdb.models[model_name].sketches['__profile__'].geometry[35], 
        mdb.models[model_name].sketches['__profile__'].geometry[36]), number1=1, 
        number2=cells_y, spacing1=20.0, spacing2=r3y-r4y, vertexList=(
        mdb.models[model_name].sketches['__profile__'].vertices[2], 
        mdb.models[model_name].sketches['__profile__'].vertices[3], 
        mdb.models[model_name].sketches['__profile__'].vertices[12], 
        mdb.models[model_name].sketches['__profile__'].vertices[13], 
        mdb.models[model_name].sketches['__profile__'].vertices[23], 
        mdb.models[model_name].sketches['__profile__'].vertices[32], 
        mdb.models[model_name].sketches['__profile__'].vertices[41], 
        mdb.models[model_name].sketches['__profile__'].vertices[42], 
        mdb.models[model_name].sketches['__profile__'].vertices[51], 
        mdb.models[model_name].sketches['__profile__'].vertices[60], 
        mdb.models[model_name].sketches['__profile__'].vertices[69]))
    
    # top & bottom portion
    line_tol=40
    mdb.models[model_name].sketches['__profile__'].Line(point1=(-abs(r3x-r4x)*7-line_tol,0), point2=(abs(r1x-r2x)*8+abs(r3x-r4x)*7+line_tol, 0))
    mdb.models[model_name].sketches['__profile__'].Line(point1=(-abs(r3x-r4x)*7-line_tol,height), point2=(abs(r1x-r2x)*8+abs(r3x-r4x)*7+line_tol, height))
    
    mdb.models[model_name].Part(dimensionality=THREE_D, name='Part-1', type=DEFORMABLE_BODY)
    mdb.models[model_name].parts['Part-1'].BaseShellExtrude(depth=depth_length, sketch=mdb.models[model_name].sketches['__profile__'])
    del mdb.models[model_name].sketches['__profile__']      
    
    #material and property
    #effective density = density * svf
    
    mat_name="Al"
    mdb.models[model_name].Material(name=mat_name)
    mdb.models[model_name].materials[mat_name].Elastic(table=((72000.0, 0.32),))
    mdb.models[model_name].materials[mat_name].Plastic(hardening=JOHNSON_COOK, 
        table=((520.0, 477.0, 0.52, 1.0, 893.0, 403.0), ))
    mdb.models[model_name].materials[mat_name].Density(table=((density, ), ))
    
    mdb.models[model_name].HomogeneousShellSection(idealization=NO_IDEALIZATION, 
        integrationRule=SIMPSON, material=mat_name, name='Section-1', 
        nodalThicknessField='', numIntPts=5, poissonDefinition=DEFAULT, 
        preIntegrate=OFF, temperature=GRADIENT, thickness=thickness, thicknessField='', 
        thicknessModulus=None, thicknessType=UNIFORM, useDensity=OFF)
    mdb.models[model_name].parts['Part-1'].SectionAssignment(offset=0.0, 
        offsetField='', offsetType=MIDDLE_SURFACE, region=Region(
        faces=mdb.models[model_name].parts['Part-1'].faces.getSequenceFromMask(
        mask=('[#ffffffff:6 #7ffff ]', ), )), sectionName='Section-1', 
        thicknessAssignment=FROM_SECTION)
    
    #assembly
    mdb.models[model_name].rootAssembly.DatumCsysByDefault(CARTESIAN)
    mdb.models[model_name].rootAssembly.Instance(dependent=ON, name='Part-1-1', 
        part=mdb.models[model_name].parts['Part-1'])
    
    #ref points
    mdb.models[model_name].rootAssembly.ReferencePoint(point=
        mdb.models[model_name].rootAssembly.instances['Part-1-1'].vertices[214])
    mdb.models[model_name].rootAssembly.ReferencePoint(point=
        mdb.models[model_name].rootAssembly.instances['Part-1-1'].vertices[208])
    
    mdb.models[model_name].RigidBody(bodyRegion=Region(
        faces=mdb.models[model_name].rootAssembly.instances['Part-1-1'].faces.getSequenceFromMask(
        mask=('[#8400210 #1 #0 #200 #0 #10000000 #6 ]', ), )), name='Constraint-1', 
        refPointRegion=Region(referencePoints=(
        mdb.models[model_name].rootAssembly.referencePoints[4], )))
    
    mdb.models[model_name].RigidBody(bodyRegion=Region(
        faces=mdb.models[model_name].rootAssembly.instances['Part-1-1'].faces.getSequenceFromMask(
        mask=('[#0:4 #18400 #e18600 ]', ), )), name='Constraint-2', refPointRegion=
        Region(referencePoints=(
        mdb.models[model_name].rootAssembly.referencePoints[5], )))
    
    # step
    mdb.models[model_name].ExplicitDynamicsStep(improvedDtMethod=ON, name='Step-1', 
        previous='Initial', timePeriod=step_time)
    
    # mdb.models[model_name].ExplicitDynamicsStep(improvedDtMethod=ON, massScaling=((
    #     SEMI_AUTOMATIC, MODEL, AT_BEGINNING, mass_scale, 0.0, None, 0, 0, 0.0, 0.0, 0, 
    #     None), ), name='Step-1', previous='Initial', timePeriod=step_time)
    
    mdb.models[model_name].fieldOutputRequests['F-Output-1'].setValues(numIntervals=
        400, variables=('S','U','LE','RF'))
    
    #contact
    mdb.models[model_name].ContactProperty('fric')
    mdb.models[model_name].interactionProperties['fric'].TangentialBehavior(
        dependencies=0, directionality=ISOTROPIC, elasticSlipStiffness=None, 
        formulation=PENALTY, fraction=0.005, maximumElasticSlip=FRACTION, 
        pressureDependency=OFF, shearStressLimit=None, slipRateDependency=OFF, 
        table=((0.2, ), ), temperatureDependency=OFF)
    mdb.models[model_name].interactionProperties['fric'].NormalBehavior(
        allowSeparation=ON, constraintEnforcementMethod=DEFAULT, 
        pressureOverclosure=HARD)
    mdb.models[model_name].ContactExp(createStepName='Initial', name='Int-1')
    mdb.models[model_name].interactions['Int-1'].includedPairs.setValuesInStep(
        stepName='Initial', useAllstar=ON)
    mdb.models[model_name].interactions['Int-1'].contactPropertyAssignments.appendInStep(
        assignments=((GLOBAL, SELF, 'fric'), ), stepName='Initial')
    
    #BC
    mdb.models[model_name].EncastreBC(createStepName='Initial', localCsys=None, 
        name='BC-1', region=Region(referencePoints=(
        mdb.models[model_name].rootAssembly.referencePoints[5], )))
    
    mdb.models[model_name].VelocityBC(amplitude=UNSET, createStepName='Step-1', 
        distributionType=UNIFORM, fieldName='', localCsys=None, name='BC-2', 
        region=Region(referencePoints=(
        mdb.models[model_name].rootAssembly.referencePoints[4], )), v1=0.0, v2=
        velocity, v3=0.0, vr1=0.0, vr2=0.0, vr3=0.0)
    
    mdb.models[model_name].DisplacementBC(amplitude=UNSET, createStepName='Initial'
        , distributionType=UNIFORM, fieldName='', localCsys=None, name='BC-4', 
        region=Region(
        faces=mdb.models[model_name].rootAssembly.instances['Part-1-1'].faces.getSequenceFromMask(
        mask=(
        '[#f7bffdef #fffffffe #ffffffff #fffffdff #fffe7bff #ef1e79ff #7fff9 ]', ), 
        ), 
        edges=mdb.models[model_name].rootAssembly.instances['Part-1-1'].edges.getSequenceFromMask(
        mask=(
        '[#f3fe8ffd #cff7ffff #ff3ef9fb #ffffffff:2 #fffd7fff #ffffffff:2 #ff1fbfff', 
        ' #ffffffff #fd7fffff #fe1ff3fb #ffffffff #f61fffd7 #ee01fb87 #fffc1fe7', 
        ' #ffffff ]', ), ), 
        vertices=mdb.models[model_name].rootAssembly.instances['Part-1-1'].vertices.getSequenceFromMask(
        mask=(
        '[#9fffe4f9 #fffff3f3 #fc3fffff #ffffffff #ffffff33 #3ffffcc3 #ff30333c', 
        ' #f ]', ), )), u1=UNSET, u2=UNSET, u3=SET, ur1=UNSET, ur2=UNSET, ur3=
        UNSET)
    
    # mdb.models[model_name].SmoothStepAmplitude(data=((0.0, 0.0), (step_time, 1.0)), name=
    #     'Amp-1', timeSpan=STEP)
    # mdb.models[model_name].boundaryConditions['BC-2'].setValues(amplitude='Amp-1')
    
    #mesh
    mdb.models[model_name].parts['Part-1'].seedPart(deviationFactor=0.1, 
        minSizeFactor=0.1, size=mesh_auxetic)
    mdb.models[model_name].parts['Part-1'].generateMesh()
    mdb.models[model_name].parts['Part-1'].setElementType(elemTypes=(ElemType(
        elemCode=S4R, elemLibrary=EXPLICIT, secondOrderAccuracy=OFF), ElemType(
        elemCode=S3R, elemLibrary=EXPLICIT)), regions=(
        mdb.models[model_name].parts['Part-1'].faces.getSequenceFromMask((
        '[#1ffff ]', ), ), ))
    mdb.models[model_name].rootAssembly.regenerate()
    
    #creating set with ids
    # Arbitrary coordinates
    arbitrary_point = (0, mesh_auxetic, mesh_auxetic)
    assembly=mdb.models[model_name].rootAssembly
    session.viewports['Viewport: 1'].setValues(displayedObject=assembly)
    mass=mdb.models[model_name].parts['Part-1'].getMassProperties()['mass']*1e6 #grams
    mass_solid=height*width*depth_length*density*1e6
    
    # svf_real=mass/mass_solid
    # ro_effective=svf_real*density
    
    # Get the model and instance
    myModel = mdb.models[model_name]
    myInstance = myModel.rootAssembly.instances['Part-1-1']
    
    print("\nMODEL OLUSTURULDU -> RUN VERIYORUM")
    # JOB
    if job_run:
        job_object=mdb.Job(activateLoadBalancing=False, atTime=None, contactPrint=OFF, 
            description='', echoPrint=OFF, explicitPrecision=SINGLE, historyPrint=OFF, 
            memory=95, memoryUnits=PERCENTAGE, model=model_name, modelPrint=OFF, 
            multiprocessingMode=DEFAULT, name=model_name, nodalOutputPrecision=SINGLE, 
            numCpus=8, numDomains=8, parallelizationMethodExplicit=DOMAIN, queue=None, 
            resultsFormat=ODB, scratch='', type=ANALYSIS, userSubroutine='', waitHours=
            0, waitMinutes=0)
        job_object.submit(consistencyChecking=OFF)
        job_object.waitForCompletion()
      
        #odb stuff, FORCE-DISPLACEMENT CURVE
        odb = session.openOdb(model_name+".odb")
        
        laststep=odb.steps['Step-1']
        # rp=odb.rootAssembly.referencePoints[4]
        # rp2=odb.rootAssembly.nodeSets['REFERENCE_POINT_        2'] #alt
        # rp1=odb.rootAssembly.nodeSets['REFERENCE_POINT_        1'] #üst
        
        rp=odb.rootAssembly.nodeSets['REFERENCE_POINT_        1']
        
        with open(model_name+".curve","w") as fout:
            pass
        with open(model_name+".curve","a") as fout:
            for x in range(len(laststep.frames)):
                lastframe=laststep.frames[x]
                rf_data=lastframe.fieldOutputs["RF"].getSubset(region=rp)
                u_data=lastframe.fieldOutputs["U"].getSubset(region=rp)
                
                for u,rf in zip(u_data.values,rf_data.values):
                    stress_data=rf.data[1]*-1/(width*depth_length)
                    strain_data=u.data[1]*-1/height
                    fout.write(str(u.data[1]*-1)+","+str(rf.data[1]*-1)+"," +str(strain_data)+","+str(stress_data)+"\n")
        
        #writing
        with open("data.log","a") as fout:
            fout.write(model_name+" "+str(p1)+" "+str(p2)+" "+str(p3)+" "+str(p4)+" " \
                        +str(l1)+" "+str(l2)+" "+str(l3)+" "+str(l4)+" " \
                        +str(height)+" " + str(width)+" "+ str(thickness)+" "\
                        +str(theta1)+" "+str(theta2)+" "+str(svf)+" "+str(ro_effective)+" " \
                        +str(mass)+"\n")
                
        odb.close()
        
        # REMOVING FILES DUE TO STORAGE REASONS
        for file in os.listdir(os.getcwd()):
            if not file.endswith(".inp") and \
            not file.endswith(".curve") and \
            not file.endswith("data.log"):
                try:
                    os.remove(file)
                except:
                    pass
          
        print(str(count) + " COZDUM, DIGER MODELE GECIYORUM EGER VARSA\n\n")
       
then=time.time()
dt=then-now
with open("data.log","a") as fout:
    fout.write("Total time: %.2f seconds" %dt) 
    
print("\n-----------------------DONE-----------------------\nTotal time: %.2f seconds\n" %dt)
print("Empty pass: %.0f" %empty_pass)


