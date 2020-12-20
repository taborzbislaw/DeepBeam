import numpy as np
import config
from scipy.interpolate import interpn
import os
from sklearn.decomposition import PCA

#########################################################################################
# Functions reads Primo 3D dose file
# and returns doses in 3D dose array and a grid of coordinates.
# It is assumed that the dose was calculated over a grid with regular spacing

def readSimulatedDoseFile(name):
    f = open(name,'r')
    items = f.readlines()
    f.close()

    shapes = list(map(int,items[7].split()[1:]))
    steps = list(map(float,items[9].split()[1:]))
    #print(items[7].split())
    #print(items[9].split())
    N = shapes[0]

    grid = (np.arange(steps[0]/2,shapes[0]*steps[0],steps[0]),np.arange(steps[1]/2,shapes[1]*steps[1],steps[1]),np.arange(steps[2]/2,shapes[2]*steps[2],steps[2]))
    lines = [ l for l in items if not (l.startswith('#') or len(l)<5)]

    #print(len(lines))
    dose = np.zeros((N,N,N),dtype=np.float32)

    for i,l in enumerate(lines):
        z = i//(N*N)
        y = (i-z*N*N)//N
        x = i - z*N*N - y*N
        dose[x,y,z] = float(l.split(' ')[0])

    return grid,dose

#########################################################################################

#########################################################################################
# Function
def createProfiles(doseFileName):
    grid,dose = readSimulatedDoseFile(doseFileName)

    #interpolation grid
    x = [s for s in np.arange(config.minSimulatedRange,config.minSimulatedRange + config.spaceStep/2.,config.spaceStep)]  

    pointsZ = [np.array([config.midPoint, config.midPoint, s]) for s in x]
    profiles = [interpn(grid, dose, pointsZ)]

    for d in config.profileDepths:
        pointsX = [np.array([s, config.midPoint, d]) for s in x]
        pointsY = [np.array([config.midPoint, s, d]) for s in x]
        profiles.append((interpn(grid, dose, pointsX)+interpn(grid, dose, pointsY))/2)

    MAX = 100
    m = np.max(profiles[0])

    for n,profile in enumerate(profiles):
        profiles[n] = profile/m*MAX

    return x,profiles


#########################################################################################
#Helper function
def createName(t,ext):
    name = '_'.join(t) + ext
    return name
#########################################################################################


#########################################################################################
# Read simulated 1D profiles from npz files

# For each data point in a points list a file with simulated profiles is read for three fields: 3x3, 10x10, and 30x30
# The simulations were run for 50x50x50cm water phantom with voxel size equal to 0.5x0.5x0.5cm
# The center of the first voxel is at (0.25,0.25,0.25) the last one is at (49.75,49.75,49.75)

# Each file contains six simulated profiles: 
# 1. central depth profile dose[25,25,x]
# 2. five lateral profiles at depths in [1.4, 5, 10, 20, 30]cm
# Each lateral profile is an average over two perpendicular profiles i.e. it is equal to (dose[25,x,depth]+dose[x,25,depth])*0.5

# The profiles are interpolated at 0.1cm in the range from 0.3cm to 49.7cm, both ends included
#
# readProfiles returns a list 'profiles' of three arrays each of size (495,6): 
# profiles[0] for 3x3 field, profiles[1] for 10x10 field, profiles[2] for 30x30 field, 

def readProfiles(DIR, points):
    profiles3 = []
    profiles10 = []
    profiles30 = []

    for point in points:
        name = DIR + createName(point+('3',),'.npz')
        file = np.load(name)
        profiles3.append(file[file.files[1]])

        name = DIR + createName(point+('10',),'.npz')
        file = np.load(name)
        profiles10.append(file[file.files[1]])

        name = DIR + createName(point+('30',),'.npz')
        file = np.load(name)
        profiles30.append(file[file.files[1]])

    profiles30 = np.asarray(profiles30)
    profiles10 = np.asarray(profiles10)
    profiles3 = np.asarray(profiles3)

    profiles = []
    profiles.append(profiles3)
    profiles.append(profiles10)
    profiles.append(profiles30)

    return profiles

#########################################################################################

# reads measured profiles from *.dat files
def readMeasuredDoseFile(name):  #returns array of shape (N,4) - first three columns are coordinates and the last one is measured dose
    f = open(name,'r')
    items = f.readlines()
    f.close()
    lines = [ list(map(float,l.split('    ')[1:]))[0:4] for l in items if not (l.startswith('#') or len(l)<5)]
    measuredDose = np.asarray(lines,dtype=np.float32)
    return measuredDose

#########################################################################################
def allPCAResults():
    dataPoints = [(str(e),str(se),str(s),str(an)) for e in config.simulatedEnergies for se in config.simulatedEnergyDispersions for s in config.simulatedSourceSizes 
              for an in config.simulatedAngularDivergences]
    profiles = readProfiles(config.profileDIR,dataPoints)

    means = []
    for field in range(3):
        means.append(np.mean(profiles[field],0))

    diff = []
    for field in range(3):
        diff.append(profiles[field] - np.stack([means[field] for _ in range(profiles[field].shape[0])]))

    fieldFeatures = []
    fieldPCAModels = []
    for field in range(3):
        profilePCAModels = []
        profileFeatures = []
        for slice in range(diff[field].shape[1]):
            pca = PCA(n_components=config.numbefOfPCAFeatures)
            X = diff[field][:,slice,:]
            pca.fit(X)
            X_projected = pca.transform(X)
            profileFeatures.append(X_projected)
            profilePCAModels.append(pca)
        fieldFeatures.append(profileFeatures)
        fieldPCAModels.append(profilePCAModels)
        
    return means,fieldFeatures,fieldPCAModels

##################################################################################

def reconstruct(xStart,allMeans,allFieldFeatures,allFieldPCAModels):

    x0 = np.zeros((4,),dtype=np.float)
    x0[0] = xStart[0]
    x0[1] = 0.5
    x0[2] = xStart[2]
    x0[3] = xStart[3]

    x0 = [p if p > c else c for (p,c) in zip(x0,config.minimum)]
    x0 = [p if p < c else c for (p,c) in zip(x0,config.maximum)]

    values = np.zeros((len(config.simulatedEnergies),len(config.simulatedEnergyDispersions),
                       len(config.simulatedSourceSizes),len(config.simulatedAngularDivergences),3),dtype=np.float)
    reconstructed_profiles = []
    for nfield,FIELD in enumerate(config.analyzedProfiles):
        if FIELD != None:
            for PROFILE in FIELD:
                n = 0
                for nE,E in enumerate(config.simulatedEnergies):
                    for nse,se in enumerate(config.simulatedEnergyDispersions):
                        for ns,s in enumerate(config.simulatedSourceSizes):
                            for na,a in enumerate(config.simulatedAngularDivergences):
                                np.copyto(values[nE,nse,ns,na,:],allFieldFeatures[nfield][PROFILE][n,:])
                                n = n+1

                grid = (config.simulatedEnergies, config.simulatedEnergyDispersions, config.simulatedSourceSizes,config.simulatedAngularDivergences)
                features = interpn(grid, values, x0)
                X_Recon = allFieldPCAModels[nfield][PROFILE].inverse_transform(features)[0,:] + allMeans[nfield][PROFILE]
                reconstructed_profiles.append(X_Recon)
    return reconstructed_profiles

####################################################################################
def difference(xStart,clinicalProfiles,allMeans,allFieldFeatures,allFieldPCAModels):

    x0 = np.zeros((4,),dtype=np.float)
    x0[0] = xStart[0]
    x0[1] = 0.5
    x0[2] = xStart[2]
    x0[3] = xStart[3]

    x0 = [p if p > c else c for (p,c) in zip(x0,config.minimum)]
    x0 = [p if p < c else c for (p,c) in zip(x0,config.maximum)]

    values = np.zeros((len(config.simulatedEnergies),len(config.simulatedEnergyDispersions),
                       len(config.simulatedSourceSizes),len(config.simulatedAngularDivergences),3),dtype=np.float)
    reconstructedProfiles = []
    for nfield,FIELD in enumerate(config.analyzedProfiles):
        if FIELD != None:
            for PROFILE in FIELD:
                n = 0
                for nE,E in enumerate(config.simulatedEnergies):
                    for nse,se in enumerate(config.simulatedEnergyDispersions):
                        for ns,s in enumerate(config.simulatedSourceSizes):
                            for na,a in enumerate(config.simulatedAngularDivergences):
                                np.copyto(values[nE,nse,ns,na,:],allFieldFeatures[nfield][PROFILE][n,:])
                                n = n+1

                grid = (config.simulatedEnergies, config.simulatedEnergyDispersions, config.simulatedSourceSizes,config.simulatedAngularDivergences)
                features = interpn(grid, values, x0)
                X_Recon = allFieldPCAModels[nfield][PROFILE].inverse_transform(features)[0,:] + allMeans[nfield][PROFILE]
                reconstructedProfiles.append(X_Recon)
    
    sum = 0.0
    for n in range(len(clinicalProfiles)):
        diff = np.sum((clinicalProfiles[n][config.allRanges[n][0]:config.allRanges[n][1]] - reconstructedProfiles[n][config.allRanges[n][0]:config.allRanges[n][1]])*
                      (clinicalProfiles[n][config.allRanges[n][0]:config.allRanges[n][1]] - reconstructedProfiles[n][config.allRanges[n][0]:config.allRanges[n][1]]))
        sum = sum + diff

    return sum

