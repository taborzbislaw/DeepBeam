from scipy.optimize import Bounds

#############################################################################################
# Custom settings of experimental design: selection of profiles used for training and testing

#For training and testing based on all simulated data
#analyzedProfiles = [[0,1,2,3,4,5],[0,1,2,3,4,5],[0,1,2,3,4,5]]
#analyzedRanges = [[[0,495],[0,495],[0,495],[0,495],[0,495],[0,495]],
#                  [[0,495],[0,495],[0,495],[0,495],[0,495],[0,495]],
#                  [[0,495],[0,495],[0,495],[0,495],[0,495],[0,495]]]


#For clinical testing based on data in ./Measured/Method1
analyzedProfiles = [[0,1],[0,1],[1]]
analyzedRanges = [[[0,348],[177,318]],
                  [[0,348],[132,363]],
                  [[2,493]]]

#For clinical testing based on data in ./Measured/Method2
#analyzedProfiles = [None,[0,3],[1,3]]
#analyzedRanges = [None,[[0,298],[159,336]],[[65,430],[50,445]]]

#For clinical testing based on data in ./Measured/Method3
#analyzedProfiles = [None,[0,1,3],[0,1,3]]
#analyzedRanges = [None,[[0,298],[166,329],[159,336]],[[0,297],[65,430],[50,445]]]

    
allRanges = [r for field in analyzedRanges if field!=None for r in field]
    
trainingGoals = [0,1,2,3]  #I am training models for energy, FWHM of energy spectrum, source size and angular divergence                

#################################

profileDIR = './TrainingProfiles/'
saveDIR = './Models/'
modelDIR = './Models/'
testProfilesDIR = './TestProfiles/'

meansFileName = 'trainMeans.npy'
goalsFileName = 'trainGoals.npy'
trainFeaturesFileName = 'trainFeatures.npy'
groundTruthFileName = 'testGoals.txt'

SEED = 10

#################################

simulatedEnergies = [5.6, 5.8, 6.0, 6.2, 6.4]
simulatedEnergyDispersions = [0, 0.5, 1]
simulatedSourceSizes = [0.0, 0.1, 0.2, 0.3, 0.4]
simulatedAngularDivergences = [0 , 1 , 2, 3]
simulatedFileSizes = [3,10,30]

minimum = [5.6,0.0,0.0,0.0]
maximum = [6.4,1.0,0.4,3.0]

bounds = Bounds([5.6,0.0,0.0,0.0],[6.4,1.0,0.4,3.0])

#################################

numbefOfPCAFeatures = 3

#################################
# Phantom

minSimulatedRange = 0.3
maxSimulatedRange = 49.7
spaceStep = 0.1
numOfSimulatedProfileSamples = 495
midPoint = 25      # half of a cubic phantom size
profileDepths = [1.4,5.,10.,20.,30.]

#################################
#Fraction of simulated data used for training

FRACTION = 1.0
