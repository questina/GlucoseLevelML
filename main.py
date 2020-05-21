import subprocess
import numpy as np
import arff
import random as rnd
import pandas as pd
from Summary import *
from imblearn.over_sampling import SMOTE

RANGE_LIST_FILEPATH = 'Ranges.txt'
VOICE_SAMPLE_DIR = 'SoundFiles/7'
ARFF_FILES_DIR = 'ArffFiles'
CV_FOLDER = 'Dataset/CV/'
WEKA_FILE = 'WekaFile.txt'
OUTPUTFOLDER = 'Output'
UsingSMOTE = False
NORMALIZE = True
NormalizeFilter = 'Standardize'
noFolds = 10


class VoiceSample: #class of each voice sample which contains root, filename and the level of glucose
    def __init__(self, root, filename):
        self.root = root
        self.fileName = filename
        self.glucoseLevel = float(filename.split('_')[0].replace(',', '.'))

    def InFolder(self, folderName):
        return self.root == folderName

    # returns the number of interval that was met: 0 - outside the range, 1 - inside the range
    def InRange(self, glucoseLevelRanges):
        if len(glucoseLevelRanges) > 1:
            if self.glucoseLevel < glucoseLevelRanges[0] or self.glucoseLevel > glucoseLevelRanges[1]:
                return 0
            return 1
        else:
            if self.glucoseLevel < glucoseLevelRanges[0]:
                return 0
            return 1


def AddAllWavFilesToList(startFolder): #creates list of elements of class VoiceSample
    fileList = []
    for root, _, files in os.walk(startFolder):
        for f in files:
            if f[-4:] == '.wav':
                fileList.append(VoiceSample(root, f))  # first number in the name of file - glucose level
    return fileList

def ReadWekaOptionsFromLine(line):
    wekaOptions = []
    internalCounter = 0
    tempList = line.split(' ')

    for counter in range(0, len(tempList)):
        if (internalCounter > counter):
            continue
        if (tempList[counter][0] == '\"'):
            internalCounter = counter
            oneLine = tempList[counter][1:] + " "
            internalCounter += 1
            while ('\"' not in tempList[internalCounter]):
                oneLine += tempList[internalCounter] + " "
                internalCounter += 1
            lastLine = tempList[internalCounter][:-1]
            while ('\"' in lastLine):
                lastLine = lastLine[:-1]
            oneLine += lastLine
            internalCounter += 1
            if ('\n' not in oneLine):
                wekaOptions.append(oneLine)
        else:
            if ('\n' not in tempList[counter]):
                wekaOptions.append(tempList[counter])
    return wekaOptions

def ReadAllWekaOptions(filepath):
    print('Reading weka options...')
    containerFile = open(filepath, 'r')
    wekaOptions = list()
    lines = containerFile.readlines()
    for line in lines:
        wekaOptions.append([line.replace('\n', '')])

    return wekaOptions


def LoadAllGlucoseLevelRanges(filepath):
    print('Reading all levels from', filepath)
    glucoseLevelRanges = []
    glucoseLevelsFile = open(filepath, 'r')
    for levelRange in glucoseLevelsFile:
        glucoseLevelRanges.append(list(map(int,
                                           levelRange.replace('[', '').replace(']', '').replace('\n', '').split(','))))
    return glucoseLevelRanges


def SplitVoiceSamplesOnFolders(startFolder, voiceSamples):
    voiceSamplesDict = dict()

    for root, _, _ in os.walk(startFolder):
        fileList = []
        for sample in voiceSamples:
            if (sample.InFolder(root)):
                fileList.append(sample)
        if (len(fileList) != 0):
            directory = root.split('/')[-1]
            voiceSamplesDict[directory] = fileList
    return voiceSamplesDict


def CleanUpAttributes(buffertLines, listOfRanges):
    stringPositions = list()

    attributesStarted = False
    counter = 0
    cleanedLines = list()
    for l in buffertLines[:-1]:
        if (l.startswith('@relation')):
            l = l.replace('SMILEfeaturesLive', 'glucose_level')
        if (l.startswith('@attribute') and attributesStarted != True):
            counter = 0
            attributesStarted = True
        if l.startswith('@attribute emotion unknown'):
            l = l.replace('emotion', 'glucose_level')
            glucoseString = '{'
            for level in listOfRanges:
                glucoseString += level + ','
            glucoseString = glucoseString[:-1] + '}'
            l = l.replace('unknown', glucoseString)
        if 'string' in l:
            stringPositions.append(counter)
        else:
            cleanedLines.append(l)
        counter += 1
    return stringPositions, cleanedLines

def CreateOutputFolder(openSmileConf, glucoseLevelRanges, dataFolderName):
    pathName = 'PersonOutputJ23/' + openSmileConf + '/'
    for level in glucoseLevelRanges:
        pathName = pathName + str(level)
    pathName = pathName[:-1] + '/'
    pathName = pathName + dataFolderName

    if not os.path.exists(pathName):
        os.makedirs(pathName)
    return pathName

def RemoveStringAttributesFromData(line, stringPositions):
    listToRemove = list()
    lineAsList = line.split(',')

    for x in stringPositions:
        listToRemove.append(lineAsList[x])

    tempLine = list()

    for s in lineAsList:
        if s not in listToRemove:
            tempLine.append(s)

    finishedString = ''

    for s in tempLine:
        finishedString += s + ','

    finishedString = finishedString[:-1]

    return finishedString


def CreateStringRepresentationOfGlucoseLevels(glucoseLevelRanges):
    listOfRanges = list()
    if len(glucoseLevelRanges) <= 1:
        listOfRanges.append(str(glucoseLevelRanges[-1]) + '<')
    else:
        for counter in range(0, len(glucoseLevelRanges) - 1):
            listOfRanges.append(str(glucoseLevelRanges[counter]) + '<='
                                + str(glucoseLevelRanges[counter + 1]))
    listOfRanges.append('other')
    return listOfRanges



def RunOpenSmileEmoLarge(voiceSamples, glucoseLevelRanges, pathToDir):
    listOfRanges = CreateStringRepresentationOfGlucoseLevels(glucoseLevelRanges)
    print('list of ranges =', listOfRanges)
    arffFile = open(pathToDir + '/emoLarge.arff', 'w+')

    subProcess = subprocess.Popen(['./opensmile-2.3.0/SMILExtract', '-noconsoleoutput', '-C',
                                   './opensmile-2.3.0/config/emo_large.conf', '-O', 'buffer', '-I',
                                   voiceSamples[0].root + '/' + voiceSamples[0].fileName])
    subProcess.wait()
    buf = open('buffer', 'r')
    lines = buf.readlines()
    stringDataPositions, cleanedLines = CleanUpAttributes(lines[:-1], listOfRanges)
    data = RemoveStringAttributesFromData(lines[-1], stringDataPositions)
    data = data.replace('unknown', listOfRanges[voiceSamples[0].InRange(glucoseLevelRanges)])
    for line in cleanedLines:
        arffFile.write(line)
    arffFile.write('\n')
    arffFile.write(data)
    buf.close()

    for sample in voiceSamples[1:]:
        subProcess = subprocess.Popen(['./opensmile-2.3.0/SMILExtract', '-noconsoleoutput', '-C',
                                       './opensmile-2.3.0/config/emo_large.conf', '-O', 'buffer', '-I',
                                       sample.root + '/' + sample.fileName])
        subProcess.wait()
        buf = open('buffer', 'r')
        lines = buf.readlines()
        data = RemoveStringAttributesFromData(lines[-1], stringDataPositions)
        data = data.replace('unknown', listOfRanges[sample.InRange(glucoseLevelRanges)])
        arffFile.write(data)
        buf.close()
    os.remove('buffer')
    arffFile.close()
    return pathToDir + '/emoLarge.arff', listOfRanges


def NormalizeArffFiles(arffFileName, filterMethod):
    if not NORMALIZE:
        return None
    print("Applying %s filter to %s..." % (filterMethod, 'emoLarge.arff'))
    errorFile = open('errorFile', 'a')
    subProcess = subprocess.Popen([
        'java', '-cp', './weka-3-8-4/weka.jar',
        'weka.filters.unsupervised.attribute.%s' % filterMethod,
        '-i', arffFileName,
        '-o', arffFileName.replace('.arff','_norm.arff'),
    ], stderr=errorFile)

    subProcess.wait()
    errorFile.close()

    return arffFileName.replace('.arff','_norm.arff')

def readArffFile(fileName):
    try:
        arffFile = open(fileName,'r')
        content = arff.load(arffFile)
    finally:
        arffFile.close()
    return content

def CountSamples(filename):
    content = readArffFile(filename)
    return len(content['data'])

def customSplit(no_of_splits, arr, arr_norm = None): #splitting on folders for CV
    if arr_norm is None:
        arr_norm = arr
    itemsPerSplit = int(len(arr) / no_of_splits)
    set_of_indexes = [i for i in range(0,len(arr))]
    rnd.shuffle(set_of_indexes)
    trainTest = []
    testIndexes = []
    trainIndexes = []

    testData = []
    trainData = []

    if itemsPerSplit > 1:
        testIndex = np.array(set_of_indexes[0:itemsPerSplit])
    else:
        testIndex = [0]

    for i in range(no_of_splits):
        if i == no_of_splits-1:
            itemsPerSplit = itemsPerSplit + len(arr)-(no_of_splits*itemsPerSplit)
            testIndex = set_of_indexes[-(itemsPerSplit):]
        else:
            for k in range(itemsPerSplit):
                testIndex[k] = set_of_indexes[k+i*itemsPerSplit]
        for j in range(len(arr)):
            if j in testIndex:
                testIndexes.append(j)
                testData.append(arr[j])
            else:
                trainIndexes.append(j)
                trainData.append(arr_norm[j])

        trainTest.append({
            'fold': i + 1,
            'data': {
                'test': testData,
                'train': trainData
            },
            'indexes': {
                'test': testIndex,
                'train': trainIndexes
            }
        })

        testIndexes = []
        trainIndexes = []
        testData = []
        trainData = []

    return trainTest

def CleanArff(arffFileName):
    print('Run feature selection...')
    pathSplit = arffFileName.split('/')

    cleanedFile = '{}/cleaned{}'.format('/'.join(pathSplit[:-1]), pathSplit[-1])
    errorFile = open('errorFile', 'a')

    subProcess = subprocess.Popen([
        'java', '-cp', './weka-3-8-4/weka.jar',
        'weka.filters.supervised.attribute.AttributeSelection',
        '-E', 'weka.attributeSelection.CfsSubsetEval',
        '-S', 'weka.attributeSelection.BestFirst',
        '-i', arffFileName,
        '-o', cleanedFile
    ], stderr=errorFile)

    subProcess.wait()
    errorFile.close()

    return cleanedFile

def getIndexesOfFeatures(cleanedTrainFile, testFile):
    print('Get a list of wanted features')
    cleanedFileContent = readArffFile(cleanedTrainFile);
    testFileContent = readArffFile(testFile)

    selectedFeatures = cleanedFileContent['attributes']
    allFeaturesOnTestFile = testFileContent['attributes']

    featuresToKeep = []

    for index, feature in enumerate(allFeaturesOnTestFile):
        if feature in selectedFeatures:
            featuresToKeep.append(index+1)

    return featuresToKeep

def selectFeaturesFromTestArrfFile(arffFileName, featuresToKeep):
    print('Run removing features from {}...'.format(arffFileName))
    pathSplit = arffFileName.split('/')

    cleanedFile = '{}/cleaned{}'.format('/'.join(pathSplit[:-1]), pathSplit[-1])
    errorFile = open('errorFile', 'a')

    indexesOfFeaturesToKeep = ','.join(list(map(lambda index: str(index), featuresToKeep)))
    subProcess = subprocess.Popen([
        'java', '-cp', './weka-3-8-4/weka.jar',
        'weka.filters.unsupervised.attribute.Remove',
        '-R', indexesOfFeaturesToKeep,
        '-V',
        '-i', arffFileName,
        '-o', cleanedFile
    ], stderr=errorFile)

    subProcess.wait()
    errorFile.close()

    return cleanedFile

def PreparingCVS(arffFileName, arffFileNameNorm = None):
    FOLDER_PREFIX = CV_FOLDER
    createFolder(FOLDER_PREFIX)
    FOLDER_PREFIX += 'fold'

    content = readArffFile(arffFileName)
    if arffFileNameNorm is not None:
        content_norm = readArffFile(arffFileNameNorm)
    else:
        content_norm = None
    dataObject = customSplit(noFolds, content['data'], content_norm['data'] if content_norm is not None else None)
    for i in range(noFolds):
        trainObj = {
            'relation': content['relation'],
            'attributes': content['attributes'],
            'data': dataObject[i]['data']['train']
        }

        testObj = {
            'relation': content['relation'],
            'attributes': content['attributes'],
            'data': dataObject[i]['data']['test']
        }
        fold = i + 1
        dirPath = '{}{}'.format(FOLDER_PREFIX, fold)
        dirPath = createFolder(dirPath)
        trainFileName = '{}{}/train.arff'.format(FOLDER_PREFIX, fold)
        testFileName = '{}{}/test.arff'.format(FOLDER_PREFIX, fold)
        trainFile = open(trainFileName, 'w+')
        testFile = open(testFileName, 'w+')
        arff.dump(trainObj, trainFile)
        arff.dump(testObj, testFile)
        cleanedTrainFile = CleanArff(trainFileName)
        wantedFeatures = getIndexesOfFeatures(cleanedTrainFile, testFileName)
        cleanedTestFile = selectFeaturesFromTestArrfFile(testFileName, wantedFeatures)


def MakeArffFileFromPandas(arffFileName, data):
    arffFile = open(arffFileName, 'r')
    arffFile_SMOTE = open(arffFileName.replace('.arff','_SMOTE.arff'), 'w+')
    lines = arffFile.readlines()
    for line in lines:
        if not line.startswith('@data'):
            arffFile_SMOTE.write(line)
        else:
            arffFile_SMOTE.write(line)
            break
    arffFile_SMOTE.write('\n')
    arffFile.close()
    for row in data:
        dataline = ''
        for num in row:
            dataline += str(num) + ','
        dataline = dataline[:-1] + '\n'
        arffFile_SMOTE.write(dataline)
    arffFile_SMOTE.close()

def useSmote(arffFileName):
    print('using SMOTE')
    arffFile = open(arffFileName,'r')
    lines = arffFile.readlines()
    StartData = False
    data = []
    columns = []
    for line in lines:
        if line.startswith('@') and not line.startswith('@relation') and not line.startswith('@data'):
            tmp = line.replace('@attribute ', '').replace('\n', '')
            tmp = tmp[:tmp.find(' ')]
            columns.append(tmp)
        if line.startswith('@data'):
            StartData = True
            continue
        if StartData:
            if not line.startswith('\n'):
                arr = line.replace(' ', '').replace('\n', '').split(',')
                data.append(arr)
    arffFile.close()
    data = np.array(data)
    df = pd.DataFrame(data, columns=columns)
    X = df.drop("glucose_level", axis=1)
    y = df[["glucose_level"]]
    oversample = SMOTE(k_neighbors=1)
    X, y = oversample.fit_resample(X, y)
    new_data = np.array(pd.concat([X, y], axis=1))
    MakeArffFileFromPandas(arffFileName,new_data)
    return arffFileName.replace('.arff','_SMOTE.arff')


def RunWekaInstanceTTS(options, trainFile, testFile, outputPath, outputIdentifier):
    print('Running Weka Instance in ', outputPath)
    errorFile = open('errorFile', 'a')
    outputFile = open(outputPath + '/WekaOutput_' + str(outputIdentifier) + '.wd', 'w+')
    if UsingSMOTE:
        trainFile = useSmote(trainFile)
    print(trainFile)
    print('options =', options)
    subProcess = subprocess.Popen(['java', '-cp', './weka-3-8-4/weka.jar'] + options + [
        '-t', trainFile,
        '-T', testFile
    ], stdout=outputFile, stderr=errorFile)
    subProcess.wait()
    outputFile.close()
    errorFile.close()

def runCrossValidation(classifiers, arffFileName, arffFileNameNorm = None):
    print('Running Cross Validation for {} with classifier {}'.format(arffFileName, classifiers))
    NumInstances = CountSamples(arffFileName)
    print(NumInstances)
    PreparingCVS(arffFileName, arffFileNameNorm)
    print('Train and test data is prepared')
    print(classifiers)
    createFolder(OUTPUTFOLDER)


def runClassifiers(classifiers,level):
    for classifier in classifiers:
        if len(classifier) > 0:
            _classifier = classifier[0].split('.')[-1]
            pathName = createFolder(
                '{}/{}/Custom/{}/{}'.format(OUTPUTFOLDER, 'CrossValidation', level[0], _classifier))

            for subdir, dirs, files in os.walk(CV_FOLDER):
                for dir in dirs:
                    testArff = '{}{}/{}'.format(subdir, dir, 'cleanedtest.arff')
                    trainArff = '{}{}/{}'.format(subdir, dir, 'cleanedtrain.arff')

                    outputIdentifier = dir.split('_')[-1]
                    outputIdentifier = '{}_{}'.format(classifier[0].split('.')[-1], outputIdentifier)

                    print('Running cross validation for {}'.format(outputIdentifier))

                    RunWekaInstanceTTS(classifier, trainArff, testArff, pathName, outputIdentifier)
        else:
            print('Empty line added to the WekaFile.txt. Please remove')


def AttributeExtraction():
    voiceSamples = AddAllWavFilesToList(VOICE_SAMPLE_DIR)
    levels = LoadAllGlucoseLevelRanges(RANGE_LIST_FILEPATH)
    level = levels[0]
    pathName = CreateOutputFolder('emo_large', level, 'All')
    arffFileName, list_of_ranges = RunOpenSmileEmoLarge(voiceSamples, level, pathName)
    return level, arffFileName, list_of_ranges


def UAR(OutputFolder, list_of_ranges):
    outputfile = open(OutputFolder, "r")
    lines = outputfile.readlines()
    for line in lines:
        if line.find(list_of_ranges[0]) != -1:
            new_line = line.split('\t')
            TP = int(new_line[0])
            FP = int(new_line[1])
        if line.find(list_of_ranges[1]) != -1:
            new_line = line.split('\t')
            FN = int(new_line[0])
            TN = int(new_line[1])
    print(OutputFolder.split('/')[-2], end = ': ')
    outputfile.close()
    if TP-FP >= 0 and TN-FN >= 0:
        return (TP/(TP+FP)+TN/(TN+FN))/2
    else:
        print('Confusion matrix is not relevant')
        return -1

def main():
    classifier = ReadAllWekaOptions(WEKA_FILE)
    level, testFileName, list_of_ranges = AttributeExtraction()
    arffFileNameNorm = NormalizeArffFiles(testFileName, NormalizeFilter)
    runCrossValidation(classifier, testFileName, arffFileNameNorm)
    runClassifiers(classifier, level)
    SummerizeOutput()
    return 0


if __name__ == '__main__':
    print('Starting...')
    main()
    print('Finished...')
