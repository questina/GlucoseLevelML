import re
import os

def createFolder(folderPath):
    if not os.path.exists(folderPath):
        os.makedirs(folderPath)

    return folderPath

def __createSummaryFolderFor(summaryFolder, reportDirectory):
    pathSplit = reportDirectory.split('/')
    dirPath = '{}/{}'.format(summaryFolder, '/'.join(pathSplit[1:]))

    return createFolder(dirPath)


# create a dictionary of all the files in their respective directories
def __createDirectoryDictionary(reportDirectory):
    dirDict = dict()

    for sub, dirs, files in os.walk(reportDirectory):
        for file in files:
            if sub in dirDict:
                dirDict[sub].append(file)
            else:
                dirDict[sub] = [file]

    return dirDict


# prepare/geneate summary of the files
def __getSummaryOf(repGroup, reportFilePath, addTestResult, isCustom, isCrossPersonTest=False):
    summaryReportLines = list()

    if isCustom:
        delimeters = {
            'entryCueText': '=== Error on test data ===',
            'start': '=== Confusion Matrix ===',
            'end': 'Time taken to test model on test data'
        }
    else:
        delimeters = {
            'entryCueText': '=== Stratified cross-validation ===',
            'start': '=== Confusion Matrix ===',
            'end': 'Time taken to perform cross-validation'
        }

    if isCrossPersonTest:
        delimeters = {
            'entryCueText': '=== Error on test data ===',
            'start': '=== Error on test data ===',
            'end': 'Time taken to test model on test data'
        }

    with open(reportFilePath, 'r', encoding='utf-8') as reportFile:
        lines = reportFile.readlines()
        entryCueFound = False
        ignoreLine = True
        pathSplit = repGroup.split('/')

        sectionTitle = reportFilePath.split('/')[-1].replace('.wd', '');
        print('pathSplit = ', pathSplit)
        sectionTitle = '{}_level {}_for_{}\n'.format(sectionTitle, pathSplit[-2], pathSplit[-1]).replace('_',
                                                                                                         ' ').upper()

        summaryReportLines.append(sectionTitle)
        summaryReportLines.append('-----------------------------------------------------------\n')

        for line in lines:
            if entryCueFound == False:
                entryCueFound = (line.strip() == delimeters['entryCueText'])

            if (entryCueFound == True):
                if ignoreLine == True and line.strip() == delimeters['start']:
                    ignoreLine = False

                if isCustom and ignoreLine == False and line.split(':')[0].strip() == delimeters['end']:
                    ignoreLine = True
                    break

                if addTestResult == False and ignoreLine == False and line.split(':')[0].strip() == delimeters['end']:
                    ignoreLine = True
                    break

                if ignoreLine == False:
                    summaryReportLines.append(line)
        # else:
        #	entryCueFound = (line.strip() == delimeters['entryCueText'])

        summaryReportLines.append('\n')

    reportFile.close()

    return summaryReportLines


# write the summaries into an output file
def __writeSummary(summaryFilePath, summaryReportLines):
    try:
        summaryFile = open(summaryFilePath, 'w+', encoding='utf-8')
        summaryFile.write(''.join(summaryReportLines))
        summaryFile.close()
    except FileNotFoundError:
        print(summaryFilePath + " does not exist")


# merge the reports of the person tests into one file
def __getPersonTestSummaryMerge(reportFilePath):
    content = []

    with open(reportFilePath, 'r', encoding='utf-8') as reportFile:
        content.extend(reportFile.readlines())

    content.append('\n')

    return content


# run summary for the person cross test output folder
def __summerizePersonCrossTest(summaryFolder, dirDict, addTestResult=False):
    __summerize(summaryFolder, dirDict, addTestResult)

    personTestSummaryFolder = '{}/{}'.format(summaryFolder, 'PersonCrossTest')
    createFolder(personTestSummaryFolder)

    summaryDict = dict()

    for sub, dirs, files in os.walk(personTestSummaryFolder):
        for file in files:
            filePath = '{}/{}'.format(sub, file)
            key = '/'.join(sub.split('/')[:-1])

            if key in summaryDict:
                summaryDict[key].extend(__getPersonTestSummaryMerge(filePath))
            else:
                summaryDict[key] = __getPersonTestSummaryMerge(filePath)

    mergedSummary = [];

    for key in summaryDict:
        summaryPath = createFolder('{}/Summary'.format(key))
        summaryFilePath = '{}/summary.txt'.format(summaryPath)
        __writeSummary(summaryFilePath, summaryDict[key])
        mergedSummary.extend(summaryDict[key])

    __writeSummary('{}/mergedSummary.txt'.format(personTestSummaryFolder), mergedSummary)


# create the summaries of all the output files
def __summerize(summaryFolder, dirDict, addTestResult=False):
    for repGroup in dirDict:
        subSplit = repGroup.split('/')
        isCustom = 'Custom' in subSplit

        '''if isCustom:
            orderdFiles = {}

            # TODO:: recreate groups

            for file in dirDict[repGroup]:
                orderdFiles[int(file.split('_')[2].replace('.wd', ''))] = file

            dirDict[repGroup] = [orderdFiles[key] for key in sorted(orderdFiles.keys())]'''

        '''if isCustom:
            models = defaultdict(list)

            for file in dirDict[repGroup]:
                model = file.split("_")[1]

                if model in models:
                    models[model].append(file)
                else:
                    models[model] = [file]
            print(models)'''

        summaryPath = __createSummaryFolderFor(summaryFolder, repGroup)
        summaryReportLines = list()

        for file in dirDict[repGroup]:
            if file.split('.')[-1] == 'wd':
                reportFilePath = '{}/{}'.format(repGroup, file)

                isCrossPersonTest = 'PersonCrossTest' in subSplit
                summaryReportLines.extend(
                    __getSummaryOf(repGroup, reportFilePath, addTestResult, isCustom, isCrossPersonTest))
            else:
                print('Invalid report file type. SKIPPED!...')

        summaryFilePath = '{}/{}'.format(summaryPath, 'summary.txt')
        __writeSummary(summaryFilePath, summaryReportLines)


# generates summary report on all the output files
def __genAllReports(summaryFolder, dirDict, addTestResult):
    __summerize(summaryFolder, dirDict, addTestResult)
    #__summerizePersonCrossTest(summaryFolder, dirDict, addTestResult)


def summarizeCrossValidationReports(path):
    print('SUMMERIZING REPORTS... ')
    LINESOFINTEREST = "^\s+\d.*$[\d<]*"

    for root, dirs, files in os.walk(path):
        for file in files:
            if not file[:-4] == 'totalled':
                filePath = '{}/{}'.format(root, file)

                print('Summing up {}'.format(filePath))
                with open(filePath, 'r', encoding='utf-8') as summaryFile:
                    lines = summaryFile.readlines()
                    TP, TN, FP, FN = 0, 0, 0, 0
                    classA, classB = None, None

                    for line in lines:
                        if re.match(LINESOFINTEREST, line):
                            lineArray = list(filter(lambda i: i is not '', line.strip().split(' ')))

                            if lineArray[3] == 'a':
                                if classA == None:
                                    classA = lineArray[-1]

                                TP += int(lineArray[0])
                                FN += int(lineArray[1])
                            else:
                                if classB == None:
                                    classB = lineArray[-1]

                                FP += int(lineArray[0])
                                TN += int(lineArray[1])

                    with open('{}/totalled{}'.format(root, file[-4:]), 'w',
                              encoding='utf-8') as totalledConfusionMatrixFile:
                        rootSplit = root.split('/');

                        content = [
                            'Summarized confusion matrix for {} at level {}\n'.format(rootSplit[4].upper(),
                                                                                      rootSplit[3]),
                            '\n',
                            ' a \t b \t <-- classified as\n',
                            '{}\t{}\t{}\n'.format(TP, FN, classA),
                            '{}\t{}\t{}\n'.format(FP, TN, classB),
                        ]

                        totalledConfusionMatrixFile.writelines(content)

                    totalledConfusionMatrixFile.close()

                summaryFile.close()


# triggers report generation for the summaryFolder submitted.
def summarizeOutputFor(summaryFolder, reportDirectory, addTestResult=False):
    if not os.path.exists(reportDirectory):
        raise Exception('{} does not exits. Check the filepath'.format(reportDirectory))

    dirDict = __createDirectoryDictionary(reportDirectory)

    outputType = reportDirectory.split('/')[-1]
    print('Consolidating reports for {}s'.format(outputType))

    {
        'Output': lambda: __genAllReports(summaryFolder, dirDict, addTestResult),
        'CrossValidation': lambda: __summerize(summaryFolder, dirDict, addTestResult),
        'PersonCrossTest': lambda: __summerizePersonCrossTest(summaryFolder, dirDict, addTestResult),
    }[outputType]()

    print('Done')

def SummerizeOutput():
    print('Generating summary files from the outputs')
    summaryFolder = createFolder('OutputSummary')

    ALL_OUTPUT = 'Output'
    CROSS_VAL_OUTPUT = 'Output/CrossValidation'
    PERSON_CROSS_TEST_OUTPUT = 'Output/PersonCrossTest'

    # summarize all the reports in the output folder
    summarizeOutputFor(summaryFolder, ALL_OUTPUT, addTestResult=True)
    #summarizeOutputFor(summaryFolder, CROSS_VAL_OUTPUT, addTestResult=False)
    # summarizeOutputFor(summaryFolder, PERSON_CROSS_TEST_OUTPUT, addTestResult=False)

    summarizeCrossValidationReports('OutputSummary/CrossValidation/Custom')