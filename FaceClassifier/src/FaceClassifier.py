# FaceClassifier class takes a series of train face images, sets weights, and predicts if an image is a face or not (via test face images)
# @author Earl Lontok

# PERCEPTRON CLASSIFICATION ALGORITHM - f(x) = w1(f1) + w2(f2) + w3(f3) + ... + wn(fn)
# this approach defines each feature to be the pixels comprising each 70 x 60 image
# therefore, each f is either 1 (if pixel is '#') or 0 (if pixel is ' ')

################################ GLOBAL VARIABLES ################################

# each image in the image files spans 70 rows and 60 columns
totalRows = 70 
totalCols = 60

# these lists will store all faces and labels from the input TRAIN files
# a label is provided by humans - a label is essentially the "correct answer" - 0 = not face, 1 = face
trainFaces = []
trainLabels = []

# a weight is associated with each feature (pixel in the 70 x 60 image) - these weights and the bias will be adjusted during the training process
# once the perceptron classification algorithm has been trained, these weights will be used to determine if an image is a face or not
weights = [[0] * totalCols for i in range(totalRows)]
bias = 0

# these lists will store all faces and labels from the input TEST files
testFaces = []
testLabels = []
totalCorrect = 0

################################ TESTING ################################

# iterates through all input test faces and determines whether each face is a face or not
def test():

    global totalCorrect
    for i in range(0, len(testFaces)):
        face = testFaces[i]
        function = calcFunc(face)
        label = testLabels[i]
        if function <= 0 and label == 0:
            print("CORRECT - Predicted: FALSE | Answer: FALSE")
            totalCorrect += 1
        elif function > 0 and label == 1:
            print("CORRECT - Predicted: TRUE | Answer: TRUE")
            totalCorrect += 1
        elif function <= 0 and label == 1:
            print("INCORRECT - Predicted: FALSE | Answer: TRUE")
        else:
            print("INCORRECT - Predicted: TRUE | Answer: FALSE")

################################ TRAINING ################################

# iterates through all input train faces several times and sets the weights
def train():

    for i in range(0, 10): # iterating through all faces several times will result in better weights - results in a better algorithm
        for j in range(0, len(trainFaces)): # iterate through all input train faces
            face = trainFaces[j]
            function = calcFunc(face)
            label = trainLabels[j]
            if function <= 0 and label == 1: # algorithm says not face but answer is face - increase weights
                increaseWeights(face)
            elif function > 0 and label == 0: # algorithm says face but answer is not face - decrease weights
                decreaseWeights(face)

            # there are 2 more cases: (function <= 0 and label == 0) and (function > 0 and label == 1)
            # in the perceptron classification algorithm, the weights should remain the same if the function and label agree
            # therefore, these 2 cases are omitted

# calculates the function for a given face
def calcFunc(face):

    # for a given feature, there is an associated weight
    # a feature is defined by pixel - f = 1 if pixel is '#' and f = 0 if pixel is ' '
    # because each feature is either 1 or 0, only the weight is considered if a pixel contains a '#'
    # for example, w * (1) = w and w * (0) = 0

    function = 0
    for i in range(0, len(face)):
        for j in range(0, len(face[i]) - 1): # the -1 accounts for the new line symbol (\n) at the end of each line
            c = face[i][j]
            if c == '#':
                weight = weights[i][j]
                function += weight
    return function + bias

# increases the weights - if a pixel contains a '#', then the associated weight is incremented by 1
def increaseWeights(face):

    global bias
    for i in range(0, len(face)):
        for j in range(0, len(face[i]) - 1): # the -1 accounts for the new line symbol (\n) at the end of each line
            c = face[i][j]
            if c == '#':
                weights[i][j] += 1
    bias += 1

# decreases the weights - if a pixel contains a '#', then the associated weight is decremented by 1
def decreaseWeights(face):

    global bias
    for i in range(0, len(face)):
        for j in range(0, len(face[i]) - 1): # the -1 accounts for the new line symbol (\n) at the end of each line
            c = face[i][j]
            if c == '#':
                weights[i][j] -= 1
    bias -= 1

################################ TRAIN FILE EXTRACTION ################################

# parses the train file and stores each face into the trainFaces list
def extractTrainFaces(filePath):

    file = open(filePath, "r")
    ctr = 0
    face = []
    for line in file:
        face.append(line)
        ctr += 1
        if ctr != 0 and ctr % totalRows == 0: # each face spans 70 rows and 60 cols - a face is stored and reset every 70 rows
            trainFaces.append(face)
            face = []

# parses the train labels file and stores each label into the trainLabels list
def extractTrainLabels(filePath):

    file = open(filePath, "r")
    for line in file:
        label = int(line)
        trainLabels.append(label)

################################ TEST FILE EXTRACTION ################################

# parses the test file and stores each face into the testFaces list
def extractTestFaces(filePath):

    file = open(filePath, 'r')
    ctr = 0
    face = []
    for line in file:
        face.append(line)
        ctr += 1
        if ctr != 0 and ctr % totalRows == 0: # each face spans 70 rows and 60 cols - a face is stored and reset every 70 rows
            testFaces.append(face)
            face = []

# parses the test labels file and stores each label into the testLabels list
def extractTestLabels(filePath):
    
    file = open(filePath, 'r')
    for line in file:
        label = int(line)
        testLabels.append(label)

################################ PRINTING ################################

# prints the input face
def printFace(face):

    for line in face:
        for c in line:
            print(c, end="")

# prints the weights associated to each pixel (70 rows x 60 cols)
def printWeights():

    for i in range(0, len(weights)):
        for j in range(0, len(weights[i])):
            print(weights[i][j], end=" ")
        print()

################################ MAIN ################################

# drives the FaceClassifier class
if __name__ == "__main__":

    trainFacesFile = "faceData/train"
    extractTrainFaces(trainFacesFile)
    trainLabelsFile = "faceData/trainLabels"
    extractTrainLabels(trainLabelsFile)

    testFacesFile = "faceData/test"
    extractTestFaces(testFacesFile)
    testLabelsFile = "faceData/testLabels"
    extractTestLabels(testLabelsFile)

    train()
    test()

    print("Success Rate: " + str(totalCorrect / len(testFaces) * 100) + "%")