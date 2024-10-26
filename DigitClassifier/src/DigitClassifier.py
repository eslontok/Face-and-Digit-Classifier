# DigitClassifier class takes a series of train digit images, sets weights for each digit (0 - 9), and predicts which digit is displayed on a given image (via test digit images)
# @ author Earl Lontok

# MULTICLASS PERCEPTRON CLASSIFICATION ALGORITHM - f(x) = w1(f1) + w2(f2) + w3(f3) + ... + wn(fn) for each class
# there are 10 digits total (0 - 9) - each digit will have its own perceptron (each digit has its own set of weights)
# this approach defines each feature to be the pixels comprising each 28 x 28 image
# therefore, each f is either 1 (if pixel is not ' ') or 0 (if pixel is ' ')

################################ GLOBAL VARIABLES ################################

# each image in the images files spans 28 rows and 28 columns
totalRows = 28
totalCols = 28

# these lists will store all digits and labels from the input TRAIN files
# a label is provided by humans - a label is essentially the "correct answer"
trainDigits = []
trainLabels = []

# a weight is associated with each feature (pixel in the 28 x 28 image) - these weights and the biases will be adjusted during the training process
# because digits is multiclass (0 - 9), each digit will have its own set of weights
# once the perceptron classification algorithm has been trained for each digit, these weights will be used to determine which digit is displayed on a given image
weights = [[[0 for k in range(totalCols)] for j in range(totalRows)] for i in range(10)] # 3d list is initialized using list comprehension
bias = [0] * 10

# these lists will store all digits and labels from the input TEST files
testDigits = []
testLabels = []
totalCorrect = 0

################################ TESTING ################################

# iterates through all input test digits and determines which digit is displayed for each image
def test():

    global totalCorrect
    for i in range(0, len(testDigits)):
        digit = testDigits[i]
        function = calcFunc(digit)
        label = testLabels[i]
        if function == label:
            print("CORRECT - Predicted: " + str(function) + " | Answer: " + str(label))
            totalCorrect += 1
        else:
            print("INCORRECT - Predicted: " + str(function) + " | Answer: " + str(label))

################################ TRAINING ################################

# iterates through all input train digits several times and sets the weights for each digit
def train():

    for i in range(0, 100): # iterating through all digits several times will result in better weights - results in a better algorithm
        for j in range(0, len(trainDigits)): # iterate through all input train digits
            digit = trainDigits[j]
            function = calcFunc(digit)
            label = trainLabels[j]
            if(function != label): # algorithm returns a digit different from the label
                adjustWeights(digit, function, label) # decrease weights associated with function and increase weights associated with label

            # there is 1 more case: function == label
            # in the perceptron classification algorithm, the weights should remain the same if the function and label agree
            # therefore, this case is omitted

# calculates the function for each digit (0 - 9) - the function with the highest score is the predicted digit
def calcFunc(digit):

    # the following applies for each digit (0 - 9):
    # a feature is defined by pixel - f = 1 if pixel is not ' ' and f = 0 if pixel is ' '
    # because each feature is either 1 or 0, only the weight is considered if a pixel is not ' '
    # for example, w * (1) = w and w * (0) = 0

    predictedLabel = 0
    maxFunction = 0
    for i in range(0, len(weights)): # iterate through each digit (each digit has its own set of weights)
        function = 0
        for j in range(0, totalRows):
            for k in range(0, totalCols):
                c = digit[j][k]
                if c != ' ':
                    weight = weights[i][j][k] # each digit has its own set of weights (weights[i] is the set of weights for digit i)
                    function += weight
        function += bias[i]
        if function > maxFunction: # keep track of the function with the highest score (this will determine the predicted digit)
            maxFunction = function
            predictedLabel = i
    return predictedLabel

# decreases the weights associated with the function and increases the weights associated with the label
# recall that a feature is defined by pixel - action is only done if not ' '
def adjustWeights(digit, function, label):

    for i in range(0, totalRows):
        for j in range(0, totalCols):
            c = digit[i][j]
            if c != ' ':
                weights[function][i][j] -= 1 # decrease the weights associated with the function (wrong answer)
                weights[label][i][j] += 1 # increase the weights associated with the label (correct answer)
    bias[function] -= 1
    bias[label] += 1

################################ TRAIN FILE EXTRACTION ################################

# parses the train file and stores each digit into the trainDigits list
def extractTrainDigits(filePath):

    file = open(filePath, "r")
    ctr = 0
    digit = []
    for line in file:
        digit.append(line)
        ctr += 1
        if ctr != 0 and ctr % totalRows == 0:  # each digit spans 28 rows and 28 cols - a digit is stored and reset every 28 rows
            trainDigits.append(digit)
            digit = []

# parses the train labels file and stores each label into the trainLabels list
def extractTrainLabels(filePath):

    file = open(filePath, "r")
    for line in file:
        label = int(line)
        trainLabels.append(label)

################################ TEST FILE EXTRACTION ################################

# parses the test file and stores each digit into the testDigits list
def extractTestDigits(filePath):

    file = open(filePath, "r")
    ctr = 0
    digit = []
    for line in file:
        digit.append(line)
        ctr += 1
        if ctr != 0 and ctr % totalRows == 0: # each digit spans 28 rows and 28 cols - a digit is stored and reset every 28 rows
            testDigits.append(digit)
            digit = []

# parses the test labels file and stores each label into the testLabels list
def extractTestLabels(filePath):

    file = open(filePath, "r")
    for line in file:
        label = int(line)
        testLabels.append(label)

################################ PRINTING ################################

# prints the input digit
def printDigit(digit):

    for line in digit:
        for c in line:
            print(c, end="")

# prints the weights associated with each pixel (28 rows x 28 cols) for the input digit
def printWeights(digit):

    digitWeights = weights[digit]
    for i in range(0, len(digitWeights)):
        for j in range(0, len(digitWeights[i])):
            print(digitWeights[i][j], end=" ")
        print()

################################ MAIN ################################

# drives the DigitClassifier class
if __name__ == "__main__":

    trainDigitsFile = "digitData/train"
    extractTrainDigits(trainDigitsFile)
    trainLabelsFile = "digitData/trainLabels"
    extractTrainLabels(trainLabelsFile)

    testDigitsFile = "digitData/test"
    extractTestDigits(testDigitsFile)
    testLabelsFile = "digitData/testLabels"
    extractTestLabels(testLabelsFile)

    train()
    test()

    print("Success Rate: " + str(totalCorrect / len(testDigits) * 100) + "%")