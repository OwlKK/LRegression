import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt


def openFileDF():
    with open('LifeExpectancy.csv'):
        df = pd.read_csv('LifeExpectancy.csv', header=0)
    return df


# return countries and LEV as a list
def openFileLists():
    with open('LifeExpectancy.csv'):
        df = pd.read_csv('LifeExpectancy.csv')
        lifeExpectancyValueList = df['Life expectancy '].values.tolist()
        countriesList = df['Country'].values.tolist()

    return countriesList, lifeExpectancyValueList


# split into train and test sets - sklearn, DROP SAMPLES WITH MISSING VALUES
def sklearnSplit(dataFrame=openFileDF()):
    # split into train and test 1/4
    trainSetInner, testSetInner = train_test_split(dataFrame, test_size=0.8)

    trainSetInner = trainSetInner.dropna(subset=['GDP', 'Total expenditure', 'Alcohol', 'Life expectancy '])
    return trainSetInner, testSetInner


# split into train and test sets - random, DROP SAMPLES WITH MISSING VALUES
def randomSplit(dataFrame=openFileDF()):
    # split into train and test 1/4 RANDOMLY
    trainSetInner = dataFrame.sample(frac=0.2)
    testSetInner = dataFrame.drop(trainSetInner.index)

    trainSetInner = trainSetInner.dropna(subset=['GDP', 'Total expenditure', 'Alcohol', 'Life expectancy '])
    return trainSetInner, testSetInner


# Number of records, histograms and statistic information
def showInformation(trainSetInner, testSetInner):
    # number of records in each dataset
    print("Train set NoR: " + str(len(trainSetInner.index)))
    print("Test set NoR: " + str(len(testSetInner.index)) + "\n")

    # histogram of life expectancy
    trainSetInner.hist(column='Life expectancy ')
    plt.title('Train set')
    testSetInner.hist(column='Life expectancy ')
    plt.title('Test set')
    plt.show()

    # statistic information of life expectancy

    print('Train set' + "\n" +
          'mean: ' + str(trainSetInner['Life expectancy '].mean()) + "\n" +
          'mode: ' + str(trainSetInner['Life expectancy '].mode()) + "\n" +
          'median: ' + str(trainSetInner['Life expectancy '].median()) + "\n")

    print('Test set' + "\n" +
          'mean: ' + str(testSetInner['Life expectancy '].mean()) + "\n" +
          'mode: ' + str(testSetInner['Life expectancy '].mode()) + "\n" +
          'median: ' + str(testSetInner['Life expectancy '].median()) + "\n")


def showInformationBoth():
    print('Random split')
    showInformation(randomSplit()[0], randomSplit()[1])

    print('\n' + 'sklearnSplit')
    showInformation(sklearnSplit()[0], sklearnSplit()[1])


# I have chosen random split

def fitModels(trainSetInner):
    xTrainList = []
    yTrainList = []
    modelList = []

    for param in ['GDP', 'Total expenditure', 'Alcohol']:
        xTrain = trainSetInner[[param]]
        yTrain = trainSetInner['Life expectancy ']
        model = LinearRegression().fit(xTrain.values, yTrain.values)  # GH told me to add .value 's

        xTrainList.append(xTrain)
        yTrainList.append(yTrain)
        modelList.append(model)

    return xTrainList, yTrainList, modelList


# Find coefficients and score
def coeffAndScore(trainSetInner):
    loopNumber = 0
    for param in ['GDP', 'Total expenditure', 'Alcohol']:
        xTrain = fitModels(trainSetInner)[0][loopNumber]
        yTrain = fitModels(trainSetInner)[1][loopNumber]
        model = fitModels(trainSetInner)[2][loopNumber]

        print("\n")
        print(f"Model for {param}:")
        print(f"Coefficient: {round(model.coef_[0], 5)}")
        print(f"Intercept: {round(model.intercept_, 5)}")
        print(f"Score: {round(model.score(xTrain, yTrain), 5)}")

        loopNumber = loopNumber + 1


def plotting(trainSetInner):
    loopNumber = 0
    for param in ['GDP', 'Total expenditure', 'Alcohol']:
        xTrain = fitModels(trainSetInner)[0][loopNumber]
        yTrain = fitModels(trainSetInner)[1][loopNumber]
        model = fitModels(trainSetInner)[2][loopNumber]

        plt.scatter(xTrain, yTrain)

        label_ = f"{round(model.coef_[0], 5)}x + {round(model.intercept_, 5)}"

        plt.plot(xTrain, model.predict(xTrain), color='red',
                 label=label_)
        plt.title(f"{param} - Life expectancy")
        plt.xlabel(param)
        plt.ylabel('Life expectancy ')

        plt.legend(loc='best')
        plt.show()

        loopNumber = loopNumber + 1


def predict(testSetInner):
    xTestList = []
    yTestList = []
    yPredictList = []

    loopNumber = 0

    testSetInner = testSetInner.dropna(subset=['GDP', 'Total expenditure', 'Alcohol', 'Life expectancy '])

    for param in ['GDP', 'Total expenditure', 'Alcohol']:
        model = fitModels(testSetInner)[2][loopNumber]

        xTest = testSetInner[[param]].values
        yTest = testSetInner[['Life expectancy ']].values
        yPredict = model.predict(xTest)

        xTestList.append(xTest)
        yTestList.append(yTest)
        yPredictList.append(yPredict)

        loopNumber = loopNumber + 1

    return xTestList, yTestList, yPredictList


def errors(testSetInner):
    xTestList, yTestList, yPredictList = predict(testSetInner)

    loopNumber = 0

    for param in ['GDP', 'Total expenditure', 'Alcohol']:
        error = yTestList[loopNumber] - yPredictList[loopNumber]
        errorMean = round(error.mean(), 4)
        errorStandardDev = round(error.std(), 4)

        print('\n')
        print(f"{param} errors")
        print(f"Mean error: {errorMean}")
        print(f"Standard deviation: {errorStandardDev}")

        loopNumber = loopNumber + 1

    print('\n')


# ADDITIONAL QUESTS

def chooseFour(trainSetInner):
    corr = trainSetInner.corr(numeric_only=True)
    corr = corr['Life expectancy '].abs().sort_values(ascending=False)
    topFour = corr.index[1:5]
    print(f'Top 4 parameters: {topFour}')

    return topFour


def modelFour(trainSetInner, topFourInner):
    trainSetInner = trainSetInner.dropna(
        subset=['Schooling', 'Income composition of resources', 'Adult Mortality', ' BMI ', 'Life expectancy '])

    X_train = trainSetInner[topFourInner].values
    y_train = trainSetInner['Life expectancy '].values
    LReg = LinearRegression().fit(X_train, y_train)  # ValueError: Input X contains NaN.

    return LReg, X_train, y_train


def infoFour(trainSetInner, testSetInner, topFourInner):
    LReg, X_train, y_train = modelFour(trainSetInner, topFourInner)
    print('Multilinar model: \n')
    print(f'Coefficents: {LReg.coef_}')
    print(f'Intercept: {LReg.intercept_}')
    print(f'Score: {LReg.score(X_train, y_train)}')
    print('\n')

    testSetInner = testSetInner.dropna(subset=['Schooling', 'Income composition of resources',
                                               'Adult Mortality', ' BMI ', 'Life expectancy '])

    X_test = testSetInner[topFourInner].values
    y_test = testSetInner['Life expectancy '].values
    y_predict = LReg.predict(X_test)
    error = y_test - y_predict
    print('Multilinear model: ')
    print('Mean error: ' + str(round(error.mean(), 4)))
    print('Standard deviation : ' + str(round(error.std(), 4)))
    print('\n')


def compareResults():
    print('\n')
    print('Multilinear: Lower mean error and standard deviation compared to simple regression ')
    print('Multilinear model has greater accuracy')


# MMMMMMMMMMMMAAAAAAAAAAAAAAAIIIIIIIIIIIIINNNNNNNNNNNNN

# Read data
data = openFileDF()
countries, LEValue = openFileLists()

# Unify train and test sets for everything
trainSet, testSet = randomSplit()

# Show information about sets
showInformationBoth()

# Print three highest 'Life Expectancy' countries
threeHighest = sorted(zip(LEValue, countries), reverse=True)[:3]
print("Countries with highest LE scores: " + str(threeHighest))

coeffAndScore(trainSet)
plotting(trainSet)

predict(testSet)
errors(testSet)

top4 = chooseFour(trainSet)
mf = modelFour(trainSet, top4)

infoFour(trainSet, testSet, top4)

compareResults()

