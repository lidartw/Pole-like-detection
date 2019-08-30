import numpy as np
import os
import pickle

from sklearn.metrics import confusion_matrix
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis


def getPaths(directoryPath, objName):
    paths = []
    for file in os.listdir(directoryPath):
        if file.startswith(objName) and file.endswith(".csv"):
            paths.append(os.path.join(directoryPath, file))
    return paths

def getEigen(cov):
    eigenValue = np.linalg.eigvals(cov)
    eigenValue = -np.sort(-eigenValue)
    return eigenValue/sum(eigenValue)

def getVecMean(vec):
    return np.array([np.mean(vec[i]) for i in range(len(vec))])

def getFeatures():
    global dataSource, objectsName, dataPath
    
    XYZColumn = (0,1,2)
    heightColumn = 2
    features = []

    tags = []
    for objIndex in range(len(objectsName)):

        for path in getPaths(dataPath, objectsName[objIndex]):
            csv = np.genfromtxt (path, delimiter=",", usecols=XYZColumn)
            cov = np.cov(csv.T)
            eigenValue = getEigen(cov)
            
            e1 = eigenValue[0]
            e2 = eigenValue[1]
            e3 = eigenValue[2]
            
            x1 = max(point[heightColumn] for point in csv) - min(point[heightColumn] for point in csv)
            x2 = eigenValue[2]/(eigenValue[0]*eigenValue[1])
            x3 = eigenValue[1]/eigenValue[2]
            x4 = eigenValue[0]*eigenValue[2]/(eigenValue[1]**2)
    
            features.append([x1, x2, x3, x4])
            tags.append(objIndex)
            
    return features, tags


def train_test_split(data, tag, test_size = 0.25):
    global objectsName
    
    dataNumbersInClasses = []
    for i in range(len(objectsName)):
        dataNumbersInClasses.append(tag.count(i))
    
    X_train = []
    y_train = []
    X_test = []
    y_test = []
    
    if test_size == 1:
        test_size = 0.9999999
    if test_size == 0:
        test_size = 0.0000001
        
    for i in range(len(dataNumbersInClasses)):
        cur_numbers = dataNumbersInClasses[i]
        Xtmp = data[:cur_numbers]
        ytmp = tag[:cur_numbers]
        data = data[cur_numbers:]
        tag = tag[cur_numbers:]

        partialIndex = int((cur_numbers - 1) * test_size) + 1
        X_train += Xtmp[partialIndex:]
        X_test += Xtmp[:partialIndex]
        y_train += ytmp[partialIndex:]
        y_test += ytmp[:partialIndex]
    
    return X_train, X_test, y_train, y_test


#dataPath = "/home/augie/Downloads/sydney-urban-objects-dataset/objects/"
#objectsName = ["traffic_light", "traffic_sign", "tree", "car"]
#dataSource = "sydney"

dataPath = "Data/"  ## Btter change to full path
objectsName = ["Pole", "PS", "SL", "SS", "STL", "TL_", "TLA", "Other"]
dataSource = "iii"
featuresNum = 4

X, y = getFeatures()
X_train, X_test, y_train, y_test = train_test_split(X, y)


names = ["Nearest Neighbors", "Linear SVM", "RBF SVM", "Gaussian Process",
         "Decision Tree", "Random Forest", "Neural Net", "AdaBoost",
         "Naive Bayes", "QDA","LDA"]

classifiers = [
    KNeighborsClassifier(3),
    SVC(kernel="linear", C=0.025),
    SVC(gamma=2, C=1),
    GaussianProcessClassifier(1.0 * RBF(1.0)),
    DecisionTreeClassifier(max_depth=5),
    RandomForestClassifier(max_depth=10, n_estimators=100, max_features=2),
    MLPClassifier(alpha=1),
    AdaBoostClassifier(),
    GaussianNB(),
    QuadraticDiscriminantAnalysis(),
    LinearDiscriminantAnalysis()
    ]

for name, clf in zip(names, classifiers):
    
    clf.fit(X_train, y_train) 
    predicted = []
    for feature, tag in zip(X_test, y_test):
        predicted.append(clf.predict([feature]))
    CM = confusion_matrix(y_test, predicted, labels=[i for i in range(len(objectsName))])
    
    
    #print results_______
    maxObjNameLength = max(len(name) for name in objectsName) #max name length of objects
    print(name, end="\n\n")
    print(" "*(maxObjNameLength), end = "")
    for name in objectsName:
        print(name.center(maxObjNameLength+1), end = "")
    print("EC")
    
    for row, name in zip(range(len(CM)), objectsName):
        print(name.rjust(maxObjNameLength), end = "")
        for column in range(len(CM[0])):
            print(str(CM[row][column]).center(maxObjNameLength+1) , end = "")
        print("{0:.2f}".format(sum(CM[row][i] for i in range(len(CM[row])) if i != row ) / sum(CM[row])))
    print("EO".rjust(maxObjNameLength), end = "")
    for j in range(len(CM[0])):
        columnSum = sum(CM[i][j] for i in range(len(CM)))
        if (columnSum == 0):
            print("0".center(maxObjNameLength+1), end ="")
        else:
            print("{0:.2f}".format((columnSum - CM[j][j])/ columnSum).center(maxObjNameLength+1), end = "")
    print()
    print("ACC".rjust(maxObjNameLength), end = "" )
    print("{0:.2f}".format(clf.score(X_test, y_test)).center(maxObjNameLength+1)  )
    print("---------------------------------------------")
    #print results_______


a = classifiers[9]


with open("ss.pkl","wb") as f:
    pickle.dump(a,f)


objectsName = ["a","b","c","d"]
XX, yy = getFeatures()


a.predict(XX)


for clf in classifiers:
    print(clf.predict(XX))

