#!/usr/bin/python
#The above comment might give you issues
import pandas as pd
import math
from tqdm import tqdm

#All running times are reported for MacBook air i5 1.3GHz with two cores


def knn(train_data, test_data, k, metric):
    for index, row_x in tqdm(test_data.iterrows(), total=test_data.shape[0]):
        # neighbors list
        neighbors = []

        # loop through every row in training to find neighbors
        for i, row_y in train_data.iterrows():
            l2 = metric(row_x, row_y, metric)
            neighbors.append((row_y, l2))

        k_nearest = sorted(neighbors, key=lambda tup: tup[1])[:k]

        positive_count = sum([1 for row, dist in k_nearest if row[len(row)-1] == ' <=50K'])
        test_data.at[index, 'prediction'] = (' <=50K' if positive_count > k//2 else ' >50K')
    return test_data

def metric(row_x, row_y, metric):
    # returns the distance between two points in different metrics
    if metric == 0:
        return l1_distance(row_x, row_y)
    elif metric == 1:
        return l2_distance(row_x, row_y)
    elif metric == 2:
        return l4_distance(row_x, row_y)
    elif metric == 3:
        return wack_distance(row_x, row_y)
    else:
        return wacker_distance(row_x, row_y)


def l1_distance(row_x, row_y):
    # Manhattan metric: sum of |x_1 - x_2|
    s=0
    for i in range(len(row_y)):
        if isinstance(row_x[i], str):   # if the values are str, the distance is 0 if same, and 1 if different
            s += (0 if row_x[i] == row_y[i] else 1)
        else:
            s += abs(row_x[i] - row_y[i])
    return s

def l2_distance(row_x, row_y):
    # Euclidean metric: sqrt of sum of square differences
    s = 0
    for i in range(len(row_y)):
        if isinstance(row_x[i], str):   # if the values are str, the distance is 0 if same, and 1 if different
            s += (0 if row_x[i] == row_y[i] else 1)
        else:
            s += (row_x[i] - row_y[i])**2
    return math.sqrt(s)

def l4_distance(row_x, row_y):
    # lp metric where p=4: same as Euclidean, to the power of 4
    s = 0
    for i in range(len(row_y)):
        if isinstance(row_x[i], str):   # if the values are str, the distance is 0 if same, and 1 if different
            s += (0 if row_x[i] == row_y[i] else 1)
        else:
            s += (row_x[i] - row_y[i])**4
    return math.sqrt(math.sqrt(s))


def wack_distance(row_x, row_y):
    # Inverted Euclidean because why the fuck not: (sqrt(|x_1 - x_2|))^2
    s = 0
    for i in range(len(row_y)):
        if isinstance(row_x[i], str):   # if the values are str, the distance is 0 if same, and 1 if different
            s += (0 if row_x[i] == row_y[i] else 1)
        else:
            s += math.sqrt(abs(row_x[i] - row_y[i]))
    return s**2

def wacker_distance(row_x, row_y):
    # Even wacker distance: (abs(sqrt(abs(x_1)) - sqrt(abs(x_2))))^2
    s = 0
    for i in range(len(row_y)):
        if isinstance(row_x[i], str):   # if the values are str, the distance is 0 if same, and 1 if different
            s += (0 if row_x[i] == row_y[i] else 1)
        else:
            s += abs(math.sqrt(abs(row_x[i])) - math.sqrt(abs(row_y[i])))
    return s**2




def accuracy(df):
    s = 0
    for i,row in df.iterrows():
        if row[0] == row[1]:
            s+=1
    
    return float(s) / float(df.shape[0])
    #return sum([1 for i, row in df.iterrows() if row[0] == row[1]]) / df.shape[0] #works only in python 3


def recordResult(acc, i, j, k, m):
    #Write experimental results into 'accuracy.txt' file, for experimental use
    output = "Accuracy for attempt:(" +  str(i) + "," + str(j) + ") with k: " + str(k) + " and metric: " + str(m) + " is: " + str(acc)
    print(output)
    f = open('accuracy.txt', 'a')
    f.write('%s'%str(output))
    f.write('\n')

    if m == 4:
        f.write('\n')
        f.write('\n')

    f.close()

def runAlgo(k, metric, frac, df):
    # Executes algorithm for given parameters
    train = int(0.65 * df.shape[0] / frac) 
    train_data = df.loc[:train,]

    if frac > 1:
        test = train*1.5
        test_data = df.loc[train:test,] 

    else:
        test_data = df.loc[train:]


    print
    print('ignore this warnning')
    print

    test_data['prediction'] = pd.Series([None for i in range(df.shape[0])])

    print
    print

    result = knn(train_data, test_data, k, metric)

    acc = accuracy(result[['result', 'prediction']])

    print

    print 'Run finished, accuracy is: ', acc
    print('The result has been saved in accuracy.txt')


    f = open('accuracy.txt', 'a')
    f.write('Accuracy = %s'%str(acc))
    f.write('\n')
    f.close()


    from sklearn.metrics import confusion_matrix
    cm = confusion_matrix(result['result'], result['prediction'])

    f = open('accuracy2.txt', 'a')
    f.write(str(cm))
    f.write('\n')
    f.close()
    print('Confusion matrix is recorded at accuracy2.txt')
    print







if __name__ == '__main__':
    df = pd.read_csv('adult.data')

    df.columns = ['age', 'work-class', 'fnlwgt', 'education', 'education-num', 'marital-status', 'occupation', 'relationship', 'race', 'sex', 'capital-gain', 'capital-loss', 'hours-per-week', 'native-country', 'result']

    # age normalization
    mu = df['age'].mean()
    sig = df['age'].std()
    df['age'] = pd.Series([(x - mu) / sig for x in df['age']])

    # fnlwgt normalization
    mu = df['fnlwgt'].mean()
    sig = df['fnlwgt'].std()
    df['fnlwgt'] = pd.Series([(x - mu) / sig for x in df['fnlwgt']])

    # education-num normalization
    mu = df['education-num'].mean()
    sig = df['education-num'].std()
    df['education-num'] = pd.Series([(x - mu) / sig for x in df['education-num']])

    # capital-gain normalization
    mu = df['capital-gain'].mean()
    sig = df['capital-gain'].std()
    df['capital-gain'] = pd.Series([(x - mu) / sig for x in df['capital-gain']])

    # capital-loss normalization
    mu = df['capital-loss'].mean()
    sig = df['capital-loss'].std()
    df['capital-loss'] = pd.Series([(x - mu) / sig for x in df['capital-loss']])

    # hours-per-week normalization
    mu = df['hours-per-week'].mean()
    sig = df['hours-per-week'].std()
    df['hours-per-week'] = pd.Series([(x - mu) / sig for x in df['hours-per-week']])

    #For experimental run, start comment here
    print 
    print('Chose an odd value of k(11 is recommended)')
    k = int(input())
    print 
    print('Chose a metric out of the following:(5 is recommended)')
    print 
    print('1 - Manhattan Distance')
    print('2 - Euclidean Distance')
    print('3 - l4 Distance')
    print('4 - Experimental metric #1')
    print('5 - Experimental metric #2')
    print 
    metric = int(input()) -1

    print 
    print('What fraction of observations do you want to consider?(integer)')
    print('1 means all data, 10 means a tenth of the data, so on(50 is recommended, this takes a while)')
    print 
    frac = int(input())

    

    runAlgo(k, metric, frac, df)



    '''


    
    #EXPERIMENTAL RUN - roughly 10 seconds per iteration, total running time of about 10 hours
    
    #Uncomment this batch to run the experimental run, make sure to comment the code above
    #

    for l in range(0, 9):
        #iterating through prime values of k
        if l == 0:
            k = 1
        elif l == 1:
            k = 3
        elif l == 2:
            k = 5
        elif l == 3:
            k = 7
        elif l == 4:
            k = 11
        elif l == 5:
            k = 13
        elif l == 6:
            k = 17
        elif l == 7:
            k = 19
        elif l == 8:
            k = 23

        for j in range(7, 11):
            #500 observations per experiment, even split between train/test
            #feel free to adjust, but notice that k=11 and norm = 4 will remain best
            start = 250*j
            middle = 250*(j+1)
            end = 250*(j+2)

            for m in range(0, 5):
                train_data = df.loc[start:middle,]
                test_data = df.loc[middle:end]
                test_data['prediction'] = pd.Series([None for i in range(df.shape[0])])

                result = knn(train_data, test_data, k, m)
                acc = accuracy(result[['result', 'prediction']])
                recordResult(acc, l, j, k, m)
    '''
