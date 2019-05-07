#!/usr/bin/python
#The above comment might give you issues
import pandas as pd
import math
from tqdm import tqdm

#All running times are reported for MacBook air i5 1.3GHz with two cores


def knn(train_data, test_data, k, norm):
    for index, row_x in tqdm(test_data.iterrows(), total=test_data.shape[0]):
        # neighbors list
        neighbors = []

        # loop through every row in training to find neighbors
        for i, row_y in train_data.iterrows():
            l2 = metric(row_x, row_y, norm)
            neighbors.append((row_y, l2))

        k_nearest = sorted(neighbors, key=lambda tup: tup[1])[:k]

        positive_count = sum([1 for row, dist in k_nearest if row[len(row)-1] == ' <=50K'])
        test_data.at[index, 'prediction'] = (' <=50K' if positive_count > k//2 else ' >50K')
    return test_data

def metric(row_x, row_y, norm):
    # returns the distance between two points in different norms
    if norm == 0:
        return l1_distance(row_x, row_y)
    elif norm == 1:
        return l2_distance(row_x, row_y)
    elif norm == 2:
        return l4_distance(row_x, row_y)
    elif norm == 3:
        return wack_distance(row_x, row_y)
    else:
        return wacker_distance(row_x, row_y)


def l1_distance(row_x, row_y):
    # Manhattan norm: sum of |x_1 - x_2|
    s=0
    for i in range(len(row_y)):
        if isinstance(row_x[i], str):   # if the values are str, the distance is 0 if same, and 1 if different
            s += (0 if row_x[i] == row_y[i] else 1)
        else:
            s += abs(row_x[i] - row_y[i])
    return s

def l2_distance(row_x, row_y):
    # Euclidean norm: sqrt of sum of square differences
    s = 0
    for i in range(len(row_y)):
        if isinstance(row_x[i], str):   # if the values are str, the distance is 0 if same, and 1 if different
            s += (0 if row_x[i] == row_y[i] else 1)
        else:
            s += (row_x[i] - row_y[i])**2
    return math.sqrt(s)

def l4_distance(row_x, row_y):
    # lp norm where p=4: same as Euclidean, to the power of 4
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
    output = "Accuracy for attempt:(" +  str(i) + "," + str(j) + ") with k: " + str(k) + " and norm: " + str(m) + " is: " + str(acc)
    print(output)
    f = open('accuracy.txt', 'a')
    f.write('%s'%str(output))
    f.write('\n')

    if m == 4:
        f.write('\n')
        f.write('\n')

    f.close()





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


    #train = int(0.65 * df.shape[0]) #takes roughly 30 hours


    train = int(0.5 * df.shape[0]) #takes roughly 4 hours -- divide by 20 for quicker run
    test = train*1.5

    train_data = df.loc[:train,]
    #test_data = df.loc[train:test,] #For shorter experiment uncomment this
    test_data = df.loc[train:] #For shorter experiment comment this
    test_data['prediction'] = pd.Series([None for i in range(df.shape[0])])

    # result = knn(train_data.loc[:10000,], test_data.loc[train:train+100,], 21)
    result = knn(train_data, test_data, 11, 4)

    acc = accuracy(result[['result', 'prediction']])


    f = open('accuracy.txt', 'a')
    f.write('Accuracy = %s'%str(acc))
    f.close()

    
    #For experimental run, end comment here


    '''
    #EXPERIMENTAL RUN - roughly 10 seconds per iteration, total running time of about 10 hours
    
    #Uncomment this batch to run the experimental run, make sure to comment the code above and the code below
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

