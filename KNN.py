import pandas as pd
import math
from tqdm import tqdm


def knn(train_data, test_data, k):
    for index, row_x in tqdm(test_data.iterrows(), total=test_data.shape[0]):
        # neighbors list
        neighbors = []

        # loop through every row in training to find neighbors
        for i, row_y in train_data.iterrows():
            l2 = l2_distance(row_x, row_y)
            neighbors.append((row_y, l2))

        k_nearest = sorted(neighbors, key=lambda tup: tup[1])[:k]

        positive_count = sum([1 for row, dist in k_nearest if row[len(row)-1] == ' <=50K'])
        test_data.at[index, 'prediction'] = (' <=50K' if positive_count > k//2 else ' >50K')
    return test_data


def l2_distance(row_x, row_y):
    s = 0
    for i in range(len(row_y)):
        if isinstance(row_x[i], str):   # if the values are str, the distance is 0 if same, and 1 if different
            s += (0 if row_x[i] == row_y[i] else 1)
        else:
            s += (row_x[i] - row_y[i])**2
    return math.sqrt(s)

def accuracy(df):
    return sum([1 for i, row in df.iterrows() if row[0] == row[1]]) / df.shape[0]




if __name__ == '__main__':
    df = pd.read_csv('adult.data')
    # test = pd.read_csv('adult.test')

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

    train = int(0.65 * df.shape[0])

    train_data = df.loc[:train,]
    test_data = df.loc[train:,]
    test_data['prediction'] = pd.Series([None for i in range(df.shape[0])])

    # result = knn(train_data.loc[:500,], test_data.loc[train:train+500,], 15)
    result = knn(train_data, test_data, 15)

    acc = accuracy(result[['result', 'prediction']])
    print(acc)
