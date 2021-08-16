import pandas as pd
import numpy as np
from scipy import spatial
import operator

r_cols = ['user_id', 'movie_id', 'rating']
ratings = pd.read_csv('C:\\Users\\Shreyas\\MLCourse\\ml-100k\\u.data', sep='\t', names=r_cols, usecols=range(3))
# print(ratings)

movieGrpBy = ratings.groupby('movie_id').agg({'rating': [np.size, np.mean]})
# print(movieGrpBy.head(10))
movieRating = pd.DataFrame(movieGrpBy['rating']['size'])  # here we are taking the size of the rating column
# print(movieRating)
movieRating = movieRating.apply(lambda x: (x - np.min(x)) / (np.max(x) - np.min(
    x)))  # here we are taking max 'size' and min 'size' and then performing the calculation for each of the movie
# print(movieRating.head())

# writing the algo for the seperation of the contents of file
movieDict = {}
with open(r'C:\\Users\\Shreyas\\MLCourse\\ml-100k\\u.item', encoding='ISO-8859-1') as f:
    temp = ''
    for line in f:
        fields = line.rstrip('\n').split(
            '|')  # here we are skipping all the whitespaces and spliting the lines as '|' this character arrives.This gives a continous line information
        movieID = int(fields[0])
        name = str(fields[1])
        genres = fields[5:25]
        genres = map(int, genres)
        movieDict[movieID] = (name, np.array(list(genres)), movieRating.loc[movieID].get('size'), movieGrpBy.loc[movieID].rating.get('mean'))


# print(movieDict[9])

def computeKNNvalue(a, b):
    genreA = a[1]
    genraB = b[1]
    genreDistance = spatial.distance.cosine(genreA, genraB)
    popularityScoreA = a[2]
    popularityScoreB = b[2]
    AbsolutePopluarity = abs(popularityScoreA - popularityScoreB)
    return genreDistance + AbsolutePopluarity


# value=computeKNNvalue(movieDict[8],movieDict[6])
# print(movieDict[6],movieDict[8])
# print(value)

def GetNeighbors(movieID, K):
    distances = []
    for movie in movieDict:
        if (movie != movieID):#here we are looking if the same movie pops up or not if it does continue or else do this...
            dist = computeKNNvalue(movieDict[movieID], movieDict[movie])
            distances.append((movie, dist))#dist gives cosine dist from computeKNN
        else:
            continue
    distances.sort(key=operator.itemgetter(1))
    # print(distances)
    neighbors = []
    for x in range(K):
        neighbors.append(distances[x][0])#This selects the first 10 selections
    # print(neighbors)
    return neighbors


K = 10
avgrating = 0
neighbors = GetNeighbors(1, K)
for neighbor in neighbors:
    avgrating += movieDict[neighbor][3]#here movieDict has 3 main things name,size,mean so movieDict[neighbor][3] means neighbor is MOVIE NUMBER while 3 is mean of the MOVIE
    print(movieDict[neighbor][0] + " " + str(movieDict[neighbor][3]))
avgrating /= K
