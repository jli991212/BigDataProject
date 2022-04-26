from collections import defaultdict
import os
import random
import numpy
from Matrix import ALS_Matrix
from sklearn.model_selection import train_test_split
import seaborn as sns
import matplotlib.pyplot as plt



os.chdir(os.path.split(os.path.realpath(__file__))[0])
BASE_PATH = os.path.abspath("..")

class ALS(object):
    def __init__(self):
        #ininlization for als model
        self.userID = ''
        self.itemID = ''
        self.userID_dictionary = ''
        self.itemID_dictionary = ''
        self.userMatrix = ''
        self.itemMatrix = ''
        self.userItem = ''
        self.Myshape = ''
        self.rmse = ''
    def processing(self, data):
        self.userID = list((set(map(lambda userid: userid[0], data))))
        self.userID_dictionary = dict(map(lambda userid: userid[::-1],
                                      enumerate(self.userID)))

        self.itemID = list((set(map(lambda itemid: itemid[1], data))))
        self.itemID_dictionary = dict(map(lambda itemid: itemid[::-1],
                                      enumerate(self.itemID)))
        usershape=len(self.userID)
        itemshape=len(self.itemID)
        self.Myshape = (usershape, itemshape)

        userRatings = defaultdict(lambda: defaultdict(int))
        itemRatings = defaultdict(lambda: defaultdict(int))
        for line in data:
            userId, itemId, point = line
            userRatings[userId][itemId] = point
            itemRatings[itemId][userId] = point

        err_msg = "exists user do not give any ratings to items"
        assert len(self.userID) == len(userRatings), err_msg

        err_msg = "exists item do not receive any ratings from users"
        assert len(self.itemID) == len(itemRatings), err_msg
        return userRatings, itemRatings

    def user_cal_ratings(self, users, itemRatings):
        def func(line, itemId):
            userID = iter(itemRatings[itemId].keys())
            rat_point = iter(itemRatings[itemId].values())
            temp = map(lambda x: self.userID_dictionary[x], userID)
            my_line = map(lambda x: line[x], temp)
            a=0.0
            for x,y in zip(my_line, rat_point):
                a+=x*y
            return a

        result = [[func(line, itemId) for itemId in self.itemID]
               for line in users.inputdata]
        return ALS_Matrix(result)

    def item_cal_ratings(self, items, userRatings):
        def func(line, userId):
            itemID = iter(userRatings[userId].keys())
            rat_point = iter(userRatings[userId].values())
            temp = map(lambda x: self.itemID_dictionary[x], itemID)
            my_line = map(lambda x: line[x], temp)
            a=0.0
            for x,y in zip(my_line, rat_point):
                a+=x*y
            return a

        result = [[func(line, userId) for userId in self.userID]
               for line in items.inputdata]
        return ALS_Matrix(result)

    def init_matrix(self, row, col):
        data = numpy.random.random((row, col))
        return ALS_Matrix(data)

    def cal_rmse(self, userRatings):

        userlen, itemlen = self.Myshape
        mse = 0.0
        total_rat = sum(list(map(len, userRatings.values())))
        for i in range(userlen):
            for j in range(itemlen):
                userId = self.userID[i]
                itemId = self.itemID[j]
                match_rat = userRatings[userId][itemId]
                if match_rat > 0:
                    rating_hat = self.userMatrix.column(i).transpose.matrix_mul(self.itemMatrix.column(j)).inputdata[0][0]
                    mse += (match_rat - rating_hat) ** 2 / total_rat
        rmse=mse**0.5
        return rmse

    def fit(self, X, rank, max_iteration=10):
        mselist=[]
        userRatings, itemRatings = self.processing(X)
        self.userItem = {rank: set(x.keys()) for rank, x in userRatings.items()}
        userlen, itemlen = self.Myshape

        error_msg = "the possible rank can not exceed the rank in the original Matrix"
        assert rank < min(userlen, itemlen), error_msg

        self.userMatrix = self.init_matrix(rank, userlen)

        for iteration in range(max_iteration):
            if iteration % 2:
                items = self.itemMatrix
                self.userMatrix = self.item_cal_ratings(
                    items.matrix_mul(items.transpose).inverse.matrix_mul(items),
                    userRatings
                )
            else:
                users = self.userMatrix
                self.itemMatrix = self.user_cal_ratings(
                    users.matrix_mul(users.transpose).inverse.matrix_mul(users),
                    itemRatings
                )
            rmse = self.cal_rmse(userRatings)
            mselist.append(rmse)
        print(" after %d runs  root sqaure error is: %.6f" % (max_iteration, rmse))

        self.rmse = rmse
        return mselist

    def get_prediction(self, userId, n):
        user = self.userMatrix.column(self.userID_dictionary[userId]).transpose

        items_col = enumerate(user.matrix_mul(self.itemMatrix).inputdata[0])
        checked_items = self.userItem[userId]
        points = filter(lambda x: x[0] not in checked_items, map(lambda x: (self.itemID[x[0]], x[1]), items_col))
        result=sorted(points, key=lambda x: x[1], reverse=True)[:n]
        return result

    def predict(self, userID, n=10):
        return [self.get_prediction(userId, n) for userId in userID]


def mseplot(iterations, train_rmse,test_rmse,rank,max_iteration):
    sns.set_context("paper", font_scale=1.3)
    sns.set_style('white')
    aa = [x for x in range(iterations)]
    plt.figure(figsize=(16, 9))
    # set title
    plt.title('rmse with rank:{0},max_iteration:{1}'.format(rank,max_iteration))
    plt.plot(aa, train_rmse, marker='.', label="train rmse")
    plt.plot(aa, test_rmse, marker='.', label="test rmse")
    plt.tight_layout()
    sns.despine(top=True)
    plt.subplots_adjust(left=0.07)
    plt.ylabel('rmse', size=15)
    plt.xlabel('iterations', size=15)
    plt.legend(fontsize=15)
    plt.savefig('loss{0}{1}.png'.format(rank,max_iteration))
    plt.close()

def load_music_ratings():

    file_name = "yahoo-music"
    path = os.path.join(BASE_PATH, "GroupProject", "%s.csv" % file_name)
    f = open(path)
    lines = iter(f)
    col_names = ", ".join(next(lines)[:-1].split(","))
    data = [[float(x) if i == 2 else int(x)
             for i, x in enumerate(line[:-1].split(","))]
            for line in lines]
    f.close()

    return data
def prediction_string(itemId, point):
    return "item ID :%d point:%.2f" % (itemId, point)

def main():
    print("Als model generating")
    X = load_music_ratings()
    test,train=train_test_split(X,train_size=0.8,random_state=42)
    max_iterations = [5,10,15]
    ranks = [3,4,5]
    for rank in ranks:
        for max_iteration in max_iterations:
            model = ALS()
            train_mse_list=model.fit(train, rank, max_iteration)
            test_mse_list=model.fit(test, rank, max_iteration)
            mseplot(max_iteration - 1, train_mse_list[1:],test_mse_list[1:],rank,max_iteration)
            print("Showing the predictions of users")
            userID = list(list(zip(*test))[0])
            trainuserID=list(list(zip(*train))[0])
            for i in userID:
                if i not in trainuserID:
                    userID.remove(i)
            predictions = model.predict(userID[0:10], n=5)
            for userId, point in zip(userID, predictions):
                predict_string = [prediction_string(itemID, point)
                               for itemID, point in point]

                print("User id:%d recommendation artist list: %s" % (userId, predict_string))

if __name__ == "__main__":
    main()