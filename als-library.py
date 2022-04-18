from collections import defaultdict
from numpy import *
import os
import numpy as np
import random
from Matrix import Matrix

class ALS(object):
    def __init__(self):
        self.user_ids = None
        self.item_ids = None
        self.user_ids_dict = None
        self.item_ids_dict = None
        self.user_matrix = None
        self.item_matrix = None
        self.user_items = None
        self.shape = None
        self.rmse = None

    def _process_data(self, X):
        self.user_ids = tuple((set(map(lambda x: x[0], X))))
        self.user_ids_dict = dict(map(lambda x: x[::-1],
                                      enumerate(self.user_ids)))

        self.item_ids = tuple((set(map(lambda x: x[1], X))))
        self.item_ids_dict = dict(map(lambda x: x[::-1],
                                      enumerate(self.item_ids)))

        self.shape = (len(self.user_ids), len(self.item_ids))

        ratings = defaultdict(lambda: defaultdict(int))
        ratings_T = defaultdict(lambda: defaultdict(int))
        for row in X:
            user_id, item_id, rating = row
            ratings[user_id][item_id] = rating
            ratings_T[item_id][user_id] = rating

        err_msg = "Length of user_ids %d and ratings %d not match!" % (
            len(self.user_ids), len(ratings))
        assert len(self.user_ids) == len(ratings), err_msg

        err_msg = "Length of item_ids %d and ratings_T %d not match!" % (
            len(self.item_ids), len(ratings_T))
        assert len(self.item_ids) == len(ratings_T), err_msg
        return ratings, ratings_T

    def _users_mul_ratings(self, users, ratings_T):
        def f(users_row, item_id):
            user_ids = iter(ratings_T[item_id].keys())
            scores = iter(ratings_T[item_id].values())
            col_nos = map(lambda x: self.user_ids_dict[x], user_ids)
            _users_row = map(lambda x: users_row[x], col_nos)
            return sum(a * b for a, b in zip(_users_row, scores))

        ret = [[f(users_row, item_id) for item_id in self.item_ids]
               for users_row in users.data]
        return Matrix(ret)

    def _items_mul_ratings(self, items, ratings):
        def f(items_row, user_id):
            item_ids = iter(ratings[user_id].keys())
            scores = iter(ratings[user_id].values())
            col_nos = map(lambda x: self.item_ids_dict[x], item_ids)
            _items_row = map(lambda x: items_row[x], col_nos)
            return sum(a * b for a, b in zip(_items_row, scores))

        ret = [[f(items_row, user_id) for user_id in self.user_ids]
               for items_row in items.data]
        return Matrix(ret)

    def _gen_random_matrix(self, n_rows, n_colums):
        data = [[random.random() for _ in range(n_colums)] for _ in range(n_rows)]
        return Matrix(data)

    def _get_rmse(self, ratings):
        m, n = self.shape
        mse = 0.0
        n_elements = sum(list(map(len, ratings.values())))
        for i in range(m):
            for j in range(n):
                user_id = self.user_ids[i]
                item_id = self.item_ids[j]
                rating = ratings[user_id][item_id]
                if rating > 0:
                    user_row = self.user_matrix.col(i).transpose
                    item_col = self.item_matrix.col(j)
                    rating_hat = user_row.mat_mul(item_col).data[0][0]
                    square_error = (rating - rating_hat) ** 2
                    mse += square_error / n_elements
        return mse ** 0.5

    def fit(self, X, k, max_iter=10):
        ratings, ratings_T = self._process_data(X)
        self.user_items = {k: set(v.keys()) for k, v in ratings.items()}
        m, n = self.shape

        error_msg = "Parameter k must be less than the rank of original matrix"
        assert k < min(m, n), error_msg

        self.user_matrix = self._gen_random_matrix(k, m)

        for i in range(max_iter):
            if i % 2:
                items = self.item_matrix
                self.user_matrix = self._items_mul_ratings(
                    items.mat_mul(items.transpose).inverse.mat_mul(items),
                    ratings
                )
            else:
                users = self.user_matrix
                self.item_matrix = self._users_mul_ratings(
                    users.mat_mul(users.transpose).inverse.mat_mul(users),
                    ratings_T
                )
            rmse = self._get_rmse(ratings)
            print("Iterations: %d, RMSE: %.6f" % (i + 1, rmse))

        self.rmse = rmse

    def _predict(self, user_id, n_items):
        users_col = self.user_matrix.col(self.user_ids_dict[user_id])
        users_col = users_col.transpose

        items_col = enumerate(users_col.mat_mul(self.item_matrix).data[0])
        items_scores = map(lambda x: (self.item_ids[x[0]], x[1]), items_col)
        viewed_items = self.user_items[user_id]
        items_scores = filter(lambda x: x[0] not in viewed_items, items_scores)

        return sorted(items_scores, key=lambda x: x[1], reverse=True)[:n_items]

    def predict(self, user_ids, n_items=10):
        return [self._predict(user_id, n_items) for user_id in user_ids]



if __name__ == "__main__":
    def main():
        print("Tesing the accuracy of ALS...")

        os.chdir(os.path.split(os.path.realpath(__file__))[0])
        BASE_PATH = os.path.abspath("..")
        def load_movie_ratings():
            """Load movie ratings data for recommedation.
            Returns:
                list -- userId, movieId, rating
            """

            file_name = "movie_ratings"
            path = os.path.join(BASE_PATH, "GroupProject", "%s.csv" % file_name)
            f = open(path)
            lines = iter(f)
            col_names = ", ".join(next(lines)[:-1].split(",")[:-1])
            print("The column names are: %s." % col_names)
            data = [[float(x) if i == 2 else int(x)
                     for i, x in enumerate(line[:-1].split(",")[:-1])]
                    for line in lines]
            f.close()

            return data

        def format_prediction(item_id, score):
            return "item_id:%d score:%.2f" % (item_id, score)
        X = load_movie_ratings()
        #X=1
        model = ALS()
        model.fit(X, k=3, max_iter=5)

        print("Showing the predictions of users...")

        user_ids = range(1, 5)
        predictions = model.predict(user_ids, n_items=2)
        for user_id, prediction in zip(user_ids, predictions):
            _prediction = [format_prediction(item_id, score)
                           for item_id, score in prediction]
            print("User id:%d recommedation: %s" % (user_id, _prediction))
    main()