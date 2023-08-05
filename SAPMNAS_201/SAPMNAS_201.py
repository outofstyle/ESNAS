import os
import time
import numpy as np
import torch
import random
from nats_bench import create
from sklearn.utils import shuffle


np.random.seed(3)
random.seed(3)
torch.backends.cudnn.deterministic = True
from pymoo.util.nds.fast_non_dominated_sort import fast_non_dominated_sort
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_squared_error
from sklearn.cluster import KMeans
from time import time
import copy
import math
from pymoo.util.nds.fast_non_dominated_sort import fast_non_dominated_sort
from pymoo.indicators.hv import Hypervolume
from pymoo.factory import get_decision_making
import pandas as pd
from sklearn import preprocessing
from sklearn.neural_network import MLPRegressor
#from Latin import latin
#api = create('/home/liugroup/semi_regssionor/coreg-master/NATS-tss-v1_0-3ffb9-simple', 'tss', fast_mode=True, verbose=True) #/home/liugroup/semi_regssionor/coreg-master

import collections
import sys
from surrogate_model import load_surrogate_model
sys.setrecursionlimit(10000)
import argparse




choice = lambda x: x[np.random.randint(len(x))] if isinstance(
    x, tuple) else choice(tuple(x))


def get_correlation(prediction, target):
    import scipy.stats as stats

    rmse = np.sqrt(((prediction - target) ** 2).mean())
    rho, _ = stats.spearmanr(prediction, target)
    tau, _ = stats.kendalltau(prediction, target)

    return rmse, rho, tau



class EvolutionSearcher(object):

    def __init__(self, args):
        self.args = args
        self.pop = np.zeros((args.population_num, 6))
        self.true_acc = np.zeros((args.population_num, 1))
        self.max_epochs = args.max_epochs
        self.select_num = args.select_num
        self.population_num = args.population_num
        self.m_prob = args.m_prob
        self.crossover_num = args.crossover_num
        self.mutation_num = args.mutation_num
        self.train_size = args.train_size
        self.epoch = 0
        self.candidates = []

        self.max_iters = 50
        self.pool_size = 500
        self.h1, self.h2,self.h1_temp, self.h2_temp = load_surrogate_model(args.model)
        self.h1_k = KNeighborsRegressor(n_neighbors=3, p=2)
        self.h2_k = KNeighborsRegressor(n_neighbors=3, p=5)

        self.h1_k_temp = KNeighborsRegressor(n_neighbors=3, p=2)
        self.h2_k_temp = KNeighborsRegressor(n_neighbors=3, p=5)
        self.h3 = KNeighborsRegressor(n_neighbors=5, p=5)
        self.pop_acc = np.zeros((args.population_num, 1))
        self.tmp_confidence = []
        self.X = []
        self.y = []
        self.h = KNeighborsRegressor(n_neighbors=3, p=3)
        self.tmp_acc = np.zeros((args.population_num, 1))
        self.tmp_acc1 = np.zeros((args.population_num, 1))
        self.predict_acc = []
        self.L1_X = []
        self.L1_y = []
        self.L2_X = []
        self.L2_y = []
        self.U1_X = []
        self.U1_y = []
        self.U2_X = []
        self.U2_y = []
        self.all_pop = []
        self.all_tmp_acc = []
        self.all_true_acc = []
        self.archive2 = []
        self.archive4 = []
        self.front = []
        self.true_front_acc = []
        self.err_predictor1 = None
        self.err_predictor2 = None
        self.t_acc = np.zeros((self.population_num, 2))



    def add_data(self, X, y):
        """
        Adds data and splits into labeled and unlabeled.
        """
        # self.X, self.y = load_data(data_dir)
        self.X = np.array(X, dtype=object)
        self.y = np.array(y, dtype=object)
        X_list = []
        X_list1 = []
        y_list = []
        y_list1 = []
        estimator = KMeans(n_clusters=2)
        res = estimator.fit_predict(self.X)
        for i in range(0, len(self.X)):
            if res[i] == 0:
                X_list.append(self.X[i])
                y_list.append(self.y[i])
            else:
                X_list1.append(self.X[i])
                y_list1.append(self.y[i])
        self.L1_X = np.array(X_list)
        self.L1_y = np.array(y_list)
        self.L2_X = np.array(X_list1)
        self.L2_y = np.array(y_list1)
        self.U1_X = np.array(X_list1)
        self.U1_y = np.array(y_list1)
        self.U2_X = np.array(X_list)
        self.U2_y = np.array(y_list)

    def _run_iteration(self, t, t0):
        """
        Run t-th iteration of co-training, returns stop_training=True if
        no more unlabeled points are added to label sets.
        """
        stop_training = False
        print('Started iteration {}: {:0.2f}s'.format(t, time() - t0))
        self._find_points_to_add()
        print(self.L1_X.shape[0], self.L2_X.shape[0])
        added = self._add_points()
        if added:
            self._fit_and_evaluate()
            self._remove_from_unlabeled()
            self._get_pool()
            if self.U1_X.shape[0] == 0 and self.U2_X.shape[0]==0:
                return stop_training
        else:
            stop_training = True
        return stop_training

    def _add_points(self):
        """
        Adds new examples to training sets.
        """
        added = False
        if self.to_add['x2'] is not None:
            self.L2_X = np.vstack((self.L2_X, self.to_add['x2']))
            self.L2_y = np.vstack((self.L2_y.reshape(self.L2_y.shape[0], -1), self.to_add['y2'].reshape(1, -1)))
            added = True
        if self.to_add['x1'] is not None:
            self.L1_X = np.vstack((self.L1_X, self.to_add['x1']))
            self.L1_y = np.vstack((self.L1_y.reshape(self.L1_y.shape[0], -1), self.to_add['y1'].reshape(1, -1)))
            added = True
        return added

    def _compute_delta(self, omega, L_X, L_y, h, h_temp):
        """
        Computes the improvement in MSE among the neighbors of the point being
        evaluated.
        """
        delta = 0
        for idx_o in omega:
            delta += (L_y[idx_o].reshape(1, -1) -
                      h.predict(L_X[idx_o].reshape(1, -1))) ** 2
            delta -= (L_y[idx_o].reshape(1, -1) -
                      h_temp.predict(L_X[idx_o].reshape(1, -1))) ** 2

        return delta

    def _compute_deltas(self, L_X, L_y, h, h_temp, idx):
        """
        Computes the improvements in local MSE for all points in pool.
        """
        L_y = L_y.reshape(L_y.shape[0], -1)
        self.h.fit(L_X, L_y.ravel())
        if idx == 1:
            deltas = np.zeros((self.U1_X_pool.shape[0],))
            if self.U1_X_pool.shape[0] ==0:
                deltas = [-1, -1]
            else:
                for idx_u, x_u in enumerate(self.U1_X_pool):
                    # Make prediction
                    x_u = x_u.reshape(1, -1)
                    # Compute neighbors
                    y_u_hat = h.predict(x_u)
                    y_u_hat = y_u_hat.reshape(1, -1)
                    omega = self.h.kneighbors(x_u, return_distance=False)[0]
                    # Retrain regressor after adding unlabeled point
                    X_temp = np.vstack((L_X, x_u))
                    # omega = random.sample(range(0, len(L_X)), 3)
                    y_temp = np.vstack((L_y, y_u_hat))  # use predicted y_u_hat
                    h_temp.fit(X_temp, y_temp.ravel())
                    delta = self._compute_delta(omega, L_X, L_y, h, h_temp)
                    deltas[idx_u] = delta
        else:
            deltas = np.zeros((self.U2_X_pool.shape[0],))
            # lcb = np.zeros((self.U2_X_pool.shape[0],))
            if self.U2_X_pool.shape[0] ==0:
                deltas = [-1, -1]
            else:
                for idx_u, x_u in enumerate(self.U2_X_pool):
                    # Make prediction
                    x_u = x_u.reshape(1, -1)
                    y_u_hat = h.predict(x_u)
                    y_u_hat = y_u_hat.reshape(1, -1)
                    # Compute neighbors-
                    omega = self.h.kneighbors(x_u, return_distance=False)[0]
                    # Retrain regressor after adding unlabeled point
                    X_temp = np.vstack((L_X, x_u))
                    y_temp = np.vstack((L_y, y_u_hat))  # use predicted y_u_hat
                    h_temp.fit(X_temp, y_temp.ravel())
                    # omega = random.sample(range(0, len(L_X)), 3)
                    delta = self._compute_delta(omega, L_X, L_y, h, h_temp)
                    deltas[idx_u] = delta
        return deltas  # lcb



    def _find_points_to_add(self):
        """
        Finds unlabeled points (if any) to add to training sets.
        """
        self.to_add = {'x1': None, 'y1': None, 'idx1': None,
                       'x2': None, 'y2': None, 'idx2': None}

        # Keep track of added idxs
        added1_idxs = []
        added2_idxs = []
        for idx_h in [1, 2]:

            if idx_h == 1:
                h = self.h1_k
                h_temp = self.h1_k_temp
                L_X, L_y = self.L1_X, self.L1_y
                deltas = self._compute_deltas(L_X, L_y, h, h_temp, idx_h)
                # Add largest delta (improvement)
                sort_idxs = np.argsort(deltas)[::-1]  # max to min
                sort_idxs1 = np.argsort(deltas)[::-1][0:10]
                max_idx = sort_idxs[0]
                #if max_idx in added1_idxs: max_idx = sort_idxs[1]
                if deltas[max_idx] >= 0:
                    added1_idxs.append(max_idx)
                    x_u = self.U1_X[max_idx].reshape(1, -1)
                    y_u = self.U1_y[max_idx].reshape(1, -1)
                    self.to_add['x' + str(idx_h)] = x_u
                    self.to_add['y' + str(idx_h)] = y_u
                    self.to_add['idx' + str(idx_h)] = self.U1_idx_pool[max_idx]
            elif idx_h == 2:
                h = self.h2_k
                h_temp = self.h2_k_temp
                L_X, L_y = self.L2_X, self.L2_y
                deltas = self._compute_deltas(L_X, L_y, h, h_temp, idx_h)
                # Add largest delta (improvement)
                sort_idxs = np.argsort(deltas)[::-1]  # max to min
                max_idx = sort_idxs[0]
                # sort_gp = np.argsort(lcb)[::-1]
                #if max_idx in added2_idxs: max_idx = sort_idxs[1]
                if deltas[max_idx] >= 0:
                    added2_idxs.append(max_idx)
                    x_u = self.U2_X[max_idx].reshape(1, -1)
                    y_u = self.U2_y[max_idx].reshape(1, -1)
                    self.to_add['x' + str(idx_h)] = x_u
                    self.to_add['y' + str(idx_h)] = y_u
                    self.to_add['idx' + str(idx_h)] = self.U2_idx_pool[max_idx]

    def _fit_and_evaluate(self, verbose=True):
        """
        Fits h1 and h2 and evaluates metrics.
        """
        #self.h1_s.fit(self.L1_X, self.L1_y.ravel())
        #self.h2_s.fit(self.L2_X, self.L2_y.ravel())
        self.h1_k.fit(self.L1_X, self.L1_y.ravel())
        self.h2_k.fit(self.L2_X, self.L2_y.ravel())
        #self._evaluate_metrics(verbose)

    def _get_pool(self):
        """
        Gets unlabeled pool and indices of unlabeled.
        """
        self.U1_X_pool, self.U1_y_pool, self.U1_idx_pool = self.U1_X, self.U1_y, range(self.U1_y.size)
        self.U2_X_pool, self.U2_y_pool, self.U2_idx_pool = self.U2_X, self.U2_y, range(self.U2_y.size)

    def _remove_from_unlabeled(self):
        # Remove added examples from unlabeled
        to_remove1 = []
        to_remove2 = []
        if self.to_add['idx1'] is not None:
            to_remove1.append(self.to_add['idx1'])
            if self.to_add['idx1'] > self.U1_X.shape[0]:
                print(self.to_add['idx1'])
                print(self.U1_X.shape[0])
        if self.to_add['idx2'] is not None:
            to_remove2.append(self.to_add['idx2'])
            if self.to_add['idx2'] > self.U2_X.shape[0]:
                print(self.to_add['idx2'])
                print(self.U2_X.shape[0])
        self.U1_X = np.delete(self.U1_X, to_remove1, axis=0)
        self.U1_y = np.delete(self.U1_y, to_remove1, axis=0)
        self.U2_X = np.delete(self.U2_X, to_remove2, axis=0)
        self.U2_y = np.delete(self.U2_y, to_remove2, axis=0)

    def _run_iteration_coreg(self, t, t0):
        """
        Run t-th iteration of co-training, returns stop_training=True if
        no more unlabeled points are added to label sets.
        """
        stop_training = False
        print('Started iteration {}: {:0.2f}s'.format(t, time() - t0))

        #self._fit_and_evaluate_coreg(t)
        self.h1.fit(self.L1_X, self.L1_y.ravel())
        self.h2.fit(self.L2_X, self.L2_y.ravel())
        if self.U_X.shape[0] == 0:
            stop_training = True
            return stop_training
        self._find_points_to_add_coreg()
        if self.U_X.shape[0] == 0:
            stop_training = True
            return stop_training
        added = self._add_points_coreg()
        if added:
            self._fit_and_evaluate_coreg(t)
            self._remove_from_unlabeled_coreg()
            self._get_pool_coreg()
            if self.U_X.shape[0] == 0:
                stop_training = True
                return stop_training
        else:
            stop_training = True
        return stop_training

    def _add_points_coreg(self):
        """
        Adds new examples to training sets.
        """
        added = False
        if self.to_add['x1'] is not None:
            self.L2_X = np.vstack((self.L2_X, self.to_add['x1']))
            self.L2_y = np.vstack((self.L2_y, self.to_add['y1']))
            added = True
        if self.to_add['x2'] is not None:
            self.L1_X = np.vstack((self.L1_X, self.to_add['x2']))
            self.L1_y = np.vstack((self.L1_y, self.to_add['y2']))
            added = True
        return added

    def _compute_delta_coreg(self, omega, L_X, L_y, h, h_temp):
        """
        Computes the improvement in MSE among the neighbors of the point being
        evaluated.
        """
        delta = 0
        for idx_o in omega:
            delta += (L_y[idx_o].reshape(1, -1) -
                      h.predict(L_X[idx_o].reshape(1, -1))) ** 2
            delta -= (L_y[idx_o].reshape(1, -1) -
                      h_temp.predict(L_X[idx_o].reshape(1, -1))) ** 2
        return delta

    def _compute_deltas_coreg(self, L_X, L_y, h, h_temp):
        """
        Computes the improvements in local MSE for all points in pool.
        """
        L_y = L_y.reshape((L_y.shape[0], -1))
        self.h.fit(L_X, L_y.ravel())
        deltas = np.zeros((self.U_X.shape[0],))
        if self.U_X.shape[0] == 0:
            deltas = [-1, -1]
        for idx_u, x_u in enumerate(self.U_X):
            # Make prediction
            x_u = x_u.reshape(1, -1)
            y_u_hat = h.predict(x_u)
            y_u_hat = y_u_hat.reshape(1, -1)
            # Compute neighbors
            omega = self.h.kneighbors(x_u, return_distance=False)[0]
            # Retrain regressor after adding unlabeled point
            X_temp = np.vstack((L_X, x_u))
            y_temp = np.vstack((L_y, y_u_hat))  # use predicted y_u_hat
            h_temp.fit(X_temp, y_temp.ravel())
            delta = 0
            for idx_o in omega:
                delta += (L_y[idx_o].reshape(1, -1) -
                          h.predict(L_X[idx_o].reshape(1, -1))) ** 2
                delta -= (L_y[idx_o].reshape(1, -1) -
                          h_temp.predict(L_X[idx_o].reshape(1, -1))) ** 2
            deltas[idx_u] = delta
        return deltas

    def _find_points_to_add_coreg(self):
        """
        Finds unlabeled points (if any) to add to training sets.
        """
        self.to_add = {'x1': None, 'y1': None, 'idx1': None,
                       'x2': None, 'y2': None, 'idx2': None}
        # Keep track of added idxs
        added_idxs = []
        for idx_h in [1, 2]:
            if idx_h == 1:
                h = self.h1
                h_temp = self.h1_temp
                L_X, L_y = self.L1_X, self.L1_y

            elif idx_h == 2:
                h = self.h2
                h_temp = self.h2_temp
                L_X, L_y = self.L2_X, self.L2_y

            deltas = self._compute_deltas_coreg(L_X, L_y, h, h_temp)
            # Add largest delta (improvement)
            sort_idxs = np.argsort(deltas)[::-1]  # max to min
            max_idx = sort_idxs[0]
            if max_idx in added_idxs: max_idx = sort_idxs[1]
            if deltas[max_idx] > 0:
                added_idxs.append(max_idx)
                x_u = self.U_X[max_idx].reshape(1, -1)
                y_u_hat = h.predict(x_u).reshape(1, -1)
                self.to_add['x' + str(idx_h)] = x_u
                self.to_add['y' + str(idx_h)] = y_u_hat
                self.to_add['idx' + str(idx_h)] = self.U_idx_pool[max_idx]

    def _fit_and_evaluate_coreg(self, t):
        """
        Fits h1 and h2 and evaluates metrics.
        """
        # self.h1_r.fit(self.L1_X, self.L1_y.ravel())
        # self.h2_r.fit(self.L2_X, self.L2_y.ravel())
        # unlabel = []
        # label = []
        # unlabel1 = []
        # label1 = []
        # if self.to_add['x1'] is not None:
        #     train_labels = self.h1_r.predict(self.LL2_X)
        #
        #     mse = mean_squared_error(self.LL2_y[:].flatten(), train_labels[:].flatten())
        #     y_ob = [self.to_add['y1'], mse]
        #     self.archive2.append(y_ob)
        # if self.to_add['x2'] is not None:
        #     train_labels1 = self.h2_r.predict(self.LL1_X)
        #     mse = mean_squared_error(self.LL1_y[:].flatten(), train_labels1[:].flatten())
        #     y_ob = [self.to_add['y2'], mse]
        #     self.archive4.append(y_ob)

        if self.to_add['x1'] is not None:
            f_labels = self.h1.predict(self.sample)
            self.h1.fit(self.L1_X, self.L1_y.ravel())
            h_labels = self.h1.predict(self.sample)

            f_mse = mean_squared_error(self.sample_acc[:].flatten(), f_labels[:].flatten())
            h_mse = mean_squared_error(self.sample_acc[:].flatten(), h_labels[:].flatten())

            y_ob = [self.to_add['y1'], f_mse-h_mse]   #self.to_add['x1'],
            self.archive2.append(y_ob)
        if self.to_add['x2'] is not None:
            f_labels1 = self.h2.predict(self.sample)
            self.h2.fit(self.L2_X, self.L2_y.ravel())
            h_labels1 = self.h2.predict(self.sample)
            f_mse1 = mean_squared_error(self.sample_acc[:].flatten(), f_labels1[:].flatten())
            h_mse1 = mean_squared_error(self.sample_acc[:].flatten(), h_labels1[:].flatten())
            y_ob = [self.to_add['y2'], f_mse1-h_mse1]   #self.to_add['x2'],
            self.archive4.append(y_ob)
        #self._evaluate_metrics_coreg()


    def _get_pool_coreg(self):
        """
        Gets unlabeled pool and indices of unlabeled.
        """
        self.U_idx_pool = range(self.U_X.shape[0])

    def _remove_from_unlabeled_coreg(self):
        # Remove added examples from unlabeled
        to_remove = []
        if self.to_add['idx1'] is not None:
            to_remove.append(self.to_add['idx1'])
        if self.to_add['idx2'] is not None:
            to_remove.append(self.to_add['idx2'])
        self.U_X = np.delete(self.U_X, to_remove, axis=0)

    def get_crossover(self):
        new_pop = np.zeros((self.population_num, self.pop.shape[1]))
        for i in range(self.population_num):
            indi1 = int(np.floor(np.random.random() * self.population_num))
            while indi1 == i:
                indi1 = int(np.floor(np.random.random() * self.population_num))

            # 随机选取杂交点，然后交换数组
            cpoint = random.randint(1, 4)
            temp1 = []
            temp2 = []
            temp1.extend(self.pop[indi1][0:cpoint])
            temp1.extend(self.pop[i][cpoint:self.pop.shape[1]])
            temp2.extend(self.pop[i][0:cpoint])
            temp2.extend(self.pop[indi1][cpoint:self.pop.shape[1]])
            #new_pop[indi1] = temp1[:]
            new_pop[i] = temp2[:]
        self.pop = copy.deepcopy(new_pop)
        """

            for j, indi in enumerate(self.X_shuffled):
                if (temp1[:] == indi).all():
                    new_pop[indi1] = temp1[:]
                if (temp2[:] == indi).all():
                    new_pop[indi2] = temp2[:]
            while new_pop[indi1] is None or new_pop[indi2] is None:
                cpoint = random.randint(0, self.pop.shape[1])
                temp1 = []
                temp2 = []
                temp1.extend(self.pop[indi1][0:cpoint])
                temp1.extend(self.pop[indi2][cpoint:self.pop.shape[1]])
                temp2.extend(self.pop[indi2][0:cpoint])
                temp2.extend(self.pop[indi1][cpoint:self.pop.shape[1]])

                for j, indi in enumerate(self.X_shuffled):
                    if (temp1[:] == indi).all():
                        new_pop[indi1] = temp1[:]
                    if (temp2[:] == indi).all():
                        new_pop[indi2] = temp2[:]
                        """


    def get_mutation(self):
        new_pop = copy.deepcopy(self.pop)
        px = self.population_num
        py = self.pop.shape[1]
        # 每条染色体随便选一个杂交
        for i in range(px):
            if (random.random() > 0.1):
                mpoint = random.randint(0, 5)
                x = self.pop[i][mpoint]
                r = random.randint(0, 4)
                while r == x:
                    r = random.randint(0, 4)
                new_pop[i][mpoint] = r
        unique_pop = []
        new_pop = copy.deepcopy(new_pop.astype(np.int))
        pop_arch = [''.join(map(str, new_pop[i])) for i in range(len(new_pop))]
        arch = [''.join(map(str, self.archive[i])) for i in range(len(self.archive))]
        for i in range(self.population_num):
            tmp = new_pop[i]
            str_tmp = ''.join(map(str, tmp))
            while str_tmp in unique_pop or str_tmp in arch:
                indi1 = int(np.floor(np.random.random() * self.population_num))
                while indi1 == i:
                    indi1 = int(np.floor(np.random.random() * self.population_num))

                # 随机选取杂交点，然后交换数组
                cpoint = random.randint(1, 4)
                temp1 = []
                temp2 = []
                temp1.extend(self.pop[indi1][0:cpoint])
                temp1.extend(self.pop[i][cpoint:self.pop.shape[1]])
                temp2.extend(self.pop[i][0:cpoint])
                temp2.extend(self.pop[indi1][cpoint:self.pop.shape[1]])
                # new_pop[indi1] = temp1[:]
                tmp = temp2[:]
                tmp0 = copy.deepcopy(np.array(tmp).astype(np.int))
                str_tmp = ''.join(map(str, tmp0))
            new_pop[i] = tmp
            unique_pop.append(str_tmp)
            #self.archive.append(new_pop[i])

        self.pop = copy.deepcopy(new_pop)


        """
                for j, indi in enumerate(self.X_shuffled):
                    if (new_pop[i] == indi).all():
                        new_pop[i] = new_pop[i]
                    else:
                        new_pop[i] = pop[i]
                        """

    def get_acc(self, pop):
        pop_acc = np.zeros((len(pop), 1))
        for i in range(0, len(pop)):
            for j, indi in enumerate(self.X_all):
                if (pop[i] == indi).all():
                    pop_acc[i] = self.y_all[j]
        self.true_acc = pop_acc

    def environment_selection(self):
        pop_list = []
        acc_list = []
        true_list = []
        acc = np.zeros((self.population_num*2, 1))
        acc1 = []
        for i in range(0, self.population_num):
            pop_list.append(self.parent_pop[i])
            acc_list.append(self.parentt_acc[i])
        for i in range(0, self.population_num):
            pop_list.append(self.pop[i])
            acc_list.append(self.t_acc[i])
        pop_list = np.array(pop_list)
        for c, indi in enumerate(self.X_shuffled):
            for k, ins in enumerate(pop_list):
                if (pop_list[k] == self.X_shuffled[c]).all():
                    acc[k] = self.y_shuffled[c]
        tmp_list = copy.deepcopy(acc_list)
        tmp_list = np.array(tmp_list)

        max_acc, min_acc = np.max(tmp_list[:, 0]), np.min(tmp_list[:, 0])
        max_conf, min_conf = np.max(tmp_list[:, 1]), np.min(tmp_list[:, 1])
        for i in range(0, self.population_num * 2):
            if acc_list[i][0] == 0.0:
                    tmp_list[i][1] = -1.0
        max_conf, min_conf = np.max(tmp_list[:, 1]), np.min(tmp_list[:, 1])
        for i in range(0, self.population_num * 2):
            #tmp_list[i][0] = (tmp_list[i][0]-min_acc)/(max_acc-min_acc)
            tmp_list[i][1] = (tmp_list[i][1] - min_conf) / (max_conf - min_conf)
        for i in range(0, self.population_num * 2):
            if max_acc >= 1.0:
                tmp_list[i][0] = max_acc + 0.5 - tmp_list[i][0]
            else:
                tmp_list[i][0] = 1.0 - tmp_list[i][0]
            tmp_list[i][1] = 1.0 - tmp_list[i][1]
        #tmp_list[:,0] = tmp_list[:, 0]*100.0
        tmp_list = np.array(tmp_list)
        acc_list = np.array(acc_list)
        if max_acc >= 1.0:
            ref_point = np.array([max_acc+0.5, 1.0])
        else:
            ref_point = np.array([1.0, 1.0])
        ind = Hypervolume(ref_point=ref_point)
        #acc_idx = acc_list.reshape(acc_list.shape[0],).argsort()[::-1][0:100]
        fronts = fast_non_dominated_sort(F=tmp_list)
        t = 0
        acc_idx = []
        best_acc = 0
        a = np.ones((1, 2))
        for k, front in enumerate(fronts):
            if k == 0:
                hv = ind.do(tmp_list[front])
                hv1 = ind.do(acc_list[front])
                '''save the result on the pareto frontier'''
                dataframe = pd.DataFrame({'hv': [hv, hv1]})
                dataframe.to_csv("new_conf_RF/201_100_hv_indicator3_c100_610.csv", index=False, sep=',', mode='a+')
                dataframe = pd.DataFrame({'pred_acc': acc_list[front,0], 'confidence': acc_list[front, 1]})
                dataframe.to_csv("new_conf_RF/100_pred_front_seed3_c100_610.csv", index=False, sep=',', mode='a+')
                dataframe = pd.DataFrame({'pred_acc': acc[front].flatten()})
                dataframe.to_csv("new_conf_RF/100_true_seed3_c100_610.csv", index=False, sep=',', mode='a+')

                #dataframe = pd.DataFrame({'pred_acc':111, 'confidence': 111})
                #dataframe.to_csv("test.csv", index=False, sep=',')
            if t + len(front) <= self.population_num:
                    acc_idx = acc_idx + front
                    t = t + len(front)

            else:
                sample = tmp_list[front, :]
                new_sample = sample[:, 1].argsort()[::-1][0:(self.population_num - t)]
                a = np.array(front)[new_sample]
                acc_idx = acc_idx + a.tolist()
                t = t + len(a)
            if t >= self.population_num:
                break
        for i in range(0, self.population_num):
            self.pop[i] = pop_list[acc_idx[i]]
            self.t_acc[i] = acc_list[acc_idx[i]]
            self.true_acc[i] = acc[acc_idx[i]]
        '''
        if self.epoch == 0 or self.epoch == 29:
            dataframe = pd.DataFrame({'pred_acc': self.t_acc[:, 0], 'confidence': self.t_acc[:, 1]})
            dataframe.to_csv("RF/lastpop_pred.csv", index=False, sep=',', mode='a+')
            dataframe = pd.DataFrame({'pred_acc': self.true_acc.flatten()})
            dataframe.to_csv("RF/lastpop_true.csv", index=False, sep=',', mode='a+')
        '''
        self.all_pop.append(pop_list)
        self.all_tmp_acc.append(acc_list[:,0].flatten())
        self.all_true_acc.append(acc.flatten())
        self.all_pop0 = np.array(self.all_pop)
        self.all_tmp_acc0=np.array(self.all_tmp_acc)
        self.all_true_acc0=np.array(self.all_true_acc)

    def search(self):
        print(
            'population_num = {} select_num = {} mutation_num = {} crossover_num = {} random_num = {} max_epochs = {}'.format(
                self.population_num, self.select_num, self.mutation_num, self.crossover_num,
                self.population_num - self.mutation_num - self.crossover_num, self.max_epochs))
        '''run the save_encoding_bench_201 '''
        self.X_all = pd.read_csv("D:\Download\TSNAS\Bench_201\Bench_201_cifar100_X.csv") #ImageNet
        self.y_all = pd.read_csv("D:\Download\TSNAS\Bench_201\Bench_201_cifar100_test_y.csv")
        self.X_all = np.array(self.X_all)
        self.y_all = np.array(self.y_all)
        self.X_shuffled, self.y_shuffled = shuffle(self.X_all, self.y_all)
        #self.sample = random.sample(self.X_shuffled.tolist(), 500)
        #self.sample = np.array(self.sample)
        self.sample = self.X_shuffled[0:self.train_size]
        self.sample_acc = self.y_shuffled[0:self.train_size]/100.0
        # for c, indi in enumerate(self.X_all):
        #     for k, ins in enumerate(self.sample):
        #         if (self.sample[k] == self.X_all[c]).all():
        #             self.sample_acc[k] = self.y_all[c]
        '''
        self.sample = latin(self.train_size, 6, 0, 4)
        self.sample_acc = np.zeros(self.train_size, 1)
        for i in range(0, self.train_size):
            for j in range(0, self.sample.shape[1]):
                self.sample[i][j] = round(self.sample[i][j])
        for c, indi in enumerate(self.X_all):
            for k, ins in enumerate(self.sample):
                if (self.sample[k] == self.X_all[c]).all():
                    self.sample_acc[k] = self.y_all[c]
                    '''

        samples, samples_acc = shuffle(self.sample, self.sample_acc, random_state=False)
        self.pop = samples[0:self.population_num]


        MAX = max(self.sample_acc)
        print(MAX)
        self.archive = []
        self.archive_acc =[]
        for i in range(0, self.sample.shape[0]):
            self.archive.append(self.sample[i])
            self.archive_acc.append(self.sample_acc[i])
        self.tmp_acc = np.zeros((self.population_num, 1))
        t0 = time()
        self.add_data(self.archive, self.archive_acc)

        self._fit_and_evaluate()
        self._get_pool()
        for t in range(0, int(self.train_size/2)+50):
            stop_training = self._run_iteration(t, t0)
            if stop_training:
                break
        self.LL1_X = copy.deepcopy(self.L1_X)
        self.LL1_y = copy.deepcopy(self.L1_y)
        self.LL2_X = copy.deepcopy(self.L2_X)
        self.LL2_y = copy.deepcopy(self.L2_y)
        self.archive2 = []
        self.archive4 = []
        self.get_crossover()
        self.get_mutation()
        self.U_X = copy.deepcopy(self.pop)
        self._get_pool_coreg()
        for t in range(0, 50):
            stop_training = self._run_iteration_coreg(t, t0)
            if stop_training:
                break
        self.tmp_acc = np.zeros((self.population_num, 1))
        self.t_acc = np.zeros((self.population_num, 2))
        for c, indi in enumerate(self.pop):
            for k, ins in enumerate(self.L1_X):
                if (self.pop[c] == self.L1_X[k]).all():
                    self.tmp_acc[c] = float(self.L1_y[k])

        for c, indi in enumerate(self.pop):
            for k, ins in enumerate(self.L2_X):
                if (self.pop[c] == self.L2_X[k]).all():
                    self.tmp_acc[c] = float(self.L2_y[k])

        for c, indi in enumerate(self.tmp_acc):
            for k, ins in enumerate(self.archive2):
                if self.tmp_acc[c] == self.archive2[k][0]:
                    self.t_acc[c][0] = self.archive2[k][0]
                    self.t_acc[c][1] = self.archive2[k][1]

        for c, indi in enumerate(self.tmp_acc):
            for k, ins in enumerate(self.archive4):
                if self.tmp_acc[c] == self.archive4[k][0]:
                    self.t_acc[c][0] = self.archive4[k][0]
                    self.t_acc[c][1] = self.archive4[k][1]

        while self.epoch < self.max_epochs:
            print('epoch = {}'.format(self.epoch))
            self.L1_X = self.LL1_X
            self.L1_y = self.LL1_y
            self.L2_X = self.LL2_X
            self.L2_y = self.LL2_y
            self.parent_pop = copy.deepcopy(self.pop)
            #self.parent_acc = copy.deepcopy(self.tmp_acc)

            self.parentt_acc = copy.deepcopy(self.t_acc)
            self.get_crossover()
            self.get_mutation()
            pop_num = (self.epoch+1)*self.population_num
            #self.pop = self.X_all[pop_num:pop_num+self.population_num]
            #self.pop_acc = self.y_all[pop_num:pop_num+self.population_num]
            self.U_X = copy.deepcopy(self.pop)
            self._get_pool_coreg()
            for t in range(0, 50):
                stop_training = self._run_iteration_coreg(t, t0)
                if stop_training:
                    break
            self.tmp_acc = np.zeros((self.population_num, 1))
            self.t_acc = np.zeros((self.population_num, 2))
            for c, indi in enumerate(self.pop):
                for k, ins in enumerate(self.L1_X):
                    if (self.pop[c] == self.L1_X[k]).all():
                        self.tmp_acc[c] = float(self.L1_y[k])

            for c, indi in enumerate(self.pop):
                for k, ins in enumerate(self.L2_X):
                    if (self.pop[c] == self.L2_X[k]).all():
                        self.tmp_acc[c] = float(self.L2_y[k])

            for c, indi in enumerate(self.tmp_acc):
                for k, ins in enumerate(self.archive2):
                    if self.tmp_acc[c] == self.archive2[k][0]:
                        self.t_acc[c][0] = self.archive2[k][0]
                        self.t_acc[c][1] = self.archive2[k][1]

            for c, indi in enumerate(self.tmp_acc):
                for k, ins in enumerate(self.archive4):
                    if self.tmp_acc[c] == self.archive4[k][0]:
                        self.t_acc[c][0] = self.archive4[k][0]
                        self.t_acc[c][1] = self.archive4[k][1]
            self.environment_selection()
            print('predict best acc', max(self.all_tmp_acc0.flatten()))
            print('true best acc', max(self.all_true_acc0.flatten()))
            a = np.argmax(self.all_tmp_acc0.flatten())
            b = np.argmax(self.all_true_acc0.flatten())
            print('true acc of predict best acc location', self.all_true_acc0.flatten()[a])
            print('true best acc location:', (self.all_tmp_acc0.flatten()[b]))

            self.epoch += 1


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--log-dir', type=str, default='log')
    parser.add_argument('--max-epochs', type=int, default=30)
    parser.add_argument('--train_size', type=int, default=100)
    parser.add_argument('--select-num', type=int, default=10)
    parser.add_argument('--population-num', type=int, default=100)
    parser.add_argument('--m_prob', type=float, default=0.1)
    parser.add_argument('--crossover-num', type=int, default=25)
    parser.add_argument('--mutation-num', type=int, default=25)
    parser.add_argument('--flops-limit', type=float, default=330 * 1e6)
    parser.add_argument('--max-train-iters', type=int, default=200)
    parser.add_argument('--max-test-iters', type=int, default=40)
    parser.add_argument('--train-batch-size', type=int, default=128)
    parser.add_argument('--test-batch-size', type=int, default=200)
    parser.add_argument('--model', type=str, default='rf', help='choice between xgb, mlp, rf, svr, knns')
    args = parser.parse_args()

    searcher = EvolutionSearcher(args)

    searcher.search()


if __name__ == '__main__':
    main()

