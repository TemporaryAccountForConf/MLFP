#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
from sklearn.metrics import euclidean_distances
from scipy import optimize
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import f1_score


class mlfp():
    def __init__(self, k=3, mu=0.5, c=0, g=1, nFtsOut=None, maxCst=int(1e7),
                 randomState=None, max_iter=500, diag=True, const1=False):
        self.k = k
        self.mu = mu
        self.c = c
        self.g = g
        self.nFtsOut_ = nFtsOut
        self.maxCst = maxCst
        self.randomState = randomState
        self.max_iter = max_iter
        self.diag = diag
        self.const1 = const1
        self.store_L = []

    def fit(self, X, y):
        self.X = X

        self.idx_pos = np.where(y == 1)[0]  # indexes of pos examples
        self.idx_neg = np.where(y != 1)[0]  # indexes of other examples
        self.nb_pos = len(self.idx_pos)
        self.nb_neg = len(self.idx_neg)
        self.m = self.nb_neg + self.nb_pos

        if self.nb_pos <= 1:
            print("Error, there should be at least 2 positive examples")
            return

        # Initialize the number of neighbors
        if self.k >= self.nb_pos:
            self.k = self.nb_pos - 1  # maximum possible number of neighbors
        if self.k <= 0:
            self.k = 1  # we need at least one neighbor

        # Initialize matrix L

        if self.diag:
            self.L = np.random.rand(X.shape[1])
            if self.const1:
                m = max(self.L.max(), -self.L.min())
                self.L = self.g*self.L/m
        else:
            self.L = np.random.rand(X.shape[1], X.shape[1])
            if self.const1:
                evals, _ = np.linalg.eigh(np.dot(self.L.T, self.L))
                m = max(evals.max(), -evals.min())
                self.L = self.g*self.L/(m**0.5)

        if self.ctrl:
            self.store_L.append(self.L.flatten())

        # Positive Positive Pairs
        D = euclidean_distances(self.X[self.idx_pos], squared=True)
        np.fill_diagonal(D, np.inf)
        Didx = np.argsort(D)  # indexes for matrix D sorted ascending
        self.SimP_i = []
        self.SimP_j = []
        for idxI in range(len(self.idx_pos)):  # for each positive example
            idxIdxJ = 0
            while idxIdxJ < self.k:
                idxJ = Didx[idxI][idxIdxJ]
                self.SimP_i.append(self.idx_pos[idxI])
                self.SimP_j.append(self.idx_pos[idxJ])
                idxIdxJ += 1
        self.SimP_i = np.array(self.SimP_i)
        self.SimP_j = np.array(self.SimP_j)

        # Negative Negative Pairs
        D = euclidean_distances(self.X[self.idx_neg], squared=True)
        np.fill_diagonal(D, np.inf)
        Didx = np.argsort(D)  # indexes for matrix D sorted ascending
        self.SimN_i = []
        self.SimN_j = []
        for idxI in range(len(self.idx_neg)):  # for each negative example
            idxIdxJ = 0
            while idxIdxJ < self.k:
                idxJ = Didx[idxI][idxIdxJ]
                self.SimN_i.append(self.idx_neg[idxI])
                self.SimN_j.append(self.idx_neg[idxJ])
                idxIdxJ += 1
        self.SimN_i = np.array(self.SimN_i)
        self.SimN_j = np.array(self.SimN_j)

        # Positive Negative Pairs
        D = euclidean_distances(self.X[self.idx_pos], self.X[self.idx_neg],
                                squared=True)
        Didx = np.argsort(D)
        self.DisP_i = []
        self.DisP_j = []
        for idxI in range(len(self.idx_pos)):  # for each posiive example
            idxIdxJ = 0
            while idxIdxJ < self.k:
                idxJ = Didx[idxI][idxIdxJ]
                self.DisP_i.append(self.idx_pos[idxI])
                self.DisP_j.append(self.idx_neg[idxJ])
                idxIdxJ += 1
        self.DisP_i = np.array(self.DisP_i)
        self.DisP_j = np.array(self.DisP_j)

        # Negative Positive Pairs
        D = euclidean_distances(self.X[self.idx_neg], self.X[self.idx_pos],
                                squared=True)
        Didx = np.argsort(D)
        self.DisN_i = []
        self.DisN_j = []
        for idxI in range(len(self.idx_neg)):  # for each posiive example
            idxIdxJ = 0
            while idxIdxJ < self.k:
                idxJ = Didx[idxI][idxIdxJ]
                self.DisN_i.append(self.idx_neg[idxI])
                self.DisN_j.append(self.idx_pos[idxJ])
                idxIdxJ += 1
        self.DisN_i = np.array(self.DisN_i)
        self.DisN_j = np.array(self.DisN_j)

        del(D, Didx, idxI, idxJ, idxIdxJ)

        # Call the L-BFGS-B optimizer
        L, loss, details = optimize.fmin_l_bfgs_b(
                       maxiter=200, func=self.loss_grad, x0=self.L)

        # Reshape result from optimizer
        if self.diag:
            self.L = np.diag(L)
            if self.const1:
                m = max(self.L.max(), -self.L.min())
                self.L = self.g*self.L/m
        else:
            self.L = L.reshape((self.X.shape[1], self.X.shape[1]))
            if self.const1:
                evals, _ = np.linalg.eigh(np.dot(self.L.T, self.L))
                m = max(evals.max(), -evals.min())
                self.L = self.g*self.L/(m**0.5)

        if self.ctrl:
            self.store_L.append(np.diag(self.L))

        return self

    def loss_grad(self, L):

        if self.diag:
            L = np.diag(L)
            if self.const1:
                m = max(L.max(), -L.min())
                L = self.g*L/m
        else:
            L = L.reshape((self.X.shape[1], self.X.shape[1]))
            if self.const1:
                evals, _ = np.linalg.eigh(np.dot(L, L.T))
                m = max(evals.max(), -evals.min())
                L = self.g*L/(m**0.5)

        if self.ctrl:
            self.L_ = L
            pred1 = self.predict(self.X_train)
            pred2 = self.predict(self.X_valid)
            pred3 = self.predict(self.X_test)
            self.store_f_m.append([f1_score(self.y_train, pred1),
                                   f1_score(self.y_valid, pred2),
                                   f1_score(self.y_test, pred3)])

        M = L.T.dot(L)

        if self.ctrl:
            self.store_L.append(np.diag(L))

        # Compute pairwise mahalanobis distance between positive examples
        # with the current projection matrix L
        Dm_pp = np.sum((self.X[self.SimP_i].dot(L.T) -
                        self.X[self.SimP_j].dot(L.T))**2, axis=1)

        # Compute pairwise distance between negative examples
        D_nn = np.sum((self.X[self.SimN_i] -
                       self.X[self.SimN_j])**2, axis=1)

        # Compute pairwise mahalanobis distance between negative examples and
        # positive examples with the current projection matrix L
        Dm_np = np.sum((self.X[self.DisN_i].dot(L.T) -
                        self.X[self.DisN_j].dot(L.T))**2, axis=1)

        # Compute pairwise distance between positive examples and negative
        # examples
        D_pn = np.sum((self.X[self.DisP_i].dot(L.T) -
                       self.X[self.DisP_j].dot(L.T))**2, axis=1)

        # L_FN
        temp = np.array([1 - self.c + dmpp - dpn
                         for i in range(self.nb_pos)
                         for dmpp in Dm_pp[self.k * i: self.k * i + self.k]
                         for dpn in D_pn[self.k * i: self.k * i + self.k]])
        L_FN_l = np.sum(np.maximum(temp, 0))

        idx = np.where(temp > 0)
        diff = (self.X[np.repeat(self.SimP_i, self.k)] -
                self.X[np.repeat(self.SimP_j, self.k)])
        diff = diff[idx]
        L_FN_g = 2*L.dot(diff.T.dot(diff))

        # L_FP
        temp = np.array([1 - self.c + dnn - dmnp
                         for i in range(self.nb_neg)
                         for dnn in D_nn[self.k * i: self.k * i + self.k]
                         for dmnp in Dm_np[self.k * i: self.k * i + self.k]])
        L_FP_l = np.sum(np.maximum(temp, 0))

        idx = np.where(temp > 0)
        diff = (self.X[np.repeat(self.DisN_i, self.k)] -
                self.X[np.repeat(self.DisN_j, self.k)])
        diff = diff[idx]
        L_FP_g = -2*L.dot(diff.T.dot(diff))

        # Squared Frobenius norm term
        identity = np.eye(M.shape[0])
        N_g = 4*L.dot(L.T.dot(L) - identity)
        N_l = np.sum((M-identity)**2)

        alpha = self.nb_pos / (self.nb_neg + self.nb_pos)

        loss = ((1 - alpha) / (self.m ^ 3) * L_FN_l +
                alpha / (self.m ^ 3) * L_FP_l +
                self.mu * N_l)

        gradient = ((1 - alpha) / (self.m ^ 3) * L_FN_g +
                    alpha / (self.m ^ 3) * L_FP_g +
                    self.mu * N_g)

        if self.diag:
            gradient = np.diag(gradient)

        if self.ctrl:
            self.store_loss.append((loss,
                                    self.loss_on_test(L,
                                                      self.X_valid,
                                                      self.y_valid),
                                    self.loss_on_test(L,
                                                      self.X_test,
                                                      self.y_test)))

        return loss, gradient.flatten()

    def transform(self, X):
        return X.dot(self.L.T)

    def predict(self, X):

        if self.ctrl:
            Lx = X.dot(self.L_.T)
        else:
            Lx = self.transform(X)

        # If X != X_train
        if X.shape != self.X.shape or np.sum(X - self.X) != 0:
            # Compute k nearest negative neighbors
            nn_neg = NearestNeighbors(n_neighbors=self.k)
            nn_neg.fit(self.X[self.idx_neg])
            knn_neg_dist, knn_neg = nn_neg.kneighbors(X)

            # Compute k nearest positive neighbors
            nn_pos = NearestNeighbors(n_neighbors=self.k)
            nn_pos.fit(self.X[self.idx_pos])
            knn_pos = nn_pos.kneighbors(X, return_distance=False)

        else:
            # Compute k nearest negative neighbors
            nn_neg = NearestNeighbors(n_neighbors=self.k+1)
            nn_neg.fit(self.X[self.idx_neg])
            temp_knn_neg_dist, temp_knn_neg = nn_neg.kneighbors(X)

            # Compute k nearest positive neighbors
            nn_pos = NearestNeighbors(n_neighbors=self.k+1)
            nn_pos.fit(self.X[self.idx_pos])
            temp_knn_pos = nn_pos.kneighbors(X, return_distance=False)

            knn_neg = np.zeros((len(X), self.k))
            knn_neg_dist = np.zeros((len(X), self.k))
            knn_pos = np.zeros((len(X), self.k), dtype=np.int8)
            for p in range(len(X)):
                if temp_knn_neg_dist[p, 0] == 0:
                    knn_neg[p] = temp_knn_neg[p, 1:]
                    knn_neg_dist[p] = temp_knn_neg_dist[p, 1:]
                    knn_pos[p] = temp_knn_pos[p, :-1]
                else:
                    knn_neg[p] = temp_knn_neg[p, :-1]
                    knn_neg_dist[p] = temp_knn_neg_dist[p, :-1]
                    knn_pos[p] = temp_knn_pos[p, 1:]

        # Compute the distances under L
        knn_pos_dist = np.zeros((len(X), self.k))
        for p in range(len(X)):
            if self.ctrl:
                knn_pos_dist[p] = np.sqrt(np.sum(np.square(
                        self.X[self.idx_pos][knn_pos[p]].dot(self.L_.T) -
                        Lx[p]), axis=1))
            else:
                knn_pos_dist[p] = np.sqrt(np.sum(np.square(
                        self.transform(self.X[self.idx_pos][knn_pos[p]]) -
                        Lx[p]), axis=1))

        # Sort 2Knn with their distances
        knn_sort = np.concatenate((np.ones((len(X), self.k)),
                                   np.zeros((len(X), self.k))), axis=1)
        knn_sort_dist = np.concatenate((knn_pos_dist, knn_neg_dist), axis=1)

        for p in range(len(X)):
            knn_sort[p] = knn_sort[p][np.argsort(knn_sort_dist[p])]

        # predict
        knn_sort = knn_sort[:, :self.k]
        pred = np.count_nonzero(knn_sort, axis=1) >= self.k//2+1
        return pred

    def loss_on_test(self, L_, X_t, y_t):

        idx_pos = np.where(y_t == 1)[0]  # indexes of pos examples
        idx_neg = np.where(y_t != 1)[0]  # indexes of other examples
        nb_pos = len(idx_pos)
        nb_neg = len(idx_neg)
        m = nb_neg + nb_pos

        # Positive Positive Pairs
        D = euclidean_distances(X_t[idx_pos], squared=True)
        np.fill_diagonal(D, np.inf)
        Didx = np.argsort(D)  # indexes for matrix D sorted ascending
        SimP_i = []
        SimP_j = []
        for idxI in range(len(idx_pos)):  # for each positive example
            idxIdxJ = 0
            while idxIdxJ < self.k:
                idxJ = Didx[idxI][idxIdxJ]
                SimP_i.append(idx_pos[idxI])
                SimP_j.append(idx_pos[idxJ])
                idxIdxJ += 1
        SimP_i = np.array(SimP_i)
        SimP_j = np.array(SimP_j)

        # Negative Negative Pairs
        D = euclidean_distances(X_t[idx_neg], squared=True)
        np.fill_diagonal(D, np.inf)
        Didx = np.argsort(D)  # indexes for matrix D sorted ascending
        SimN_i = []
        SimN_j = []
        for idxI in range(len(idx_neg)):  # for each negative example
            idxIdxJ = 0
            while idxIdxJ < self.k:
                idxJ = Didx[idxI][idxIdxJ]
                SimN_i.append(idx_neg[idxI])
                SimN_j.append(idx_neg[idxJ])
                idxIdxJ += 1
        SimN_i = np.array(SimN_i)
        SimN_j = np.array(SimN_j)

        # Positive Negative Pairs
        D = euclidean_distances(X_t[idx_pos], X_t[idx_neg], squared=True)
        Didx = np.argsort(D)
        DisP_i = []
        DisP_j = []
        for idxI in range(len(idx_pos)):  # for each posiive example
            idxIdxJ = 0
            while idxIdxJ < self.k:
                idxJ = Didx[idxI][idxIdxJ]
                DisP_i.append(idx_pos[idxI])
                DisP_j.append(idx_neg[idxJ])
                idxIdxJ += 1
        DisP_i = np.array(DisP_i)
        DisP_j = np.array(DisP_j)

        # Negative Positive Pairs
        D = euclidean_distances(X_t[idx_neg], X_t[idx_pos], squared=True)
        Didx = np.argsort(D)
        DisN_i = []
        DisN_j = []
        for idxI in range(len(idx_neg)):  # for each posiive example
            idxIdxJ = 0
            while idxIdxJ < self.k:
                idxJ = Didx[idxI][idxIdxJ]
                DisN_i.append(idx_neg[idxI])
                DisN_j.append(idx_pos[idxJ])
                idxIdxJ += 1
        DisN_i = np.array(DisN_i)
        DisN_j = np.array(DisN_j)

        del(D, Didx, idxI, idxJ, idxIdxJ)

        # Compute pairwise mahalanobis distance between positive examples
        # with the current projection matrix L
        Dm_pp = np.sum((X_t[SimP_i].dot(L_.T) -
                        X_t[SimP_j].dot(L_.T))**2, axis=1)

        # Compute pairwise distance between negative examples
        D_nn = np.sum((X_t[SimN_i] -
                       X_t[SimN_j])**2, axis=1)

        # Compute pairwise mahalanobis distance between negative examples and
        # positive examples with the current projection matrix L
        Dm_np = np.sum((X_t[DisN_i].dot(L_.T) -
                        X_t[DisN_j].dot(L_.T))**2, axis=1)

        # Compute pairwise distance between positive examples and negative
        # examples
        D_pn = np.sum((X_t[DisP_i].dot(L_.T) -
                       X_t[DisP_j].dot(L_.T))**2, axis=1)

        # L_FN
        temp = np.array([1 - self.c + dmpp - dpn
                         for i in range(nb_pos)
                         for dmpp in Dm_pp[self.k * i: self.k * i + self.k]
                         for dpn in D_pn[self.k * i: self.k * i + self.k]])
        L_FN_l = np.sum(np.maximum(temp, 0))

        # L_FP
        temp = np.array([1 - self.c + dnn - dmnp
                         for i in range(nb_neg)
                         for dnn in D_nn[self.k * i: self.k * i + self.k]
                         for dmnp in Dm_np[self.k * i: self.k * i + self.k]])
        L_FP_l = np.sum(np.maximum(temp, 0))

        # Squared Frobenius norm term
        M = L_.T.dot(L_)
        identity = np.eye(M.shape[0])
        N_l = np.sum((M-identity)**2)

        alpha = nb_pos / (nb_neg + nb_pos)

        loss = ((1 - alpha) / (m ^ 3) * L_FN_l +
                alpha / (m ^ 3) * L_FP_l +
                self.mu * N_l)

        return loss
