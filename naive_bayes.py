from typing import Tuple

import numpy as np
from scipy import stats


def normalize(matrix: np.array, axis: int) -> np.array:
    normalized_vals = matrix / np.sum(matrix, axis=axis, keepdims=True)
    return normalized_vals


def majority(user_answer_matrix: np.array) -> np.array:
    """
    Args
        user_answer_matrix: K*N matrix (K: #users, N: #data)

    Returns
        arraylike with length of C (prior matrix)
    """
    mode, count = stats.mode(user_answer_matrix, nan_policy="omit")
    y_majority = mode[0].astype(int)
    class_prior = 1.*np.bincount(y_majority) / len(y_majority)
    return class_prior


def initialize_conditional_probabilities(user_answer_matrix: np.array, non_zero_value: float = 1e-3) -> np.array:
    """
    Args
        user_answer_matrix: K*N matrix (K: #users, N: #data)

    Returns
        K*C*C conditional probability matrix
    """
    K, N = user_answer_matrix.shape
    C = len(np.unique(user_answer_matrix[~np.isnan(user_answer_matrix)]))

    # get true y by majority rule
    mode, count = stats.mode(user_answer_matrix, nan_policy="omit")
    y_majority = mode[0].astype(int)

    # init v
    v = np.zeros((K, C, C))
    for k in range(K):
        for c1 in range(C):
            for c2 in range(C):
                if len(y_majority[y_majority == c2]) != 0:
                    if len(y_majority[(user_answer_matrix[k] == c1) & (y_majority == c2)]) != 0:
                        v[k, c1, c2] = 1.*len(y_majority[(user_answer_matrix[k] == c1) & (
                            y_majority == c2)]) / len(y_majority[y_majority == c2])
                    else:
                        v[k, c1, c2] = non_zero_value
                else:
                    v[k, c1, c2] = 1. / C

    return normalize(v, axis=1)


def expectation(user_answer_matrix: np.array, v: np.array, p: np.array) -> Tuple[np.array, np.array]:
    """
    Args
        user_answer_matrix: K*N matrix (K: #users, N: #data)
        v: K*C*C conditional probability matrix 
        p: arraylike with length of C (prior matrix)

    Returns
        expected posterior probability (C*N matrix) and log_likelihood (int)
    """
    K, N = user_answer_matrix.shape
    rng_K = np.array(range(K))
    posterior = np.apply_along_axis(lambda x: p * np.prod(v[rng_K[~np.isnan(x)].astype(
        int), x[~np.isnan(x)].astype(int), :], axis=0), 0, user_answer_matrix)
    log_likelihood = np.sum(np.log(np.sum(posterior, axis=0)))
    posterior = normalize(posterior, axis=0)
    return posterior, log_likelihood


def maximization(user_answer_matrix: np.array, posterior: np.array) -> Tuple[np.array, np.array]:
    """
    Args
        user_answer_matrix: K*N matrix (K: #users, N: #data)
        posterior: C*N matrix
    Returns
        (updated conditional matrix (K*C*C), prior matrix (length of C))
    """
    p = np.sum(posterior, axis=1)

    K, N = user_answer_matrix.shape
    C, N = posterior.shape

    v = np.zeros((K, C, C))

    for i in range(K):
        for c1 in range(C):
            for c2 in range(C):
                v[i][c1][c2] = np.sum(posterior[c2][user_answer_matrix[i] == c1]) / \
                    np.sum(posterior[c2][~np.isnan(user_answer_matrix[i])])

    p = normalize(p, axis=0)

    return v, p


def estimate_conditional_prob(user_answer_matrix: np.array, N_iters: int = 10, halting_coefficient: int = 1) -> Tuple[np.array, np.array]:
    """
    Args
        user_answer_matrix: K*N matrix (K: #users, N: #data)
        N_iters: the number of maximum iterations
        halting_coefficient: halts when delta <= halting_coefficient * delta_prev

    Returns
        (updated conditional matrix (K*C*C), prior matrix (length of C))
    """

    v = initialize_conditional_probabilities(user_answer_matrix)
    p = majority(user_answer_matrix)

    ll = 0
    delta = -float("inf")
    for _ in range(N_iters):

        # store prev values
        ll_prev = ll
        delta_prev = delta

        # EM
        posterior, ll = expectation(user_answer_matrix, v, p)
        v, p = maximization(user_answer_matrix, posterior)

        # halting condition
        delta = ll - ll_prev
        if delta <= halting_coefficient * delta_prev:
            break

    return v, p


def posterior(user_answer_matrix, v, p) -> np.array:
    """
    return 

    Args
        user_answer_matrix: K*N matrix (K: #users, N: #data)
        v: K*C*C matrix 
        p: array or list of length C
    Returns
        posterior probability (C*N)
    """
    K, N = user_answer_matrix.shape
    _, C, _ = v.shape
    prob_map = np.zeros((C, N))

    for n in range(N):
        ans = user_answer_matrix[:, n]
        for t in range(C):
            mul = 1
            for k in range(K):
                if not np.isnan(ans[k]):
                    mul *= v[k, int(ans[k]), t]
            prob_map[t, n] = mul * p[t]

    return normalize(prob_map, axis=0)


if __name__ == "__main__":
    # step 1. define user_answer matrix (value)
    n_classes = 10
    n_users = 5
    n_data = 100
    user_answer_matrix = np.random.randint(n_classes, size=(n_users, n_data))

    # step 2. estimate conditional matrix and prior matrix
    v, p = estimate_conditional_prob(user_answer_matrix)

    # step 3. estimate posterior
    post = posterior(user_answer_matrix, v, p)

    assert post.shape == (10, 100)
