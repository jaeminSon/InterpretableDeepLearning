import numpy as np


def get_offset(epsilon: float) -> float:
    return -np.log((1 - epsilon) / epsilon)


def counterfactual_attribution_ratio(w_f_d: np.array, w_f: np.array, yhat: np.array = None, epsilon: float = 1e-6) -> float:
    """
    Args
        w_f_d: arraylike, length of c
        w_f: arraylike, length of c
        featuremap: arraylike, size of (h,w,c)
    
    Returns
        counterfactual attribution ratio: float
        
    """
    if yhat is None:
        yhat = 1-epsilon

    inv_sigma_yhat = get_offset(yhat)
    inv_sigma_epsilon = get_offset(epsilon)

    return np.exp(inv_sigma_yhat-inv_sigma_epsilon) * w_f_d.dot(w_f) / np.linalg.norm(w_f)


def get_activation_finding(w_f_d: np.array, w_f: np.array, featuremap: np.array):
    """
        Args
        w_f_d: arraylike, length of c
        w_f: arraylike, length of c
        featuremap: arraylike, size of (h,w,c)

    Returns
        3 activation maps (all size of (h,w,1))
        total activation map using the feature map, attributed activation map, residual activation
    """
    vector_finding = (w_f_d.dot(w_f) / np.linalg.norm(w_f)) * w_f
    vector_finding_expand = vector_finding[None, None]
    return np.sum(featuremap * vector_finding_expand, axis=-1, keepdims=True)


def get_activation(w_f_d: np.array, w_f: np.array, featuremap: np.array):
    """
    Args
        w_f_d: arraylike, length of c
        w_f: arraylike, length of c
        featuremap: arraylike, size of (h,w,c)

    Returns
        3 activation maps (all size of (h,w,1))
        total activation map using the feature map, attributed activation map, residual activation
    """
    w_f_d_expand = np.array(w_f_d)[None, None]  # (n_features, ) -> (1, 1, n_features)

    total_activation = np.sum(
        featuremap * w_f_d_expand, axis=-1, keepdims=True)

    activation_finding = get_activation_finding(w_f_d, w_f, featuremap)

    activation_others = total_activation - activation_finding

    return total_activation, activation_finding, activation_others


if __name__ == "__main__":

    counterfactual_attribution_ratio(
        np.random.random(128), np.random.random(128))
    counterfactual_attribution_ratio(
        np.random.random(128), np.random.random(128), 0.9)

    get_activation(np.random.random(128), np.random.random(128), np.random.random((256, 256, 128)))
