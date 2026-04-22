def zero_to_nan(x):
    return np.where(x == 0, np.nan, x).astype(float)