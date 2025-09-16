import numpy as np
def Ridge(X, y, lambd):
  # add constant
  X = np.hstack([np.ones((X.shape[0], 1)), X]) # create X.shape[0] x 1 Vector of ones and puts that as the first column in X (horizontal stack)

  # get shape
  n, p = X.shape

  # using that, calculate the analytical solution
  # CAUTION: one should not apply regularization to the intercept! so create lambda matrix and remove reg. term from the intercept column
  L = lambd*np.eye(p)
  L[0, 0] = 0  # no penalty on intercept
  Beta_hat = (np.linalg.inv(X.T@X + L) @ X.T @ y) # see formula above

  # then simply transpose and flatten it so it's just a simple vector an can easily be accessed
  Beta_hat = Beta_hat.T
  Beta_hat = Beta_hat.flatten()

  return Beta_hat