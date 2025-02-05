def miniBatchGradientDescent(X, y, theta, alpha, batch_size, num_iterations):
    m = len(y)  # Total number of training examples
    for _ in range(num_iterations):
        # Randomly sample indices for the current mini-batch (without replacement)
        indices = np.random.choice(m, batch_size, replace=False)
        X_batch = X[indices]
        y_batch = y[indices]
        
        # Compute the gradient over the mini-batch
        gradient = np.zeros_like(theta)
        for k in range(batch_size):
            xi = X_batch[k]
            yi = y_batch[k]
            prediction = np.dot(xi, theta)
            error = prediction - yi
            gradient += error * xi
        
        # Update theta with the average gradient of the mini-batch
        theta -= (alpha / batch_size) * gradient
    return theta