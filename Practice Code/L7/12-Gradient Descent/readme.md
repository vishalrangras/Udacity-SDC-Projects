#Gradient Descent Solution

def gradient_descent_update(x, gradx, learning_rate):
    """
    Performs a gradient descent update.
    """
    x = x - learning_rate * gradx
    # Return the new value for x
    return x

We adjust the old x pushing it in the direction of gradx with the force learning_rate. Subtracting learning_rate * gradx. Remember the gradient is initially in the direction of steepest ascent so subtracting learning_rate * gradx from x turns it into steepest descent. You can make sure of this yourself by replacing the subtraction with an addition.