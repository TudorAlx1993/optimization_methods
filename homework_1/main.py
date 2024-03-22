import numpy as np
import cvxpy as cp
from scipy import optimize


def main():
    seed = 33
    np.random.seed(seed)

    m = 70
    n = 50
    min_k = 10 ** 5
    ex_1(m, n, min_k)


def ex_1(m, n, min_k, epsilon=10 ** -5):
    A, b, L = generate_inputs_exercise_1(m, n, min_k)

    #x_init = np.random.rand(n, 1).astype(np.float64)
    x_init = np.zeros((n,1)).astype(np.float64)

    alpha = 0.1
    k_constant_step, x_star_constant_step, gradient_x_star_constant_step = gradient_method_constant_step_ex_1(x_init, A,
                                                                                                              b,
                                                                                                              epsilon,
                                                                                                              alpha)
    #k_ideal_step, x_star_ideal_step, gradient_x_star_ideal_step = gradient_method_ideal_step_ex_1(x_init, A, b, epsilon)
    #k_adaptive_step, x_star_adaptive_step, gradient_x_star_adaptive_step = gradient_method_adaptive_step_ex_1(x_init, A,
    #                                                                                                          b,
    #                                                                                                          epsilon)

    #x_star_scipy = np.linalg.lstsq(A, b)[0].reshape(-1, 1)

    #print(x_star_scipy.flatten())
    #print(x_star_adaptive_step.flatten())
    #print(np.linalg.norm(gradient_function_ex_1(x_star_scipy, A, b)) ** 2)
    #print(np.linalg.norm(gradient_function_ex_1(x_star_adaptive_step, A, b)) ** 2)

def gradient_method_adaptive_step_ex_1(x_init, A, b, epsilon):
    k = 0
    x_current = x_init
    gradient_current = gradient_function_ex_1(x_current, A, b)
    while np.linalg.norm(gradient_current) > epsilon:
        alpha_adaptive = alpha_adaptive_ex_1(x_current, gradient_current, A, b)
        x_current = x_current - alpha_adaptive * gradient_current
        gradient_current = gradient_function_ex_1(x_current, A, b)
        k += 1

    return k, x_current, gradient_current


def alpha_adaptive_ex_1(x, gradient, A, b):
    c, rho, alpha = np.random.rand(3)

    while (loss_function_ex_1(x - alpha * gradient, A, b)) > (
            loss_function_ex_1(x, A, b) - c * alpha * np.linalg.norm(gradient) ** 2):
        alpha = rho * alpha

    return alpha


def gradient_method_ideal_step_ex_1(x_init, A, b, epsilon):
    k = 0
    x_current = x_init
    gradient_current = gradient_function_ex_1(x_current, A, b)
    while np.linalg.norm(gradient_current) > epsilon:
        alpha_ideal = alpha_ideal_ex_1(x_current, gradient_current, A, b)
        x_current = x_current - alpha_ideal * gradient_current
        gradient_current = gradient_function_ex_1(x_current, A, b)
        k += 1

    return k, x_current, gradient_current


def alpha_ideal_ex_1(x, gradient, A, b):
    def objective_function(alpha, x, gradient, A, b):
        return loss_function_ex_1(x - alpha * gradient, A, b)

    def constraint(alpha):
        return alpha

    alpha_init = np.random.rand(1)
    optimization_result = optimize.minimize(fun=objective_function, x0=alpha_init, args=(x, gradient, A, b),
                                            constraints={'type': 'ineq', 'fun': constraint})
    alpha = optimization_result.x

    return alpha


def gradient_method_constant_step_ex_1(x_init, A, b, epsilon, alfa):
    k = 0
    x_current = x_init
    gradient_current = gradient_function_ex_1(x_current, A, b)
    while np.linalg.norm(gradient_current) > epsilon:
        x_current = x_current - alfa * gradient_current
        #print('x_current={}'.format(x_current.shape))
        print(x_init.shape)
        gradient_current = gradient_function_ex_1(x_current, A, b)
        k += 1

    return k, x_current, gradient_current


def gradient_function_ex_1(x, A, b):
    #gradient = np.matmul(A.transpose(), np.matmul(A, x) - b.reshape(-1, 1))
    gradient=A.T.dot(A.dot(x)-b)

    return gradient


def loss_function_ex_1(x, A, b):
    loss = (1.0 / 2.0) * np.linalg.norm(np.matmul(A, x) - b.reshape(-1, 1)) ** 2

    return loss


def generate_inputs_exercise_1(m, n, min_k):
    A = np.random.rand(m, n).astype(np.float64)
    b = np.random.rand(m,1).astype(np.float64)

    return A, b, None

    while True:
        hessian_matrix = np.matmul(A.transpose(), A)

        eigenvalues = np.linalg.eigvals(hessian_matrix)
        assert np.all(eigenvalues > 0)

        min_eigenvalue = np.min(eigenvalues)
        max_eigenvalue = np.max(eigenvalues)
        k = max_eigenvalue / min_eigenvalue

        if k > min_k:
            break
        A[0] *= 2

    k_numpy = np.linalg.cond(hessian_matrix)
    assert np.allclose(k, k_numpy)

    return A, b, max_eigenvalue


if __name__ == '__main__':
    main()
