import os
import pickle
import numpy as np
import cvxpy as cp
import matplotlib.pyplot as plt


def main():
    # global configurations
    seed = 13
    np.random.seed(seed)

    output_dir = './outputs'
    n = 100
    m = 10
    epsilon = 10 ** -6

    # punctul a)
    weights, X_plus, y_plus, X_minus, y_minus = ex_1(n, m, output_dir)

    # punctul b)
    X = np.vstack((X_plus, X_minus))
    y = np.concatenate((y_plus, y_minus), axis=0)

    weights_star_v1 = ex_2_v1(X, y, epsilon, output_dir)
    assert np.all(np.sign(np.dot(X, weights_star_v1)) == np.sign(y))

    A = np.multiply(-1 * y.reshape(-1, 1), X)
    b = -1 * np.ones(shape=y.shape)
    weights_star_v2 = ex_2_v2(A, b, epsilon, output_dir)
    assert np.all(np.sign(np.dot(X, weights_star_v2)) == np.sign(y))
    assert np.all(np.dot(A, weights_star_v1) <= b)
    assert np.all(np.dot(A, weights_star_v2) <= b)
    assert np.allclose(weights_star_v1, weights_star_v2, rtol=epsilon, atol=epsilon)

    weights_star_v3 = ex_2_v3(A, b, epsilon, output_dir)
    assert np.all(np.sign(np.dot(X, weights_star_v3)) == np.sign(y))
    assert np.all(np.dot(A, weights_star_v3) <= b)

    # punctul c)
    weights_star_v4, no_iterations_v4 = ex_3(A, b, epsilon=epsilon, output_dir=output_dir)
    assert np.all(np.sign(np.dot(X, weights_star_v4)) == np.sign(y))

    # punctul d)
    weights_star_v5, no_iterations_v5 = ex_4(X, y, output_dir)
    assert np.all(np.sign(np.dot(X, weights_star_v5)) == np.sign(y))
    compare_gradient_with_yakubovich_iterations(gradient_no_iters=no_iterations_v4,
                                                yakubovich_no_iters=no_iterations_v5,
                                                output_dir=os.path.join(output_dir, 'exercise_4'))


def compare_gradient_with_yakubovich_iterations(gradient_no_iters, yakubovich_no_iters, output_dir):
    file_content = 'Comparison of convergence:\n'
    file_content += '\t* gradient method: convergence after {} iterations\n'.format(gradient_no_iters)
    file_content += '\t* Yakubovich-Kaczmarz method: convergence after {} iterations\n'.format(yakubovich_no_iters)

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    file = open(os.path.join(output_dir, 'convergence_comparison.txt'), 'w')
    file.write(file_content)
    file.close()


def ex_4(X, y, output_dir=None):
    A = np.multiply(-1 * y.reshape(-1, 1), X)
    b = -1 * np.ones(shape=y.shape)

    n, m = A.shape
    no_iterations = 0
    consecutive_counter = 0
    max_consecutive_counter = 100
    x_init = np.random.rand(m)

    x_current = x_init
    while True:
        if consecutive_counter >= max_consecutive_counter:
            break

        random_index = np.random.randint(low=0, high=n, size=1)
        random_vector = A[random_index, :]
        random_vector = random_vector.reshape(-1)
        alpha = np.dot(random_vector, x_current) - b[random_index]

        if alpha > 0:
            x_current = x_current - alpha * random_vector / (np.linalg.norm(random_vector) ** 2)

        no_iterations += 1

        condition = np.all(np.sign(np.dot(X, x_current)) == np.sign(y))
        if condition:
            consecutive_counter += 1
        else:
            consecutive_counter = 0

    if output_dir is not None:
        output_dir = os.path.join(output_dir, 'exercise_4')

        data = {'weights_star': x_current}
        save_dict_to_file(data, 'weights_star.pkl', output_dir)

    return x_current, no_iterations


def ex_3(A, b, epsilon=10 ** -3, output_dir=None):
    def gradient(x, A, b):
        gradient_value = np.dot(A.transpose(), np.maximum(np.dot(A, x) - b, 0))

        return gradient_value

    n, m = A.shape
    x_init = np.random.randn(m)
    iterations = 0

    alpha = 0.05
    x_current = x_init
    while True:
        gradient_current = gradient(x_current, A, b)
        gradient_current_norm = np.linalg.norm(gradient_current)

        if gradient_current_norm < epsilon:
            break

        x_current -= alpha * gradient_current
        iterations += 1

    if output_dir is not None:
        output_dir = os.path.join(output_dir, 'exercise_3')

        data = {'weights_star': x_current}
        save_dict_to_file(data, 'weights_star.pkl', output_dir)

    return x_current, iterations


def ex_2_v3(A, b, epsilon, output_dir=None):
    n, m = A.shape

    def objective_function(x, A, b):
        function = cp.sum(cp.pos(A @ x - b) ** 2)

        return function

    x = cp.Variable(shape=m)
    min_objective_function = cp.Minimize(objective_function(x, A, b))
    problem = cp.Problem(min_objective_function)
    solver_parameters = {'abstol': epsilon, 'reltol': epsilon}
    problem.solve(solver=cp.ECOS, **solver_parameters)

    assert problem.status == 'optimal'
    x_star = x.value

    if output_dir is not None:
        output_dir = os.path.join(output_dir, 'exercise_2')

        data = {'weights_star': x_star}
        save_dict_to_file(data, 'weights_star_v3.pkl', output_dir)

    return x_star


def ex_2_v2(A, b, epsilon, output_dir=None):
    n, m = A.shape

    def objective_function(x):
        return 0

    x = cp.Variable(shape=m)
    min_objective_function = cp.Minimize(objective_function(x))
    constraints = [A @ x <= b]
    problem = cp.Problem(min_objective_function, constraints)
    solver_parameters = {'abstol': epsilon, 'reltol': epsilon}
    problem.solve(solver=cp.ECOS, **solver_parameters)

    assert problem.status == 'optimal'
    x_star = x.value

    if output_dir is not None:
        output_dir = os.path.join(output_dir, 'exercise_2')

        data = {'weights_star': x_star}
        save_dict_to_file(data, 'weights_star_v2.pkl', output_dir)

    return x_star


def ex_2_v1(X, y, epsilon, output_dir=None):
    n, m = X.shape

    def objective_function(weights):
        return 0

    weights = cp.Variable(shape=m)
    min_objective_function = cp.Minimize(objective_function(weights))
    constraints = [cp.multiply(-y[index], X[index] @ weights) <= -1 for index in range(n)]
    problem = cp.Problem(min_objective_function, constraints)
    solver_parameters = {'abstol': epsilon, 'reltol': epsilon}
    problem.solve(solver=cp.ECOS, **solver_parameters)

    assert problem.status == 'optimal'
    weights_star = weights.value

    if output_dir is not None:
        output_dir = os.path.join(output_dir, 'exercise_2')

        data = {'weights_star': weights_star}
        save_dict_to_file(data, 'weights_star_v1.pkl', output_dir)

    return weights_star


def ex_1(n, m, output_dir=None):
    weights = np.random.randn(m)

    X_plus = np.random.randn(n, m)
    y_plus = np.ones(shape=n)

    X_minus = np.random.randn(n, m)
    y_minus = -1 * np.ones(shape=n)

    for index in range(n):
        # vector_plus si vector_minus sunt referinte la liniile din matricile X_plus, respectiv X_minus
        # daca modific o valoare din vector, atunci se modifica si linia din matricea corespunzatoare
        vector_plus = X_plus[index, :]
        vector_minus = X_minus[index, :]

        while True:
            if np.dot(weights, vector_plus) > 0 and np.dot(weights, vector_minus) < 0:
                break
            else:
                vector_plus += 1
                vector_minus -= 1

    assert np.all(np.dot(X_plus, weights) > 0)
    assert np.all(np.dot(X_minus, weights) < 0)

    if output_dir is not None:
        output_dir = os.path.join(output_dir, 'exercise_1')

        data = {'weights': weights,
                'X_plus': X_plus,
                'y_plus': y_plus,
                'X_minus': X_minus,
                'y_minus': y_minus}
        save_dict_to_file(data, 'simulated_data.pkl', output_dir)

    return weights, X_plus, y_plus, X_minus, y_minus


def save_dict_to_file(data, file_name, output_dir):
    if type(data) is not dict:
        raise ValueError('parameter data should be a dictionary!')

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    file = open(os.path.join(output_dir, file_name), "wb")
    pickle.dump(data, file)
    file.close()


if __name__ == '__main__':
    main()
