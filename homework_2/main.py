import os
import scipy
import pickle
import cvxpy as cp
import numpy as np
import matplotlib.pyplot as plt


def main():
    seed = 33
    np.random.seed(seed)

    output_dir = './outputs'

    # general configurations
    n = 10 ** 3
    mean = 0.0
    sigma = 5.0
    x_init = 0.0
    p = 0.5
    b = 1.0
    epsilon = 10 ** -6

    # punctul a)
    x, y, z, v = generate_inputs(x_init, n, mean, sigma, p, b, output_dir)

    # punctul b)
    D_matrix = generate_D_matrix(n)
    lambda_values = [1, 5, 10, 15, 20,100]
    x_star_cvxpy = {}
    for lambda_value in lambda_values:
        x_star_cvxpy[lambda_value] = generate_trend_using_cvxpy(y, D_matrix, lambda_value, epsilon)
    save_data_to_pickle_file(data={'lambda_to_trend_star_dict': x_star_cvxpy},
                             file_name='data.pickle',
                             output_dir=os.path.join(output_dir, 'l1_filtering_with_cvxpy'))
    plot_optimal_trend(lambda_to_trend_star_dict=x_star_cvxpy,
                       original_trend=x,
                       plot_params={'title': 'L-1 filtering with CVXPY vs original trend', 'x_label': 'Time',
                                    'y_label': 'Trend'},
                       file_name='summary.png',
                       output_dir=os.path.join(output_dir, 'l1_filtering_with_cvxpy'))


def plot_optimal_trend(lambda_to_trend_star_dict, original_trend, plot_params, file_name, output_dir):
    if type(lambda_to_trend_star_dict) is not dict:
        raise ValueError(
            'parameter lambda_to_trend_star_dict should be a dict where the keys are the are related to the smoothing parameter and the values are related to the optimal trends!')

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    if not file_name.endswith('png'):
        file_name += '.png'

    fig, ax = plt.subplots(figsize=(15, 8))
    for lambda_value, optimal_trend in lambda_to_trend_star_dict.items():
        ax.plot(optimal_trend, label='lambda={}'.format(lambda_value))
    ax.plot(original_trend, label='original trend', linewidth=3.0)
    ax.legend()
    if 'title' in plot_params.keys():
        ax.set_title(plot_params['title'])
    if 'x_label' in plot_params.keys():
        ax.set_xlabel(plot_params['x_label'])
    if 'y_label' in plot_params.keys():
        ax.set_ylabel(plot_params['y_label'])
    fig.savefig(os.path.join(output_dir, file_name))


def generate_trend_using_cvxpy(y, D_matrix, lambda_value, epsilon):
    def objective_function(x, y, D_matrix, lambda_value):
        function = 0.5 * cp.norm(x - y) ** 2 + lambda_value * cp.norm(D_matrix @ x, p=1)

        return function

    n = y.shape[0]
    x = cp.Variable(shape=n)
    min_of_obj_function = cp.Minimize(objective_function(x, y, D_matrix, lambda_value))
    problem = cp.Problem(min_of_obj_function)
    solver_parameters = {'abstol': epsilon, 'reltol': epsilon}
    problem.solve(solver=cp.ECOS, **solver_parameters)
    assert problem.status == 'optimal'

    return x.value


def generate_D_matrix(n):
    if not (type(n) is int and n >= 3):
        raise ValueError('parameter n should be an integer greater or equal to 3')

    coefs = np.array([1, -2, 1])

    D_matrix = np.zeros(shape=(n - 2, n))
    for row_index in range(D_matrix.shape[0]):
        D_matrix[row_index][row_index:(row_index + len(coefs))] = coefs

    return D_matrix


def generate_inputs(x_init, n, mean, sigma, p, b, output_dir=None):
    if not (type(sigma) is float and sigma > 0):
        raise ValueError('parameter sigma should be strictly positive real number!')
    if not (type(p) is float and 0 < p < 1):
        raise ValueError('parameter p should be a real number between 0.0 (exclusive) and 1.0 (exclusive)!')
    if not (type(b) is float and b > 0):
        raise ValueError(
            'parameter b should be a strictly positive real number because it is used to draw uniformly distributed random samples from interval [-b,b)!')

    x = np.zeros(shape=n)
    x[0] = x_init
    z = np.random.normal(mean, sigma, size=n)
    v = np.zeros(shape=n)

    v[0] = np.random.uniform(low=0.0, high=1.0)
    for time_step in range(1, v.shape[0]):
        random_prob = np.random.uniform(low=0.0, high=1.0)

        if 0 <= random_prob < p:
            v[time_step] = v[time_step - 1]
        else:
            # generez din intervalul [-b,b) ca seria lui x sa poata si descreste in timp
            # daca faceam np.random.uniform() generarea se facea doar din intervalul [0,1), iar seria lui x nu avea cum sa scada in timp (doar crestea)
            v[time_step] = np.random.uniform(low=-b, high=b)

        x[time_step] = x[time_step - 1] + v[time_step - 1]

    y = x + z

    if output_dir is not None:
        output_dir = os.path.join(output_dir, 'simulated_data')
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        fig, (top_ax, bottom_ax) = plt.subplots(nrows=2, ncols=1, figsize=(15, 7))
        top_ax.plot(range(1, x.shape[0] + 1), x, color='red')
        top_ax.set_title('Simulated data for vector $x$')
        bottom_ax.plot(range(1, y.shape[0] + 1), y, color='blue')
        bottom_ax.set_title('Simulated data for vector $y$')
        for ax in [top_ax, bottom_ax]:
            ax.set_xlabel('Time step')
            ax.set_ylabel('Value')
        fig.tight_layout()
        fig.savefig(os.path.join(output_dir, 'simulated_data.png'))

        data = {'x': x,
                'y': y,
                'z': z,
                'v': v}
        save_data_to_pickle_file(data=data, file_name='simulated_data.pickle', output_dir=output_dir)

    return x, y, z, v


def save_data_to_pickle_file(data, file_name, output_dir):
    if type(data) is not dict:
        raise ValueError(
            'parameter data should be a dictionary where the keys denote the name of the variables and the values denote the corresponding objects!')

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    if not (file_name.endswith('.pickle') or file_name.endswith('.pkl')):
        file_name += '.pickle'

    file = open(os.path.join(output_dir, file_name), 'wb')
    pickle.dump(data, file)
    file.close()


if __name__ == '__main__':
    main()
