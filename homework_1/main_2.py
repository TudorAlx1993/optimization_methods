import os
import pickle
import numpy as np
from scipy import optimize
import matplotlib.pyplot as plt


def main():
    # seed = 33
    # np.random.seed(seed)
    output_dir = './outputs'

    m = 75
    n = 50
    min_k = 10 ** 5
    ex_1(m, n, min_k, output_dir)


def ex_1(m, n, min_k, output_dir, epsilon=10 ** -5):
    A, b, L = generate_inputs_exercise_1(m, n, min_k)

    x_init = np.random.randn(n)

    alpha = 2.0 / L * 0.9
    x_star_constant_step, no_iters_constant_step, grad_norm_hist_constant_step, loss_hist_constant_step = minimize_function_with_gradient_ex_1(
        x_init.copy(), A, b, epsilon, 'constant_step', alpha)
    x_star_ideal_step, no_iters_ideal_step, grad_norm_hist_ideal_step, loss_hist_ideal_step = minimize_function_with_gradient_ex_1(
        x_init.copy(), A, b, epsilon, 'ideal_step', alpha)
    x_star_adaptive_step, no_iters_adaptive_step, grad_norm_hist_adaptive_step, loss_hist_adaptive_step = minimize_function_with_gradient_ex_1(
        x_init.copy(), A, b, epsilon, 'adaptive_step')
    x_star_numpy = np.linalg.lstsq(A, b, rcond=None)[0]

    my_results = {'constant_step': {'x_star': x_star_constant_step, 'no_iters': no_iters_constant_step,
                                    'gradient_norm_history': grad_norm_hist_constant_step,
                                    'loss_history': loss_hist_constant_step},
                  'ideal_step': {'x_star': x_star_ideal_step, 'no_iters': no_iters_ideal_step,
                                 'gradient_norm_history': grad_norm_hist_ideal_step,
                                 'loss_history': loss_hist_ideal_step},
                  'adaptive_step': {'x_star': x_star_adaptive_step, 'no_iters': no_iters_adaptive_step,
                                    'gradient_norm_history': grad_norm_hist_adaptive_step,
                                    'loss_history': loss_hist_adaptive_step}
                  }

    for method in my_results.keys():
        assert np.allclose(my_results[method]['gradient_norm_history'][-1], 0.0, atol=epsilon) is True
        assert np.allclose(my_results[method]['loss_history'][-1], objective_function_ex1(x_star_numpy, A, b),
                           atol=epsilon) is True
        assert np.allclose(my_results[method]['x_star'], x_star_numpy, atol=epsilon) is True

    save_results_to_txt_file_ex_1(A, b, my_results, x_star_numpy, output_dir)


def save_results_to_txt_file_ex_1(A, b, my_results, x_star_numpy, output_dir):
    output_dir = os.path.join(output_dir, 'exercise_1')
    if not os.path.isdir(output_dir):
        os.makedirs(output_dir)

    generated_data = {'A': A, 'b': b}
    file = open(os.path.join(output_dir, 'generated_data.pickle'), 'wb')
    pickle.dump(generated_data, file, protocol=pickle.HIGHEST_PROTOCOL)
    file.close()

    file_content = 'My implementations:\n'
    for method in my_results.keys():
        file_content += '\t* method={}\n'.format(method)
        file_content += '\t\t* convergence after no_iterations={}\n'.format(my_results[method]['no_iters'])
        file_content += '\t\t* x_star={}\n'.format(my_results[method]['x_star'].flatten())
        file_content += '\t\t* gradient norm of f(x_star)={}\n'.format(my_results[method]['gradient_norm_history'][-1])
        file_content += '\t\t* objective function in x_star={}\n'.format(
            my_results[method]['loss_history'][-1])
    file_content += '\t* numpy implementation\n'
    file_content += '\t\t* x_star={}\n'.format(x_star_numpy.flatten())
    file_content += '\t\t* gradient norm of f(x_star)={}\n'.format(
        np.linalg.norm(gradient_function_ex_1(x_star_numpy, A, b)))
    file_content += '\t\t* objective function in x_star={}\n'.format(objective_function_ex1(x_star_numpy, A, b))

    file = open(os.path.join(output_dir, 'results.txt'), 'w')
    file.write(file_content)
    file.close()

    fig, (top_ax, bottom_ax) = plt.subplots(nrows=2, ncols=1, figsize=(15, 15))
    for method in my_results.keys():
        x_star = my_results[method]['x_star']
        loss_x_star = objective_function_ex1(x_star, A, b)
        loss_history = my_results[method]['loss_history']
        centered_loss_history = [value - loss_x_star for value in loss_history]
        gradient_norm_history = my_results[method]['gradient_norm_history']

        top_ax.semilogy(range(len(loss_history)), centered_loss_history, label=method, linewidth=5.0)
        top_ax.set_xlabel('Iteration')
        top_ax.set_ylabel('Value')
        top_ax.set_title('$f(x)-f(x*)$')
        top_ax.legend()

        bottom_ax.semilogy(range(len(gradient_norm_history)), gradient_norm_history, label=method, linewidth=5.0)
        bottom_ax.set_xlabel('Iteration')
        bottom_ax.set_ylabel('Value')
        bottom_ax.set_title('$||gradient(f(x))||$')
        bottom_ax.legend()
    fig.savefig(os.path.join(output_dir, 'plots.png'))


def minimize_function_with_gradient_ex_1(x_init, A, b, epsilon, method, alpha=None):
    iteration = 0
    gradient_norm_history = []
    loss_history = []
    x_current = x_init
    while True:
        gradient_current = gradient_function_ex_1(x_current, A, b)

        loss_current = objective_function_ex1(x_current, A, b)
        loss_history.append(loss_current)

        gradient_current_norm = np.linalg.norm(gradient_current)
        gradient_norm_history.append(gradient_current_norm)
        if gradient_current_norm < epsilon:
            break

        if method == 'constant_step' and type(alpha) is np.float64 and alpha.size == 1:
            x_current -= alpha * gradient_current
        elif method == 'ideal_step':
            x_current -= alpha_ideal_ex_1(x_current, gradient_current, A, b, alpha) * gradient_current
        elif method == 'adaptive_step':
            x_current -= alpha_adaptive_ex_1(x_current, gradient_current, A, b) * gradient_current
        else:
            raise ValueError(
                'algorithm not implemented for method={}! Method should be one of the following values: constant_step, ideal_step or adaptive_step. If method=constant_step then paramter alpha should be a float.'.format(
                    method))

        iteration += 1

    return x_current, iteration, gradient_norm_history, loss_history


def alpha_ideal_ex_1(x, gradient, A, b, alpha_init):
    def objective_function(alpha, x, gradient, A, b):
        return objective_function_ex1(x - alpha * gradient, A, b)

    def constraint(alpha):
        return alpha

    optimization_result = optimize.minimize(fun=objective_function, x0=alpha_init, args=(x, gradient, A, b),
                                            constraints={'type': 'ineq', 'fun': constraint})
    alpha = optimization_result.x

    return alpha


def alpha_adaptive_ex_1(x, gradient, A, b):
    c, rho, alpha = np.random.rand(3)

    while (objective_function_ex1(x - alpha * gradient, A, b)) > (
            objective_function_ex1(x, A, b) - c * alpha * np.linalg.norm(gradient) ** 2):
        alpha = rho * alpha

    return alpha


def gradient_function_ex_1(x, A, b):
    gradient = np.matmul(A.transpose(), np.matmul(A, x) - b)

    return gradient


def objective_function_ex1(x, A, b):
    loss = 0.5 * np.linalg.norm(np.matmul(A, x) - b) ** 2

    return loss


def generate_inputs_exercise_1(m, n, min_k):
    A = np.random.randn(m, n)
    b = np.random.randn(m)

    hessian_matrix = np.matmul(A.transpose(), A)

    eigenvalues = np.linalg.eigvals(hessian_matrix)
    assert np.all(eigenvalues > 0)

    min_eigenvalue = np.min(eigenvalues)
    max_eigenvalue = np.max(eigenvalues)
    k = max_eigenvalue / min_eigenvalue

    k_numpy = np.linalg.cond(hessian_matrix)
    assert np.allclose(k, k_numpy)

    return A, b, max_eigenvalue


if __name__ == '__main__':
    main()
