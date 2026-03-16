#include "statistics.h"
#include "mle.h"

double neg_log_likelihood_normal(const Eigen::VectorXd &params,
                                 const std::vector<double> &data)
{

    double mu = params(0);
    double sigma = params(1);

    if (sigma <= 0)
        return 1e10; // penalizar valores inválidos

    double n = data.size();
    double nll = 0.0;

    // -log L = n*log(sigma) + n/2*log(2π) + sum((yi-mu)²) / (2σ²)
    for (const double &yi : data)
    {
        nll += std::pow(yi - mu, 2);
    }
    nll = nll / (2 * sigma * sigma) + n * std::log(sigma);

    return nll;
}

Eigen::VectorXd gradient(std::function<double(const Eigen::VectorXd &)> f,
                         const Eigen::VectorXd &theta,
                         double h)
{
    int k = theta.size();
    Eigen::VectorXd grad(k);

    for (int i = 0; i < k; i++)
    {
        Eigen::VectorXd theta_plus = theta;
        Eigen::VectorXd theta_minus = theta;
        theta_plus(i) += h;
        theta_minus(i) -= h;

        // Diferencia central: más precisa que diferencia hacia adelante
        grad(i) = (f(theta_plus) - f(theta_minus)) / (2 * h);
    }
    return grad;
}

Eigen::MatrixXd hessian(std::function<double(const Eigen::VectorXd &)> f,
                        const Eigen::VectorXd &theta,
                        double h)
{
    int k = theta.size();
    Eigen::MatrixXd H(k, k);

    for (int i = 0; i < k; i++)
    {
        for (int j = 0; j < k; j++)
        {
            Eigen::VectorXd t_pp = theta, t_pm = theta,
                            t_mp = theta, t_mm = theta;
            t_pp(i) += h;
            t_pp(j) += h;
            t_pm(i) += h;
            t_pm(j) -= h;
            t_mp(i) -= h;
            t_mp(j) += h;
            t_mm(i) -= h;
            t_mm(j) -= h;

            H(i, j) = (f(t_pp) - f(t_pm) - f(t_mp) + f(t_mm)) / (4 * h * h);
        }
    }
    return H;
}

// Retorna el vector de parámetros óptimos
Eigen::VectorXd newton_raphson(
    std::function<double(const Eigen::VectorXd &)> f,
    Eigen::VectorXd theta_init,
    int max_iter,
    double tol)
{

    for (int i = 0; i < max_iter; i++)
    {
        Eigen::VectorXd numerical_gradient = gradient(f, theta_init);
        Eigen::MatrixXd numerical_hessian = hessian(f, theta_init);

        Eigen::VectorXd new_theta = theta_init - numerical_hessian.colPivHouseholderQr().solve(numerical_gradient);

        if ((new_theta - theta_init).norm() < tol)
        {
            return new_theta;
        }
        theta_init = new_theta;
    }

    throw std::runtime_error("No convergetion");
}

Eigen::VectorXd genetic_algorithm(
    std::function<double(const Eigen::VectorXd &)> f,
    const Eigen::VectorXd &lower_bounds, // límite inferior de búsqueda
    const Eigen::VectorXd &upper_bounds, // límite superior de búsqueda
    const GAParams &params)
{
    Eigen::MatrixXd population(params.population_size, lower_bounds.size());
    for (int i = 0; i < lower_bounds.size(); i++)
    {
        auto rand_number = runiform_generator(params.population_size, lower_bounds[i], upper_bounds[i]);
        population.col(i) = Eigen::Map<Eigen::VectorXd>(rand_number.get(), params.population_size);
    }

    Eigen::VectorXd fitness(params.population_size);

    for (int i = 0; i < params.population_size; i++)
    {
        fitness[i] = f(population.row(i).transpose());
    }

    Eigen::VectorXd grad(2);
    return grad;
}
