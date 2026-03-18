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
    std::mt19937 gen(std::random_device{}());
    std::uniform_int_distribution<int> pop_idx(0, params.population_size - 1);
    std::uniform_real_distribution<double> dist_uniform(0.0, 1.0);
    std::normal_distribution<double> dist_normal(0.0, params.mutation_scale);

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

    Eigen::Index init_best_idx;
    fitness.minCoeff(&init_best_idx);
    Eigen::VectorXd prev_best_variables = population.row(init_best_idx).transpose();

    auto tournament = [&]()
    {
        // first index is the 'champion'
        int idx_champion = pop_idx(gen);
        for (int t = 1; t < params.tournament_size; t++)
        {
            int idx_challenger = pop_idx(gen);
            // fight - we are minimizing
            if (fitness[idx_champion] > fitness[idx_challenger])
            {
                idx_champion = idx_challenger;
            }
        }
        // Winner of the tournament
        return idx_champion;
    };

    for (int i = 0; i < params.max_generations; i++)
    {
        Eigen::MatrixXd new_population(params.population_size, lower_bounds.size());

        // new population
        for (int j = 0; j < params.population_size; j++)
        {
            int parent_1 = tournament();
            int parent_2 = tournament();

            double w_crossover = dist_uniform(gen);
            new_population.row(j) = w_crossover * population.row(parent_1) + (1 - w_crossover) * population.row(parent_2);

            // Mutation
            for (int k = 0; k < lower_bounds.size(); k++)
            {
                if (dist_uniform(gen) < params.mutation_rate)
                {
                    new_population(j, k) += dist_normal(gen);

                    // This mutation fullfil the bounds
                    new_population(j, k) = std::max(lower_bounds(k), std::min(new_population(j, k), upper_bounds(k)));
                }
            }
        }

        // Eigen::Index prev_best_idx;
        // double prev_best_fitness = fitness.minCoeff(&prev_best_idx);
        // Eigen::VectorXd prev_best_variables(lower_bounds.size());
        // prev_best_variables = population.row(prev_best_idx).transpose();
        for (int j = 0; j < params.population_size; j++)
        {
            double new_fitness = f(new_population.row(j).transpose());
            if (fitness[j] > new_fitness)
            {
                population.row(j) = new_population.row(j);
                fitness[j] = new_fitness;
            }
        }
        Eigen::Index new_best_idx;
        double new_best_fitness = fitness.minCoeff(&new_best_idx);

        // Convergence
        double delta = (prev_best_variables - population.row(new_best_idx).transpose()).norm();
        prev_best_variables = population.row(new_best_idx).transpose();
        if (i > 0 && delta < params.tol)
        {
            std::cout << "Convergió en generación " << i << "\n";
            return population.row(new_best_idx).transpose();
        }
    }

    throw std::runtime_error("No convergetion");
}
