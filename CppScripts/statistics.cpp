#include "statistics.h"

double mean(const std::vector<double> &x)
{
    if (x.empty())
    {
        throw std::invalid_argument("Error: The vector is empty");
    }
    double sums = 0.0;
    for (const double &xi : x)
    {
        sums += xi;
    }
    return sums / x.size();
}

double variance(const std::vector<double> &x)
{
    if (x.size() < 2)
    {
        throw std::invalid_argument("Error: The vector must have at least 2 elements");
    }
    double mean_val = mean(x);

    double sums = std::accumulate(x.begin(), x.end(), 0.0, [mean_val](double current_sum, const double &x)
                                  { return current_sum + std::pow(x - mean_val, 2); });

    double result = sums / (x.size() - 1);
    return result;
}
double variance(const std::vector<double> &x, double mean_val)
{
    if (x.size() < 2)
    {
        throw std::invalid_argument("Error: The vector must have at least 2 elements");
    }
    double sums = std::accumulate(x.begin(), x.end(), 0.0, [mean_val](double current_sum, const double &x)
                                  { return current_sum + std::pow(x - mean_val, 2); });

    double result = sums / (x.size() - 1);
    return result;
}

double std_desv(const std::vector<double> &x)
{
    double result = std::sqrt(variance(x));

    return result;
}

double std_desv(const std::vector<double> &x, double mean_val)
{
    double result = std::sqrt(variance(x, mean_val));

    return result;
}

double covariance(const std::vector<double> &x, const std::vector<double> &y)
{
    if (x.size() != y.size())
    {
        throw std::invalid_argument("Error: The vector x and y must have the same length.");
    }
    double mean_x = mean(x);
    double mean_y = mean(y);
    std::vector<double> aux(x.size());

    std::transform(x.begin(), x.end(), y.begin(), aux.begin(), [mean_x, mean_y](const double &x, const double &y)
                   { return (x - mean_x) * (y - mean_y); });

    double sums = std::accumulate(aux.begin(), aux.end(), 0.0, [](double current_sum, const double &aux)
                                  { return current_sum + aux; });

    double result = sums / (aux.size() - 1);
    return result;
}

double covariance(const std::vector<double> &x, double mean_x, const std::vector<double> &y, double mean_y)
{
    if (x.size() != y.size())
    {
        throw std::invalid_argument("Error: The vector x and y must have the same length.");
    }

    std::vector<double> aux(x.size());

    std::transform(x.begin(), x.end(), y.begin(), aux.begin(), [mean_x, mean_y](const double &x, const double &y)
                   { return (x - mean_x) * (y - mean_y); });

    double sums = std::accumulate(aux.begin(), aux.end(), 0.0, [](double current_sum, const double &aux)
                                  { return current_sum + aux; });

    double result = sums / (aux.size() - 1);
    return result;
}

double correlation(const std::vector<double> &x, const std::vector<double> &y)
{
    double mean_x = mean(x);
    double mean_y = mean(y);
    double result = covariance(x, mean_x, y, mean_y) / (std_desv(x, mean_x) * std_desv(y, mean_y));
    return result;
}

std::unique_ptr<double[]> rnorm_generator(int n, double mu, double sigma)
{
    if (n <= 0 || sigma <= 0)
    {
        throw std::invalid_argument("Error: 'n' and 'sigma' must be positive!");
    }
    std::random_device rd;
    std::mt19937 gen(rd());
    std::normal_distribution<double> dist(mu, sigma);

    auto data = std::make_unique<double[]>(n);

    for (int i = 0; i < n; i++)
    {
        data[i] = dist(gen);
    }

    return data;
}

Eigen::VectorXd ols(const Eigen::MatrixXd &X, const Eigen::VectorXd &y)
{
    return ((X.transpose() * X).inverse() * X.transpose()) * (y);
}
