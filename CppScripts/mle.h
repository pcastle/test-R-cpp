#pragma once
#include <functional>
#include "statistics.h"  // para tener Eigen y el resto disponible

// Log-likelihood for Normal Distribution
double neg_log_likelihood_normal(const Eigen::VectorXd& params,
                                  const std::vector<double>& data);

// numerical gradient calculated by central differences
Eigen::VectorXd gradient(std::function<double(const Eigen::VectorXd&)> f,
                          const Eigen::VectorXd& theta,
                          double h = 1e-5);

// numerical Hessian
Eigen::MatrixXd hessian(std::function<double(const Eigen::VectorXd&)> f,
                         const Eigen::VectorXd& theta,
                         double h = 1e-5);

// Newton-Raphson optimization algorithm
Eigen::VectorXd newton_raphson(std::function<double(const Eigen::VectorXd&)> f,
                                Eigen::VectorXd theta_init,
                                int max_iter = 1000,
                                double tol = 1e-8);

                                // Parámetros del algoritmo genético
struct GAParams {
    int population_size = 100;
    int max_generations = 500;
    double mutation_rate = 0.1;      // probabilidad de mutar cada gen
    double mutation_scale = 0.5;     // escala de la mutación N(0, scale)
    int tournament_size = 3;         // individuos por torneo
    double tol = 1e-6;               // criterio de convergencia
};

Eigen::VectorXd genetic_algorithm(
    std::function<double(const Eigen::VectorXd&)> f,
    const Eigen::VectorXd& lower_bounds,   // límite inferior de búsqueda
    const Eigen::VectorXd& upper_bounds,   // límite superior de búsqueda
    const GAParams& params = GAParams());