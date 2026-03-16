#pragma once
#include <iostream>
#include <vector>
#include <cmath>
#include <algorithm>
#include <numeric>
#include <memory>
#include <random>
#include <eigen3/Eigen/Dense>

double mean(const std::vector<double> &x);
double variance(const std::vector<double> &x);
double variance(const std::vector<double> &x, double mean_val);
double std_desv(const std::vector<double> &x);
double std_desv(const std::vector<double> &x, double mean_val);
double covariance(const std::vector<double> &x, const std::vector<double> &y);
double covariance(const std::vector<double> &x, double mean_x, const std::vector<double> &y, double mean_y);
double correlation(const std::vector<double> &x, const std::vector<double> &y);
// Generate n randoms numbers from a uniform distribution between a and b
std::unique_ptr<double[]> runiform_generator(int n, double a, double b);
// Generate n randoms numbers from a normal distribution with mean mu and standard desviation sigma
std::unique_ptr<double[]> rnorm_generator(int n, double mu, double sigma);
Eigen::VectorXd ols(const Eigen::MatrixXd &X, const Eigen::VectorXd &y);