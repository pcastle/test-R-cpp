#pragma once
#include <iostream>
#include <vector>
#include <cmath>
#include <algorithm>
#include <numeric>

double mean(const std::vector<double> &x);
double variance(const std::vector<double> &x);
double variance(const std::vector<double> &x, double mean_val);
double std_desv(const std::vector<double> &x);
double std_desv(const std::vector<double> &x, double mean_val);
double covariance(const std::vector<double> &x, const std::vector<double> &y);
double covariance(const std::vector<double> &x, double mean_x, const std::vector<double> &y, double mean_y);
double correlation(const std::vector<double> &x, const std::vector<double> &y);