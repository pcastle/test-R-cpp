// [[Rcpp::depends(RcppEigen)]]
#include <RcppEigen.h>

// El atributo [[Rcpp::export]] le dice a Rcpp que exporte esta función a R
// [[Rcpp::export]]
double mean_cpp(Rcpp::NumericVector x)
{
    if (x.size() == 0)
        Rcpp::stop("Error: vector vacío"); // equivalente a stop() en R

    double sum = 0.0;
    for (double xi : x)
        sum += xi;
    return sum / x.size();
}

// [[Rcpp::export]]
double variance_cpp(Rcpp::NumericVector x)
{
    if (x.size() < 2)
        Rcpp::stop("Error: se necesitan al menos 2 elementos");

    double m = mean_cpp(x);
    double sum = 0.0;
    for (double xi : x)
        sum += (xi - m) * (xi - m);
    return sum / (x.size() - 1);
}

// [[Rcpp::export]]
double std_desv_cpp(Rcpp::NumericVector x)
{
    return std::sqrt(variance_cpp(x));
}

// [[Rcpp::export]]
double correlation_cpp(Rcpp::NumericVector x, Rcpp::NumericVector y)
{
    if (x.size() != y.size())
    {
        Rcpp::stop("Error: The vector x and y must have the same length.");
    }
    double mean_x = mean(x);
    double mean_y = mean(y);
    std::vector<double> aux(x.size());

    std::transform(x.begin(), x.end(), y.begin(), aux.begin(), [mean_x, mean_y](const double &x, const double &y)
                   { return (x - mean_x) * (y - mean_y); });

    double sums = std::accumulate(aux.begin(), aux.end(), 0.0, [](double current_sum, const double &aux)
                                  { return current_sum + aux; });

    double result = sums / (aux.size() - 1);
    return result / (std_desv_cpp(x) * std_desv_cpp(y));
}