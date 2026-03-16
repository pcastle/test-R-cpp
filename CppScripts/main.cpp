#include "statistics.h"
#include <iomanip>
#include "mle.h"

int main()
{
    std::vector<double> x = {1, 2, 3, 4};
    std::vector<double> y = {4, 3, 2, 1};

    try
    {
        std::vector<double> vacio = {};
        mean(vacio);
    }
    catch (const std::invalid_argument &e)
    {
        std::cout << e.what() << "\n";
    }
    std::cout << mean(x) << std::endl;
    std::cout << variance(x) << std::endl;
    std::cout << std_desv(x) << std::endl;
    std::cout << correlation(x, y) << std::endl;

    int n = 10000;
    auto datos = rnorm_generator(n, 5.0, 2.0); // N(5, 2)

    // Convertir a vector para usar tus funciones
    std::vector<double> v(datos.get(), datos.get() + n);

    std::cout << "Media:    " << mean(v) << "\n";     // ≈ 5.0
    std::cout << "Std Dev:  " << std_desv(v) << "\n"; // ≈ 2.0

    // Parámetros poblacionales conocidos
    double mu = 5.0, sigma = 2.0;

    for (int n : {1000, 10000, 100000, 1000000})
    {
        auto datos = rnorm_generator(n, mu, sigma);
        std::vector<double> v(datos.get(), datos.get() + n);

        std::cout << "n = " << std::setw(7) << n
                  << "  media = " << std::fixed << std::setprecision(4) << mean(v)
                  << "  sd = " << std_desv(v) << "\n";
    }

    // Vector columna (como en R: x <- c(1,2,3))
    Eigen::VectorXd x1(3);
    x1 << 1.0, 2.0, 3.0;

    // Matriz (como en R: A <- matrix(c(...), nrow=2, ncol=2))
    Eigen::MatrixXd A(2, 2);
    A << 1.0, 2.0,
        3.0, 4.0;

    std::cout << "Vector x:\n"
              << x1 << "\n\n";
    std::cout << "Matriz A:\n"
              << A << "\n\n";

    // Transpuesta
    std::cout << "A transpuesta:\n"
              << A.transpose() << "\n\n";

    // Inversa
    std::cout << "A inversa:\n"
              << A.inverse() << "\n\n";

    // Multiplicación matricial (como A %*% B en R)
    Eigen::MatrixXd B = A * A;
    std::cout << "A * A:\n"
              << B << "\n";

    Eigen::MatrixXd X(5, 2);
    X << 1, 1.0,
        1, 2.0,
        1, 3.0,
        1, 4.0,
        1, 5.0;

    Eigen::VectorXd y1(5);
    y1 << 3.1, 4.9, 7.2, 8.8, 11.1; // ≈ 1 + 2x

    Eigen::VectorXd beta = ols(X, y1);
    std::cout << "beta_0 (intercepto): " << beta(0) << "\n"; // ≈ 1
    std::cout << "beta_1 (pendiente):  " << beta(1) << "\n"; // ≈ 2

    // Generar datos N(3.0, 1.5)
    auto raw = rnorm_generator(500, 3.0, 1.5);
    std::vector<double> data(raw.get(), raw.get() + 500);

    // Función objetivo: bind los datos al neg_log_likelihood
    auto f = [&data](const Eigen::VectorXd &p)
    {
        return neg_log_likelihood_normal(p, data);
    };

    // Punto inicial alejado de la verdad
    Eigen::VectorXd theta_init(2);
    theta_init << mean(data), std_desv(data);

    Eigen::VectorXd theta_hat = newton_raphson(f, theta_init);

    std::cout << "n = " << std::setw(6) << n
              << "  mu = " << std::setprecision(4) << theta_hat(0)
              << "  sigma = " << theta_hat(1) << "\n";

    return 0;
}