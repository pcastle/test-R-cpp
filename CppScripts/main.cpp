#include "statistics.h"
#include <iomanip>
#include <sstream>
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

    // ─── Datos ────────────────────────────────────────────────────────────
    const double TRUE_MU = 3.0;
    const double TRUE_SIGMA = 1.5;
    const int N = 500;

    auto raw = rnorm_generator(N, TRUE_MU, TRUE_SIGMA);
    std::vector<double> data(raw.get(), raw.get() + N);

    auto f = [&data](const Eigen::VectorXd &p)
    {
        return neg_log_likelihood_normal(p, data);
    };

    // ─── Bounds para GA ───────────────────────────────────────────────────
    Eigen::VectorXd lower(2), upper(2);
    lower << -10.0, 0.01;
    upper << 10.0, 10.0;

    GAParams ga_params;
    ga_params.population_size = 200;
    ga_params.max_generations = 1000;
    ga_params.tol = 1e-8;

    LMParams lm_params;

    // ─── Helper para imprimir una fila ────────────────────────────────────
    auto print_row = [&](const std::string &method,
                         bool ok,
                         double mu, double sigma)
    {
        std::cout << std::left << std::setw(20) << method
                  << std::right << std::fixed << std::setprecision(4);
        if (ok)
        {
            std::cout << std::setw(10) << mu
                      << std::setw(10) << sigma
                      << std::setw(12) << std::abs(mu - TRUE_MU)
                      << std::setw(12) << std::abs(sigma - TRUE_SIGMA)
                      << std::setw(8) << "OK";
        }
        else
        {
            std::cout << std::setw(10) << "—"
                      << std::setw(10) << "—"
                      << std::setw(12) << "—"
                      << std::setw(12) << "—"
                      << std::setw(8) << "FALLÓ";
        }
        std::cout << "\n";
    };

    auto print_header = [&](const std::string &title)
    {
        std::cout << "\n=== " << title << " ===\n";
        std::cout << std::left << std::setw(20) << "Método"
                  << std::right << std::setw(10) << "mu"
                  << std::setw(10) << "sigma"
                  << std::setw(12) << "err_mu"
                  << std::setw(12) << "err_sigma"
                  << std::setw(8) << "estado"
                  << "\n"
                  << std::string(72, '-') << "\n";
    };

    // ─── Comparación con distintos puntos iniciales ───────────────────────
    std::vector<std::pair<std::string, Eigen::VectorXd>> theta_inits;

    Eigen::VectorXd t1(2);
    t1 << 0.0, 1.0;
    theta_inits.push_back({"Punto inicial bueno (0, 1)", t1});
    Eigen::VectorXd t2(2);
    t2 << 10.0, 5.0;
    theta_inits.push_back({"Punto inicial malo (10, 5)", t2});
    Eigen::VectorXd t3(2);
    t3 << -5.0, 0.1;
    theta_inits.push_back({"Punto inicial malo (-5, 0.1)", t3});

    for (auto &[title, theta0] : theta_inits)
    {
        print_header(title);

        // Newton-Raphson
        try
        {
            auto nr = newton_raphson(f, theta0);
            print_row("Newton-Raphson", true, nr(0), nr(1));
        }
        catch (...)
        {
            print_row("Newton-Raphson", false, 0, 0);
        }

        // Levenberg-Marquardt
        try
        {
            auto lm = levenberg_marquardt(f, theta0, lm_params);
            print_row("Levenberg-M", true, lm(0), lm(1));
        }
        catch (...)
        {
            print_row("Levenberg-M", false, 0, 0);
        }

        // Genético (no usa punto inicial)
        try
        {
            auto ga = genetic_algorithm(f, lower, upper, ga_params);
            print_row("Genético", true, ga(0), ga(1));
        }
        catch (...)
        {
            print_row("Genético", false, 0, 0);
        }
    }

    // ─── Estabilidad: 10 corridas con punto inicial malo ─────────────────
    std::cout << "\n=== Estabilidad (10 corridas, punto inicial (10, 5)) ===\n";
    std::cout << std::left << std::setw(5) << "Run"
              << std::right << std::setw(12) << "NR_mu"
              << std::setw(10) << "LM_mu"
              << std::setw(10) << "GA_mu"
              << std::setw(12) << "NR_sigma"
              << std::setw(10) << "LM_sigma"
              << std::setw(10) << "GA_sigma"
              << "\n"
              << std::string(72, '-') << "\n";

    Eigen::VectorXd theta_bad(2);
    theta_bad << 10.0, 5.0;

    for (int run = 0; run < 10; run++)
    {
        auto raw_i = rnorm_generator(N, TRUE_MU, TRUE_SIGMA);
        std::vector<double> data_i(raw_i.get(), raw_i.get() + N);
        auto fi = [&data_i](const Eigen::VectorXd &p)
        {
            return neg_log_likelihood_normal(p, data_i);
        };

        auto fmt = [](bool ok, double v) -> std::string
        {
            if (!ok)
                return "  FALLÓ";
            std::ostringstream ss;
            ss << std::fixed << std::setprecision(4) << v;
            return ss.str();
        };

        bool nr_ok = false, lm_ok = false, ga_ok = false;
        double nr_mu = 0, nr_sigma = 0;
        double lm_mu = 0, lm_sigma = 0;
        double ga_mu = 0, ga_sigma = 0;

        try
        {
            auto r = newton_raphson(fi, theta_bad);
            nr_ok = true;
            nr_mu = r(0);
            nr_sigma = r(1);
        }
        catch (...)
        {
        }
        try
        {
            auto r = levenberg_marquardt(fi, theta_bad, lm_params);
            lm_ok = true;
            lm_mu = r(0);
            lm_sigma = r(1);
        }
        catch (...)
        {
        }
        try
        {
            auto r = genetic_algorithm(fi, lower, upper, ga_params);
            ga_ok = true;
            ga_mu = r(0);
            ga_sigma = r(1);
        }
        catch (...)
        {
        }

        std::cout << std::left << std::setw(5) << run + 1
                  << std::right << std::setw(12) << fmt(nr_ok, nr_mu)
                  << std::setw(10) << fmt(lm_ok, lm_mu)
                  << std::setw(10) << fmt(ga_ok, ga_mu)
                  << std::setw(12) << fmt(nr_ok, nr_sigma)
                  << std::setw(10) << fmt(lm_ok, lm_sigma)
                  << std::setw(10) << fmt(ga_ok, ga_sigma)
                  << "\n";
    }

    for (int n : {100, 500, 1000, 10000})
    {
        auto raw_i = rnorm_generator(n, 3.0, 1.5);
        std::vector<double> data_i(raw_i.get(), raw_i.get() + n);
        auto fi = [&data_i](const Eigen::VectorXd &p)
        {
            return neg_log_likelihood_normal(p, data_i);
        };

        try
        {
            auto lm = levenberg_marquardt(fi, theta_bad, lm_params);
            std::cout << "n=" << std::setw(6) << n
                      << "  mu=" << lm(0)
                      << "  sigma=" << lm(1) << "\n";
        }
        catch (const std::exception &e)
        {
            std::cout << "n=" << std::setw(6) << n
                      << "  FALLÓ: " << e.what() << "\n";
        }
    }

    return 0;
}