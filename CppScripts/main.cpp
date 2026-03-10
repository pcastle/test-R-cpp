#include "statistics.h"

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

    return 0;
}