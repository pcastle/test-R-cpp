# install.packages(c("Rcpp","RcppEigen"))
library(Rcpp)
sourceCpp("CppScripts/statistics.cpp")

sourceCpp("CppScripts/rcpp_statistics.cpp")

x <- c(2.5, 3.1, 4.8, 2.0, 5.3)
mean_cpp(x)      # llama tu C++
mean(x)          # R nativo — mismo resultado


x <- rnorm(1000, mean=5, sd=2)
y <- rnorm(1000, mean=3, sd=1)

# Comparar
cat("mean:   R =", mean(x),        "C++ =", mean_cpp(x), "\n")
cat("var:    R =", var(x),         "C++ =", variance_cpp(x), "\n")
cat("sd:     R =", sd(x),          "C++ =", std_desv_cpp(x), "\n")
cat("cor:    R =", cor(x, y),      "C++ =", correlation_cpp(x, y), "\n")
