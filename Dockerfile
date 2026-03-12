FROM rocker/r-base:4.5.2


RUN apt-get update && apt-get install -y \
    g++ gcc cmake make \
    libcurl4-openssl-dev \
    libssl-dev libeigen3-dev \
    libxml2-dev \
    git \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/* \
    && Rscript -e "install.packages(c('renv', 'languageserver'), repos='https://cran.r-project.org')"

WORKDIR /workspace
CMD ["/bin/bash"]