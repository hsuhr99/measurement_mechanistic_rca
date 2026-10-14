# -------------------------
# Base image
# -------------------------
FROM python:3.11-bullseye
# -------------------------
# Install system libraries and R
# -------------------------
ENV R_VERSION=4.4.0 \
    DEBIAN_FRONTEND=noninteractive



RUN apt-get update -qq && apt-get -y install --no-install-recommends \
    ca-certificates \
    build-essential \
    gfortran \
    libreadline-dev \
    xorg-dev \
    libbz2-dev \
    liblzma-dev \
    curl \
    libxml2-dev \
    libcairo2-dev \
    libsqlite3-dev \
    libmariadbd-dev \
    libpq-dev \
    libssh2-1-dev \
    unixodbc-dev \
    libcurl4-openssl-dev \
    libssl-dev \
    libsodium-dev \
    wget \
    graphviz \
    graphviz-dev \
    libxml2-dev \
    libssl-dev \
    libpcre2-dev \
    libbz2-dev \
    liblzma-dev \
    libzstd-dev \
    libtirpc-dev \
    zlib1g-dev \
    liblz4-dev \
    libicu-dev \
    git \
    wget \
    libblas-dev \
    liblapack-dev \
    && rm -rf /var/lib/apt/lists/*

# Download and install R
RUN wget -c https://cran.r-project.org/src/base/R-4/R-${R_VERSION}.tar.gz \
    && tar -xf R-${R_VERSION}.tar.gz \
    && cd R-${R_VERSION} \
    && ./configure --enable-R-shlib\
    && make -j$(nproc) \
    && make install \
    && cd .. \
    && rm -rf R-${R_VERSION} R-${R_VERSION}.tar.gz

# -------------------------
# Install R package qgam
# -------------------------
# RUN Rscript -e "install.packages('mgcv', repos='https://cloud.r-project.org', dependencies=TRUE)"
RUN Rscript -e "install.packages('qgam', repos='https://cloud.r-project.org', dependencies=TRUE)"
RUN Rscript -e "install.packages('SID', repos='https://cloud.r-project.org', dependencies=TRUE)"
RUN Rscript -e "library(qgam); print('qgam installed successfully!')"


# -------------------------
# Set workdir
# -------------------------
WORKDIR /app

# -------------------------
# Upgrade pip, setuptools, wheel to ensure prebuilt wheels are used
# -------------------------
RUN pip install --upgrade pip setuptools wheel

# -------------------------
# Copy Python requirements and install
# -------------------------
    # Install PyTorch first
RUN pip install \
    torch==2.8.0+cpu \
    torchvision==0.23.0 \
    --index-url https://download.pytorch.org/whl/cpu
# RUN pip install --use-pep517 gpy>=1.13

RUN pip install numpy==1.26.4
RUN pip install GPy==1.13.2

RUN pip install \
    torch-scatter==2.1.2 \
    torch-sparse==0.6.18 \
    torch-geometric==2.6.1 \
    -f https://data.pyg.org/whl/torch-2.8.0+cpu.html

# Then install remaining requirements (excluding torch / torchvision)
# RUN pip install --use-pep517 gpy>=1.13
COPY requirements.txt .
RUN pip install -r requirements.txt 


# -------------------------
# Copy project code
# -------------------------
COPY . .

# -------------------------
# Install dodiscover in editable mode
# -------------------------
WORKDIR /app/src/competitors/dodiscover
RUN pip install -e .

# -------------------------
# Return to main directory
# -------------------------
WORKDIR /app


ENV LD_LIBRARY_PATH=/usr/local/lib/R/lib
ENV R_HOME=/usr/local/lib/R
# -------------------------
# Default interactive shell
# -------------------------
CMD ["/bin/bash"]
