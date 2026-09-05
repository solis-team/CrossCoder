# Use miniconda for better package management
FROM continuumio/miniconda3:latest

# install git, g++ and other dependencies
RUN apt-get update && apt-get install -y \
    git \
    g++ \
    zip \
    unzip \
    procps \
    r-base \
    libgdal-dev \
    libfreetype6-dev \
    libpng-dev \
    pkg-config \
    libsndfile1 \
    && rm -rf /var/lib/apt/lists/*

# Create a Python 3.7 conda environment
RUN conda create -n bigcodebench python=3.7 -y

# Activate the environment and install packages
SHELL ["conda", "run", "-n", "bigcodebench", "/bin/bash", "-c"]

# upgrade pip in the conda environment
RUN pip install --upgrade pip

# Add a new user "bigcodebenchuser"
RUN useradd -m -s /bin/bash bigcodebenchuser

RUN rm -rf /bigcodebench

# Acquire benchmark code from the local build context
COPY . /bigcodebench

# Install numpy first (critical dependency)
RUN pip install numpy==1.19.5 pyarrow==12.0.1

# Install bigcodebench without dependencies
RUN cd /bigcodebench && pip install . --no-deps

# Install core dependencies
RUN pip install \
    appdirs \
    fire \
    wget \
    termcolor \
    tempdir==0.7.1 \
    tqdm \
    pqdm \
    gradio-client \
    datasets==2.13.2 \
    transformers==4.20.0 \
    rich

# Install evaluation libraries - with platform=amd64, pre-built wheels will be used
# Core libraries first
RUN pip install \
    chardet==4.0.0 \
    python-dateutil==2.7.3 \
    pytz==2021.1 \
    six==1.16.0

# Scientific computing stack - will use pre-built wheels on amd64
RUN pip install \
    scipy==1.5.4 \
    scikit-learn==0.19.2 \
    matplotlib==3.3.4 \
    seaborn==0.9.0

# Data processing
RUN pip install \
    pandas==1.3.5 \
    openpyxl==3.0.0 \
    prettytable==2.0.0 \
    xmltodict==0.12.0 \
    PyYAML==5.3.0

# Web frameworks
RUN pip install \
    Flask==1.1.4 \
    Flask-Login==0.5.0 \
    Flask-WTF==0.15.1 \
    WTForms==2.3.1 \
    Werkzeug==1.0.1

# Audio/Image processing
# Use newer opencv-python version that has pre-built wheels for Python 3.7
RUN pip install \
    opencv-python==4.5.5.64 \
    soundfile==0.10.1 \
    Pillow==8.4.0

# NLP and other utilities
RUN pip install \
    nltk==3.6.7 \
    textblob==0.15.3 \
    Faker==8.1.0

# Geo and mapping
RUN pip install \
    geopy==2.1.0 \
    folium==0.12.1

# Audio analysis
RUN pip install \
    numba==0.53.1 \
    librosa==0.8.1

# HTTP requests
RUN pip install requests==2.27.1

WORKDIR /app

RUN chown -R bigcodebenchuser:bigcodebenchuser /app

RUN chmod -R 777 /app

USER bigcodebenchuser

# Update ENTRYPOINT to use conda run
ENTRYPOINT ["conda", "run", "--no-capture-output", "-n", "bigcodebench", "python", "-m", "bigcodebench.evaluate"]

CMD ["sh", "-c", "pids=$(ps -u $(id -u) -o pid,comm | grep 'bigcodebench' | awk '{print $1}'); if [ -n \"$pids\" ]; then echo $pids | xargs -r kill; fi; rm -rf /tmp/*"]
