FROM tensorflow/tensorflow
WORKDIR /server
COPY . /server
RUN apt-get update && apt-get install -y \
    wget \
    build-essential \
    cmake \
    git \
    unzip \
    pkg-config \
    python-dev \
    python-opencv \
    libopencv-dev \
    libav-tools  \
    libjpeg-dev \
    libpng-dev \
    libtiff-dev \
    libjasper-dev \
    libgtk2.0-dev \
    python-numpy \
    python-pycurl \
    libatlas-base-dev \
    gfortran \
    webp \
    python-opencv \
    qt5-default \
    libvtk6-dev \
    zlib1g-dev
RUN wget https://github.com/opencv/opencv/archive/4.0.1.zip
RUN unzip 4.0.1.zip
RUN cd opencv-4.0.1 && \
    mkdir build && \
    cd build && \
    cmake .. && \
    make -j4 && \
    make install && \
    ldconfig
RUN pip install avro
RUN pip install keras
RUN pip install opencv-python
EXPOSE 8080