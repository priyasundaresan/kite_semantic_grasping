FROM nvidia/cuda:11.1.1-cudnn8-devel-ubuntu20.04
RUN apt-key del 7fa2af80
RUN apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/3bf863cc.pub

RUN rm /etc/apt/sources.list.d/*
RUN apt-get clean 

#RUN apt-get update && apt-get install -y --no-install-recommends \
#      git \
#      libgl1-mesa-glx \
#      libhdf5-dev \
#      libglib2.0-0 \
#      openmpi-bin \
#      wget \
#      && \
#    rm -rf /var/lib/apt/lists/*

ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y --no-install-recommends \
    apt-utils git curl vim wget \
    build-essential cmake \
    libopenblas-dev \
    libboost-dev \
    libboost-all-dev \
    mesa-common-dev \
    libglu1-mesa-dev \
    freeglut3-dev

RUN apt-get install -y libsm6 libxext6 libxrender-dev

RUN apt-get update && apt-get install -y software-properties-common && add-apt-repository ppa:deadsnakes/ppa && \
    apt-get update && apt-get install -y python3.6 python3.6-dev python3-pip
RUN apt install -y python3.6-distutils
RUN ln -sfn /usr/bin/python3.6 /usr/bin/python3 && ln -sfn /usr/bin/python3 /usr/bin/python && ln -sfn /usr/bin/pip3 /usr/bin/pip

RUN pip3 install --no-cache-dir --upgrade pip setuptools && \
    echo "alias python='python3'" >> /root/.bash_aliases && \
    echo "alias pip='pip3'" >> /root/.bash_aliases

RUN apt-get update \
    && apt-get install -y wget \
    && wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
ENV PATH="/root/miniconda3/bin:$PATH"
RUN mkdir /root/.conda && bash Miniconda3-latest-Linux-x86_64.sh -b

ADD env.yml /tmp/env.yml
RUN conda env create -f /tmp/env.yml

WORKDIR /host
CMD ["/bin/bash"]

