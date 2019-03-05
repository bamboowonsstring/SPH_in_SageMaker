# Build an image that can do training and inference in SageMaker
# This is a Python 2 image that uses the nginx, gunicorn, flask stack
# for serving inferences in a stable way.

FROM nvidia/opencl:devel-ubuntu16.04

MAINTAINER bamboowonsstring <extra.excramattion1@gmail.com>


RUN apt-get -y update && apt-get install -y\
         wget \
         python \
         python-pip \
         ca-certificates \
         python-dev 
#    && rm -rf /var/lib/apt/lists/*
# RUN wget https://bootstrap.pypa.io/ez_setup.py -O - | python3
#R UN apt-get install libgomp1
RUN apt-get install -y build-essential python-dev python-numpy \
    python-mako cython python-pytest mayavi2 python-virtualenv
#RUN apt-get install nvidia-modprobe
#RUN pip install pocl
#RUN apt-get install opencl-headers

#For pysph
RUN pip install pytools
RUN pip install numpy-stl
#for pyopencl
#RUN apt-get install libnuma1
#RUN pip install pybind11 
#RUN pip install pyopencl
RUN pip install PySPH

ENV PYTHONUNBUFFERED=TRUE
ENV PYTHONDONTWRITEBYTECODE=TRUE
ENV PATH="/opt/program:${PATH}"

# Set up the program in the image
COPY calculate.py /opt/program/
COPY train /opt/program/
WORKDIR /opt/program

