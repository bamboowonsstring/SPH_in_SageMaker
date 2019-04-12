# Build an image that can do training and inference in SageMaker
# This is a Python 2 image that uses the nginx, gunicorn, flask stack
# for serving inferences in a stable way.

FROM nvidia/opencl:devel-ubuntu16.04

MAINTAINER bamboowonsstring <extra.excramattion1@gmail.com>


RUN apt-get -y update && apt-get install -y\
         wget \
         python3 \
         python3-pip \
         ca-certificates \
         python3-dev 
RUN apt-get install -y build-essential 
RUN apt-get install -y python3-numpy
RUN apt-get install -y python3-pytest 
RUN apt-get install -y cython3

#For pysph
RUN pip3 install pytools
RUN pip3 install h5py
RUN pip3 install pysph

ENV PYTHONUNBUFFERED=TRUE
ENV PYTHONDONTWRITEBYTECODE=TRUE
ENV PATH="/opt/program:${PATH}"

# Set up the program in the image
COPY calculate.py /opt/program/
COPY train /opt/program/
WORKDIR /opt/program

