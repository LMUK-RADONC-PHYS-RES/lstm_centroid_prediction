# Dockerfile for centroid_prediction project

# inherit from monai image
FROM projectmonai/monai:0.5.2

# create install folder inside container
RUN mkdir /requirements_installation

# add requirements file to container
ADD requirements.txt /requirements_installation

# update to latest pip version 
RUN pip install -U pip

# check python version --> Python 3.8.5
RUN python --version

# install the required python packages
RUN pip install -r /requirements_installation/requirements.txt

# add some missing libraries for cv2
RUN apt-get update 
RUN apt-get install ffmpeg libsm6 libxext6  -y


