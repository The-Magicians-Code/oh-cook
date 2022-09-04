FROM nvcr.io/nvidia/pytorch:22.01-py3

ARG DEBIAN_FRONTEND=noninteractive
RUN apt-get update -y

# RUN apt-get upgrade -y
# RUN apt-get install neofetch -y

# Works, but without Gstreamer
RUN pip install opencv-python==4.5.5.64
# RUN apt-get install ffmpeg libsm6 libxext6 -y
RUN apt-get install -y libsm6 libxext6 libxrender-dev

# Broken, probably
# RUN apt-get install gstreamer1.0* -y

# RUN apt install ubuntu-restricted-extras -y
# RUN apt install libgstreamer1.0-dev libgstreamer-plugins-base1.0-dev -y

RUN pip install scikit-image pycuda

# RUN git clone https://github.com/opencv/opencv.git
##################### This bit builds nicely, but throws Gstreamer error
# COPY opencv /opencv
# WORKDIR /opencv/
# RUN git checkout 4.5.0
# RUN mkdir build
# WORKDIR /opencv/build
# RUN cmake -D CMAKE_BUILD_TYPE=RELEASE \
#     -D INSTALL_PYTHON_EXAMPLES=ON \
#     -D INSTALL_C_EXAMPLES=OFF \
#     -D PYTHON_EXECUTABLE=$(which python3) \
#     -D BUILD_opencv_python2=OFF \
#     -D CMAKE_INSTALL_PREFIX=$(python3 -c "import sys; print(sys.prefix)") \
#     -D PYTHON3_EXECUTABLE=$(which python3) \
#     -D PYTHON3_INCLUDE_DIR=$(python3 -c "from distutils.sysconfig import get_python_inc; print(get_python_inc())") \
#     -D PYTHON3_PACKAGES_PATH=$(python3 -c "from distutils.sysconfig import get_python_lib; print(get_python_lib())") \
#     -D WITH_GSTREAMER=ON \
#     -D BUILD_EXAMPLES=OFF ..

# RUN make -j$(nproc)
# RUN make install
# RUN ldconfig

WORKDIR /workspace
# COPY /pytorch /workspace/torching
EXPOSE 3000