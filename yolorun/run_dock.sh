#!/usr/bin/env bash
docker build -t cont2 .
docker run --gpus all --ipc=host --ulimit memlock=-1 --ulimit stack=67108864 -it --rm -v $PWD/yolov5:/workspace -p 3001:3001 cont2
# docker run -it --rm -p 3000:3000 cont1
# docker run --gpus all --ipc=host --ulimit memlock=-1 --ulimit stack=67108864 -it --rm -p 3001:3001 cont2
