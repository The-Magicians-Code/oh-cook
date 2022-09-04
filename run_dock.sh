#!/usr/bin/env bash
docker build -t cont1 .
docker run -it --rm -v $PWD:/workspace/torching -p 3000:3000 cont1
# docker run -it --rm -p 3000:3000 cont1