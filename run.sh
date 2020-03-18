#!/usr/bin/env bash

docker run -it --name fiit_oznal --rm -p 8888:8888 -v $(pwd):/project/ fiit_oznal
