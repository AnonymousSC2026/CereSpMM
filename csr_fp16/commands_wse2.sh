#!/usr/bin/env bash

set -e

cslc --arch=wse3 ./layout.csl --fabric-dims=57,57 --fabric-offsets=4,1 \
--params=h:2,w:3,Mt:2,Kt:2,Nt:2,MAX_META_LEN:3,MAX_PTR_LEN:3 \
--memcpy --channels=1 -o out

cs_python run.py --name out
