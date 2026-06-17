#!/bin/bash

holdout_domain=$1
split=$2

python encode.py --config config.yaml --split ${split} --holdout_domain ${holdout_domain}

