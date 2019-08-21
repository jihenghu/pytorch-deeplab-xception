#!/usr/bin/env bash

# need to build and install cocoapi on local computer
# otherwise, it will have error: No module named 'pycocotools'
# for each os system (Mac or Ubuntu, need to rebuild pycocotools)

dir=${PWD}

cd ~/codes/github_public_repositories/cocoapi

cd PythonAPI
make

cp -r pycocotools ${dir}/.