#! /bin/bash

trtexec_path=$1
model_path=$2
precision=$3

engine_path="${model_path%.onnx}.engine"

${trtexec_path} -
    -onnx=${model_path} \ 
    --saveEngine=${engine_path} \ 
    --fp${precision} \ 
    --profilingVerbosity=detailed  \
    --timingCacheFile=${engine_path}.cache