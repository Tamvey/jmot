#!/bin/bash
trtexec_path=$1
model_path=$2

base_name=$(basename "$model_path" .engine)

mkdir -p profile &&

${trtexec_path} \
        --verbose \
        --noDataTransfers \
        --useCudaGraph \
        --separateProfileRun \
        --useSpinWait \
        --loadEngine=${model_path} \
        --exportTimes=/profile/${base_name}_time.json \
        --exportProfile=/profile/${base_name}_profile.json \
        --exportLayerInfo=/profile/${base_name}_graph.json \
        --timingCacheFile=/profile/${base_name}_cache.json \
        --profilingVerbosity=detailed 
        # --minShapes=input:1x3x640x640 \
        # --optShapes=input:8x3x640x640 \
        # --maxShapes=input:16x3x640x640