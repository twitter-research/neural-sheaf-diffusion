#!/bin/bash

for i in {0..7}
do
    CUDA_VISIBLE_DEVICES=$((i % 8)) wandb agent "$1"/research-repo-sheaf_exp/"$2" &
done