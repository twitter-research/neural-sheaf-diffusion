#!/bin/sh

python -m exp.run \
    --add_lp=True \
    --d=3 \
    --dataset=squirrel \
    --dropout=0 \
    --early_stopping=100 \
    --epochs=1000 \
    --folds=10 \
    --hidden_channels=32 \
    --input_dropout=0.7 \
    --layers=5 \
    --lr=0.01 \
    --model=BundleSheaf \
    --orth=householder \
    --second_linear=True \
    --weight_decay=0.00011215791366362148 \
    --left_weights=True \
    --right_weights=True \
    --use_act=True \
    --normalised=True \
    --edge_weights=True \
    --stop_strategy=acc \
    --entity="${ENTITY}" 