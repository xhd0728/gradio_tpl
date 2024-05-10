export CUDA_VISIBLE_DEVICES=0

set -ex

python src/run.py \
    --ip 0.0.0.0 \
    --port 65001 \
    --queue
