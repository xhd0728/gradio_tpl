@echo off
set CUDA_VISIBLE_DEVICES=0

python src/run.py ^
--ip 127.0.0.1 ^
--port 65001 ^
--queue true
