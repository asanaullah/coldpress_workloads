python -m coldpress.distributed.run --nproc_per_node=2 --nnodes=2 train.py --dataset=mnist \
   --train-test-split=0.8 --epochs=2 --batch-size=128 --hidden-size=4096 --lr=0.01
