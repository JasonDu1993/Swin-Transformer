CUDA_VISIBLE_DEVICES=0,1,2,3,5,6,7 python -m torch.distributed.launch --nproc_per_node 7 --master_port 12345 main.py \
  --cfg configs/swin_small_patch4_window7_224.yaml --batch-size 32 --accumulation-steps 4
