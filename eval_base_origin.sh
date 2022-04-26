CUDA_VISIBLE_DEVICES=4,5,6,7 python -m torch.distributed.launch --nproc_per_node 4 --master_port 12346  main.py --eval single\
  --cfg configs/swin_base_patch4_window7_224_22k.yaml  --resume /zhoudu/checkpoints/swin/swin_base_patch4_window7_224_22k.pth --batch-size 128
