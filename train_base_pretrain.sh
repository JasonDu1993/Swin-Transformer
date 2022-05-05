CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python -m torch.distributed.launch --nproc_per_node 8 --master_port 12345  main.py \
  --cfg configs/swin_base_patch4_window7_224_22k_ali.yaml  --pretrained /zhoudu/checkpoints/swin/swin_base_patch4_window7_224_22k.pth \
  --batch-size 512 --accumulation-steps 1 --amp-opt-level O2 --use-checkpoint
