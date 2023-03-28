python finetune_blip2.py \
--batch-size 4 \
--lr 1e-6 \
--lr-scheduler 200000 \
--out-dir experiments \
--exp-name blip2_proj_new \
--warm-up-iter 1000 \
--config-path blip2/config/blip2_filetune_proj.yaml \
--val-every-epoch 5 \
--nodebug