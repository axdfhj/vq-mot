python finetune_blip2.py \
--batch-size 2 \
--lr 1e-6 \
--lr-scheduler 200000 \
--out-dir experiments \
--exp-name blip2_vit \
--warm-up-iter 1000 \
--config-path blip2/config/blip2_finetune_vit.yaml \
--val-every-epoch 2 \
--nodebug