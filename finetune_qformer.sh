python finetune_blip2.py \
--batch-size 8 \
--lr 1e-6 \
--lr-scheduler 200000 \
--out-dir experiments \
--exp-name blip2_qformer \
--warm-up-iter 1000 \
--config-path blip2/config/blip2_finetune_qformer.yaml \
--val-every-epoch 3 \
--nodebug