python train_vae.py \
--batch-size 512 \
--lr 2e-4 \
--lr-scheduler 50000 \
--nb-code 512 \
--down-t 2 \
--depth 3 \
--dilation-growth-rate 3 \
--out-dir experiments \
--dataname t2m \
--quantizer ema_reset \
--loss-vel 0.5 \
--recons-loss l1_smooth \
--h36m \
--nodebug \
--exp-name vanilla_64frames_allmot_bs512_extralossweight_10_50