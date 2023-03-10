python3 train.py  \
--exp-name debug \
--batch-size 2 \
--nb-code 512 \
--resume-pth output/VQVAE/net_last.pth \
--vq-name VQVAE \
--out-dir experiments \
--lr-scheduler 150000 \
--lr 1.5e-5 \
--dataname t2m \
--weight-decay 4.5e-2 \
--down-t 2 \
--depth 3 \
--quantizer ema_reset \
--dilation-growth-rate 3 \
--vq-act relu