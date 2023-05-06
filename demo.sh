# python3 demo.py --checkpoint_dir /HOME/lyh/vq-mot/experiments/vanilla_randlen_extramot_recons_flag_lossweight3e-1/checkpoints/epoch-599-step-0.pth --replication_times 2
python3 demo.py \
--checkpoint_dir /HOME/lyh/vq-mot/experiments/cleandata_h36m_flag3d_textaug_weight04_gtext00/checkpoints/epoch-859-step-0.pth \
--replication_times 2 \
--output_dir /HOME/lyh/vq-mot/experiments/cleandata_h36m_flag3d_textaug_weight04_gtext00/zero-shot-flag3d
