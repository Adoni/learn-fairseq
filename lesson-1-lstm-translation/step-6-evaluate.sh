fairseq-generate easy-dataset/preprocessed \
    --path checkpoints/checkpoint_best.pt \
    --batch-size 128 --beam 5 \
    --user-dir ./my_fairseq_module \
