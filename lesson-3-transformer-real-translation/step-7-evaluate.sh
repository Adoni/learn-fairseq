export CUDA_VISIBLE_DEVICES=6
fairseq-generate iwslt14/preprocessed \
    --path checkpoints/checkpoint_best.pt \
    --task my_translation \
    --batch-size 128 --beam 5 \
    --user-dir ./my_fairseq_module \
    --skip-invalid-size-inputs-valid-test