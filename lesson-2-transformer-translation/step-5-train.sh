export CUDA_VISIBLE_DEVICES=6
fairseq-train easy-dataset/preprocessed \
  --user-dir ./my_fairseq_module \
  --task my_translation \
  --arch my_small_transformer \
  --optimizer adam \
  --lr 5e-4 --lr-scheduler inverse_sqrt --warmup-updates 4000 \
  --max-tokens 12000 \
  --max-epoch 200 \
  --save-interval 50 \
  --tensorboard-logdir logs
