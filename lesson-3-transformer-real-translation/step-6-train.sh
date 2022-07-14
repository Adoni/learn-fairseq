export CUDA_VISIBLE_DEVICES=0,1,2,3
fairseq-train iwslt14/preprocessed \
  --user-dir ./my_fairseq_module \
  --task my_db_translation \
  --arch my_small_transformer \
  --optimizer adam \
  --lr 5e-4 --lr-scheduler inverse_sqrt --warmup-updates 4000 \
  --max-tokens 12000 \
  --max-epoch 20 \
  --save-interval 5 \
  --tensorboard-logdir logs \
  --skip-invalid-size-inputs-valid-test \
  --num-workers 1 \
  --fp16
