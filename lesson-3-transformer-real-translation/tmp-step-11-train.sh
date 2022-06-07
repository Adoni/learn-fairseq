#export CUDA_VISIBLE_DEVICES=0,1,2,4
fairseq-train iwslt14/tmp-preprocessed \
  --task translation \
  --arch transformer \
  --optimizer adam \
  --lr 5e-4 --lr-scheduler inverse_sqrt --warmup-updates 4000 \
  --max-tokens 12000 \
  --max-epoch 20 \
  --save-interval 5 \
  --tensorboard-logdir logs \
  --skip-invalid-size-inputs-valid-test \
  --batch-size 128\
  --source-lang src \
  --target-lang tgt \
  --fp16
