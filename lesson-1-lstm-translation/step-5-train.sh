export CUDA_VISIBLE_DEVICES=0
fairseq-train easy-dataset/preprocessed \
  --user-dir ./my_fairseq_module \
  --arch tutorial_simple_lstm \
  --encoder-dropout 0.2 --decoder-dropout 0.2 \
  --optimizer adam --lr 0.005 --lr-shrink 0.5 \
  --max-tokens 12000 \
  --max-epoch 100 \
  --source-lang in \
  --target-lang out