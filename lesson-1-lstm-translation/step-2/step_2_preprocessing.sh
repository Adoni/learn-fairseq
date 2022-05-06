RAW=easy-dataset/raw_data
fairseq-preprocess --source-lang in --target-lang out \
    --trainpref $RAW/train --validpref $RAW/valid --testpref $RAW/test \
    --destdir easy-dataset/preprocessed