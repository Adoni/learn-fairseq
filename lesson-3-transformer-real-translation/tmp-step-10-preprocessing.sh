RAW=iwslt14/tmp-bpe
fairseq-preprocess --source-lang src --target-lang tgt \
    --trainpref $RAW/train --validpref $RAW/valid --testpref $RAW/test \
    --destdir iwslt14/tmp-preprocessed