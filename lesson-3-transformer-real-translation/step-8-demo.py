# encoding: utf-8
"""
@author: Xiaofei Sun
@contact: xiaofei_sun@shannonai.com
@time: 2022/05/03
@desc: 这只飞很懒
"""

import fastBPE
from fairseq.hub_utils import GeneratorHubInterface

from my_fairseq_module.model.my_transformer_model import MyTransformerModel


def run():
    model: GeneratorHubInterface = MyTransformerModel.from_pretrained(
        model_name_or_path="checkpoints",
        checkpoint_file="checkpoint_best.pt",
        tokenizer=None,
    )
    print(type(model))
    src_bpe = fastBPE.fastBPE("./codes.bpe", "./vocab.src.40000")
    tgt_bpe = fastBPE.fastBPE("./codes.bpe", "./vocab.tgt.40000")
    bpe_symbol = "@@ "
    while True:
        sentence = input('\nInput: ')
        translation = model.translate([src_bpe.apply([sentence])[0]])  # 注意这里是特地写的繁琐的，为了强调二者接收的都是list，而不是单个句子
        print(translation)
        # 下面这一行模仿了fairseq例fastBPE的处理方法，参见 https://github.com/facebookresearch/fairseq/blob/main/fairseq/data/encoders/fastbpe.py
        print((translation[0] + " ").replace(bpe_symbol, "").rstrip())


if __name__ == '__main__':
    run()
