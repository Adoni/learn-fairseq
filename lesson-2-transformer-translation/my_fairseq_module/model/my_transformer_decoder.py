# encoding: utf-8
"""
@author: Xiaofei Sun

@time: 2022/05/09
@desc: 这只飞很懒
"""
from fairseq.models.transformer import TransformerDecoder


class MyTransformerDecoder(TransformerDecoder):
    def __init__(self, args, dictionary, embed_tokens, no_encoder_attn=False):
        super(MyTransformerDecoder, self).__init__(args, dictionary, embed_tokens, no_encoder_attn)
        print("Here is my decoder")
