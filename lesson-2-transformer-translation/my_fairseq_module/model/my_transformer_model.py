# encoding: utf-8
"""
@author: Xiaofei Sun
@contact: xiaofei_sun@shannonai.com
@time: 2022/05/09
@desc: 这只飞很懒
"""
from fairseq.models import register_model
from fairseq.models.transformer import TransformerModel

from .my_transformer_decoder import MyTransformerDecoder
from .my_transformer_encoder import MyTransformerEncoder


@register_model("my_transformer")
class MyTransformerModel(TransformerModel):
    def __init__(self, args, encoder, decoder):
        super(MyTransformerModel, self).__init__(args, encoder, decoder)

    @classmethod
    def build_encoder(cls, args, src_dict, embed_tokens):
        return MyTransformerEncoder(args, src_dict, embed_tokens)

    @classmethod
    def build_decoder(cls, args, tgt_dict, embed_tokens):
        return MyTransformerDecoder(
            args,
            tgt_dict,
            embed_tokens,
            no_encoder_attn=getattr(args, "no_cross_attention", False),
        )
