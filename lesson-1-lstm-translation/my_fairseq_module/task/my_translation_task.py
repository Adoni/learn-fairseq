# encoding: utf-8
"""
@author: Xiaofei Sun
@contact: xiaofei_sun@shannonai.com
@time: 2022/05/06
@desc: 这只飞很懒
"""
from fairseq.tasks import register_task
from fairseq.tasks.translation import TranslationTask


@register_task('my_translation_task')
class SimpleClassificationTask(TranslationTask):
    pass
