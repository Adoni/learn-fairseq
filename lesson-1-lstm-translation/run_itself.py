# encoding: utf-8
"""
@author: Xiaofei Sun
@contact: xiaofei_sun@shannonai.com
@time: 2022/06/08
@desc: 这只飞很懒
"""
# !/home/sunxiaofei/anaconda3/envs/seq/bin/python
# -*- coding: utf-8 -*-
import re
import sys

from fairseq_cli.train import cli_main

if __name__ == '__main__':
    sys.argv[0] = re.sub(r'(-script\.pyw|\.exe)?$', '', sys.argv[0])
    sys.exit(cli_main())
