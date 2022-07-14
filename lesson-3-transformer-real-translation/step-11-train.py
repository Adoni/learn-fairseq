# encoding: utf-8
"""
@author: Xiaofei Sun

@time: 2022/06/22
@desc: 这只飞很懒
"""
# encoding: utf-8
"""
@author: Xiaofei Sun

@time: 2022/06/09
@desc: 这只飞很懒
"""
import sys

from config import config
from fairseq_cli.train import cli_main


def run():
    sys.argv += [
        config.fairseq_data_path,
        "--task", "translation",
        "--arch", "transformer",
        # "--arch", "auto_encoder_transformer",
        "--optimizer", "adam",
        "--lr", "5e-4",
        "--lr-scheduler",
        "inverse_sqrt",
        "--warmup-updates", "40000",
        "--max-tokens", "12000",
        "--max-epoch", "40",
        "--save-interval", "1",
        "--tensorboard-logdir", "logs",
        "--skip-invalid-size-inputs-valid-test",
        "--source-lang", "src",
        "--target-lang", "tgt",
        "--clip-norm", str(1.0),
        "--user-dir", "./models",
        "--keep-last-epochs", "3",
        "--fp16"
    ]
    print(sys.argv)
    cli_main()


if __name__ == '__main__':
    run()
