# -*- coding: utf-8 -*-
# ---------------------

import torch.backends.cudnn as cudnn
from conf import Conf
from trainer import Trainer
import time

import glob
from pathlib import Path
from retry import retry
import click

cudnn.benchmark = True


@click.command()
@click.option('--exp_path', type=str, default="./conf/experiments/")
@retry(tries=2, delay=2)
def scheduler(exp_path):
    for i, file in enumerate(sorted(glob.glob(exp_path + "*.yaml"))):
        time.sleep(5)
        exp_name = Path(file).stem
        cnf = Conf(conf_file_path=file, exp_name=exp_name, seed=666, log=False)
        print("\n Starting experiment: " + exp_name + "\n")
        trainer = Trainer(cnf=cnf)
        try:
            trainer.run()
        except Exception as e:
            print(e)
        del trainer
        print("\n Starting next experiment...\n")


if __name__ == '__main__':
    scheduler()
