# -*- coding: utf-8 -*-
# ---------------------

import click
import torch.backends.cudnn as cudnn

from conf import Conf
from trainer import Trainer
from inference import TS

cudnn.benchmark = True


@click.command()
@click.option('--exp_name', type=str, default=None)
@click.option('--conf_file_path', type=str, default=None)
@click.option('--seed', type=int, default=None)
@click.option('--inference', type=bool, default=False)
def main(exp_name, conf_file_path, seed, inference):
    # type: (str, str, int, bool) -> None

    # if `exp_name` is None,
    # ask the user to enter it
    if exp_name is None:
        exp_name = click.prompt('▶ experiment name', default='default')

    # if `exp_name` contains '!',
    # `log_each_step` becomes `False`
    log_each_step = True
    if '!' in exp_name:
        exp_name = exp_name.replace('!', '')
        log_each_step = False

    # if `exp_name` contains a '@' character,
    # the number following '@' is considered as
    # the desired random seed for the experiment
    split = exp_name.split('@')
    if len(split) == 2:
        seed = int(split[1])
        exp_name = split[0]

    cnf = Conf(conf_file_path=conf_file_path, seed=seed, exp_name=exp_name, log=log_each_step)
    print(f'\n{cnf}')

    print(f'\n▶ Starting Experiment \'{exp_name}\' [seed: {cnf.seed}]')

    if inference:
        ts_model = TS(cnf=cnf)
        ts_model.test()
    else:
        trainer = Trainer(cnf=cnf)
        trainer.run()


if __name__ == '__main__':
    main()
