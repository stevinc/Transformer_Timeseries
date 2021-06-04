# -*- coding: utf-8 -*-
# ---------------------

import subprocess

import click
from path import Path


# -----------------------------
# Template of the Slurm script
# -----------------------------
TEMPLATE = '''#!/bin/bash
#SBATCH --job-name=**exp**
#SBATCH --output=**project**/slurm/log/out.**exp**.txt
#SBATCH --error=**project**/slurm/log/err.**exp**.txt
#SBATCH --open-mode=append
#SBATCH --partition=prod
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1

source activate python3

cd **project**
srun python -u main.py --exp_name '**exp**!' --conf_file_path '**cnf**'
'''


@click.command()
def main():
    """
    (1) creates slurm script
    (2) saves it in 'slurm/<exp_name>.sh
    (3) runs it if `sbatch` == True
    """

    out_err_log_dir_path = Path('slurm/log')
    if not out_err_log_dir_path.exists():
        out_err_log_dir_path.makedirs()

    exp_name = click.prompt('▶ experiment name', type=str)
    if Path(f'conf/{exp_name}.yaml').exists():
        conf_file_name = click.prompt('▶ conf file name', default=f'{exp_name}.yaml')
    else:
        conf_file_name = click.prompt('▶ conf file name', default='default.yaml')

    if '/' in conf_file_name:
        conf_file_path = conf_file_name
    else:
        conf_file_path = f'conf/{conf_file_name}'
    project_dir_path = Path('.').abspath()

    text = TEMPLATE
    text = text.replace('**exp**', exp_name)
    text = text.replace('**cnf**', conf_file_path)
    text = text.replace('**project**', project_dir_path)
    if 'flanzi' in project_dir_path:
        text = text.replace('source activate python3', '#source activate python3')

    print('\n-------------------------------------\n')
    print(text)

    out_file_path = Path('slurm') / exp_name + '.sh'
    out_file_path.write_text(text=text)

    print('-------------------------------------\n')
    if click.confirm('▶ sbatch now?', default=True):
        print('\n-------------------------------------\n')
        command = f'sbatch {out_file_path}'
        process = subprocess.Popen(command.split(), stdout=subprocess.PIPE)
        output, error = process.communicate()
        print('▶', output.decode())
        if error:
            print('▶ [ERROR] - ', error.decode())


if __name__ == '__main__':
    main()
