#!/usr/bin/python

#SBATCH --job-name=MYTEST_LV0-Opt
#SBATCH --output=slurm-%A_%a_%N.out
#SBATCH --partition=xyz
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4 #corresponds to Gurobi threads
#SBATCH --mem=30000
#SBATCH --array=0-2
##SBATCH --ntasks-per-node=3 #total mem per node = cpus-per-task*mem-per-cpu*ntasks-per-node ; -> 4*5GB*3 = 60GB


import os,multiprocessing
import sys
import datetime
import time,random,yaml
import itertools
import vresutils.file_io_helper as io_helper

import logging
logger = logging.getLogger(__name__)

NOW = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")

with open("options.yml","r") as yml_:
    options = yaml.load(yml_)
run_name = 'opteu_MYTEST'
options['run_name'] = run_name

results_dir_name = os.path.join(options['results_dir_name'], options['run_name'])

io_helper.ensure_mkdir(results_dir_name)


#store copies of all files, in case they get changed during the run
storedir = os.path.join(results_dir_name,'slurm-{}'.format(os.environ['SLURM_ARRAY_JOB_ID']))
io_helper.ensure_mkdir(storedir)
stored = {}
for file_name in ["opt_ws_network.py","parameter_batch.py","options.yml"]:#,"blind_process_sector.py"]:
    stored[file_name] = os.path.join(storedir,
                                     '{}-{}'.format(file_name,os.environ['SLURM_ARRAY_JOB_ID']))
    io_helper.copy_without_overwrite(file_name,stored[file_name])

def run_opt(p):

    logger.info('run_opt args: {}'.format(p))

    #reopen to avoid shared memory issues with different threads modifying the same dictionary
    with open(stored["options.yml"],"r") as yml_:
        options = yaml.load(yml_)
    options['run_name'] = run_name

    if p=='Opt':
        options['line_volume_limit_max'] = None
    options['line_volume_limit_factor'] = p


    options['results_suffix'] = 'MYTEST_LV{}_c0_{nam}-{time}'.format(
            options['line_volume_limit_factor'],
            nam=options['run_name'],time=NOW)

    options['solver_options']['logfile'] = os.path.join(results_dir_name,
                                                        'solver_{suffix}.log'.format(suffix=options['results_suffix']))

    options['opt_log_file'] = os.path.join(results_dir_name,
                                           'opt_log_{suffix}.out'.format(suffix=options['results_suffix']))

    options['options_file'] = os.path.join(results_dir_name,
                                          'options_{suffix}.yml'.format(suffix=options['results_suffix']))


    command = ("python {script} {options}".format(
                script=stored['opt_ws_network.py'],
                options=options['options_file']))

    #time.sleep(2.*random.random())
    time.sleep(int(os.environ['SLURM_ARRAY_TASK_ID']))

    io_helper.ensure_mkdir(results_dir_name)


    # save default files
    for file_name in stored.iterkeys():
        fn,ext=os.path.splitext(file_name)
        os.system('cp {} {}'.format(
            stored[file_name],
            os.path.join(results_dir_name, '{fn}_{suffix}{ext}'.format(
                suffix=options['results_suffix'],fn=fn,ext=ext))))
    
    # overwrite local default options file
    with open(options["options_file"],"w") as yml_:
        yaml.dump(options,yml_)

    logger.info('Running command:\n'+command)
    os.system(command)

    return

if __name__ == '__main__':
    ps = [0.,0.25,'Opt']

    assert int(os.environ['SLURM_ARRAY_TASK_MAX']) == len(ps)-1

    run_opt(ps[int(os.environ['SLURM_ARRAY_TASK_ID'])])


    #move log file to results dir:
    os.system('mv slurm-{A}_{a}_{N}.out {outname_}-{A}_{a}_{N}_p{p}-{time}.out'.format(
        A=os.environ['SLURM_ARRAY_JOB_ID'],
        a=os.environ['SLURM_ARRAY_TASK_ID'],
        N=os.environ['SLURMD_NODENAME'],
        p=ps[int(os.environ['SLURM_ARRAY_TASK_ID'])],
        outname_=os.path.join(storedir,'slurm'),
        time=NOW))
