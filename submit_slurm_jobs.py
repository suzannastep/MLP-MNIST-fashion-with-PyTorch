import os
import numpy as np
from collections import OrderedDict
from itertools import product

# creates a folder in the current working directory
def mkdir(dir):
    if not os.path.exists(dir):
        os.mkdir(dir)

# given input parameters, writes a .sh file and submits a sbatch job
def make_sbatch(code_file,jobname,parameternames,parameters):
    params = ""
    paramname = "/"
    for param,name in zip(parameters,parameternames):
        paramname += "_" + name + "_" + str(param)
        params += f"--{name} {param} "
    paramname = paramname.replace(" ", "_")
    filename = jobname + paramname
    params += f"--filename {filename}"
    print(params)
    print(paramname)
    mkdir("./"+filename)
    command = f"python -W error::UserWarning {code_file} "
    with open(filename+f"{paramname}.sh",'w') as fh:
        # the .sh file header may be different depending on the cluster
        fh.writelines('#!/bin/bash')
        fh.writelines('\n\n#SBATCH --job-name={}'.format(filename))
        fh.writelines('\n#SBATCH --partition=general')
        fh.writelines('\n#SBATCH --gres=gpu:1')
        fh.writelines(f'\n#SBATCH --output=log/{filename}.out')
        fh.writelines(f'\n#SBATCH --error=log/{filename}.err')
        fh.writelines('\necho "$date Starting Job"')
        fh.writelines('\necho "SLURM Info: Job name:${SLURM_JOB_NAME}"')
        fh.writelines('\necho "    JOB ID: ${SLURM_JOB_ID}"')
        fh.writelines('\necho "    Host list: ${SLURM_JOB_NODELIST}"')
        fh.writelines('\necho "    CUDA_VISIBLE_DEVICES: ${CUDA_VISIBLE_DEVICES}"')
        fh.writelines('\nwhich python\n\n')
        fh.writelines(command+params)
    return filename+f"/{paramname}.sh"

def run_sbatch(job_file):
    os.system('sbatch {}'.format(job_file))
                                                                            
if __name__ == "__main__":
    output_path = '/net/projects/willettlab/sueparkinson/deeprelu' #absolute path to where to save results
    os.chdir(output_path)

    code_file = '/home/sueparkinson/deeprelu/MLP-MNIST-fashion-with-PyTorch/main.py'
    jobname = "fashionMNIST"
    parameters = OrderedDict()
    parameters['model'] = ['H','Hml']
                            #['A'  ,'B'  ,'C'  ,'D'  ,'E'  ,'F'  ,'G',
                            #'Aml','Bml','Cml','Dml','Eml','Fml','Gml',]
    parameters['wd'] = [1e-3,1e-4,1e-5,1e-6,0]
    # create folder in the current working directory
    mkdir(jobname)
    mkdir("log")
    mkdir(f"log/{jobname}")
    # run file w all different combos of parameters we want to try
    for paramsetting in product(*parameters.values()):
        job_file = make_sbatch(code_file,jobname,parameters.keys(),paramsetting)
        print("created",job_file)
        run_sbatch(job_file)
        print("running",job_file)