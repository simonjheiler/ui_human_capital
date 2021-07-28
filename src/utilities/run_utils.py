import subprocess

import matlab.engine
from bld.project_paths import project_paths_join as ppj


#####################################################
# PARAMETERS
#####################################################

python_path = r"C:\Programs\Anaconda3\envs\labor-markets\python.exe"
stata_path = r"C:\Program Files\stata14\StataSE-64"
r_path = r"C:\Programs\Anaconda3\envs\labor-markets\Scripts\RScript.exe"


#####################################################
# FUNCTIONS
#####################################################


def run_py_script(filename, params):

    cmd = [python_path, filename]
    for param in params:
        cmd.append(param)

    return subprocess.call(cmd)


def run_r_script(filename, params):

    cmd = [r_path, "--vanilla", filename]
    for param in params:
        cmd.append(param)

    return subprocess.call(cmd)


def run_stata_script(filename, params):

    cmd = [stata_path, "/e", "do", filename]
    for param in params:
        cmd.append(param)

    return subprocess.call(cmd)


def run_matlab_script(filename, params):

    eng = matlab.engine.start_matlab()

    return eng.filename(nargout=0)


#####################################################
# SCRIPT
#####################################################

if __name__ == "__main__":

    run_py = False
    run_r = False
    run_stata = False
    run_matlab = True

    # script = ppj("IN_DATA_MGMT", "format_data_cps_monthly.py")
    # script = ppj("PROJECT_ROOT", "src", "model_analysis", "probit_regression.r")
    # script = ppj("PROJECT_ROOT", "src", "model_analysis", "aggregate_regression_mr_old.do")
    script = ppj("PROJECT_ROOT", "src", "model_analysis", "analytics", "Main.m")

    parameters = []

    if run_py:
        run_py_script(script, parameters)
    elif run_r:
        run_r_script(script, parameters)
    elif run_stata:
        run_stata_script(script, parameters)
    elif run_matlab:
        run_matlab_script(script, parameters)
    else:
        print("select program to run script")
