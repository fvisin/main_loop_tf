from subprocess import check_output
from main import Experiment


__version__ = check_output('git rev-parse HEAD',
                           shell=True).strip().decode('ascii')
