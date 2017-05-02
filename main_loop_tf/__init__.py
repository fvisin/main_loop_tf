from subprocess import check_output


__version__ = check_output('git rev-parse HEAD',
                           shell=True).strip().decode('ascii')
