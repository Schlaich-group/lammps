import jinja2
import numpy as np
import matplotlib.pyplot as plt
import scipy.constants as constants
import os
import shutil
import multiprocessing as mp

WINDOW_MIN = 150
WINDOW_MAX = 180

INITIAL_F = np.exp(4)**(1/2**6)
F_STEP_MAX = 17
ACCURACY_FACTOR = 500

NR_STEPS_MC = 100000

TEMPLATE_FILENAME = 'in.wang_landau.fcc.template'

# pick the middle of the window
START_NR = (WINDOW_MAX + WINDOW_MIN) // 2

windows = np.arange(WINDOW_MIN, WINDOW_MAX+1)

# write the windows to a file, together with a number of ones (qs) and zeros (hs)
# delimited by a tab
np.savetxt('qs.dat', np.column_stack((windows, np.zeros_like(windows),
                                      np.zeros_like(windows))),
           delimiter='\t')

# create the template for the lammps input file from in.wang_landau.template
template = jinja2.Template(open(TEMPLATE_FILENAME).read())


# render the initial f and number of steps
def create_and_run(f_fac, steps, min_n=WINDOW_MIN, max_n=WINDOW_MAX,
                   read_data=False, start_n=0, fcc=True):
    with open('in.wang_landau', 'w') as f:
        f.write(template.render(f_fac=f_fac, steps=steps, min_n=min_n,
                                max_n=max_n, read_data=read_data,
                                start_n=start_n, fcc=fcc))

    # run lammps
    import subprocess
    # subprocess call without output
    call = subprocess.call(['../../../../build/lmp', '-in', 'in.wang_landau'],
                           stdout=subprocess.DEVNULL,
                           stderr=subprocess.DEVNULL)

    # check the return code of the subprocess call
    if call == 0:
        print('LAMMPS converged.')
        return True
    else:
        return False


def convergence_check(f):
    # Do a convergence check on qs.dat, see if the histogram is flat 
    # (h.min == 1/ln(f))
    hist = np.loadtxt('qs.dat', delimiter='\t')
    return hist[:, 2].min() >= ACCURACY_FACTOR / np.sqrt(np.log(f))


def reset_hs():
    qs = np.loadtxt('qs.dat', delimiter='\t')
    qs[:, 2] = 0
    np.savetxt('qs.dat', qs, delimiter='\t')


create_and_run(INITIAL_F, NR_STEPS_MC, start_n=START_NR)
# print(convergence_check(INITIAL_F))

f = INITIAL_F
f_update = 1
print(f'Starting to run, f is {f} we are at f_{f_update}')
while True:
    # check for convergence, if converged, make f more fine grained by 
    # sqrt(f)
    if convergence_check(f):
        # write out the qs to its own file, per f_update
        np.savetxt(f'qs_f_{f_update}.dat', np.loadtxt('qs.dat',
                                                      delimiter='\t'),
                   delimiter='\t')

        f = np.sqrt(f)
        f_update += 1

        # if f is too fine grained, break
        if f_update > 15:
            break

        reset_hs()
        print(f'New update to f, f is now {f} we are at f_{f_update}')
        print(f'This means we will step through {NR_STEPS_MC} steps')

    print(f'Running with f={f} for {NR_STEPS_MC} steps')
    if create_and_run(f, NR_STEPS_MC, read_data=True):
        converged = convergence_check(f)
        print(f'Converged: {converged}')
