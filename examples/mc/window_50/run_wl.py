import jinja2
import numpy as np
import matplotlib.pyplot as plt
import scipy.constants as constants

WINDOW_MIN = 0
WINDOW_MAX = 50
INITIAL_F = np.exp(4)**(1/2**2)
F_STEP_MAX = 15
ACCURACY_FACTOR = 500

NR_STEPS_MC = 100000

# pick the middle of the window
START_NR = (WINDOW_MAX - WINDOW_MIN) // 2

windows = np.arange(WINDOW_MIN, WINDOW_MAX+1)

# write the windows to a file, together with a number of ones (qs) and zeros (hs)
# delimited by a tab
np.savetxt('qs.dat', np.column_stack((windows, np.zeros_like(windows),
                                      np.zeros_like(windows))),
           delimiter='\t')

# create the template for the lammps input file from in.wang_landau.template
template = jinja2.Template(open('in.wang_landau.template').read())


# render the initial f and number of steps
def create_and_run(f_fac, steps, min_n=WINDOW_MIN, max_n=WINDOW_MAX,
                   read_data=False, start_n=0):
    with open('in.wang_landau', 'w') as f:
        f.write(template.render(f_fac=f_fac, steps=steps, min_n=WINDOW_MIN, 
                                max_n=WINDOW_MAX, read_data=read_data, 
                                start_n=start_n))

    # run lammps
    import subprocess
    # subprocess call without output
    subprocess.call(['../../../build/lmp', '-in', 'in.wang_landau'], 
                    stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


def convergence_check(f):
    # Do a convergence check on qs.dat, see if the histogram is flat (h.min == 1/ln(f))
    hist = np.loadtxt('qs.dat', delimiter='\t')
    return hist[:, 2].min() > ACCURACY_FACTOR / np.sqrt(np.log(f))


def reset_hs():
    qs = np.loadtxt('qs.dat', delimiter='\t')
    qs[:, 2] = 0
    np.savetxt('qs.dat', qs, delimiter='\t')


create_and_run(INITIAL_F, 100, start_n=START_NR)
# print(convergence_check(INITIAL_F))

f = INITIAL_F
f_update = 1
print(f'Starting to run, f is {f} we are at f_{f_update}')
while True:
    # check for convergence, if converged, make f more fine grained by sqrt(f)
    if convergence_check(f):
        # write out the qs to its own file, per f_update
        np.savetxt(f'qs_f_{f_update}.dat', np.loadtxt('qs.dat', delimiter='\t'))

        f = np.sqrt(f)
        f_update += 1

        # if f is too fine grained, break
        if f_update > 15:
            break

        reset_hs()
        print(f'New update to f, f is now {f} we are at f_{f_update}')
        print(f'This means we will step through {NR_STEPS_MC} steps')

    print(f'Running with f={f} for {NR_STEPS_MC} steps')
    create_and_run(f, NR_STEPS_MC, read_data=True)

def omega(f, temp, Qarr):
    """ all energies in Kelvin
        fugacity in bar!
    """
    mass = 28 * constants.atomic_mass
    db = constants.Planck / np.sqrt(2*np.pi * mass * constants.Boltzmann * temp)
    mu = temp * np.log((f*constants.bar)*db**3/(constants.Boltzmann*temp))
    N = Qarr[:,0]
    lnQ = Qarr[:,1]
    return (-temp*lnQ - mu*N)

fig, ax = plt.subplots(2)

qs = np.loadtxt('qs.dat', delimiter='\t')[:, 1]
hs = np.loadtxt('qs.dat', delimiter='\t')[:, 2]

ax[0].plot(windows, hs)
ax[0].set_ylim(0, hs.max()+ 100)
ax[1].plot(windows, qs)
plt.show()
