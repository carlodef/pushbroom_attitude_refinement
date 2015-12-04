This Python package provides implementations of the orbiting pushbroom camera
simulator and the attitude refinement algorithm described in the IPOL paper
*Attitude Refinement for Orbiting Pushbroom Cameras: a Simple Polynomial
Fitting Method* by Carlo de Franchis, Enric Meinhardt-Llopis, Daniel Greslou
and Gabriele Facciolo.

This is version `20151203`, released on 03 december 2015. Updated versions will
be distributed on github at
<http://www.github.com/carlodef/pushbroom_attitude_refinement>

This package is written by Carlo de Franchis <carlo.de-franchis@m4x.org> with
contributions from Gabriele Facciolo <facciolo@cmla.ens-cachan.fr> and Enric
Meinhardt-Llopis <enric.meinhardt@cmla.ens-cachan.fr>.

The module `pushbroom_simulator.py` is based on a code written by Daniel
Greslou <daniel.greslou@cnes.fr>.

The package is distributed under the terms of the AGPLv3 license.


### List of files

    run_attitude_refinement_simulation.py
    attitude_refinement_from_gcp.py
    pushbroom_simulator.py
    utils.py
    params.json.example

### Required dependencies

This Python package depends on `numpy` and `cvxopt`. The sources of these two
packages are included in the `3rdparty` directory. Instructions on how to
compile them are given below.

#### numpy

    cd 3rdparty
    tar xf numpy-1.9.2.tar.gz
    cd numpy-1.9.2

    <!---check the INSTALL.txt of numpy to correctly choose between the "gnu"
    and "gnu95" fortran compilers-->
    python setup.py build --fcompiler=gnu95
    python setup.py install --prefix=../local/

#### cvxopt

    cd 3rdparty
    tar xf cvxopt-1.1.7.tar.gz
    cd cvxopt-1.1.7
    python setup.py build
    python setup.py install --prefix=../local/


### Optional dependencies

To produce the plots shown on the IPOL demo associated to the paper,
`matplotlib` is needed. The easiest way is probably to install it through your
favourite package manager, but if you want to compile it from source here are
the instructions:

    cd 3rdparty
    wget https://downloads.sourceforge.net/project/matplotlib/matplotlib/matplotlib-1.4.3/matplotlib-1.4.3.tar.gz
    tar xf matplotlib.tar.gz
    cd matplotlib
    python setup.py build
    python setup.py install --prefix=../local/


### Usage example

To run an attitude refinement simulation, set the parameters of the camera and
the image coordinates of the control points in the file `params.json`, as shown
in the file `params.json.example`. Then run the Python script
`run_attitude_refinement_simulation.py` with the path to the json file as unique
argument:

    cp params.json.example params.json
    python run_attitude_refinement_simulation.py params.json

This will print some debug info about the optimization on `stderr`, and print
the attitudes Root Mean Square Errors (RMSE, ie L2 norm of the error) on
`stdout`. If `matplotlib` is available, three plots will be saved to the files
`attitude_residuals.pdf`, `localization_errors.png` and
`attitude_estimated_vs_measured_vs_truth.pdf`
