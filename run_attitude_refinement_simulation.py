#!/usr/bin/env python
# -*- coding: utf-8 -*-
# pylint: disable=C0103
# pylint: disable=W0403

# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published
# by the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Affero General Public License for more details.
#
# You should have received a copy of the GNU Affero General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.

"""
Attitudes refinement for orbiting pushbroom cameras.

This is the main module, used to run a simulation of the pushbroom attitude
refinement problem from one single image with ground control points.

Copyright (C) 2015, Carlo de Franchis <carlo.de-franchis@m4x.org>
"""

import sys
import json
import numpy as np

import attitude_refinement_from_gcp as arfg
import pushbroom_simulator as ps


# standard pushbroom cameras params
# sources:
#   Pleiades Imagery User Guide v2.0 (october 2012)
#   http://www.exelisvis.com/docs/ParametersForDigitalCamerasPushbroomSensors.html
#   http://www.satimagingcorp.com/satellite-sensors/worldview-{1,2,3}
#   https://directory.eoportal.org/web/eoportal/satellite-missions/v-w-x-y-z/worldview-{1,2,3}
STD_INSTRUMENTS = {
    'pleiades': {'f': 12.905,
                 'w': 13,
                 'n': 30000,
                 'dwell_time': 0.00007385}, # from our Pleiades dimap files
    'wv1': {'f': 8.8,
            'w': 8,
            'n': 35000,
            'dwell_time': 0.0001},  # missing info
    'wv2': {'f': 13.311,
            'w': 8,
            'n': 35000,
            'dwell_time': 0.0001},  # missing info
    'wv3': {'f': 12.9,  # missing info
            'w': 8,  # missing info
            'n': 35000,  # missing info
            'dwell_time': 0.0001}  # missing info
}

STD_ORBITS = {
    'pleiades': {'a': 694,
                 'i': 98.2,
                 'lon': 30},
    'wv1': {'a': 496,
            'lon': 30},
    'wv2': {'a': 770,
            'lon': 30},
    'wv3': {'a': 617,
            'lon': 30}
}

STD_VIEW = {'duration': 2.5, 'pso_0': 180}


def main(params_file, gcpfile=None):
    """
    This is an example of json parameters file:

    {
      "camera": {
        "instrument": "pleiades",
        "orbit": "pleiades",
        "view": { "psi_x": 5, "psi_y": 1, "gamma": 192 }
      },
      "normalized_points" : False,
      "points" : [[0, 0], [1000, 3000], [2000, 6000], [3000, 9000]],
      "sigma" : [0.2, 0.1],
      "perturbation_degree": 3,
      "perturbation_amplitude": 50
    }
    """
    # read the json configuration file
    f = open(params_file)
    params = json.load(f)
    f.close()

    # define the camera
    c = params['camera']
    if type(c['instrument'] == str):
        c['instrument'] = STD_INSTRUMENTS[c['instrument']]
    if type(c['orbit'] == str):
        c['orbit'] = STD_ORBITS[c['orbit']]
    c['view'].update(STD_VIEW)
    cam = ps.instantiate_camera(c)

    # prepare control points by sampling random altitudes with 1000m amplitude
    pts = np.array(params['points'])
    pts = np.hstack([pts, 1000 * np.random.rand(pts.shape[0], 1)])
    if params.has_key('normalized_points') and params['normalized_points']:
        nrows = c['view']['duration'] / c['instrument']['dwell_time']
        ncols = c['instrument']['n']
        pts[:, 0] *= nrows
        pts[:, 1] *= ncols

    # run the simulation
    sigma = params['sigma'] if params.has_key('sigma') else (0, 0)
    a = params['perturbation_amplitude']
    d = params['perturbation_degree']
    out = arfg.simulate_single_image_problem(cam, pts, deg=d, amplitude=a,
                                             sigma=sigma, plots='demo',
                                             gcpfile=gcpfile)

    # print outputs
    print 'RMSE (in microrad) of the roll (measured, ie before correction): ', out[0]
    print '---- --- --------- -- --- ---- (estimated, ie after correction): ', out[1]
    print '---- --- --------- -- --- pitch (measured): ', out[2]
    print '---- --- --------- -- --- ----- (estimated): ', out[3]
    print 'RMSE (in meters) of the localization (measured): ', out[4]
    print '---- --- ------- -- --- ------------ (estimated): ', out[5]


if __name__ == '__main__':

    if len(sys.argv) == 2:
        main(sys.argv[1])
    elif len(sys.argv) == 3:
        main(sys.argv[1], sys.argv[2])
    else:
        print """
        Incorrect syntax, use:
          > %s params.json [gcpfile]

          Launches a simulation of the pushbroom attitude estimation problem
          from one single image with ground control points.
        """ % sys.argv[0]
        sys.exit(1)
