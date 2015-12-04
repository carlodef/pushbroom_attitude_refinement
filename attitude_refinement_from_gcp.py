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
Attitudes refinement for orbiting pushbroom cameras with ground control points.

Copyright (C) 2015, Carlo de Franchis <carlo.de-franchis@m4x.org>
Copyright (C) 2015, Gabriele Facciolo <facciolo@cmla.ens-cachan.fr>
"""

from __future__ import print_function
import os
import sys
import copy
import numpy as np

import matplotlib
matplotlib.use('Agg')
import pylab

import utils
import pushbroom_simulator as ps


def roll_and_pitch_mapping_a_to_b(a, b):
    """
    Computes the roll and pitch needed to map a point on another, on a sphere.

    The two input points are given by their cartesian coordinates (x, y, z). The
    computed angles phi and psi are such that the rotation obtained by composing
    an elemental rotation of phi about the x-axis with an elemental rotation of
    psi about the new y-axis maps a to b.

    Args:
        a, b: two array-like objects of length 3.

    Returns:
        phi, psi: the two roll and pitch angles such that R_x(phi)R_y(psi)a = b.
    """
    # checks that the input 3d points are on the same sphere
    np.testing.assert_allclose(np.linalg.norm(a), np.linalg.norm(b))

    # compute the roll angle phi
    x = utils.solve_acos_plus_bsin_plus_c(b[1], b[2], -a[1])

    # compute the pitch angle psi
    bb = np.dot(np.linalg.inv(utils.elemental_rotation_3d('x', x)), b)
    y = np.arctan2(bb[0], bb[2]) - np.arctan2(a[0], a[2])

    # check that the composition of R_x(phi) and R_y(psi) maps a to b
    np.testing.assert_allclose(np.dot(utils.rotation_3d('xyz', x, y, 0), a), b)
    return x, y


def roll_and_pitch_from_world_to_image_correspondence(camera, correspondence):
    """
    Computes the instantaneous roll and pitch of an image row from one gcp.

    Args:
        camera: instance of the ps.PushbroomCamera class
        correspondence: list [r, c, h, lon, lat] which defines a correpondence
            between the pixel (r, c) at row r and column c and the 3-space point
            whose geographic coordinates, in degrees, are (lon, lat) and height
            above the Earth, in meters, is h.

    Returns:
        roll and pitch values, in radians
    """
    r, c, h, lon, lat = correspondence
    t = r * camera.instrument.dwell_time  # time associated to the row

    # a: coordinates of the '3d image point' located on the image plane
    R = utils.elemental_rotation_3d('z', np.polyval(camera.poly_psi, t))  # yaw
    a = np.dot(R, camera.instrument.sight_vector(c))

    # b: coordinates of the ground control point
    b = np.dot(camera.rotational_to_orbital(t).T,
               utils.geo_to_cartesian(lon, lat, h))
    b += np.array([0, 0, camera.orbit.radius])
    b /= np.linalg.norm(b)

    # compute the needed roll and pitch angles
    return roll_and_pitch_mapping_a_to_b(a, b)


def localization_error(camera_measured, camera_true, alt, subsampling=100):
    """
    Compute localization errors for a given camera wrt a given true camera.

    Args:
        camera_measured: measured camera
        camera_true: true camera, the error is computed with respect to that one
        alt: altitude (above the sea level) at which the localization error is
            computed
        subsampling (default 100): number of localization error samples

    Returns:
        list of localization errors.
    """
    out = []
    c = camera_true.instrument.n_pix / 2  # middle of the row
    for r in xrange(0, np.round(camera_true.lig_f).astype(int), subsampling):
        lon1, lat1 = camera_measured.locdir(r, c, alt)
        lon2, lat2 = camera_true.locdir(r, c, alt)
        radius = ps.PhysicalConstants.earth_radius + alt
        out.append(utils.haversine(radius, lon1, lat1, lon2, lat2))
    return out


def perturbate_camera(camera, amplitude, deg=3, perturbations_poly=None):
    """
    Add a polynomial perturbation to the attitudes of a camera.

    Args:
        camera: instance of the ps.PushbroomCamera class
        amplitude: amplitude, in rad, of the perturbation added to the
            camera roll and pitch attitude functions.
        deg (default 3): degree of the polynomial error added to the attitude
            functions
        perturbations_poly (default None): array_like of size Nx2.
            The first column for the roll and the second for the pitch. If not
            None, overwrites the 'deg' and 'amplitude' parameters.

    Returns:
        camera: a copy of the input camera, with modified attitudes
    """
    if perturbations_poly is None:
        # compute perturbation samples
        t = np.linspace(0, camera.duration, deg+1)
        x = amplitude * (2*np.random.rand(len(t), 2) - 1)

        # fit attitude polynomials
        try:
            perturbations_poly = np.polyfit(t, x, deg)
        except np.linalg.LinAlgError as e:
            print('polynomial fitting failed: ', e)
            return

    out = copy.deepcopy(camera)
    out.poly_phi += np.poly1d(perturbations_poly[:, 0])
    out.poly_theta += np.poly1d(perturbations_poly[:, 1])
    return out


def estimate_camera(gcp, measured_camera, max_attitude_error, model=3):
    """
    Estimates a camera from world to image correspondences and initialization.

    Args:
        gcp: array_like of size Nx5, each line contains a world to image
            correspondence given as (row, col, alt, lon, lat)
        measured_camera: instance of the ps.PushbroomCamera class, with noisy
            attitudes.
        max_attitude_error: maximal error in the noisy roll and pitch functions
            of the measured camera.
        model (default 3): degree of the polynoms used to fit the roll and pitch
            samples.

    Returns:
        an instance of the ps.PushbroomCamera class, with the estimated
        attitudes.
    """
    # compute roll and pitch sample values from the measured attitudes
    t = gcp[:, 0] * measured_camera.instrument.dwell_time
    measured_data = np.vstack([measured_camera.poly_phi(t),
                               measured_camera.poly_theta(t)]).T

    # compute roll and pitch sample values estimated from each correspondence
    estimated_data = []
    for m in gcp:
        phi, psi = roll_and_pitch_from_world_to_image_correspondence(measured_camera, m)
        estimated_data.append([phi, psi])
    estimated_data = np.array(estimated_data)

    # discard the estimated attitude samples that are too far away from the
    # measured ones
    corrections = estimated_data - measured_data
    ind = (np.abs(corrections) < max_attitude_error).all(axis=1)
    # keep only the lines [True, True]
    b = corrections[ind]
    n = ind.sum()

    out = copy.deepcopy(measured_camera)
    fitted_correction = np.array([[0, 0]])

    # estimate attitude correction coefficients by constrained linear
    # least-squares. The degree of the polynomial is 'model', but there are less
    # than model+1 points, the degree is lowered.
    if n:
        # the constraint is -h < Gx < h
        tt = np.linspace(0, measured_camera.duration)
        G = np.vander(tt, N=min(model+1, n), increasing=True)
        h = max_attitude_error * np.ones(len(G))

        # linear least squares: min |Ax - b|
        A = np.vander(t[ind], N=min(model+1, n), increasing=True)
        fitted_correction = np.zeros([min(model+1, n), 2])
        try:
            fitted_correction[:, 0] = utils.linear_least_squares(A, b[:, 0], G, h)[::-1]
            fitted_correction[:, 1] = utils.linear_least_squares(A, b[:, 1], G, h)[::-1]
        except np.linalg.LinAlgError as e:
            print('polynomial fitting failed: ', e, file=sys.stderr)
        except AssertionError as e:
            print('Assertion failed: ', e, file=sys.stderr)

    print('fitted_correction: ', fitted_correction, file=sys.stderr)
    out.poly_phi += np.poly1d(fitted_correction[:, 0])
    out.poly_theta += np.poly1d(fitted_correction[:, 1])

    return out, estimated_data


def simulate_single_image_problem(camera, points, deg=3, amplitude=50,
                                  perturbations_poly=None, sigma=None, model=3,
                                  gcpfile=None, plots=None):
    """
    Simulates the attitudes estimation problem for a single camera.

    The attitude corrections are estimated from a set of 3D-2D correspondences
    computed from the known camera model. The corrected attitudes are then
    compared to the true model to evaluate the performance of the estimation
    process.

    The 3D-2D correspondences are computed by back-projecting the image points
    at the given altitudes, with the given camera.

    Args:
        camera: instance of the ps.PushbroomCamera class.
        points: array_like of size Nx3. One triplet (i, j, h) per line,
            were i, j are the row, col pixel coordinates and h is the altitude.
        deg (default 3): degree of the perturbation polynomial.
        amplitude (default 50): amplitude, in radians, of the perturbation
            added to the camera roll and pitch attitude functions.
        perturbations_poly (default None): array_like of size Nx2.
            The first column for the roll and the second for the pitch. If not
            None, overwrites the 'deg' and 'amplitude' parameters.
        sigma (default None): tuple of length 2. The first element is the std
            deviation of the gaussian noise added to the two coordinates of the
            image points (in pixels), while the second is the std deviation of
            the gaussian noise added to the world points coordinates (in
            meters).
        model (default 3): degree of the polynoms used to fit the roll and pitch
            samples.
        gcpfile (default None): path to a txt file where to store the gcp
            generated for this experiment. It's an output file
        plots (default None): set to 'demo' or 'paper' to produce the needed
            matplotlib plots

    Returns
        list of length 6, containing the RMSE before and after correction of the
        roll, pitch (in micro radians) and localization (in meters).
    """
    # list of image to 3-space correspondences, stored as an Nx5 numpy array
    # each line contains (row, col, h, lon, lat)
    geo = camera.locdir_list(points)
    gcp = np.hstack([points, geo])
    n = len(gcp)

    # add gaussian noise to image and world points coordinates
    if sigma is not None:
        gcp[:, :2] += sigma[0] * np.random.randn(n, 2)
        gcp[:, 2] += sigma[1] * np.random.randn(n)
        r = ps.PhysicalConstants.earth_radius
        gcp[:, 3:] += np.degrees(sigma[1] / r) * np.random.randn(n, 2)

    if gcpfile is not None:
        np.savetxt(gcpfile, gcp)

    # estimate camera from world to image correspondences and noisy attitudes
    camera_measured = perturbate_camera(camera, amplitude, deg,
                                        perturbations_poly)
    camera_estimated, samples_estimated = estimate_camera(gcp, camera_measured,
                                                          amplitude, model)

    # distance to ground truth
    z = np.mean(points[:, 2])
    err_loc_i = localization_error(camera_measured, camera, z, subsampling=100)
    err_loc_f = localization_error(camera_estimated, camera, z, subsampling=100)
    err_roll_i = camera_measured.poly_phi - camera.poly_phi
    err_roll_f = camera_estimated.poly_phi - camera.poly_phi
    err_pitch_i = camera_measured.poly_theta - camera.poly_theta
    err_pitch_f = camera_estimated.poly_theta - camera.poly_theta

    cameras = (camera, camera_measured, camera_estimated)
    errors = (err_loc_i, err_loc_f)
    if plots == 'demo':
        plots_for_ipol_demo(cameras, samples_estimated, gcp, errors)
    if plots == 'paper':
        plots_for_ipol_paper(cameras, samples_estimated, gcp, errors)
        pts_x = points[:, 1] / camera.instrument.n_pix
        pts_y = [1 - y for y in points[:, 0] / camera.lig_f]
        plot_points(pts_x, pts_y, 'paper_figs/points.pdf')

    # compute the rmse
    out = []
    for x in [err_roll_i, err_roll_f, err_pitch_i, err_pitch_f]:
        out.append(1e6 * utils.poly_l2_norm(x, 0, camera.duration))
    for x in [err_loc_i, err_loc_f]:
        out.append(np.linalg.norm(x) / np.sqrt(len(x)))
    return out


def plots_for_ipol_demo(cameras, samples_estimated, gcp, errors):
    """
    Create plots shown in the online ipol demo.

    Args:
        cameras: tuple containing 3 camera objects (instances of the
            ps.PushbroomCamera), ordered as (true, measured, estimated)
        samples_estimated: roll, pitch samples estimated pointwise for each gcp
        gcp: list of ground control points, given as [row, col, alt]
        errors: tuple containing the initial and final errors, ordered as
            (error_i, error_f).
    """
    camera_true, camera_measured, camera_estimated = cameras
    t = gcp[:, 0] * camera_true.instrument.dwell_time
    tt = np.linspace(0, camera_true.duration, 100)
    samples_true = np.vstack([np.polyval(camera_true.poly_phi, t),
                              np.polyval(camera_true.poly_theta, t)]).T

    # plot roll/pitch differences in microradians
    pylab.subplot(211)
    gt = camera_true.poly_phi(tt)
    pylab.grid(True)
    pylab.ylabel('roll error (micro rad)')
    pylab.plot(t, 1e6*(samples_estimated[:, 0] - samples_true[:, 0]), 'bo')
    pylab.plot(tt, np.zeros(len(tt)), 'g-', label='True model')
    pylab.plot(tt, 1e6*(camera_measured.poly_phi(tt) - gt), 'r-', label='Measured model')
    pylab.plot(tt, 1e6*(camera_estimated.poly_phi(tt) - gt), 'b-', label='Estimated model')
    lgd = pylab.legend(loc='upper left', bbox_to_anchor=(1, 1))
    pylab.xlim(0, camera_true.duration)

    pylab.subplot(212)
    gt = camera_true.poly_theta(tt)
    pylab.grid(True)
    pylab.xlabel('time (s)')
    pylab.ylabel('pitch error (micro rad)')
    pylab.plot(t, 1e6*(samples_estimated[:, 1] - samples_true[:, 1]), 'bo')
    pylab.plot(tt, np.zeros(len(tt)), 'g-', label='True model')
    pylab.plot(tt, 1e6*(camera_measured.poly_theta(tt) - gt), 'r-', label='Measured model')
    pylab.plot(tt, 1e6*(camera_estimated.poly_theta(tt) - gt), 'b-', label='Estimated model')
    pylab.xlim(0, camera_true.duration)

    pylab.savefig('attitude_residuals.png', bbox_extra_artists=(lgd,),
                  bbox_inches='tight')
    pylab.clf()

    # plot roll true model, measured model and estimated model
    pylab.subplot(211)
    pylab.grid(True)
    pylab.ylabel('roll angle (rad)')
    pylab.plot(t, samples_true[:, 0], 'go')
    pylab.plot(t, samples_estimated[:, 0], 'bo')
    pylab.plot(tt, camera_true.poly_phi(tt), 'g-', label='True model')
    pylab.plot(tt, camera_measured.poly_phi(tt), 'r-', label='Measured model')
    pylab.plot(tt, camera_estimated.poly_phi(tt), 'b-', label='Estimated model')
    lgd = pylab.legend(loc='upper left', bbox_to_anchor=(1, 1))
    pylab.xlim(0, camera_true.duration)

    pylab.subplot(212)
    pylab.grid(True)
    pylab.xlabel('time (s)')
    pylab.ylabel('pitch angle (rad)')
    pylab.plot(t, samples_true[:, 1], 'go')
    pylab.plot(t, samples_estimated[:, 1], 'bo')
    pylab.plot(tt, camera_true.poly_theta(tt), 'g-', label='True model')
    pylab.plot(tt, camera_measured.poly_theta(tt), 'r-', label='Measured model')
    pylab.plot(tt, camera_estimated.poly_theta(tt), 'b-', label='Estimated model')
    pylab.xlim(0, camera_true.duration)

    pylab.savefig('attitude_estimated_vs_measured_vs_truth.png',
                  bbox_extra_artists=(lgd,), bbox_inches='tight')
    pylab.clf()

    # plot localization error
    error_i, error_f = errors
    tt = np.linspace(0, camera_true.duration, len(error_i))
    pylab.grid(True)
    pylab.xlabel('time (s)')
    pylab.ylabel('localization error (meters)')
    pylab.plot(tt, error_i, 'r-', label='Measured model')
    pylab.plot(tt, error_f, 'b-', label='Estimated model')
    lgd = pylab.legend(loc='upper left', bbox_to_anchor=(1, 1))
    pylab.xlim(0, camera_true.duration)
    pylab.savefig('localization_errors.png', bbox_extra_artists=(lgd,),
                  bbox_inches='tight')
    pylab.clf()


def plots_for_ipol_paper(cameras, samples_estimated, gcp, errors):
    """
    Create plots included in the ipol paper.

    Args:
        cameras: tuple containing 3 camera objects (instances of the
            ps.PushbroomCamera), ordered as (true, measured, estimated)
        samples_estimated: roll, pitch samples estimated pointwise for each gcp
        gcp: list of ground control points, given as [row, col, alt]
        errors: tuple containing the initial and final errors, ordered as
            (error_i, error_f).
    """
    if not os.path.isdir('paper_figs'):
        os.makedirs('paper_figs')

    camera_true, camera_measured, camera_estimated = cameras
    t = gcp[:, 0] * camera_true.instrument.dwell_time
    tt = np.linspace(0, camera_true.duration, 100)
    samples_true = np.vstack([np.polyval(camera_true.poly_phi, t),
                              np.polyval(camera_true.poly_theta, t)]).T

    pylab.clf()
    # plot roll/pitch differences in microradians
    pylab.grid(True)
    pylab.plot(t, 1e6*(samples_estimated[:, 0] - samples_true[:, 0]), 'bo')
    gt = camera_true.poly_phi(tt)
    pylab.plot(tt, np.zeros(len(tt)), 'g-')
    pylab.plot(tt, 1e6*(camera_measured.poly_phi(tt) - gt), 'r-')
    pylab.plot(tt, 1e6*(camera_estimated.poly_phi(tt) - gt), 'b-')
    pylab.xlim(0, camera_true.duration)
    pylab.savefig('paper_figs/roll_error.pdf', bbox_inches='tight')
    pylab.clf()

    pylab.grid(True)
    pylab.plot(t, 1e6*(samples_estimated[:, 1] - samples_true[:, 1]), 'bo')
    gt = camera_true.poly_theta(tt)
    pylab.plot(tt, np.zeros(len(tt)), 'g-', label='True model')
    pylab.plot(tt, 1e6*(camera_measured.poly_theta(tt) - gt), 'r-')
    pylab.plot(tt, 1e6*(camera_estimated.poly_theta(tt) - gt), 'b-')
    pylab.xlim(0, camera_true.duration)
    pylab.savefig('paper_figs/pitch_error.pdf', bbox_inches='tight')
    pylab.clf()

    # plot localization error
    error_i, error_f = errors
    tt = np.linspace(0, camera_true.duration, len(error_i))
    pylab.grid(True)
    pylab.xlim(0, camera_true.duration)
    pylab.plot(tt, error_i, 'r-')
    pylab.plot(tt, error_f, 'b-')
    pylab.savefig('paper_figs/localization_error.pdf', bbox_inches='tight')
    pylab.clf()


def plot_points(pts_x, pts_y, out=None):
    """
    Plot points as small red circles on a white background.
    """
    pylab.plot(pts_x, pts_y, marker='o', color='r', ls='')
    pylab.xlim(-.05, 1.05)
    pylab.ylim(-.05, 1.05)
    pylab.tick_params(axis='both',       # changes apply to both x and y axes
                      which='both',      # both major and minor ticks are affected
                      bottom='off',      # ticks along the bottom edge are off
                      top='off',
                      left='off',
                      right='off',
                      labelleft='off',
                      labelbottom='off') # labels along the bottom edge are off
    if out:
        pylab.savefig(out, bbox_inches='tight')
    else:
        pylab.savefig('points.png', bbox_inches='tight')
