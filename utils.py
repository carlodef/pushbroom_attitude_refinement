# -*- coding: utf-8 -*-
# pylint: disable=W0141
# pylint: disable=W0403
# pylint: disable=C0103

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
Utility functions for the pushbroom camera simulator and attitudes refinement
modules.

Copyright (C) 2015, Carlo de Franchis <carlo.de-franchis@m4x.org>
"""

from __future__ import print_function
import sys
import numpy as np
import cvxopt

import pushbroom_simulator as ps


def lon_lat(p):
    """
    Computes the lon, lat of a point given by its cartesian coordinates.

    The input coordinates are given in an Earth-centered frame.

    Args:
        p: numpy 1x3 array containing the coordinates of the input point
    """
    x = p[0]
    y = p[1]
    z = p[2]

    # latitude depends only on x*x + y*y and z, longitude only on y and x
    # the numpy arctan2 function takes care of all the border cases
    lat = np.arctan2(z, np.sqrt(x*x + y*y))
    lon = np.arctan2(y, x)

    return np.degrees(lon), np.degrees(lat)


def lon_lat_to_cartesian(r, lon, lat):
    """
    Args:
        lon, lat: in degrees
    """
    lon, lat = map(np.radians, (lon, lat))
    x = np.cos(lat) * np.cos(lon)
    y = np.cos(lat) * np.sin(lon)
    z = np.sin(lat)
    return r * np.array([x, y, z])


def geo_to_cartesian(lon, lat, h):
    """
    Args:
        lon,lat: in degrees
        h: in meters, above the Earth spherical surface
    """
    r = ps.PhysicalConstants.earth_radius
    out = lon_lat_to_cartesian(r, lon, lat)
    return (1 + h / np.linalg.norm(out)) * out


def rotational_to_local_geographic(lon, lat):
    """
    Computes change of basis matrix from rotational to local geographic frame.

    Args:
        lon, lat: longitude and latitude of the current point on the ground, in
            degrees

    Returns:
        numpy 3x3 array representing the change of basis matrix
    """
    return rotation_3d('zyx', lon * np.pi/180, -lat * np.pi/180, 0)


def intersect_line_and_plane(v, p, n):
    """
    Computes the intersection between a line and a plane.

    The line passes through the origin and is defined by a 3-space vector. The
    plane is defined by a normal vector and a point.

    Args:
        v: 3-space vector directing the line
        n: 3-space vector normal to the plane
        p: 3-space point located on the plane
        All the arguments are 1x3 numpy arrays

    Returns:
        1x3 numpy array containing the coordinates of the intersection point
    """
    t = np.dot(p, n) / np.dot(v, n)
    return t * v


def intersect_line_and_sphere(u, c, r):
    """
    Computes the first intersection between a directed line and a sphere.

    The line passes through the origin and is defined by a 3-space vector. The
    sphere is defined by its center and radius. The problem is solved by
    searching for a value of t such that || -c + t*u || = r.

    This leads to the equation x*t^2 + y*t + z = 0, where x is the squared norm
    of u, y is -2*np.dot(u, c), and z is the squared norm of c minus the square
    of r.

    Args:
        u: coordinates of a 3-space vector that gives the direction of the line
        c: coordinates of the center of the sphere
        r: radius of the sphere

    Returns:
        the 3-space coordinates of the first intersection point. If there is no
        intersection, returns None.
    """
    x = np.linalg.norm(u)**2
    y = -2 * np.dot(u, c)
    z = np.linalg.norm(c)**2 - r*r

    if y*y - 4*x*z < 0:
        # no real solutions, ie no intersection
        return None
    else:
        poly = np.poly1d([x, y, z])
        t = min(poly.r)
        if t < 0:
            # one intersection point is behind us, ie wrong direction of u
            return None
        else:
            return t * np.array(u)


def elemental_rotation_3d(axis, t):
    """
    Args:
        axis: 'x', 'y', or 'z'
        t: angle in radians
    """
    c = np.cos(t)
    s = np.sin(t)
    if axis == 'x':
        return np.array([[1, 0, 0],
                         [0, c, -s],
                         [0, s, c]])
    if axis == 'y':
        return np.array([[c, 0, s],
                         [0, 1, 0],
                         [-s, 0, c]])
    if axis == 'z':
        return np.array([[c, -s, 0],
                         [s, c, 0],
                         [0, 0, 1]])
    return np.array([[1, 0, 0],
                     [0, 1, 0],
                     [0, 0, 1]])


def rotation_3d(axes, roll, pitch, yaw):
    """
    Computes the matrix of the 3-space rotation defined by 3 intrinsic angles.

    The input angles define elemental rotations R1, R2, and R3 about the 3 axes
    of an euclidian basis oriented using the right hand rule. The product
    R1*R2*R3 of these elemental rotations represents a rotation whose intrinsic
    roll, pitch and yaw angles are those given as input. The axes associated to
    roll, pitch and yaw are given by the first argument. For examples if the
    first argument is 'xyz' then the rotation is the one obtained by applying
    first the roll about the x axis, then the pitch about the new y axis, and
    last the yaw about the new new z axis.

    Args:
        axes: string that can take six different values: 'xyz', 'xzy', 'yxz',
            'yzx', 'zxy' or 'zyx'.
        roll, pitch, yam: three angles

    Returns:
        numpy 3x3 array representing the output rotation
    """
    if axes not in ['xyz', 'xzy', 'yxz', 'yzx', 'zxy', 'zyx']:
        print("ERROR rotation_3d: bad value for axes argument")
        return
    r1 = elemental_rotation_3d(axes[0], roll)
    r2 = elemental_rotation_3d(axes[1], pitch)
    r3 = elemental_rotation_3d(axes[2], yaw)
    return np.dot(r1, np.dot(r2, r3))


def haversine(r, lon1, lat1, lon2, lat2):
    """
    Computes the great circle distance between two points on a sphere from
    their longitudes and latitudes.

    Args:
        r: sphere radius
        lon1, lat1: geographic coordinates of the first point, in degrees
        lon2, lat2: geographic coordinates of the second point, in degrees

    Returns:
        distance, in the same unit as the input radius
    """
    # convert degrees to radians
    lon1, lat1, lon2, lat2 = map(np.radians, [lon1, lat1, lon2, lat2])

    # haversine formula
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = np.sin(dlat/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2)**2
    return r * 2*np.arcsin(np.sqrt(a))


def linear_least_squares(A, b, G=None, h=None):
    """
    Finds a vector x that minimizes the euclidean norm of Ax-b.

    An optional linear inequality constraint can be imposed: the output x will
    be such that -h <= Gx <= h component-wise. It uses cvxopt.solvers.qp, which
    seems to be equivalent to Matlab lsqlin.

    Args:
        A, b: array_like of size (n, m) and (n, 1)
        G, h (optional): array_like of size (p, m) and (p, 1)

    Returns:
        the optimal x
    """
    P = np.dot(A.T, A)
    q = -np.dot(A.T, b)
    cvxopt.solvers.options['show_progress'] = False
    cvxopt.solvers.options['abstol'] = 1e-25
    cvxopt.solvers.options['reltol'] = 1e-25
    cvxopt.solvers.options['feastol'] = 1e-15

    if G is None:
        out = cvxopt.solvers.qp(cvxopt.matrix(P), cvxopt.matrix(q))
        x = np.array(out['x']).flatten()
        np.testing.assert_allclose(x, np.linalg.lstsq(A, b)[0])
    else:
        GG = np.vstack([G, -G])
        hh = np.hstack([h, h])
        out = cvxopt.solvers.qp(cvxopt.matrix(P), cvxopt.matrix(q),
                                cvxopt.matrix(GG), cvxopt.matrix(hh))
        x = np.array(out['x']).flatten()
        try:
            np.testing.assert_array_less(np.dot(G, x), h)
            np.testing.assert_array_less(-h, np.dot(G, x))
        except AssertionError as e:
            print('Assertion failed: ', e, file=sys.stderr)

    print("cvxopt optimization status: ", out['status'], file=sys.stderr)
    print("linear_least_squares min: ", np.linalg.norm(np.dot(A, x) - b),
          file=sys.stderr)
    return x


def poly_l2_norm(p, a, b):
    """
    Computes the L2 norm of a polynomial function on a given interval.

    Args:
        p: np.poly1d object
        a, b: two float defining the intervals. If a > b, they are swapped.

    Returns:
        the square root of the integral of the square of p over [a, b] or [b, a]
    """
    if a > b:
        a, b = b, a

    q = p * p
    Q = q.integ()  # use of the np.poly1d.integ method
    return np.sqrt(Q(b) - Q(a))


def solve_acos_plus_bsin_plus_c(a, b, c):
    """
    Solves a*cos(x) + b*sin(x) + c = 0 for x in [-pi/2, pi/2].

    The method used is to change variable by setting t = sin(x).

    Args:
        a, b, c: coefficients of the equation

    Returns:
        A solution x if it exists, None in the other cases.
    """
    out = None
    if a*a + b*b - c*c > 0:
        poly = np.poly1d([a*a + b*b, 2*b*c, c*c - a*a])
        if not any([-1 <= t <= 1 for t in poly.r]):
            print('no solutions between -1 and 1: ', poly.r)
        elif all([-1 <= t <= 1 for t in poly.r]):
            i = np.argmin(np.abs(np.array([a*np.sqrt(1-t*t) + b*t + c for t in
                                           poly.r])))
            out = np.arcsin(poly.r[i])
        else:
            if -1 <= poly.r[0] <= 1:
                out = np.arcsin(poly.r[0])
            else:
                np.arcsin(poly.r[1])

    # check that the solution is in [-pi/2, pi/2] and satisfies the equation
    assert -np.pi/2 < out < np.pi/2
    np.testing.assert_allclose(a*np.cos(out) + b*np.sin(out) + c, 0, atol=1e-12)
    return out
