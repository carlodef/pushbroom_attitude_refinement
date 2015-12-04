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
Orbiting pushbroom camera simulator adapted from the module SIMI-PDV_v1.2.py
of Daniel Greslou <daniel.greslou@cnes.fr>

Copyright (C) 2015, Carlo de Franchis <carlo.de-franchis@m4x.org>

Five different reference frames may be used in this simulator. All of them
are used with a Cartesian coordinate system described below:

* Earth-centered inertial frame:
    z: the Earth rotation axis, oriented towards the North pole,
	  x: lies in the equatorial plane and is permanently fixed in a direction
	     relative to the celestial sphere. It does not rotate with the Earth.

* Earth-centered rotational frame:
    is equal to the Earth-centered inertial frame at t=0, then rotates with
    the Earth around its z axis.

* local geographic frame:
    x: directed towards zenith (ie normal to the ground, pointing to the sky)
    y: directed towards east
    z: directed towards north
    The orientation of the local geographic frame depends on its
    position.

* orbital frame:
	  z: directed towards the center of the Earth
	  x: given by the satellite movement.

* camera frame:
    obtained from the orbital frame by applying the composition of 3
    rotations:
      - roll (rotation about x axis), angle phi,
      - pitch (rotation about y axis), angle theta,
      - yaw (rotation about z axis), angle psi.
	  It is equal to the orbital frame when the attitude (ie the triplet roll,
	  pitch, yaw) is zero.
"""

import numpy as np

import utils


class PhysicalConstants(object):
    """
    Container for physical constants.
    """
    earth_radius = 6378000.0  # meters
    earth_gravitational_parameter = 398600.6  # km^3/s^2
    earth_rotation_rate = 2*np.pi / (86164.10)  # rad/s, stellar day
    earth_revolution_rate = 2*np.pi / (365.256*24*3600)  # rad/s
    earth_j2 = 1.08e-3  # source fr.wikipedia.org/wiki/Orbite_héliosynchrone

    @staticmethod
    def inertial_to_rotational(t):
        """
        Computes the change of basis matrix from inertial to rotational frame.

        Args:
            t: time in seconds from a moment where rotational and inertial
                frames were equal.

        Returns:
            numpy 3x3 array representing the change of basis matrix
        """
        a = t * PhysicalConstants.earth_rotation_rate
        s = np.sin(a)
        c = np.cos(a)
        return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])


class Instrument(object):
    """
    Simple model of a pushbroom sensor.

    Attributes:
        focal: focal length in meters
        w_pix: side length of a pixel, in micrometers. Pixels are assumed to be
            squares.
        n_pix: number of pixels in the pushbroom array
        dwell_time: time spent capturing each row, in seconds
        offset: offset of the pushbroom array on the x axis with respect to the
            camera frame, in micrometers
    """

    def __init__(self, f, w, n, dwell_time, offset=0):
        """
        Inits Instrument with given attribute values
        """
        self.focal = float(f)
        self.w_pix = float(w)
        self.n_pix = int(n)
        self.dwell_time = float(dwell_time)
        self.offset = float(offset)


    def sight_vector(self, c):
        """
        Computes the sight direction of a given pixel of the pushbroom array.

        Args:
            c: index of the pixel, ie integer between 1 and n_pix

        Returns:
            numpy array of shape (1, 3), representing a unit 3-space vector.
        """
        x = self.offset
        y = self.w_pix * (c - 0.5*self.n_pix)
        z = self.focal * 1e6  # convert meters to micrometers
        v = np.array([x, y, z])
        return v / np.linalg.norm(v)


class CircularOrbit(object):
    """
    Circular orbit around the Earth.

    A circular orbit is a circular path followed by a physical body traveling
    around the Earth. Not only the radius, but also the speed, angular speed,
    potential and kinetic energy of the physical body are constant. A circular
    orbit is completely determined by its radius, inclination and ascending
    node.

    Attributes:
        altitude: orbit altitude above the Earth, in meters
        radius: orbit radius, ie Earth radius plus altitude, in meters
        period: orbit temporal period, in seconds. It is computed from the
            orbit radius
        omega: angular speed of the satellite, in radians per second
        i: orbit inclination from the equatorial plane, in radians
        lon: longitude of the ascending node in the inertial frame, in degrees
    """

    def __init__(self, a, i, lon=0):
        """
        Computes all CircularOrbit attributes from the 3 input arguments.

        Args:
            a: orbit altitude, in km
            i: orbit inclination, in degrees
            lon (optional, default is 0): longitude of the reference node, in
                degrees
        """
        self.altitude = float(a) * 1000  # meters
        self.radius = PhysicalConstants.earth_radius + self.altitude

        # computation of the orbit temporal period
        mu = PhysicalConstants.earth_gravitational_parameter
        r = self.radius / 1000  # in km, to match mu
        self.period = 2 * np.pi * np.sqrt(r**3 / mu)
        self.omega = 2 * np.pi / self.period  # angular speed in rad/s

        self.i = i * np.pi/180.0
        self.lon = lon


    def inertial_to_orbital(self, pso):
        """
        Computes the change of basis matrix from inertial to orbital frame.

        If X is the coordinate vector of a 3-space vector expressed in the
        inertial frame, and X' is the coordinate vector of the same 3-space
        vector expressed in the orbital frame, then the matrix P computed by
        this function is such that X = PX'.

        The change of basis is a simple rotation computed from the 3 angles
        lon, i and pso. The orbital plane precession is neglected.

        Args:
           pso: satellite angular position, in degrees.
        """
        # first rotation about z axis. It sends the x axis to reference lon.
        lon = self.lon * np.pi/180

        # second rotation about the new x axis, to send the z axis in the
        # orbital plane. The angle between the xz plane (x stands for the new x
        # axis) and the orbital plane is i - pi/2
        inc = self.i - np.pi/2

        # third rotation about the new y axis. It sends the new x axis to
        # the Earth-Satellite axis. The angle about y is given by -pso.
        # Additional rotation of -pi/2 about the same axis, such that the new z
        # axis points towards the Earth center.
        pos = -pso * np.pi/180 - np.pi/2
        return utils.rotation_3d('zxy', lon, inc, pos)


    def target_point_orbital_frame(self, a, b, alt=0):
        """
        Computes orbital coordinates of the ground point targeted by the
        satellite.

        Args:
            a: sight direction angle about the x axis of orbital frame, degrees
            b: angle about the y axis of orbital frame, degrees
            alt: altitude of the ground above the sphere

        Returns:
            numpy 1x3 array with the coordinates in the orbital frame
        """
        # Earth center position in the orbital frame
        c = np.array([0, 0, self.radius])

        # coordinates of the vector defined by angles a, b and coordinate z=1
        # in the orbital frame
        a_rad = a * np.pi/180
        b_rad = b * np.pi/180
        v = np.array([np.tan(b_rad), -np.tan(a_rad), 1])

        # intersection of the line directed by v with the Earth
        r = PhysicalConstants.earth_radius + alt
        return utils.intersect_line_and_sphere(v, c, r)


class SunSynchronousCircularOrbit(CircularOrbit):
    """
    Circular Sun-synchronous orbit around the Earth.

    A Sun-synchronous orbit is a geocentric orbit which combines altitude and
    inclination in such a way that an object on that orbit will appear to orbit
    in the same position, from the perspective of the Sun, during its orbit
    around the Earth. Or in other words orbit in such a way that it precesses
    once a year.

    A Sun-synchronous circular orbit is completely determined by its radius and
    ascending node. Its inclination is computed from the radius.

    Attributes:
        same attributes as the CircularOrbit class
    """

    def __init__(self, a, lon=0):
        """
        Computes the orbit inclination then creates a CircularOrbit instance.

        Args:
            a: orbit altitude, in km
            lon (optional, default is 0): longitude of the reference node, in
                degrees
        """
        r = PhysicalConstants.earth_radius / 1000  # in kilometers
        m = PhysicalConstants.earth_gravitational_parameter
        o = PhysicalConstants.earth_revolution_rate
        j = PhysicalConstants.earth_j2
        x = 1 + float(a)/r

        # formula from http://fr.wikipedia.org/wiki/Orbite_héliosynchrone
        i = 180/np.pi * np.arccos(-2.0/3.0 * o/j * x**3.5 * np.sqrt(r**3/m))

        CircularOrbit.__init__(self, a, i, lon)


class PushbroomCamera(object):
    """
    Pushbroom camera model for satellites.

    A pushbroom camera is defined by the instrument, the orbit, the duration of
    the acquisition, the position of the satellite on the orbit when the
    acquisition starts, and the evolution of the attitude angles over time.

    This is the main class of this simulator.

    Attributes:
        instrument: instance of the Instrument class
        orbit: instance of the SunSynchronousCircularOrbit class
        duration: duration of the acquisition, in seconds
        pso_0: angular position of the satellite on the circular orbit, at the
            beginning of the acquisition, in degrees. A null angle corresponds
            to the reference node. The orbit is oriented by the satellite
            motion.
        phi: list of coefficients [c_n, c_n-1, ..., c_0] of the polynom which
            defines the evolution of the roll function over time. The initial
            roll angle phi(0) is equal to c_0.
        theta: coefficients of the polynomial pitch function
        psi: coefficients of the polynomial yaw function
    """

    def __init__(self, instrument, orbit, duration, pso_0, psi_x, psi_y, cap,
                 dt=0.1, d_poly=3, slow_factor=1.0, alt=0.0):
        """
        Inits PushbroomCamera with the given attribute values.

        The attitude functions are estimated in such a way that they satisfy 3
        input parameters: the initial roll and pitch angles and pushbroom array
        direction on the ground.

        Arguments:
            instrument, orbit, duration, pso_0: see attributes description in
                the class docstring.
            psi_x: initial angle of the camera frame, with respect to the
                orbital frame, about the x axis, in degrees (roll).
            psi_y: initial angle of the camera frame, with respect to the
                orbital frame, about the y axis, in degrees (extrinsic pitch).
            cap: direction of the pushbroom array movement, on the ground, in
                degrees. North is 0, Sud is 180 and East is 90.
            dt (optional, default is 0.1): temporal sampling step used to
                estimate the ordered attitude functions (ie the attitude
                functions derived from psi_x, psi_y and cap).
            d_poly (optional, default is 3): degree of the polynomials used
                to interpolate the attitude functions from the samples
            slow_factor: slow down factor applied to the speed of the
                projection of the pushbroom array on the ground.
            alt: altitude of the ground, in addition to the Earth radius.
        """
        self.instrument = instrument
        self.orbit = orbit
        self.duration = duration
        self.pso_0 = pso_0
        self.col_i = 0
        self.col_f = instrument.n_pix
        self.lig_i = 0
        self.lig_f = np.floor(duration / self.instrument.dwell_time)

        # attitudes evolution computation
        t, phi, theta, psi = self.guidance_algorithm(psi_x, psi_y, cap, dt,
                                                     alt, slow_factor)
        self.poly_psi = np.poly1d(np.polyfit(t, psi, d_poly))
        self.poly_phi = np.poly1d(np.polyfit(t, phi, d_poly))
        self.poly_theta = np.poly1d(np.polyfit(t, theta, d_poly))


    def pso_from_date(self, t):
        """
        Compute the pso (Position sur Orbite) of the satellite at a given time.

        Args:
            t: time in seconds from the beginning of the acquisition

        Returns:
            satellite angular position (degrees) on its orbit at time t
        """
        return self.pso_0 + self.orbit.omega * t * 180.0/np.pi


    def orbital_to_camera(self, t):
        """
        Computes the change of basis matrix from orbital to camera frame.

        This change of basis is a rotation determined by roll, pitch and yaw
        angles, which are computed from t.

        Args:
            t: time in seconds from the beginning of the acquisition

        Returns:
            numpy 3x3 array representing the change of basis matrix
        """
        phi = np.polyval(self.poly_phi, t)
        theta = np.polyval(self.poly_theta, t)
        psi = np.polyval(self.poly_psi, t)
        return utils.rotation_3d('xyz', phi, theta, psi)


    def rotational_to_orbital(self, t):
        """
        Computes the change of basis matrix from rotational to orbital frame.

        Args:
            t: time in seconds from the beginning of the acquisition

        Returns:
            numpy 3x3 array representing the change of basis (rotation) matrix
        """
        pso = self.pso_from_date(t)
        a = self.orbit.inertial_to_orbital(pso)
        b = PhysicalConstants.inertial_to_rotational(t)
        return np.dot(b.T, a)


    def pushbroom_direction_on_the_ground(self, cap, p, t):
        """
        Computes the projection of the pushbroom direction on the ground.

        Args:
            cap: direction of the pushbroom array movement, on the ground, in
                degrees. North is 0, Sud is 180 and East is 90.
            p: orbital coordinates of the ground point currently targeted
            t: time elapsed since the beginning of the acquisition, in seconds

        Returns:
            the orbital coordinates of a vector giving the pushbroom movement
            direction, projected on the ground.
        """
        # coordinates of the Earth center in the orbital frame
        c = np.array([0, 0, self.orbit.radius])

        # direction of the pushbroom movement, projected on the ground, in
        # local geographic coordinates (z, easting, northing)
        v = np.array([0, np.sin(cap * np.pi/180), np.cos(cap * np.pi/180)])

        # convert v from local geographic coordinates to orbital coordinates
        rot_to_orb = self.rotational_to_orbital(t)
        p_rot = np.dot(rot_to_orb, -c + p)
        lon, lat = utils.lon_lat(p_rot)
        rot_to_geo = utils.rotational_to_local_geographic(lon, lat)
        return np.dot(rot_to_orb.T, np.dot(rot_to_geo, v))


    def attitude_from_point_and_speed(self, m, v):
        """
        Derives the attitude from a targeted point and a pushbroom direction.

        The roll, pitch, yaw angles transforming the orbital frame into the
        camera frame are computed. The camera frame is defined by a sight
        direction (given by the ground point m) and the normal vector to the
        projection of the pushbroom array on the ground (given by v). Rotations
        have to be applied in this order: roll, then pitch, then yaw.

        Args:
            m: point on Earth, given by its coordinates in the orbital frame
            v: movement of the pushbroom array, projected on the ground, given
                in the orbital frame. It has to be a unit vector.

        Returns:
            phi, theta, psi: attitude angles in radians
        """
        # computation of phi and theta is straightforward (do a drawing)
        phi = -np.arctan(m[1] / m[2])
        theta = np.arcsin(m[0] / np.linalg.norm(m))

        # computation of psi
        # normal vector to the ground in m
        n = -np.array([0, 0, self.orbit.radius]) + m
        n = n / np.linalg.norm(n)

        # vector orthogonal to v, on the ground
        w = np.cross(n, v)

        rot = utils.rotation_3d('xyz', phi, theta, 0)
        x = np.dot(rot, np.array([1, 0, 0]))
        y = np.dot(rot, np.array([0, 1, 0]))
        xx = np.cross(y, n)  # projection of x on the tangent plane
        yy = np.cross(x, n)  # projection of y on the tangent plane

        if np.dot(w, yy) == 0:
            psi = np.pi/2
        else:
            psi = np.arctan(np.dot(w, xx) / np.dot(w, yy))

        sp = np.sin(psi)
        cp = np.cos(psi)

		# correction
        sgcosgammar = np.dot(v, sp*yy - cp*xx)
        if sgcosgammar < 0:
            if psi < 0:
                psi += np.pi
            else:
                psi -= np.pi

        return phi, theta, psi


    def pixel_projection_width(self, m, x=0, y=0):
        """
        Computes the width of the projection of a pixel on the ground.

        Args:
            m: numpy 3x3 array representing the change of
               coordinates matrix between orbital frame and camera frame
            x, y (optional, default is 0, 0): coordinates of the pixel in the
                image plane

        Returns:
            width, in meters, of the projection of the pixel on the ground, in
            the pushbroom direction (orthogonal to the pushbroom movement)
        """
        # pixel width and focal length in mm
        w = self.instrument.w_pix * 0.001
        f = self.instrument.focal * 1000

        # direction of the camera, expressed in the orbital frame
        v = np.dot(m, np.array([-x, -y, f]))

        # intersection of the line directed by v with the Earth
        c = np.array([0, 0, self.orbit.radius])
        r = PhysicalConstants.earth_radius
        p = utils.intersect_line_and_sphere(v, c, r)

        # normal to the tangent plane to the Earth at the intersection point p
        n = p - c

        # projection of the left and right borders of the pixel on the ground
        v1 = np.dot(m, np.array([-(x + .5*w), -y, f]))
        v2 = np.dot(m, np.array([-(x - .5*w), -y, f]))
        p1 = utils.intersect_line_and_plane(v1, p, n)
        p2 = utils.intersect_line_and_plane(v2, p, n)

        return np.linalg.norm(p1 - p2)


    def guidance_algorithm(self, psi_x, psi_y, cap, dt, alt=0, slow_factor=1):
        """
        Computes attitude samples used to estimate the attitude functions.

        Args:
            psi_x: sight direction angle about the x axis of orbital frame, degrees
            psi_y: sight direction angle about the y axis of orbital frame, degrees
            cap: direction of the pushbroom array movement, on the ground, in
                degrees. North is 0, Sud is 180 and East is 90.
            dt: temporal sampling step
            alt (default 0): altitude used to compute the attitude samples
            slow_factor (default 1): slow down factor applied to the speed of the
                projection of the pushbroom array on the ground.

        Returns:
            t: 1d np.array containing the dates at which the attitudes are
                sampled.
            phi, theta, psi: three 1d np.array containing the roll, pitch and
                yaw samples.
        """
        # number of samples used to interpolate the attitude angles
        n = int(self.duration / dt) + 1
        theta = np.zeros(n)
        phi = np.zeros(n)
        psi = np.zeros(n)

        # dates at which the attitude angles are sampled
        t = np.linspace(0, (n-1)*dt, n)

        # p is the point on Earth spotted by the satellite, in orbital frame
        p = self.orbit.target_point_orbital_frame(psi_x, psi_y, alt)

        for k in range(n):
            # direction of pushbroom motion, on the ground, in orbital frame
            v = self.pushbroom_direction_on_the_ground(cap, p, t[k])

            # attitude
            phi[k], theta[k], psi[k] = self.attitude_from_point_and_speed(p, v)
            orb_to_cam = utils.rotation_3d('xyz', phi[k], theta[k], psi[k])

            # speed of the pushbroom motion, on the ground
            norm = self.pixel_projection_width(orb_to_cam)
            norm /= (self.instrument.dwell_time * slow_factor)

            # update the targeted point
            p += v * norm * dt
            p -= np.array([0, 0, self.orbit.radius])
            p = np.dot(self.rotational_to_orbital(t[k]), p)
            p = np.linalg.solve(self.rotational_to_orbital(t[k]+dt), p)
            p += np.array([0, 0, self.orbit.radius])

        return t, phi, theta, psi


    def locdir_at(self, phi, theta, psi, row, col, alt=0, geo=True):
        """
        Given the satellite attitude, computes lon, lat of point row, col, alt.

        Args:
            phi, theta, psi: attitude angles of the satellite camera
            row, col: pixel coordinates in the image plane
            alt: altitude of the 3d point above the Earth surface
            geo: boolean flag telling whether to return geographic
                coordinates or Cartesian coordinates.

        Returns:
            geographic coordinates (lon, lat), or Cartesian coordinates
            (x, y, z). Cartesian coordinates are computed with respect to the
            orbital frame.
        """
        # first compute coordinates of the 3-space point in the orbital frame
        orbital_to_camera = utils.rotation_3d('xyz', phi, theta, psi)
        v = np.dot(orbital_to_camera, self.instrument.sight_vector(col))
        c = np.array([0, 0, self.orbit.radius])
        r = PhysicalConstants.earth_radius + alt
        p = utils.intersect_line_and_sphere(v, c, r)
        if not geo:
            return p

        # then convert them to lon, lat, using t to get the satellite position
        if p is None:
            return p
        else:
            t = row * self.instrument.dwell_time
            mat_t_ol = self.rotational_to_orbital(t)
            return utils.lon_lat(np.dot(mat_t_ol, -c + p))


    def locdir(self, row, col, alt=0, geo=True):
        """
        Computes the 3-space coordinates of a point given its row, col and alt.

        Args:
            row, col: pixel coordinates in the image plane
            alt: altitude of the 3D point above the Earth surface
            geo: boolean flag telling whether to return geographic
                coordinates or Cartesian coordinates.

        Returns:
            geographic coordinates (lon, lat), or Cartesian coordinates
            (x, y, z). Cartesian coordinates are computed with respect to the
            orbital frame.
        """
        t = row * self.instrument.dwell_time
        phi = np.polyval(self.poly_phi, t)
        theta = np.polyval(self.poly_theta, t)
        psi = np.polyval(self.poly_psi, t)
        return self.locdir_at(phi, theta, psi, row, col, alt, geo)


    def locdir_list(self, pixels_with_altitudes):
        """
        Calls locdir on all the triplets (r, c, h) of the input list

        Args:
            pixels_with_altitudes: Nx3 numpy array, each line (r, c, h)
                containing the coordinates row, col of a pixel together with an
                altitude h.

        Returs:
            Nx2 numpy array, each line containing the (lon, lat) coordinates of
            the point defined by (r, c, h).

        """
        out = []
        for p in pixels_with_altitudes:
            lon, lat = self.locdir(p[0], p[1], p[2])
            out.append([lon, lat])
        return np.array(out)



def instantiate_camera(p):
    """
    Builds a camera object by reading its params in a dictionary.

    Args:
        p: a dictionary containing the parameters, with fields named as in the
           params.json.example file

    Returns:
        an instance of the PushbroomCamera class
    """
    i = p['instrument']
    instrument = Instrument(i['f'], i['w'], i['n'], i['dwell_time'])

    o = p['orbit']
    if o.has_key('i'):
        orbit = CircularOrbit(o['a'], o['i'], o['lon'])
    else:
        orbit = SunSynchronousCircularOrbit(o['a'], o['lon'])

    v = p['view']
    return PushbroomCamera(instrument, orbit, v['duration'], v['pso_0'],
                           v['psi_x'], v['psi_y'], v['gamma'])
