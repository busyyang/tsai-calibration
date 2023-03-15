#!/usr/bin/python
###
#
#   2-stage Tsai Calibration with Distortion.
#     Given world (3D position on the cube, in mm) and pixel (2D position in the image, in px) cooordinates as input,
#     along with the pixel size and resolution.
#	  Extracts a set of parameters that describe the camera - see calibrateDistorted(points)
#     Author: Samuel Bailey <sam@bailey.geek.nz>
#
###


from __future__ import print_function

import math

from transforms import *
from calibration import calibrate

# http://stackoverflow.com/questions/5980042/how-to-implement-the-verbose-or-v-option-into-a-script
verbose = True
printVerbose = print if verbose else lambda *a, **k: None


def error(points):
    numbers = [np.linalg.norm(np.subtract(p.pixel[:2], p.distortedPixel[:2])) for p in points]
    return {'mean': np.mean(numbers), 'median': np.median(numbers), 'min': np.amin(numbers), 'max': np.amax(numbers)}


def estimateKappa(points):
    def estimateKappaP(point):
        """
        x_u = x_d(1+k*r^2)
        y_u = y_d(1+k*r^2) where r^2 = (x_d)^2 + (y_d)^2

        (x_u)^2 + (y_u)^2 = [(x_d)^2 + (y_d)^2] [(1+k*r^2)^2]
        note u2 = (x_u)^2 + (y_u)^2, and d2 = (x_d)^2 + (y_d)^2
        so, u2 = d2[(1+k*d2)^2]
        then, k = (sqrt(u / d) - 1) / d2

        if u2>d2, the value of k should be positive, otherwise it should be nagitive.

        :param point:
        :return:
        """

        u2 = (point.projectedSensor[0] * point.projectedSensor[0]) + (
                point.projectedSensor[1] * point.projectedSensor[1])
        d2 = (point.sensor[0] * point.sensor[0]) + (point.sensor[1] * point.sensor[1])

        d = math.sqrt(d2)
        u = math.sqrt(u2)
        k = (u / d - 1) / d2
        return k

    return np.mean(list(map(estimateKappaP, points)))


def calibrateDistorted(settings, points, image):
    pixelSize = settings['pixelSize']
    resolution = settings['resolution']
    label = settings['label']
    yOffset = settings['yOffset']
    numLowDistortionPoints = settings['minLowDistortionPoints']
    numHighDistortionPoints = settings['numHighDistortionPoints']
    passes = settings['passes']

    points = pixelToSensor(points, resolution, pixelSize)

    # split the data into low/high distortion points
    points = sorted(points, key=lambda p: euclideanDistance2d(p.sensor))
    printVerbose('%d points' % len(points))
    lowDistortionPoints = points[:numLowDistortionPoints]
    printVerbose('%d low distortion points, max. distance from center of sensor = %fmm' % (
        len(lowDistortionPoints), np.max(list(map(lambda p: np.linalg.norm(p.sensor[:2]), lowDistortionPoints)))))

    highDistortionPoints = points[-numHighDistortionPoints:]
    printVerbose('%d high distortion points, min. distance from center of sensor = %fmm' % (
        len(highDistortionPoints), np.min(list(map(lambda p: np.linalg.norm(p.sensor[:2]), highDistortionPoints)))))

    kappa = 0.0  # assume K1 = 0 (no distortion) for the initial calibration

    # record some basic statistics
    errors = []
    kappas = []

    def stats():
        e = error(points)
        errors.append(e)
        kappas.append(kappa)
        print(e)
        print("kappe = ", kappa)
        return e

    # re-calibrate, re-estimate kappa, repeat
    for i in range(passes + 1):
        # use the estimated K1 to "undistort" the location of the points in the image,
        # then calibrate (using the points with low distortion.)
        params = calibrate(pixelToSensor(lowDistortionPoints, resolution, pixelSize, kappa))

        # use the new calibration parameters to re-estimate kappa (using the points with high distortion.)
        kappa = estimateKappa(worldToPixel(highDistortionPoints, params, pixelSize, resolution, yOffset, kappa))

        # project the points in world coordinates back onto the sensor then record some stats
        points = worldToPixel(points, params, pixelSize, resolution, yOffset, kappa)
        stats()

    translationVector = np.array([params['tx'], params['ty'], params['tz']], np.float64)

    return {
        'label': label,
        'params': {
            'f': params['f'],
            'rotationMatrix': params['rotationMatrix'],
            'translationVector': translationVector,
            'RT': np.dot(translationToHomogeneous(translationVector), rotationToHomogeneous(params['rotationMatrix'])),
            'K1': kappas[-1],
            'pixelSize': pixelSize,
            'resolution': resolution,
            'error': errors[-1]
        },
        'points': points,
        'image': image,
        'errors': errors,
        'kappas': kappas
    }
