#!/usr/bin/python
###
#
#   Transformation functions for the pinhole camera model.
#     Author: Samuel Bailey <sam@bailey.geek.nz>
#
###

from mmath import *
from scipy.optimize import minimize


### Transform from world coordinates to pixel coordinates


# 3d world coordinates (in mm) -> 3d camera coordinates (in mm)
def worldToCamera(points, params, yOffset):
    worldToCamera = np.dot(translationToHomogeneous([params['tx'], params['ty'] + yOffset, params['tz']]),
                           rotationToHomogeneous(params['rotationMatrix']))

    def transform(point):
        return point._replace(camera=np.dot(worldToCamera, [point.world[0], point.world[1], point.world[2], 1]))

    return list(map(transform, points))


# 3d camera coordinates (in mm) -> sensor coordinates (2d, in mm)
def projectPoints(points, pixelSize, resolution, f):
    cameraToPlane = np.array([[f, 0.0, 0.0, 0.0], [0.0, f, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0]], np.float64)

    def project_sensor(point):
        # perspective projection of the 3d point onto the plane at z=f
        p = np.dot(cameraToPlane, point.camera)
        p = p / p[2]  # perspective division

        # p is now a 2d vector from the center of the image sensor, in mm
        return point._replace(projectedSensor=p)

    def project_pixel(point):
        s = point.projectedSensor
        p = np.array([s[0] / pixelSize[0] + resolution[0] / 2, s[1] / pixelSize[1] + resolution[1] / 2])
        return point._replace(projectedPixel=p)

    points = list(map(project_sensor, points))
    return list(map(project_pixel, points))


# normalised image coordinates -> distorted normalised image coordinates
def distortPoints(points, pixelSize, resolution, kappa):
    def trans_distorted_sensor(point):
        """
        x_u=x_d(1+k*r^2),y_u=y_d(1+k*r^2), where r^2 = (x_d)^2+(y_u)^2
        (x_u)^2+(y_u)^2 = [(x_d)^2+(y_u)^2]*[(1+k*r^2)^2]
        then, u2=d2(1+k*d2)^2, now we need to know d2, that means we could minimize
            [u2-d2(1+k*d2)^2]^2
        :param point:
        :return:
        """
        u = point.projectedSensor
        ru2 = (u[0] * u[0]) + (u[1] * u[1])

        obj = lambda x: (x * ((1 + kappa * x) ** 2) - ru2) ** 2
        res = minimize(obj, np.array([ru2]), method='Powell')
        rd2 = res.x[0]

        correction = 1.0 / (1.0 + (kappa * rd2))
        return point._replace(distortedSensor=np.multiply(u, [correction, correction, 1.0]))

    def trans_distorted_pixel(point):
        d = point.distortedSensor
        return point._replace(distortedPixel=np.array([d[0] / pixelSize[0] + resolution[0] / 2,
                                                       d[1] / pixelSize[1] + resolution[1] / 2]))

    points = list(map(trans_distorted_sensor, points))
    return list(map(trans_distorted_pixel, points))


### Transform from pixel to world coordinates

# normalised image coordinates -> undistorted normalised image coordinates
def undistortPoints(points, kappa):
    def transform(point):
        x, y = point.sensor[0], point.sensor[1]
        correction = 1.0 + (kappa * ((x * x) + (y * y)))
        return point._replace(sensor=np.multiply(point.sensor, [correction, correction, 1.0]))

    return list(map(transform, points))


### Compose a few of the above together to make things easier to read
def pixelToSensor(points, resolution, pixelSize, kappa=0.0):
    def transform(point):
        return point._replace(sensor=np.array([(point.pixel[0] - resolution[0] / 2) * pixelSize[0],
                                               (point.pixel[1] - resolution[1] / 2) * pixelSize[1], 1]))

    return undistortPoints(list(map(transform, points)), kappa)


def sensorToPixel(points, pixelSize, resolution, kappa=0.0):
    return distortPoints(points, pixelSize, resolution, kappa)


def worldToSensor(points, params, pixelSize, resolution, yOffset, kappa=0.0):
    return sensorToPixel(projectPoints(worldToCamera(points, params, yOffset), pixelSize, resolution, params['f']),
                         pixelSize, resolution, kappa)


def worldToPixel(points, params, pixelSize, resolution, yOffset, kappa=0.0):
    return sensorToPixel(projectPoints(worldToCamera(points, params, yOffset), pixelSize, resolution, params['f']),
                         pixelSize, resolution, kappa)


# Distance from the origin in camera coordinates to the origin in world coordinates (in mm)
def cameraToWorldDistance(params, yOffset):
    return np.linalg.norm([params['tx'], params['ty'] - yOffset, params['tz']])
