#!/usr/bin/python
###
#
#   An implementation of Tsai's camera calibration technique.
#     Author: Samuel Bailey <sam@bailey.geek.nz>
#
###


# we need to do some basic python stuff...
from __future__ import print_function
import json
import math
import os.path
from pprint import pprint

# and some math stuff
import numpy as np

# and display some points
import matplotlib

matplotlib.use('agg')
from matplotlib.backends.backend_pdf import PdfPages

# http://stackoverflow.com/questions/5980042/how-to-implement-the-verbose-or-v-option-into-a-script
verbose = True
printVerbose = print if verbose else lambda *a, **k: None

from mmath import *
from point import Point, newPoint
from transforms import *
from calibration import calibrate
from distortion import *
from plot import *


def processStereo(leftCamera, rightCamera):
    worldPoints = [p.world for p in leftCamera['points']]
    leftParams = leftCamera['params']
    rightParams = rightCamera['params']

    print('\nCamera: %s' % leftCamera['label'])
    print('\n Left:')
    pprint(leftParams)

    print('\n Right:')
    pprint(rightParams)

    baseline = np.linalg.norm(leftParams['translationVector'] - rightParams['translationVector'])

    print('\n baseline:')
    pprint(baseline)

    print('\n distance to camera:')
    camera_midpoint = np.linalg.norm((leftParams['translationVector'] + rightParams['translationVector']) / 2.0)
    print(camera_midpoint)

    return {
        'baseline': baseline,
        'points': worldPoints,
    }


def openFile(settings, folder):
    dataFilename = os.path.join(folder, 'config.json')

    with open(dataFilename) as dataFile:
        data = json.load(dataFile)

    for n in ['pixelSize', 'resolution', 'label']:
        settings[n] = data[n]

    def readCsvLine(csvLines: str):
        points = []
        for csvLine in csvLines:
            values = [float(v) for v in csvLine.strip().split(',')]
            points.append(
                {'left': newPoint({'world': values[:3], 'pixel': values[-4:-2]}),
                 'right': newPoint({'world': values[:3], 'pixel': values[-2:]})}
            )
        return points

    with open(os.path.join(folder, data['points'])) as csvFile:
        points = readCsvLine(csvFile.readlines())

    os.path.join(folder, data['images'][0])
    leftImage = plt.imread(os.path.join(folder, data['images'][0]))
    rightImage = plt.imread(os.path.join(folder, data['images'][1]))

    leftPoints = [o['left'] for o in points]
    rightPoints = [o['right'] for o in points]
    leftCamera = calibrateDistorted(settings, leftPoints, leftImage)
    rightCamera = calibrateDistorted(settings, rightPoints, rightImage)
    world = processStereo(leftCamera, rightCamera)

    return {'left': leftCamera, 'right': rightCamera, 'world': world}


def openFolders(settings):
    stats = [openFile(settings, folder) for folder in settings['folders']]

    with PdfPages(settings['outputFilename']) as pdf:
        def p(s, l):
            pdf.savefig(plotPoints(s['points'], s['image'], l))

        def f(x):
            p(x['left'], 'Left Sensor')
            p(x['right'], 'Right Sensor')
            pdf.savefig(displayStereo(x))
            pdf.savefig(displayStereoSide(x))

        [f(stat) for stat in stats]


def main():

    settings = {
        'camera': 'Siemens Carm',
        'yOffset': 0.0,
        'minLowDistortionPoints': 16,
        'numHighDistortionPoints': 60,
        'passes': 8,
        'folders': ['data/3d/shoot2'],
        'outputFilename': 'data/3d/shoot2/output.pdf'
    }
    """
    settings = {
        'camera': 'GoPro Hero 3+ Stereo',
        'yOffset': 0.0,
        'minLowDistortionPoints': 16,
        'numHighDistortionPoints': 8,
        'passes': 8,
        'folders': ['data/3d/shoot1'],
        'outputFilename': 'output/output.pdf'
    }
    """
    openFolders(settings)


if __name__ == '__main__':
    main()
