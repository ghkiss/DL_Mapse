import numpy as np
import torch
import scipy.ndimage as nd
from scipy import signal, interpolate

import matplotlib.pyplot as plt

#plt.ion()

class PostProcessor(object):

    def __init__(self,coordinateExtract=None, rotate=None, peakDetect=None):

        if not coordinateExtract:
            self.coordinateExtract = CoordinateExtractor()
        else:
            self.coordinateExtract = coordinateExtract
        if not rotate:
            self.rotate = Rotation()
        else:
            self.rotate = rotate
        if not peakDetect:
            self.peakDetect = PeakDetection()
        else:
            self.peakDetect = peakDetect

    def __call__(self, predicted_pmap, reference, left, pixelsize, scale_correction):
        try:
            coordinates = self.coordinateExtract(predicted_pmap)
            #coordinates = self.rotate(coordinates)

            print(scale_correction)
            coordinates = coordinates * scale_correction
            print(reference)
            print(coordinates)
            mean_dist = 0
            for i in range(reference.shape[0]):
                mean_dist += np.sqrt((reference[i,0]-(coordinates[i,0,0]+left))**2+(reference[i,1]-coordinates[i,0,1])**2)
                mean_dist += np.sqrt(np.power((reference[i,2]-(coordinates[i,1,0]+left)),2)+np.power((reference[i,3]-coordinates[i,1,1]),2))

            mean_dist /= (2*reference.shape[0])
            print(mean_dist)


            left_movement, right_movement = self.extractHorisontalMovement(coordinates)

            left_es, left_ed, lp_l = self.peakDetect(left_movement)
            right_es, right_ed, lp_r = self.peakDetect(right_movement)

            plt.clf()
            plt.plot(left_movement['t'], left_movement['y'])
            plt.plot(right_movement['t'], right_movement['y'])
            plt.plot(lp_l)
            plt.plot(lp_r)
            plt.plot(left_movement['t'][left_es], left_movement['y'][left_es], marker='x')
            plt.plot(left_movement['t'][left_ed], left_movement['y'][left_ed], marker='x')
            plt.plot(right_movement['t'][right_es], right_movement['y'][right_es], marker='x')
            plt.plot(right_movement['t'][right_ed], right_movement['y'][right_ed], marker='x')
            plt.grid(b=True, which='both')
            plt.show()
            left_mapse = self.mapseCalc(left_movement['y'][left_es], left_movement['y'][left_ed], pixelsize)
            right_mapse = self.mapseCalc(right_movement['y'][right_es], right_movement['y'][right_ed], pixelsize)
        except:
            left_mapse = np.nan
            right_mapse = np.nan

        mapse = {'left':left_mapse, 'right':right_mapse}
        measurement_info = {'left_es':left_es,'left_ed':left_ed,
                            'right_es':right_es,'right_ed':right_ed,
                            'left_movement':left_movement,'right_movement':right_movement}

        print(mapse)

        return mapse, measurement_info


    def extractHorisontalMovement(self, coordinates):

        left_movement = {'y': np.array([]), 't': np.array([], dtype=int), 'nan': np.array([], dtype=int)}
        right_movement = {'y': np.array([]), 't': np.array([], dtype=int), 'nan': np.array([], dtype=int)}

        for i, point in enumerate([left_movement, right_movement]):
            for j, value in enumerate(coordinates[:,i,1]):
                if np.isnan(value):
                    point['nan'] = np.append(point['nan'],1)
                else:
                    point['y'] = np.append(point['y'],value)
                    point['t'] = np.append(point['t'],j)
                    point['nan'] = np.append(point['nan'],0)

        return left_movement, right_movement


    def mapseCalc(self, es, ed, pixelsize):
        if es.size==0 or ed.size==0:
            mapse = np.nan
        else:
            mapse = np.abs(np.mean(es) - np.mean(ed))*pixelsize

        return mapse

class CoordinateExtractor(object):

    def __init__(self, method="argmax", threshold=0.5):
        self.method = method
        self.threshold = threshold

    def __call__(self, predicted_pmap):
        coordinates = np.empty((predicted_pmap.shape[0],2,2))
        if self.method == "argmax":
            for i in range(predicted_pmap.shape[0]):
                left_argmax_idx = torch.argmax(predicted_pmap[i,0,:,:])
                right_argmax_idx = torch.argmax(predicted_pmap[i,1,:,:])
                left_point = (left_argmax_idx / predicted_pmap.shape[-2],
                              left_argmax_idx % predicted_pmap.shape[-1])
                right_point = (right_argmax_idx / predicted_pmap.shape[-2],
                               right_argmax_idx % predicted_pmap.shape[-1])

                coordinates[i,0,0] = left_point[1]
                coordinates[i,0,1] = left_point[0]
                coordinates[i,1,0] = right_point[1]
                coordinates[i,1,1] = right_point[0]

        else:
            predicted_pmap = predicted_pmap.ge(self.threshold).numpy()
            for i in range(predicted_pmap.shape[0]):
                left_point = nd.center_of_mass(predicted_pmap[i,0,:,:])
                right_point = nd.center_of_mass(predicted_pmap[i,1,:,:])

                coordinates[i,0,0] = left_point[1]
                coordinates[i,0,1] = left_point[0]
                coordinates[i,1,0] = right_point[1]
                coordinates[i,1,1] = right_point[0]

        return coordinates

class Rotation(object):
    def __inti__(self):
        print()

    def __call__(self, coordinates):
        rotated_coordinates = np.empty(coordinates.shape)

        for i in range(coordinates.shape[1]):
            x = coordinates[:,i,0]
            y = coordinates[:,i,1]

            if np.isnan(y).all():
                raise Exception('NAN in before rotation')
            origin_idx = np.nanargmin(y)
            origin = np.array([x[origin_idx], y[origin_idx]])

            x_nnan = x[~np.isnan(x)]
            y_nnan= y[~np.isnan(y)]

            ind = np.argsort(y_nnan)[-2:]

            mean = np.array([np.nanmean(x_nnan[ind]), np.nanmean(y_nnan[ind])])
            mean_vector = np.array([(mean[0]-origin[0]),
                                    np.abs(origin[1]-mean[1])])
            correction_vector = np.array([0., np.abs(origin[1]-mean[1])])

            if np.sum(mean_vector)+np.sum(correction_vector) > 0.:
                angle = np.arccos(np.dot(mean_vector,correction_vector)/np.dot(np.linalg.norm(mean_vector),
                                                                               np.linalg.norm(correction_vector)))
                if mean_vector[0] < 0.:
                    angle = -angle
            else:
                angle = np.nan

            cos, sin = np.cos(angle), np.sin(angle)

            R = np.array([[cos, -sin],[sin, cos]])

            for j in range(x.shape[0]):
                rotated_point = np.dot(R, np.array([x[j], y[j]]))
                rotated_coordinates[j,i,0] = rotated_point[0]
                rotated_coordinates[j,i,1] = rotated_point[1]

        return rotated_coordinates

class PeakDetection(object):

    def __init__(self, border_coeff=0.2, peak_distance=8):
        self.peak_distance = peak_distance
        self.border_coeff = border_coeff

    def __call__(self, point):
        border = (np.max(point['y'])-np.min(point['y']))*self.border_coeff
        pad_front = np.ones((self.peak_distance,))*point['y'][0]
        pad_back = np.ones((self.peak_distance,))*point['y'][-1]
        padded = np.concatenate((pad_front,point['y'],pad_back))
        y_lp = np.convolve(padded, np.ones((2*self.peak_distance+1,))/(2*self.peak_distance+1), mode='valid')

        peaks_es, _ = signal.find_peaks(-point['y'], distance=self.peak_distance, height=-y_lp+border)
        peaks_ed, _ = signal.find_peaks(point['y'], distance=self.peak_distance, height=y_lp+border)

        peaks_es, peaks_ed = self.filterPeaks(peaks_es, peaks_ed)
        return peaks_es, peaks_ed, y_lp

    def filterPeaks(self, peaks_es, peaks_ed):
        peaks_es, peaks_ed, temp_es, temp_ed = peaks_es.tolist(), peaks_ed.tolist(), [], []

        while peaks_es and peaks_ed:
            if peaks_ed[0] > peaks_es[0]:
                if len(peaks_es) == 1:
                    if peaks_ed[0] > peaks_es[0]:
                        temp_es.append(peaks_es[0])
                        temp_ed.append(peaks_ed[0])
                        peaks_es.pop(0)
                        peaks_ed.pop(0)
                        continue
                if peaks_ed[0] < peaks_es[1]:
                    temp_es.append(peaks_es[0])
                    temp_ed.append(peaks_ed[0])
                    peaks_es.pop(0)
                    peaks_ed.pop(0)
                else:
                    peaks_es.pop(0)
            else:
                peaks_ed.pop(0)

        return np.asarray(temp_es), np.asarray(temp_ed)
