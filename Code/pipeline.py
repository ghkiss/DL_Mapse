import os
import h5py
import numpy as np
import torch
import matplotlib.pyplot as plt
from scipy import stats
import scipy.ndimage as nd
import cv2

from Models import models
from preProcessor import PreProcessor
from postProcessor import PostProcessor


class Pipeline(object):

    def __init__(self, preprocess, landmarkDetect, postprocess):
        self.preprocess = preprocess
        self.landmarkDetect = landmarkDetect
        self.postprocess = postprocess

    def __call__(self, sequence, reference, pixelsize):
        seq, scale_correction, width = self.preprocess(sequence)
        predicted_sequence = self.landmarkDetect(seq)
        ret_cor = self.postprocess(predicted_sequence, reference, width, pixelsize, scale_correction)


        for i in range(sequence.shape[0]):
            plt.clf()
            img = sequence[i,:,:]
            cv2.circle(img,(100,100),200,(255),-1)
            plt.imshow(img, cmap='gray')
            #plt.imshow(sequence[i,:,:], cmap='gray')
            plt.scatter(ret_cor[i,0], ret_cor[i,1],c='r')
            plt.scatter(ret_cor[i,2], ret_cor[i,3],c='r')
            plt.pause(0.5)

        return mapse


class LandmarkDetector(object):

    def __init__(self, model, model_seq_len):
        self.model = model
        self.model_seq_len = model_seq_len

    def __call__(self, sequence):
        predicted_sequence = torch.empty((sequence.shape[0]-(self.model_seq_len-1),2,sequence.shape[-2],sequence.shape[-1])).float()

        for frame in range(sequence.shape[0]-(self.model_seq_len-1)):
            model_input = self.fetch_input(sequence, frame)
            prediction_masks = torch.sigmoid(self.model(model_input))
            predicted_sequence[frame,:,:,:] = prediction_masks[0,:,:,:]

            #plt.clf()
            #plt.imshow(model_input[0,0,1,:,:], cmap='gray')
            #plt.show()

        return predicted_sequence

    def fetch_input(self, sequence, frame):
        if self.model_seq_len > 1:
            model_input = sequence[frame:frame+self.model_seq_len,:,:]
            model_input = model_input.unsqueeze(0).unsqueeze(0)
        else:
            model_input = sequence[frame,:,:]
            model_input = model_input.unsqueeze(0).unsqueeze(0).unsqueeze(0)

        return model_input


def main():

    file_dir = "../../../master-project/Project/Data/final-data/test"
    model_path = "../Weights/best_true_weights.pth"

    files = []
    for root, dirs, file in os.walk(file_dir):
        files.extend(file)
        break
    files.sort()
    print(files)

    eps = 1e-10
    model_seq_len = 3

    model = models.Model(model_seq_len)
    model.load_state_dict(torch.load(model_path, map_location='cpu')['model_state_dict'])
    device = torch.device('cpu')
    model = model.to(device)
    model.eval()
    torch.set_grad_enabled(False)


    preprocess = PreProcessor(model_seq_len)
    landmark_detector = LandmarkDetector(model, model_seq_len)
    postprocess = PostProcessor()

    pipeline = Pipeline(preprocess,landmark_detector,postprocess)

    euclidean, x, y = np.array(()), np.array(()), np.array(())
    tp, tn, fn, fp = np.zeros((2)), np.zeros((2)), np.zeros((2)), np.zeros((2))
    ref_scat, pred_scat = np.array(()), np.array(())
    for i, file in enumerate(files):
        #if file[-5:-3] == "2c":
        #    continue
        #if i!=5:
        #    continue
        print("File {}/{}".format(i+1, len(files)))
        file_path = os.path.join(file_dir, file)
        print(file)
        raw_file = h5py.File(file_path, 'r')
        """
        sequence = np.array(raw_file['tissue']['data'])
        pixelsize = np.array(raw_file['tissue']['pixelsize'])
        """
        sequence = np.array(raw_file['images'])
        ref = np.array(raw_file['reference'])
        pixelsize = 0.5

        """
        for j in range(sequence.shape[0]):
            refl = np.zeros(sequence.shape[-2:])
            refr = np.zeros(sequence.shape[-2:])
            refl[int(ref[j,1]),int(ref[j,0])] = 1
            refr[int(ref[j,3]),int(ref[j,2])] = 1
            refl = nd.gaussian_filter(refl, sigma=3)
            refr = nd.gaussian_filter(refr, sigma=3)
            plt.clf()
            ax = plt.gca()
            plt.imshow(sequence[j,:,:], cmap='gray')
            plt.scatter(ref[j,0], ref[j,1], c='r')
            plt.scatter(ref[j,2], ref[j,3], c='r')
            ax.set_xlim(100,300)
            ax.set_ylim(250,50)
            plt.show()
            plt.imshow(refl, cmap=plt.jet())
            ax = plt.gca()
            ax.set_xlim(100,300)
            ax.set_ylim(250,50)
            plt.show()
            plt.imshow(refr, cmap=plt.jet())
            ax = plt.gca()
            ax.set_xlim(100,300)
            ax.set_ylim(250,50)
            plt.show()
        """

        info = pipeline(sequence, ref, pixelsize)
        euclidean = np.append(euclidean, info['euclidean'])
        x = np.append(x, info['x'])
        y = np.append(y, info['y'])
        #ref_scat = np.append(ref_scat, info['ref'])
        #pred_scat = np.append(pred_scat, info['pred'])


        #tp += info['tp']
        #tn += info['tn']
        #fp += info['fp']
        #fn += info['fn']

    #slope, inter, _, _, _ = stats.linregress(ref_scat, pred_scat)
    #line = slope*ref_scat + inter

    """
    plt.clf()
    plt.grid(b=True, which='both')
    plt.scatter(ref_scat, pred_scat, label='Estimates', c='r', s=5)
    plt.plot([25,85],[25,85], 'k--', label='Diagonal', linewidth=1)
    plt.plot(ref_scat,line, 'g--', label='Trend', linewidth=1)
    plt.legend()
    plt.title('Y-axis location of landmarks')
    plt.ylabel('Estimate [mm]')
    plt.xlabel('Reference [mm]')
    plt.show()
    """

    print()
    print("Mean euclidean distance: ", np.mean(euclidean))
    print("RMSE for x-axis: ", np.mean(x))
    print("RMSE for y-axis: ", np.mean(y))
    print("STD for euclidean distance: ", np.std(euclidean))
    print("STD for x-axis: ", np.std(x))
    print("STD for y-axis: ", np.std(y))
    print()
    print("Total sensitivity: ", np.sum(tp)/(np.sum(tp)+np.sum(fn)+eps))
    print("Total specificity: ", np.sum(tn)/(np.sum(tn)+np.sum(fp)+eps))
    print("Total precision: ", np.sum(tp)/(np.sum(tp)+np.sum(fp)+eps))
    print("Total MCC: ", ((np.sum(tp)*np.sum(tn))-(np.sum(fp)*np.sum(fn)))/(np.sqrt((np.sum(tp)+np.sum(fp))*(np.sum(tp)+np.sum(fn))*(np.sum(tn)+np.sum(fp))*(np.sum(tn)+np.sum(fn))))+eps)
    print("Left sensitivity: ", tp[0]/(tp[0] + fn[0]+eps))
    print("Left specificity: ", tn[0]/(tn[0] + fp[0]+eps))
    print("Left precision: ", tn[0]/(tn[0] + fp[0]+eps))
    print("Right sensitivity: ", tp[1]/(tp[1] + fn[1]+eps))
    print("Right specificity: ", tn[1]/(tn[1] + fp[1]+eps))
    print("Right precision: ", tn[1]/(tn[1] + fp[1]+eps))
    print(tp,tn,fp,fn)


main()
