import os
import h5py
import numpy as np
import torch
import matplotlib.pyplot as plt

from Models import models
from preProcessor import PreProcessor
from postProcessor import PostProcessor


class Pipeline(object):

    def __init__(self, preprocess, landmarkDetect, postprocess):
        self.preprocess = preprocess
        self.landmarkDetect = landmarkDetect
        self.postprocess = postprocess

    def __call__(self, sequence, reference, pixelsize):
        sequence, scale_correction, width = self.preprocess(sequence)
        predicted_sequence = self.landmarkDetect(sequence)
        mapse, measurement_info = self.postprocess(predicted_sequence, reference, width, pixelsize, scale_correction)


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

            """
            plt.clf()
            plt.imshow(prediction_masks[0,0,:,:].numpy()+prediction_masks[0,1,:,:]+model_input[0,0,1,:,:], cmap='gray')
            plt.pause(0.01)
            """

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

    file_dir = "../Data/val"
    model_path = "../Weights/best_true_weights_3len.pth"

    files = []
    for root, dirs, file in os.walk(file_dir):
        files.extend(file)
        break

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

    for file in files:
        file_path = os.path.join(file_dir, file)
        raw_file = h5py.File(file_path, 'r')
        """
        sequence = np.array(raw_file['tissue']['data'])
        pixelsize = np.array(raw_file['tissue']['pixelsize'])
        """
        sequence = np.array(raw_file['images'])
        ref = np.array(raw_file['reference'])
        pixelsize = 0.5

        pipeline(sequence, ref, pixelsize)

main()
