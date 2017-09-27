from abc import ABCMeta, abstractmethod

import numpy as np
from scipy.misc import imread
from GPyOpt.methods import BayesianOptimization
import time
import random, datetime, time, glob, cv2
from psychopy import visual, core, event, data, gui


from ImageResponse import ImageResponse


class BOMaxStim():

    """Abstract class for identifying maximizing stimuli with Bayesian
    optimization."""

    def __init__(self, stimuli, features, image_response):
        """
        Arguments:
            stimuli: iterable of raw stimuli.
            features: numpy array of stimuli features, in the same order as
                stimuli.
        """
        self.stimuli = stimuli
        self.features = features
        self.image_response = image_response

        # Make the features immutable so they can be put in a dictionary
        self.features.flags.writeable = False
        # Make a dictionary mapping features to their corresponding stimuli
        self.feature_stimuli_map = {f.data: s for
                                    f, s in zip(self.features, self.stimuli)}

        # Set up the domain for bayesian optimization
        self.domain = [{
            "name": "features",
            "type": "bandit",
            "domain": self.features
        }]

        # Set up the BO object
        self.bo = BayesianOptimization(
            f=self.get_target_from_feature,
            domain=self.domain,
            initial_design_numdata=None,
            acquisition_type="EI",
            normalize_Y=True,
            maximize=True,
        )

    def optimize(self, max_iter):
        """Run the optimization."""
        self.bo.run_optimization(max_iter)

    def get_target_from_feature(self, feature):
        """Get the target value for the given stimulus feature."""
        # First identify the stimulus corresponding to the feature
        feature.flags.writeable = False
        stimulus = self.feature_stimuli_map[feature[0].data]
        # Get the target from the stimulus
        return self.get_target_from_stimulus(stimulus)

    def get_target_from_stimulus(self, stimulus):
        """Get the target value for the given stimulus."""
        print(stimulus)
        response_data = self.image_response.get_image_response(stimulus, self.stimuli.index(stimulus)+1)
        assert(int(response_data[1][0][1]) == self.stimuli.index(stimulus)+1)
        response_data = response_data[0][0].get_data()
        #get average of samples 716-820 = 150-200ms post-stimulus
        avg = np.mean(response_data[0, :128, 716:820])
        return avg


# Load images
images = []
for i in range(1, 80):
    images.append("Stimuli/OptPilot/diffeo/s1/Im_01_{:02}.png".format(i))

# Load features
labels = np.load("Stimuli/OptPilot/features/labels.npy")
embeddings_raw = np.load("Stimuli/OptPilot/features/embeddings.npy")

embeddings = np.zeros((len(images), 128))
for label, embedding in zip(labels, embeddings_raw):
    if label.split("/")[3] != "s1":
        continue
    label_idx = int(label.split("/")[-1].split(".")[0].split("_")[-1]) - 1
    embeddings[label_idx] = embedding

window = visual.Window([800, 800], monitor="testMonitor", units="deg")
pport_addr = 0xcff8
imsize = 182

image_response = ImageResponse(list(range(1,80)), window, imsize, pport_addr)
optimizer = BOMaxStim(images, embeddings, image_response)

optimizer.optimize(100)

