# This class implements a method called compute_optimal_thresh.
# Which computes scores for every combination of parameters.

#This class was built by Claudia Ortiz-Duron
#The code that computes scores for every combination of parameters was written by
#Dr. Simon Rogers 

# Numpy is a fundamental package for scientific computing with Python
import numpy as np


class Optimal_thresh(object):
    def compute_optimal_thresh(self, scores):
        thresh_vals = np.linspace(0, 1, 100)
        best_perf = 0
        best_thresh = 0
        scores = np.array(scores)
        for t in thresh_vals:
            perf = ((scores[:, 0] >= t)*(scores[:, 1] < t)).mean()
            if perf > best_perf:
                best_perf = perf
                best_thresh = t
        return best_perf, best_thresh
