# This class contains the code developed to plot heat maps of the correctness
# of the parameters. Matplotlib python library was utilized to display the graphs.
# The number of plots created depend on the parameter, Minimum Number of Matched Peaks.
# The values of this parameter are stored in an np.array, and the program creates a
# plot for each element of the array. The class contains two methods; the first one
# plots the Minimum MS2 intensity parameter with the MZ tolerance,
# and the second one plots the mean difference.


# Dependencies to make the graphs
import matplotlib.pyplot as plt
import base64
import io
import matplotlib.gridspec as gridspec
import matplotlib.image as mpimg
import pylab as plt
# Numpy is a fundamental package for scientific computing with Python
import numpy as np


class Plots(object):
    # Initialize global variables.
    picture = 0
    picture2 = 0
    # Graph for Minimum ms2 intensity with MZ Tolerance values

    def plot_tolerance_with_ms2_vals(self, min_peaks, tol_vals, grid, ms2_vals):
        # Make a genera figure
        fig = plt.figure(figsize=(6, 5*5))
        # Set distance between graphs
        fig.subplots_adjust(hspace=0.2)
        for mp in min_peaks:
            # Add a graph
            fig.add_subplot(min_peaks.size, 1,  mp).title.set_text(
                'Minimum shared MS2 peaks = {}'.format(mp))
            # make plots
            pos = np.unravel_index(grid[mp].argmax(), grid[mp].shape)
            plt.plot(
                pos[1], pos[0], 'ro')
            plt.text(pos[1], pos[0], grid[mp][pos[0], pos[1]])
            plt.xticks(range(len(tol_vals)), tol_vals)
            plt.yticks(range(len(ms2_vals)), ms2_vals)
            plt.imshow(grid[mp], aspect='auto')
        # Set the general figure to bytes and convert it to a String to store it as a URL
        img = io.BytesIO()
        # Save general image

        plt.savefig(img, format='png')
        img.seek(0)
        graph_url = base64.b64encode(img.getvalue()).decode()
        self.picture = 'data:image/png;base64,{}'.format(graph_url)
        return self.picture

    # Graph for mean difference
    def show_mean_diff(self, all_pos, all_neg, min_peaks):
        img2 = io.BytesIO()
        # Make a genera figure
        fig = plt.figure(figsize=(5, 5*5))
        # Set distance between graphs
        fig.subplots_adjust(hspace=0.2)
        for mp in all_pos:
            # Add a graph
            fig.add_subplot(min_peaks.size, 1, mp).title.set_text(
                'Mean difference for Minimum shared MS2 peaks =. {}'.format(mp))
            # Make plots
            eg = all_pos[mp][all_pos[mp].keys()[0]]
            mean_diff = np.zeros_like(eg)
            tot = 0
            # loop over spectra
            for s in all_pos[mp]:
                mean_diff += all_pos[mp][s] - all_neg[mp][s]
                tot += 1
            mean_diff /= tot
            best = 0
            best_pos = None
            for i, r in enumerate(mean_diff):
                for j, c in enumerate(r):
                    if c > best:
                        best_pos = (i, j)
                        best = c
            plt.imshow(mean_diff, aspect='auto')
        # Save general image
        plt.savefig(img2, format='png')
        # Set the general figure to bytes and convert it to a String to store it as a URL
        img2.seek(0)
        graph_url2 = base64.b64encode(img2.getvalue()).decode()
        self.picture2 = 'data:image/png;base64,{}'.format(graph_url2)
        return self.picture2
