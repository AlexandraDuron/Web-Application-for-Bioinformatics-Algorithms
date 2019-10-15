# This class behaves such as a main method, it runs the parameter optimization process.

#This class was built by me
#The code to carry out the parameter optimization process was written by
#Dr. Simon Rogers 

# Dependencies that Molnet needs to run
from mnet import *
from mnet_utilities import optimise_noise_thresh
from scoring_functions import *
from copy import deepcopy
# Numpy is a fundamental package for scientific computing with Python
import numpy as np
# Thread dependency
import threading


# Flask dependencies
from flask import Flask, request, Flask, make_response, request


# Dependencies from the restructuring of the code
from heat_maps import Plots

from dictionary import Dictionary
from restructure import Restructure
from optimal_thresh import Optimal_thresh


class Optimize(object):
    # Initialize global variables.
    BestThreshold = 0
    bestMP = 0
    bestM = 0
    bestT = 0
    stop_threads = False
    graph_tolerance_with_ms2_vals = 0
    graph_show_mean_diff = 0

    # Method to start the parameters optimization
    def main(self, start_ms2_vals, stop_ms2_vals, step_ms2_vals, matched_peaks_array, tol_vals_array):
        # Program runs in a thread
        def run():
            sys.path.append('C:/Users/ALEXANDRA/workspace/molnet-master/files')
            input_file = '/Users/ALEXANDRA/workspace/molnet-master/files/Beer_multibeers_1_T10_POS.mzML'
            # Load file
            l = MNetLoadMZML(input_file, mz_tolerance=0.2,
                             rt_tolerance=5, get_groups=True)
            ms1, spectra = l.load_spectra()

            # Make a dictionaries to map the groups.
            # For each of the spectra, find the most similar
            # in the dataset that is not in the group.
            d = Dictionary()

            group_dict = d.make_dictionary_group_dict(l, spectra)
            neg_hits = d.make_dictionary_neg_hits(l, spectra)

            all_pos = {}
            all_neg = {}
            n_done = 0
            print("I've done the groups")

            # Compute the similarity between all positive and negative pairs using every combination of the different values of the parameters.

            # Minimum MS2 intensity
            ms2_vals = np.arange(start_ms2_vals, stop_ms2_vals, step_ms2_vals)
            # Minimum number of matched peaks:
            min_peaks = matched_peaks_array
            # MS2 tolerance
            tol_vals = tol_vals_array
            print("parameters acepted")
            print("These are the values for the Minimum MS2 intensity: ")
            print(ms2_vals)
            print("These are the values for the Minimum number of matched peaks: ")
            print(min_peaks)
            print("These are the values for MS2 tolerance: ")
            print(tol_vals)
            # Loop over the minimum number of matched peaks.
            for mp in min_peaks:

                all_pos[mp] = {}
                all_neg[mp] = {}
                for spec1 in spectra:
                    group = group_dict[spec1]
                    found = False
                    for eg_pos in group:
                        if not eg_pos == spec1:
                            found = True
                            break
                    if found:
                        pass
                    eg_neg = neg_hits[spec1][0]
                    s_pos = []
                    s_neg = []
                    all_m_vals = []
                    all_t_vals = []
                    # Loop over the minimum MS2 intensity.
                    for m in ms2_vals:
                        p1 = deepcopy(spec1)
                        p2 = deepcopy(eg_pos)
                        p3 = deepcopy(eg_neg)
                        p1.remove_small_peaks(min_ms2_intensity=m)
                        p2.remove_small_peaks(min_ms2_intensity=m)
                        p3.remove_small_peaks(min_ms2_intensity=m)
                        s_pos_t = []
                        s_neg_t = []
                        # Loop over the MS2 tolerance.
                        for t in tol_vals:
                            s, _ = fast_cosine(p1, p2, t, mp)
                            s_pos_t.append(s)
                            s, _ = fast_cosine(p1, p3, t, mp)
                            s_neg_t.append(s)
                        s_pos.append(s_pos_t)
                        s_neg.append(s_neg_t)
                    all_pos[mp][spec1] = np.array(s_pos)
                    all_neg[mp][spec1] = np.array(s_neg)
                    n_done += 1
                print("I have done a loop the for minimum number of matched peaks")
            # This code was commented due to a deficiency. The class Restructure
            # does not deliver correct results and makes the process slower.
            #r = Restructure()

            # restructure results
            n_spec = len(neg_hits)
            restructured_results = {}
            for mp in all_pos:
                restructured_results[mp] = {}
                for m in ms2_vals:
                    restructured_results[mp][m] = {}
                    for t in tol_vals:

                        restructured_results[mp][m][t] = []

            for mp in all_pos:
                sub_res = all_pos[mp]
                for spec, table in sub_res.items():
                    # table is ms2 v tol
                    for i, m in enumerate(ms2_vals):
                        for j, t in enumerate(tol_vals):
                            val = table[i, j]
                            restructured_results[mp][m][t].append(
                                [val, all_neg[mp][spec][i, j]])

            # For every combination, compute the optimal threshold and the score.
            print("I've finished the loops, I'm going to compute the optimal threshold ")

            best = None,
            best_score = 0
            best_thresh = 0
            grid = {}
            for mp in min_peaks:
                grid[mp] = np.zeros((len(ms2_vals), len(tol_vals)))
                for i, m in enumerate(ms2_vals):
                    for j, t in enumerate(tol_vals):
                        scores = restructured_results[mp][m][t]
                        # This code was commented due to a deficiency. The class Restructure
                        # does not deliver correct results and makes the process slower.
                        #scores = r.restructure_results(neg_hits, all_pos, ms2_vals, tol_vals, all_neg)
                        op = Optimal_thresh()
                        best_perf, bt = op.compute_optimal_thresh(scores)
                        grid[mp][i, j] = best_perf
                        if best == None:
                            best = (mp, m, t)

                            best_score = best_perf
                            best_thresh = bt
                        else:
                            if best_perf >= best_score:
                                best = (mp, m, t)

                                best_score = best_perf
                                best_thresh = bt
            # Check the results in the terminal
            print("This is the optimal value of the Minimum number of matched peaks: ")

            print(best[0])
            print("This is the optimal value of the Minimum MS2 intensity: ")

            print(best[1])
            print("This is the optimal value of the MS2 tolerance: ")

            print(best[2])
            print("This is the value of the optimal threshold: ")
            print(best_thresh)

            # Optimal values of parameters and the optimal threshold
            self.BestThreshold = best_thresh
            self.bestMP = best[0]
            self.bestM = best[1]
            self.bestT = best[2]

        # Make plots
            p = Plots()
            self.graph_tolerance_with_ms2_vals = p.plot_tolerance_with_ms2_vals(
                min_peaks, tol_vals, grid, ms2_vals)
            self.graph_show_mean_diff = p.show_mean_diff(
                all_pos, all_neg, min_peaks)
            print("I have done the graphs")

            # Stop thread
            self.stop_threads = True
            print("Stop Thread")

        t1 = threading.Thread(target=run)
        t1.start()
        if self.stop_threads:
            t1.join()

    # Get graphs.

    def print_graph(self):
        return self.graph_tolerance_with_ms2_vals

    def print_graph2(self):
        return self.graph_show_mean_diff

    # Get values of the optmized parameters.
    def get_BestThreshold(self):
        print(self.BestThreshold)
        return self.BestThreshold

    def get_mp(self):
        print(self.bestMP)
        return self.bestMP

    def get_m(self):
        print(self.bestM)
        return self.bestM

    def get_t(self):
        print(self.bestT)
        return self.bestT

    # Notify thread stop
    def stop_parameters_returning(self):
        while True:
            pass
            if self.stop_threads:

                break
