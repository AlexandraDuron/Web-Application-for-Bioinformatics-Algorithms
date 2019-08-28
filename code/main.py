
import numpy as np
from copy import deepcopy
from mnet import *
from copy import deepcopy
import numpy as np
import threading
from molnet_views import *
import progressbar as pb
import pyprog
from time import sleep
from flask import Flask
from flask import request
import threading
import time
import cStringIO
import pylab as plt
import numpy as np

from flask import Flask, make_response, request

from numpy import *
import numpy as np
import json
from mnet_utilities import optimise_noise_thresh
from scoring_functions import *

import matplotlib
import matplotlib.pyplot as plt
import base64
import io
import matplotlib.gridspec as gridspec
import matplotlib.image as mpimg
from optimal_thresh import compute_optimal_thresh


class Optimize(object):

    OptimalParemeters = 0
    BestThreshold = 0
    bestMP = 0
    bestM = 0
    bestT = 0
    stop_threads = False
    picture = 0
    picture2 = 0

    def main(self, start_ms2_vals, stop_ms2_vals, step_ms2_vals, matched_peaks_array, tol_vals_array):

        def run():
            sys.path.append(
                'C:/Users/ALEXANDRA/workspace/molnet-master/files')
            print("Hello world it's working spectra loading ")
            input_file = '/Users/ALEXANDRA/workspace/molnet-master/files/Beer_multibeers_1_T10_POS.mzML'

            l = MNetLoadMZML(input_file, mz_tolerance=0.2,
                             rt_tolerance=5, get_groups=True)
            ms1, spectra = l.load_spectra()

            # make a dictionary to map the groups
            group_dict = {}
            for g in l.groups:
                for s in g:
                    group_dict[s] = g
            neg_hits = {}
            pmz_tol = 5
            rt_tol = 50
            n_done = 0

            n_matched = 2
            ms2_mz_tol = 1.0

            for s in spectra:
                best = None
                max_score = 0
                this_group = group_dict[s]
                for sp in spectra:
                    if abs(s.precursor_mz - sp.precursor_mz) > pmz_tol:
                        if abs(s.rt - sp.rt) > rt_tol:
                            score, _ = fast_cosine(
                                s, sp, ms2_mz_tol, n_matched)
                            if score > max_score:
                                best = sp
                                max_score = score
                neg_hits[s] = (best, max_score)
                n_done += 1

            all_pos = {}
            all_neg = {}
            n_done = 0
            import numpy as np

            ms2_vals = np.arange(start_ms2_vals, stop_ms2_vals, step_ms2_vals)
            print("parameters for ms2_vals  were accepted ")

            min_peaks = matched_peaks_array
            print("parameters for min_peaks  were accepted ")

            tol_vals = tol_vals_array
            print("parameters for  tol_vals  were accepted ")

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

                    for m in ms2_vals:
                        p1 = deepcopy(spec1)
                        p2 = deepcopy(eg_pos)
                        p3 = deepcopy(eg_neg)
                        p1.remove_small_peaks(min_ms2_intensity=m)
                        p2.remove_small_peaks(min_ms2_intensity=m)
                        p3.remove_small_peaks(min_ms2_intensity=m)
                        s_pos_t = []
                        s_neg_t = []
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
                    if n_done % 100 == 0:
                        print n_done

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
                    for i, m in enumerate(ms2_vals):
                        for j, t in enumerate(tol_vals):
                            val = table[i, j]
                            restructured_results[mp][m][t].append(
                                [val, all_neg[mp][spec][i, j]])

            def compute_optimal_thresh(scores):
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

            best = None,
            best_score = 0
            best_thresh = 0
            grid = {}
            for mp in min_peaks:
                grid[mp] = np.zeros((len(ms2_vals), len(tol_vals)))
                for i, m in enumerate(ms2_vals):
                    for j, t in enumerate(tol_vals):
                        scores = restructured_results[mp][m][t]
                        best_perf, bt = compute_optimal_thresh(scores)
                        grid[mp][i, j] = best_perf
                        if best == None:
                            best = (mp, m, t)
                            bestMP = (mp)
                            bestM = (m)
                            bestT = (t)
                            best_score = best_perf
                            best_thresh = bt
                        else:
                            if best_perf >= best_score:
                                best = (mp, m, t)
                                bestMP = (mp)
                                bestM = (m)
                                bestT = (t)
                                best_score = best_perf
                                best_thresh = bt

            print("This is the best threshold")
            print(best, best_score, best_thresh)
            print(mp)
            print(m)
            print(t)

            self.OptimalParemeters = best
            self.BestThreshold = best_thresh
            self.bestMP = bestMP
            self.bestM = bestM
            self.bestT = bestT

            self.stop_threads = True
            print('thread killed')

            import matplotlib.pyplot as plt

            fig = plt.figure(figsize=(6, 5*5))
            fig.subplots_adjust(hspace=0.5)
            for mp in min_peaks:
                fig.add_subplot(min_peaks.size, 1,  mp).title.set_text(
                    'Min shared ms2 peaks = {}'.format(mp))

                pos = np.unravel_index(grid[mp].argmax(), grid[mp].shape)
                plt.plot(pos[1], pos[0], 'ro')
                plt.text(pos[1], pos[0], grid[mp][pos[0], pos[1]])
                plt.xticks(range(len(tol_vals)), tol_vals)
                plt.yticks(range(len(ms2_vals)), ms2_vals)
                plt.imshow(grid[mp], aspect='auto')

            img = io.BytesIO()
            plt.savefig(img, format='png')

            img.seek(0)
            graph_url = base64.b64encode(img.getvalue()).decode()

            self.picture = 'data:image/png;base64,{}'.format(graph_url)

            print("I did graph 1")

            # # min shared ms2 peaks-------------------- Plot Number 2

            img2 = io.BytesIO()
            fig = plt.figure(figsize=(5, 5*5))
            fig.subplots_adjust(hspace=0.5)

            for mp in all_pos:
                fig.add_subplot(min_peaks.size, 1, mp)

                eg = all_pos[mp][all_pos[mp].keys()[0]]
                mean_diff = np.zeros_like(eg)
                tot = 0
                for s in all_pos[mp]:  # loop over spectra
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

            plt.savefig(img2, format='png')
            img2.seek(0)
            graph_url2 = base64.b64encode(img2.getvalue()).decode()

            self.picture2 = 'data:image/png;base64,{}'.format(graph_url2)

            print("I did graph 2")

        t1 = threading.Thread(target=run)
        t1.start()
        if self.stop_threads:
            t1.join()

    def print_graph(self):
        return self.picture

    def print_graph2(self):
        return self.picture2

    def get_OptimalParemeters(self):
        print(self.OptimalParemeters)
        return self.OptimalParemeters

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

    def stop_parameters_returning(self):
        while True:
            pass
            if self.stop_threads:

                break
