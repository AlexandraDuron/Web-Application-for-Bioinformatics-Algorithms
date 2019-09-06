
# This class contains two methods to map the groups.
# The first one returns a dictionary called group_dict.
# The second method returns a dictionary
# called neg_hits. Both methods are re called in the Optimize class
# and stored in variables.

# Dependencies of Molnet that the code needs to run
from scoring_functions import *
from mnet import *


class Dictionary(object):
        # Return a dictionary called, group_dict
    def make_dictionary_group_dict(self, l, spectra):
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
                        score, _ = fast_cosine(s, sp, ms2_mz_tol, n_matched)
                        if score > max_score:
                            best = sp
                            max_score = score
            neg_hits[s] = (best, max_score)
            n_done += 1
        return group_dict

# Return a dictionary called, neg_hits
    def make_dictionary_neg_hits(self, l, spectra):
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
                        score, _ = fast_cosine(s, sp, ms2_mz_tol, n_matched)
                        if score > max_score:
                            best = sp
                            max_score = score
            neg_hits[s] = (best, max_score)
            n_done += 1
        return neg_hits
