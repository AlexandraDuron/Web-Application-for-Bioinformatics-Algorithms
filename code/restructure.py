
# This class contains a method to Re-organise results to make them easier to analyse.
# It returns a dictionary with the restructured results

#This class was built by Claudia Ortiz-Duron
#The code to Re-organise results to make them easier to analyse was written by
#Dr. Simon Rogers 


class Restructure():

    def restructure_results(neg_hits, all_pos, ms2_vals, tol_vals, all_neg):
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
        return restructured_results[mp][m][t]
