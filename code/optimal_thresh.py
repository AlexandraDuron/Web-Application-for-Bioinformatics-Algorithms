

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
