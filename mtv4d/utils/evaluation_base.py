import numpy as np


# another attr is error list of tp samples

def cal_ap_list_from_tp_list(tp_list, fp_list, conf_list, num_ap_interval=101, is_cal_ap=False, draw_pr_curve=False):
    # tp_list means whether each sample is true in positive pred samples.
    npos = len(tp_list)
    tp = np.cumsum(tp_list).astype(float)
    fp = np.cumsum(fp_list).astype(float)
    conf = np.array(conf_list)
    prec = tp / (fp + tp)
    rec = tp / float(npos)
    rec_interp = np.linspace(0, 1, num_ap_interval)  # 101 steps, from 0% to 100% recall.
    precision = np.interp(rec_interp, rec, prec, right=0)
    confidence = np.interp(rec_interp, rec, conf, right=0)
    recall = rec_interp
    if is_cal_ap:
        ap = calc_ap(precision, 0.1, 0.1)
        return ap
    if draw_pr_curve:
        import matplotlib.pyplot as plt
        plt.plot(recall, precision), plt.plot(rec_interp, conf), plt.show()
    return precision, recall, confidence


def calc_ap(precision, min_recall=0.1, min_precision=0.1) -> float:
    """ Calculated average precision in nus """
    assert 0 <= min_precision < 1
    assert 0 <= min_recall <= 1
    prec = np.copy(precision)
    prec = prec[round(100 * min_recall) + 1:]  # Clip low recalls. +1 to exclude the min recall bin.
    prec -= min_precision  # Clip low precision
    prec[prec < 0] = 0
    return float(np.mean(prec)) / (1.0 - min_precision)
