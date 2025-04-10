from scipy.spatial import distance


def chamfer_distance(pts1, pts2):
    # points shape is num x 3
    dist_mat = distance.cdist(pts1, pts2, 'euclidean')
    valid_ab = dist_mat.min(-1).mean()
    valid_ba = dist_mat.min(-2).mean()

    return (valid_ba + valid_ab) / 2