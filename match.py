import numpy as np
from utils import *
import pyflann as pf

def best_coherence_match(As , f_q, s, a_height, a_width, row ,col, b_width):
    rs = []
    rss = []

    for x in range(row+1):
        for y in range(col+1):
            if (x*b_width+y) < (row*b_width+col):
                pr_x,pr_y = s[x*b_width+y][0]+row-x , s[x*b_width+y][1]+col-y
                if 0<= pr_x <a_height and 0<= pr_y <a_width:
                    rss.append(np.array([x,y]))
                    rs.append(pr_x*a_width+pr_y)

    if len(rss)<=0:
        return np.array([-1, -1])

    r_star = np.argmin(norm(As[np.array(rs)] - f_q, ord=2, axis=1))
    [r_star_x,r_star_y] = rss[r_star]
    [crrs_x,crrs_y] = s[r_star_x*b_width+r_star_y]
    return np.array((crrs_x+row-r_star_x,crrs_y+col-r_star_y))
    
def ann(a_feature, ap_feature):
    layer = len(a_feature)
    flann = []
    for z in range(layer):
        flann.append(pf.FLANN())

    flann_params = [list([])]*layer
    s = [list([])]*layer

    for l in range(1, layer):
        s_tmp = [np.hstack([a_feature[l], ap_feature[l]])]
        s[l] = np.vstack(s_tmp)

        flann_params[l] = flann[l].build_index(s[l], algorithm='kdtree')
    return flann, flann_params, s

def best_approximate_match(flann, params, f_q,width):
    result = flann.nn_index(f_q, 1, checks=params['checks'])[0][0]
    return np.array([result//width,result%width])