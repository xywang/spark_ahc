from __future__ import print_function

"""
created by xywang in jul 2016
input data will be normalized in this script
dot product of normalized data is considered as similarity
tested on 7 methods, got same results as its single-threaded version

note that for ahc_sim, lookup table has one more column than the lookup table of ahc_dist.
Coz except for pairwise similarity S(x,y), ahc_sim needs S'(x,y) on which max() is operated.
And to compute S'(x,y), S(x,x) and S(y,y) are needed.

So the computation concerns (1) compute S(x,y) (2) compute S(x,x) and S(y,y) (3) compute S'(x,y)

note that if data is normalized, S(x,x)=1 always holds except for centroid and median methods.
centroid and median methods have to have self_sim_dict to update self similarities,
while the other five methods do not need.

calling cen() function is different from calling other functions, as its delta_i and delta_j
are dependent on count_dict[i] and count_dict[j].
"""

import os
import sys
os.environ['SPARK_HOME']="/home/hduser/spark-2.0.0-bin-hadoop2.7"
sys.path.append("/home/hduser/spark-2.0.0-bin-hadoop2.7/python")

# ================= load libraries ==================

import numpy as np
from pyspark.mllib.linalg import Vectors
from pyspark import SparkContext
from operator import add

# ================ create a SparkContext ============

sc = SparkContext(master="spark://159.84.139.244:7077", appName="iris_wOInd")

# ================ functions ========================

def swap_del_upd(del_key, i, j):
    """
    input (k,j) or (j,k)
    return (k,i) or (i,k)

    in this script, this function is used
    to convert keys in del_tb to keys in upd_tb
    """

    # get k
    ind_j = del_key.index(j)
    ind_k = 1 - ind_j
    k = del_key[ind_k]

    # sort i and k
    if k > i:
        upd_key = (i, k)
    else:
        upd_key = (k, i)

    return upd_key

def sim_(u,v):
    # return dot product as pairwise similarity
    sim_val = np.dot(u,v)
    return sim_val

def sim_sim_pr(u,v):
    # return S(u,v) and S'(u,v) as a value in rdd
    sim = sim_(u,v)
    sim_pr = sim - 0.5*(sim_(u,u)+sim_(v,v))
    return sim, sim_pr

#  ===============  7 updating schemas ===============

def single():
    al_i, al_j, beta, gamma = 0.5, 0.5, 0.0, -0.5
    #delta_i, delta_j = 0.5, 0.5
    return al_i, al_j, beta, gamma #, delta_i, delta_j

def complete():
    al_i, al_j, beta, gamma = 0.5, 0.5, 0.0, 0.5
    #delta_i, delta_j = 0.5, 0.5
    return al_i, al_j, beta, gamma #, delta_i, delta_j

def weighted():
    al_i, al_j, beta, gamma = 0.5, 0.5, 0.0, 0.0
    #delta_i, delta_j = 0.5, 0.5
    return al_i, al_j, beta, gamma #, delta_i, delta_j

def median():
    al_i, al_j, beta = 0.5, 0.5, -0.25
    return al_i, al_j, beta

def average(count_dict, i, j):
    para_i = float(count_dict[i])
    para_j = float(count_dict[j])
    summ = para_i + para_j
    al_i = para_i/summ
    al_j = para_j/summ
#     beta, gamma = 0.0, 0.0
#     delta_i, delta_j = 0.5, 0.5
    return al_i, al_j #, beta, gamma, delta_i, delta_j

def centroid(count_dict, i, j):
    para_i = float(count_dict[i])
    para_j = float(count_dict[j])
    summ = para_i + para_j
    al_i = para_i/summ
    al_j = para_j/summ
    beta = -para_i*para_j/summ**2
    delta_i = (para_i/summ)**2
    delta_j = (para_j/summ)**2
    return al_i, al_j, beta, delta_i, delta_j

def ward(count_dict, i, j, k):
    para_i = float(count_dict[i])
    para_j = float(count_dict[j])
    para_k = float(count_dict[k])
    summ = para_i + para_j + para_k
    al_i = (para_i+para_k)/summ
    al_j = (para_j+para_k)/summ
    beta = -para_k/summ
    return al_i, al_j, beta

LW = {'weighted': weighted,
      'average': average,
      'centroid': centroid,
      'median': median,
      'ward': ward,
      'single': single,
      'complete': complete}

# ================ functions called in rdd.map() ===========

def sin_com_wei(d, method):
    [al_i, al_j, beta, gamma] = LW[method]()
    c_ik = d[1][0][0]
    c_jk = d[1][1][0]
    c_ij = sim_ij
    val_sim_ij = c_ik * al_i + c_jk * al_j + c_ij * beta - abs(c_ik - c_jk) * gamma # update sim
    #val_self_sim_ij = i_self_sim_val[0][1] * delta_i + j_self_sim_val[0][1] * delta_j # update self_sim of ij
    val_sim_pr_ij = val_sim_ij - 1.0 # update sim'
    return (d[0], (val_sim_ij, val_sim_pr_ij))

def med(d, method):
    [al_i, al_j, beta] = LW[method]()
    c_ik = d[1][0][0]
    c_jk = d[1][1][0]
    c_ij = sim_ij
    val_sim_ij = c_ik * al_i + c_jk * al_j + c_ij * beta  # update sim
    ind_k = 1-d[0].index(i)
    k = d[0][ind_k]
    val_sim_pr_ij = val_sim_ij - 0.5*(self_sim_dict[i] + self_sim_dict[k]) # update sim'
    return (d[0], (val_sim_ij, val_sim_pr_ij))

def cen(d, al_i, al_j, beta):
    c_ik = d[1][0][0]
    c_jk = d[1][1][0]
    c_ij = sim_ij
    val_sim_ij = c_ik * al_i + c_jk * al_j + c_ij * beta  # update sim
    ind_k = 1-d[0].index(i)
    k = d[0][ind_k]
    val_sim_pr_ij = val_sim_ij - 0.5*(self_sim_dict[i] + self_sim_dict[k]) # update sim'
    return (d[0], (val_sim_ij, val_sim_pr_ij))

def ave(d, method):
    [al_i, al_j] = LW[method](count_dict, i, j)
    c_ik = d[1][0][0]
    c_jk = d[1][1][0]
    val_sim_ij = c_ik * al_i + c_jk * al_j
    val_sim_pr_ij = val_sim_ij - 1.0
    return (d[0], (val_sim_ij, val_sim_pr_ij))

def war(d, method):
    ind_k = 1-d[0].index(i)
    k = d[0][ind_k]
    [al_i, al_j, beta] = LW[method](count_dict, i, j, k)
    c_ik = d[1][0][0]
    c_jk = d[1][1][0]
    c_ij = sim_ij
    val_sim_ij = c_ik * al_i + c_jk * al_j + c_ij * beta
    val_sim_pr_ij = val_sim_ij - 1.0
    return (d[0], (val_sim_ij, val_sim_pr_ij))

#  ===============  RDD transformations ==============

lines = sc.textFile("hdfs://master:9000/user/xywang/x_700mb.txt").cache() #iris_wOInd.csv
# [start =========== use zipWithIndex() to add index for each row =============
dt_vec = lines.map(lambda line: (Vectors.dense([float(x) for x in line.strip().split(",")]), 1)) # (densevector(...),1)
num_points = dt_vec.values().reduce(add)
dt_vec_norm = dt_vec.keys().map(lambda l: l/np.linalg.norm(l)) # normalize the vectors
dt_tup = dt_vec_norm.zipWithIndex().map(lambda x: (x[1], x[0])).cache()
# =========== use zipWithIndex() to add index for each row =============== end]

# [start === on drive program, pay attention to point indexes ====
count_dict = {n:1 for n in range(num_points)}
self_sim_dict = count_dict.copy()           # to update self similarities for centroid and median methods
# ===== on drive program, pay attention to point indexes ==== end]

cart_tups = dt_tup.cartesian(dt_tup)
fil_cart_tups = cart_tups.filter(lambda d: d[1][0] > d[0][0]) # ((ind_x, vec),(ind_y, vec))
res_dict = fil_cart_tups.map(lambda tup: ((tup[0][0],tup[1][0]), 
                                          sim_sim_pr(tup[0][1],tup[1][1]))) # ((ind_x, ind_y), (sim, sim'))
res_dict.cache()
resPar = res_dict.getNumPartitions()
#res_dict = res_dict.repartition(8) # rdd.coalesce() will make error occur at rdd.mapPartitionsWithIndex()
#i_j_sim = res_dict.max(key=lambda d: d[1][1])

# ==== input a method =====
method = "single"
den_arr = []
ite = 0

while (ite < 10):

    # [start ============== look for optimal number of partitions
    # so that no empaty partition
    def length(ind, iterator): yield (ind, sum(1 for _ in iterator))
    par_ind_size = res_dict.mapPartitionsWithIndex(length)
    #par_ind_size_co = par_ind_size.collect()
    #from operator import add
    num_par = par_ind_size.filter(lambda d: d[1] != 0).map(lambda x:1).reduce(add) # count the number of emtpy partitions
    if resPar > num_par: # there exist empty partition
        res_dict = res_dict.repartition(num_par)
    else:
        None
    #new_par_num_co = new_par_num.
    #.map(lambda d:1).reduce(add) # or use rdd.count() here


    # [start ============== look for i_j_sim that contains max ============
    # method 1:
    def f(iterator): yield max(iterator, key=lambda d:d[1][1])
    max_list = res_dict.mapPartitions(f)
    i_j_sim = max_list.max(key=lambda d:d[1][1])

    # method 2:
    #i_j_sim = res_dict.glom().map(lambda list : max(list, key=lambda d:d[1][1])).max(lambda d:d[1][1])


    #res_dict.foreachPartition(lambda d: 0 if )

    # method 3:
    #i_j_sim = res_dict.max(key=lambda d: d[1][1]) # get one pair with max s'(x,y)

    # rdd.max() does NOT work here, because takes max on the first value in a tuple
    # ============== look for i_j_sim that contains max ============ end]

    i = i_j_sim[0][0]
    j = i_j_sim[0][1]
    sim_ij = i_j_sim[1][0] # i = 0, j = 2, sim_ij = 0.999987347282, sim_pr_ij = -1.26527175262e-05
    sim_pr_ij = i_j_sim[1][1]

    # ======== add one parent node to dendrogram =========
    den_arr.append([i, j, sim_ij, sim_pr_ij])

    # ================ build upd, del, rest tables ========
    res_dict_no_ij = res_dict.filter(lambda d: d[0] != (i,j))
    res_dict_no_ij.cache()
    upd_dict = res_dict_no_ij.filter(lambda d: i in d[0])
    upd_dict.cache()
    del_dict = res_dict_no_ij.filter(lambda d: j in d[0])
    del_dict.cache()
    rest_dict = res_dict_no_ij.filter(lambda d: (i not in d[0]) & (j not in d[0]))
    rest_dict.cache()

    # ================ update similarities ===================
    del_dict_swapped_keys = del_dict.map(lambda d: (swap_del_upd(d[0], i, j), d[1]))
    del_dict_swapped_keys.cache()

    # [srt == change join to a broadcast join by collecting del_dict to the driver and broadcasting it to all mappers ==
    upd_del = upd_dict.join(del_dict_swapped_keys) # ((ind_x,ind_y),(upd_(sim,sim'),del_(sim,sim')))

    # del_dict_swapped_keys_broadcast = sc.broadcast(del_dict_swapped_keys.collectAsMap())
    # del_dict_swapped_keys_br_v = del_dict_swapped_keys_broadcast.value

    # def join_upd_del(iterator):
    #     for ite in iterator:
    #         if ite[0] in del_dict_swapped_keys_br_v:
    #             yield (ite[0], (ite[1], del_dict_swapped_keys_br_v[ite[0]]))
    #
    # upd_del = upd_dict.mapPartitions(join_upd_del, preservesPartitioning=True)

    # this change speeds up computation by 8.3%
    # == change join to a broadcast join by collecting upd_dict to the driver and broadcasting it to all mappers == end]
    upd_del.cache()

    if (method == "single") | (method == "complete") | (method == "weighted"):
        new_dict = upd_del.map(lambda d: sin_com_wei(d, method))
    elif method == "median":
        self_sim_dict[i] = (self_sim_dict[i] + self_sim_dict[j]) * 0.25 # update self_sim of ij, coz iUj=i
        self_sim_dict.pop(j)
        new_dict = upd_del.map(lambda d: med(d, method))
    elif method == "ward":
        new_dict = upd_del.map(lambda d: war(d,method))
    elif method == "average":
        new_dict = upd_del.map(lambda d: ave(d,method))
    elif method == "centroid":
        [al_i, al_j, beta, delta_i, delta_j] = LW[method](count_dict, i, j)
        self_sim_dict[i] = self_sim_dict[i] * delta_i + self_sim_dict[j] * delta_j # update self_sim of ij, coz iUj=i
        self_sim_dict.pop(j)
        new_dict = upd_del.map(lambda d: cen(d, al_i, al_j, beta))

    new_dict.persist()
    # =============== combine upd + rest =================

    # new_dict_max = new_dict.max(key=lambda d: d[1][1])
    # rest_dict_max = rest_dict.max(key=lambda d: d[1][1])
    #
    # if new_dict_max[1][1] >= rest_dict_max[1][1]:
    #     i_j_sim = new_dict_max
    # else:
    #     i_j_sim = rest_dict_max

    res_dict = (new_dict + rest_dict)
    res_dict = res_dict.coalesce(4)
    res_dict.persist()

    count_dict[i] += count_dict[j]
    count_dict[j] = 0

    ite += 1

# upd_dict_debug = upd_dict.collect()
# del_dict_debug = del_dict.collect()
# rest_dict_debug = rest_dict.collect()
# del_dict_swapped_keys_debug = del_dict_swapped_keys.collect()
# upd_del_debug = upd_del.collect()

# ================ stop SparkContext =================
sc.stop()
# =============== evaluate results ====================

def cophe_array(n, den_arr):
    """input dendrogram array, output coph_arry from cophenetic matrix"""
    I = np.matrix(np.identity(n))
    cophe_mat = np.zeros([n,n])

    for i in range(len(den_arr)):
        row = den_arr[i,0]
        col = den_arr[i,1]
        val = den_arr[i,3] # use sim_pr, not sim here
        row_set = np.where(I[row] == 1)
        col_set = np.where(I[col] == 1)
        r_ls = np.array(row_set[1]).reshape(-1,).tolist()
        c_ls = np.array(col_set[1]).reshape(-1,).tolist()
        for j in range(len(r_ls)):
            cophe_mat[r_ls[j],c_ls]=val
        for k in range(len(c_ls)):
            cophe_mat[c_ls[k],r_ls]=val
        I[row,c_ls]=1

    iu1 = np.triu_indices(n,1)
    cophe_arr = cophe_mat[iu1]
    return cophe_arr

cophe_arr = cophe_array(num_points, np.asarray(den_arr))

import readline # error occurs if remove this line
from rpy2.robjects import r
from rpy2.robjects.numpy2ri import numpy2ri
ro = numpy2ri(cophe_arr)
r.assign("cophe_py", ro)
r("saveRDS(cophe_py, 'single.rds')")

