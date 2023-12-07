# -*- coding: utf-8 -*-
"""
Created on Thu May 14 13:14:53 2020

@author: wuest

Module implementing Renyi Entropy computation from Universal Sequence Maps
with Parzen Kernel Estimator as introduced in the paper by Vinga and Almeida [1].

[1] S. Vinga and J. S. Almeida, “Rényi continuous entropy of DNA sequences,” Journal of Theoretical Biology, vol. 231, no. 3, pp. 377–388, 2004, doi: 10.1016/j.jtbi.2004.06.030.

"""

import numpy as np
import numexpr as ne
from scipy.spatial.distance import pdist
import matplotlib.pyplot as plt
import datetime
import pathlib
import copy


# set the default variance vector to use with Gaussian density kernels
SIG2V_DEFAULT = ('1.000000e-10', '1.778279e-10', '3.162278e-10', '5.623413e-10',
                 '1.000000e-09', '1.778279e-09', '3.162278e-09', '5.623413e-09',
                 '1.000000e-08', '1.778279e-08', '3.162278e-08', '5.623413e-08',
                 '1.000000e-07', '1.778279e-07', '3.162278e-07', '5.623413e-07',
                 '1.000000e-06', '1.778279e-06', '3.162278e-06', '5.623413e-06',
                ' 1.000000e-05', '1.778279e-05', '3.162278e-05', '5.623413e-05',
                '1.000000e-04', '1.778279e-04', '3.162278e-04', '5.623413e-04',
                '1.000000e-03', '1.778279e-03', '3.162278e-03', '5.623413e-03',
                '1.000000e-02', '1.778279e-02', '3.162278e-02', '5.623413e-02',
                '1.000000e-01', '1.778279e-01', '3.162278e-01', '5.623413e-01',
                '1', '1.778279e+00', '3.162278e+00', '5.623413e+00', '10', '1.778279e+01',
                '3.162278e+01', '5.623413e+01', '100')

def renyi4d(cgr, sig2v=SIG2V_DEFAULT, refseq=None, filesave=False):
    """
    renyi4d matches exact formula of the renyi entropy algorithm of a 4d usm used by Vinga & Almeida 2004.

    Parameters
    ----------
    cgr : LIST or ARRAY-LIKE OBJECT
        ARRAY LIKE CONTAINING USM FORWARD COORDINATES AS ROWS.
    sig2v : TUPLE OR ARRAY-LIKE OBJECT, optional
        ARRAY CONTAINING VARIANCE VALUES TO USE FOR THE GAUSSIAN DENSITY KERNEL.
        The default is the tuple of values contained in sig2v_default, the same
        values used in the original paper by Vinga & Almeida (2004).
    refseq : BOOL, optional
        NAME OF SEQUENCE. The default is None.
    filesave : BOOL, optional
        OPTION TO SAVE RESULTS TO FILE. The default is False.

    Returns
    -------
    r2usm_dict : DICT
        DICT CONTAINING SIG2:ENTROPY AS KEY:VALUE PAIRS.

    """
    #convert cgr is ndarray
    cgr = np.asarray(cgr)
    #n is sequence length and d is the size of the alphabet or dimension of the USM
    n, d = cgr.shape
    #get d(i,j) the pairwise squared euclidean distance between all sample USM points ai and aj.
    dij = pdist(cgr, 'sqeuclidean')
    r2usm_dict = {}
    for i in range(len(sig2v)):
        sig2 = sig2v[i]
        G = 2*np.sum(np.exp((-1/(4*sig2))*dij))+n
        V = (1/((n**2)*16*(np.pi**2)*(sig2**2)))*G
        r2usm = np.negative(np.log(V))
        r2usm_dict[sig2] = r2usm
    sig2_list, r2usm_list = zip(*r2usm_dict.items())
    ln_sig2 = np.log(sig2_list)
    if refseq==None:
        seqname = "cgr_{}".format(datetime.datetime.now().isoformat())
    elif refseq:
        seqname = refseq
    plt.plot(ln_sig2, r2usm_list, 'bo')
    plt.title("Renyi Quadratic Entropy for {} N={} D={}".format(seqname, n, d))
    plt.xlabel('sig2')
    plt.ylabel('Renyi2 Entropy')
    plt.show()
    if filesave==True:
        fname = 'renyi2_{}'.format(seqname)
        plt.savefig(fname, format='png')

    return r2usm_dict

def renyi2usm(cgr_coords, sig2v=SIG2V_DEFAULT, refseq=None, Plot=False, filesave=False, deep_copy=False):
    """
    Calculates Renyi quadratic entropy of a set of USM forward coordinates

    Parameters
    ----------
    cgr : ARRAY LIKE CONTAINING USM FORWARD COORDINATES AS ROWS
    sig2v : ARRAY LIKE, optional
        ARRAY LIKE CONTAINING VARIANCES, SIG2, TO USE WITH GAUSSIAN KERNEL IN
        PARZEN WINDOW METHOD. The default is the tuple of values contained in
        sig2v_default, the same values used in the original paper by
        Vinga & Almeida (2004).
    refseq : STRING, optional
        NAME OF SEQUENCE. The default is None.
    Plot : BOOLEAN, optional
        OPTION TO PLOT ENTROPY VALUES AS A FUNCTION OF THE LOG OF THE KERNEL
        VARIANCES, SIG2V. ENTROPY VALUES ON THE Y AXIS AND LN(SIG2) VALUES
        ON THE X AXIS. The default is False.
    filesave : BOOLEAN, optional
        OPTION TO SAVE RESULTS TO FILE. The default is False.
    deep_copy : BOOLEAN, optional
        IF TRUE, WILL USE A DEEP COPY OF cgr_coords TO CALCULATE THE
        ENTROPY VALUES. The default is False.

    Returns
    -------
    Dictionary containing renyi quadratic entropy of the USM for each sig2 value.

    """
    sig2v = np.array(sig2v, dtype=np.float64)
    if deep_copy is True:
        #convert cgr to ndarray
        cgr=copy.deepcopy(cgr_coords)
        cgr = np.asarray(cgr)
    else:
        cgr = np.asarray(cgr_coords)
    #n is sequence length and d is the size of the alphabet or dimension of the USM
    n, d = cgr.shape
    #get d(i,j) the pairwise squared euclidean distance between all sample USM points ai and aj.
    dij = pdist(cgr, 'sqeuclidean')
    r2usm_dict = {}
    for i in range(len(sig2v)):
        sig2 = sig2v[i]
        G = 2 * np.sum(ne.evaluate('exp((-1/(4 * sig2)) * dij)'))+n
        V = (1/((n ** 2) * ((2 * np.sqrt(sig2) * (np.sqrt(np.pi))) ** d))) * G
        r2usm = np.negative(np.log(V))
        r2usm_dict[sig2] = r2usm
    if refseq==None:
        seqname = "cgr_{}".format(datetime.datetime.now().isoformat())
    elif refseq:
        seqname = refseq
    if Plot is True:
        sig2_list, r2usm_list = zip(*r2usm_dict.items())
        ln_sig2 = np.log(sig2_list)
        plt.plot(ln_sig2, r2usm_list, 'bo')
        plt.title("Renyi Quadratic Entropy for {} N={} D={}".format(seqname, n, d))
        plt.xlabel('sig2')
        plt.ylabel('Renyi2 Entropy')
        plt.show()
        if filesave==True:
            fname = 'renyi2_{}'.format(seqname)
            plt.savefig(fname, format='png')
    return r2usm_dict

def positive_asymptote(d, sig2v=SIG2V_DEFAULT):
    """
    Function to output x and y coordinates of graph asymptote of the graph
    of Renyi vs. ln sig2 as lnsig2 approaches positive infinity. This is the
    graph plotted by renyi2usm().

    See proof in doc/demo_usm_entropy.ipynb

    Parameters
    ----------
    d : INT
        DIMENSION OF THE USM FROM WHICH r2usm_dict WAS COMPUTED.
    sig2v : ARRAY-LIKE
        DEFAULT ARE THE MODULE'S DEFAULT SIGMA SQUARED VALUES

    Returns
    -------
    xvals, yvals : AN ARRAY OF X VALUES AND AN ARRAY OF Y VALUES TO BE GRAPHED

    """
    sig2_arr = np.array(sig2v)
    xvals = np.log(sig2_arr)
    yvals = np.log((2*np.sqrt(sig2_arr)*np.sqrt(np.pi))**d)
    return xvals, yvals

def negative_asymptote(d, N, sig2v=SIG2V_DEFAULT):
    """
    Function to output x and y coordinates of graph asymptote of the graph
    of Renyi vs. ln sig2 as lnsig2 approaches negative infinity. This is the
    graph plotted by renyi2usm().

    See proof in doc/demo_usm_entropy.

    Parameters
    ----------
    d : INT
        DIMENSION OF THE USM FROM WHICH r2usm_dict WAS COMPUTED.
    N : INT
        NUMBER OF COORDINATES IN THE USM MAP (IE THE LENGTH OF THE SEQUENCE
        MAPPED IN THE USM)
    sig2v : ARRAY-LIKE
        DEFAULT ARE THE MODULE'S DEFAULT SIGMA SQUARED VALUES

    Returns
    -------
    xvals, yvals : AN ARRAY OF X VALUES AND AN ARRAY OF Y VALUES TO BE GRAPHED

    """
    sig2_arr = np.array(sig2v)
    xvals = np.log(sig2_arr)
    yvals = np.log(N*((2*np.sqrt(sig2_arr)*np.sqrt(np.pi))**d))
    return xvals, yvals


def usm_density(c, L):
    """
    Calculates the subquadrant density of USM map coordinates

    Parameters
    ----------
    c : LIST OF FORWARD USM MAP COORDINATE ARRAYS
    L : L-TUPLE, NUMBER OF USM DIVISIONS IN EACH COORDINATE
    alpha : PARAMETER FOR RENYI ENTROPY

    Returns
    -------
    vector of L-tuple counts

    """
    #d is the number of bijective contractions for tuples length L
    d=2**L
    #total number of subquadrants used for density calculation
    nbin = d**2
    t=np.floor(d*c[L-1:])
    print(t)
    B, J = np.unique(t, axis=0, return_inverse=True)
    print("B and J",B, J)
    n=np.histogram(J, bins=nbin)
    return n

if __name__ == "__main__":
    #from Bio import SeqIO
    from pyusm import USM
    """
    char_val = ["a","b","c","d","e","f","g","h","i"]
    data = random.choices(char_val, k=35000)
    data_string = ""
    data_string = data_string.join(data)
    """
    cwd = pathlib.Path.cwd()
    demo_dir = cwd / "tests/"
    fname = "R1.txt"
    file_to_open = demo_dir / "test_data" / fname
    with open(file_to_open, 'r') as fhand:
        seq = fhand.read()
        print(seq, type(seq))
        data_string = str(seq)
        #print(data_string, type(data_string))
        data= list(data_string)
        #print(data)
        cgr = USM.make_usm(data, A=['A','C','G','T'])
    rn2dict = renyi2usm(cgr.fw, refseq=fname)
    rn2arrayN = list(rn2dict.items())
    rn2arrayN.sort()
    rn2arrayN = np.array(rn2arrayN, dtype=np.float64)
    rn2file = "renyi2usm_AA.rn2"
    file_to_open2 = demo_dir / "expected_output" / rn2file
    with open(file_to_open2, 'r') as fhand:
        lines = fhand.readlines()
    rn2array = []
    row = 0
    for line in lines:
        if row < 2:
            row += 1
            continue
        else:
            row += 1
            line = line.rstrip().split('\t')
            #print(line)
            rn2array.append([float(line[0].strip()), float(line[1].strip())])
    rn2arrayOG = np.array(rn2array, dtype=np.float64)
    print(np.isclose(rn2arrayN, rn2arrayOG))
    print(np.allclose(rn2arrayN, rn2arrayOG))
    print(np.array_equal(rn2arrayN, rn2arrayOG))
        #print("cgr", cgr)
        #cgr = np.asarray(cgr)
        #np.savetxt("cgr_MC0-py-2.csv", cgr, delimiter=",")
        #sig2va = np.genfromtxt(demo_dir.joinpath('sig2.csv'), delimiter=',')
        #np.set_printoptions(suppress=False)
        # print(sig2v)

        # print(sig2v == sig2va)
        # print(sig2v.dtype)
        # print(float(sig2v[0]))
#%%
        #print("renyi2usm")
        #print("----------------------------------------------------------")
        #print(renyi2usm(cgr.fw, refseq=fname))
        #print("renyi4d")
        #print("----------------------------------------------------------")
        #print(renyi4d(cgr, sig2v, refseq=fname))

