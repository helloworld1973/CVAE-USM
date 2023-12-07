# -*- coding: utf-8 -*-
"""
Created on Sun May 17 17:38:17 2020

@author: wuest
"""

import numpy as np
import copy
#from .usmutils import ngon_coords
if __package__ is None:
    import sys, pathlib
    sys.path.append(pathlib.Path(__file__).parent)
    from usmutils import ngon_coords, CGR, get_alphabet_coords
else:
    from .usmutils import ngon_coords, CGR, get_alphabet_coords

# def check_alphabet(uu, A):
#     # This function takes a user-defined list of the symbols in the alphabet, A, and compares with the set of unique symbols found in seq, uu
#     # raises exceptions if the length of uu is greater than A or if a symbol occurs in seq that is not in A.

#     # sort A for consistency across data structures
#     A.sort()
#     #d is dimension of alphabet
#     d=len(A)
#     assert d >= len(uu), "Unique sequence units greater than alphabet. List of unique sequence units: {}".format(uu)
#     # ix is list of the index in A of each symbol in uu
#     ix = []
#     for i in range(len(uu)):
#         assert uu[i] in A, "Unrecognized symbol in sequence. List of unique sequence units: {}".format(uu)
#         j = A.index(uu[i])
#         ix.append(j)
#     return ix

# def get_alphabet_coords(seq, alphabet=None, form='USM'):
#     #determine number of unique symbols in seq
#     #uu is an ndarray of the unique values in seq, sorted.
#     #J is an ndarray size = len(seq) containing the index values of uu that could recreate seq
#     uu, J = np.unique(seq, return_inverse=True)
#     # get A, a sorted list of the alphabet of the sequence
#     if alphabet is not None:
#         if isinstance(alphabet, dict):
#             coord_dict = alphabet
#             A = list(coord_dict.keys())
#         else:
#             coord_dict = None
#             # assumes alphabet is an array-like object or set like object
#             A = list(alphabet)
#         A.sort()
#         # ix is list of the index in A of each symbol in uu
#         ix = check_alphabet(uu, A)
#     else:
#         coord_dict = None
#         A = uu
#         ix = None
#     # get dimension, d, of the alphabet, which equals the number of vertices in the map
#     d = len(A)
#     # get Y, a numpy array of vertex coordinates, where each row is a map coordinate
#     if coord_dict:
#         # if user provided coord_dict, Y is constructed from the 1D arrays stored in the values of the dict
#         # get list of dict values in the order of the sorted keys in A
#         vrts = [coord_dict[alph] for alph in A]
#         Y = np.array(vrts)
#     elif not coord_dict:
#         # if no coord_dict provided, Y is constructed from the default definitions
#         if form == 'USM':
#             Y = np.identity(d)
#         elif form == 'CGR':
#             Y = ngon_coords(d)
#         # create a coord_dict with the newly constructed coord array
#         coord_dict = coord_dict_make(A, Y)
#     if ix:
#         # if some of the symbols in the user-defined alphabet are missing in seq
#         # then we remove the coordinate rows in Y for the symbols that are missing
#         # so that the index of uu used in J is associated with the correct coordinate in Y.
#         Y = Y[ix]
#     # X is an N x d array where the ith row is the vertex coordinate for to the symbol at the index i in seq
#     X = Y[J]
#     return X, coord_dict

# def coord_dict_make(alphabet, vertices):
#     #creates the coord_dict from the list of alphabet and vertices in correct order
#     coord_dict=dict(zip(alphabet, vertices))
#     return coord_dict

class USM(CGR):
    """
    Creates a container for CGR coordinates with attributes related to the alphabet of the sequence,
    coordinates of alphabet, specifying forward or backward generated coordinates and form of CGR (either USM or 2D, default is USM).

    Class
    """
    __slots__= ['fw', 'bw', 'coord_dict', 'form']
    def __init__(self, forward, backward=None, coord_dict=None, form='USM'):
        self.fw=forward
        self.bw=backward
        self.coord_dict = coord_dict
        self.form=form

    def coord_dict_make(alphabet, vertices):
        #creates the coord_dict from the list of alphabet and vertices in correct order
        coord_dict=dict(zip(alphabet, vertices))
        return coord_dict

    @classmethod
    def make_usm(cls, sequence, A=None, seed='centroid', deep_copy=True):
        """
        Calculates USM coordinates of a discrete-valued sequence of arbitrary alphabet size.

        Parameters
        ----------
        sequence : LIST OR ARRAY TYPE;
            CATEGORICAL SEQUENCE OF DATA
        A : LIST OR DICT, optional
            IF LIST, SHOULD CONTAIN ALL POSSIBLE SYMBOLS OF THE ALPHABET OF THE
            SEQUENCE. FUNCTION WILL THEN CALCULATE VERTEX COORDINATES AUTOMATICALLY.
            IF DICT, SHOULD HAVE ONE ENTRY FOR EACH POSSIBLE SYMBOL OF
            THE ALPHABET OF THE SEQUENCE WHERE THE ENTRY KEY IS THE SYMBOL AND THE
            ENTRY VALUE IS THE 1D ARRAY-LIKE COORDINATE TO BE ASSOCIATED WITH
            THAT SYMBOL.
            The default is None.
            If default, will take alphabet as set of unique characters in seq and
            calculate vertex coordinates automatically.
        seed : STRING, default 'centroid';
            INDICATES THE PROCEDURE FOR SEEDING THE USM IFS
        deep_copy : BOOL
            IF TRUE (DEFAULT) WILL USE A DEEP COPY OF THE SEQUENCE TO MAKE THE USM.

        Returns
        -------
        USM : An instance of a CGR object. CGR.fw is a list of ndarrays containing
            the forward SM coordinates for each symbol in seq

        """
        if deep_copy is True:
            seq=copy.deepcopy(sequence)
        else:
            seq = sequence
        N = len(seq)
        # X is an N x d array where the ith row is the vertex coordinate for the symbol at the index i in seq
        # each key:value in coord_dict is a symbol:vertex coordinate array for
        # each symbol in the alphabet of seq
        X, coord_dict = get_alphabet_coords(seq, alphabet=A, form='USM')
        d = len(coord_dict)
        X_rev = copy.deepcopy(np.flip(X, 0))
        if seed=='centroid':
            f=[np.repeat(0.5, d)]
            b=[np.repeat(0.5, d)]
            for i in range(N):
                u=0.5*f[i]+0.5*X[i]
                v=0.5*b[i]+0.5*X_rev[i]
                f.append(u)
                b.append(v)
        elif seed=='circular':
            c=[np.repeat(0.5, d)]
            for i in range(N):
                u=0.5*c[i]+0.5*X[i]
                c.append(u)
            b=[c[-2]]
            for i in range(N):
                u=0.5*b[i]+0.5*X_rev[i]
                b.append(u)
            f=[b[-2]]
            for i in range(N):
                u=0.5*f[i]+0.5*X[i]
                f.append(u)
        elif seed=='rand':
            c=np.random(0,1,d)
            f=[c]
            b=[c]
            for i in range(N):
                u=0.5*f[i]+0.5*X[i]
                v=0.5*b[i]+0.5*X_rev[i]
                f.append(u)
                b.append(v)
        #print(c)
        #fl = [arr.tolist() for arr in f]
        fl = np.asarray(f)
        #bl = [arr.tolist() for arr in b]
        bl = np.asarray(b)
        USM=cls(forward=fl[1:], backward=bl[1:], coord_dict=coord_dict, form='USM')
        #vert_coords=list(map(tuple, np.identity(d)))
        #USM.coord_dictMake(A,vert_coords)
        return USM


    @classmethod
    def cgr2d(cls, seq, A=None):
        """
        Calculates 2D CGR coordinates according to the method for estimating a
        bijective contraction ratio proposed by Almeida & Vinga (2009).
        First estimates the contraction ratio to use based on the size of the
        alphabet of the input sequence.

        Parameters
        ----------
        seq : LIST-LIKE OBJECT
            CONTAINS THE SEQUENCE TO BE GRAPHED.
        A : LIST OR DICT
            IF LIST, SHOULD CONTAIN ALL POSSIBLE SYMBOLS OF THE ALPHABET OF THE
            SEQUENCE. FUNCTION WILL THEN CALCULATE VERTEX COORDINATES AUTOMATICALLY.
            IF DICT, SHOULD HAVE ONE ENTRY FOR EACH POSSIBLE SYMBOL OF
            THE ALPHABET OF THE SEQUENCE WHERE THE ENTRY KEY IS THE SYMBOL AND THE
            ENTRY VALUE IS THE 1D ARRAY-LIKE COORDINATE TO BE ASSOCIATED WITH
            THAT SYMBOL.
            The default is None.
            If default, will take alphabet as set of unique characters in seq and
            calculate vertex coordinates automatically.

        Returns
        -------
        CGR : a USM object containing the coordinate array and alphabet-dict.

        References
        ----------
        Almeida, J. S., &#38; Vinga, S. (2009). Biological sequences as pictures: A generic two dimensional solution for iterated maps.
            BMC Bioinformatics, 10(100), 1–7. https://doi.org/10.1186/1471-2105-10-100

        """
        # N=len(seq)
        # #uu is the set of unique symbols in seq and J inverse sequence of uu indices that recreate seq
        # uu, J = np.unique(seq, return_inverse=True)

        # if A:
        #     if isinstance(A, list):
        #         A.sort()
        #         #print("A", A)
        #         #d is dimension of alphabet
        #         d=len(A)
        #         ix = check_alphabet(uu, A)
        #         #get coordinates for vertices of an equilateral d-gon
        #         x_vals, y_vals = ngon_coords(d)
        #         # Y= np.column_stack((x_vals, y_vals))
        #         # Y=Y[ix]
        #     elif isinstance(A, dict):
        #         a, vrts = zip(*A.items())
        #         d = len(a)
        #         ix = check_alphabet(uu, a)
        #         x_vals, y_vals = zip(*vrts)
        #     # x_vals become the first column and y_vals the second
        #     # Y is an N x 2 array and each row in Y is an x,y pair
        #     Y= np.column_stack((x_vals, y_vals))
        #     Y=Y[ix]
        # else:
        #     #get number of unique symbols in seq
        #     d=len(uu)
        #     A=uu
        #     #get coordinates for vertices of an equilateral d-gon
        #     """
        #     radians=[]
        #     for k in range(d):
        #         rad = (2*np.pi*k)/d
        #         radians.append(rad)
        #     x_vals= np.cos(radians)
        #     y_vals=np.sin(radians)
        #     """
        #     x_vals, y_vals = ngon_coords(d)
        #     Y= np.column_stack((x_vals, y_vals))
        # X=Y[J]
        N = len(seq)
        # X is an N x d array where the ith row is the vertex coordinate for the symbol at the index i in seq
        # each key:value in coord_dict is a symbol:vertex coordinate array for
        # each symbol in the alphabet of seq
        X, coord_dict = get_alphabet_coords(seq, alphabet=A, form='CGR')
        d = len(coord_dict)
        k=np.rint(((d+2)/4))
        #s is the contraction ratio
        s=(2*np.cos((np.pi*(0.5-(k/d))))-2*np.cos((np.pi*(0.5-(1/(2*d)))))*np.cos(((2*k-1)*(np.pi/(2*d))))*(1+(np.tan(((2*k-1)*(np.pi/(2*d))))/np.tan((np.pi-(d+2*k-2)*(np.pi/(2*d)))))))/(2*np.cos((np.pi*(0.5-(k/d)))))
        c=[np.asarray((0,0))]
        for i in range(N):
            ci1=c[i]+s*(X[i]-c[i])
            c.append(ci1)
        cgr2d=cls(forward=c[1:], coord_dict=coord_dict, form='CGR')
        # verts=np.column_stack((x_vals, y_vals))
        # vert_coords=list(map(tuple, verts))
        # cgr2d.coord_dict_make(A, vert_coords)
        return cgr2d

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


if __name__ == '__main__':
        cwd = pathlib.Path.cwd()
        demo_dir = cwd / "tests/"
        fname = "MC0.txt"
        file_to_open = demo_dir / "test_data" / fname
        with open(file_to_open, 'r') as fhand:
            seq = fhand.read()
            data= list(seq)
            #print(data)
            mc0usm = USM.make_usm(data, A=['A','C','G','T'])
        og_mc0_coords = 'og_MC0_coords.csv'
        file_to_open2 = demo_dir / "expected_output" / og_mc0_coords
        og_mc0 = np.genfromtxt(file_to_open2, delimiter=',')
        print(np.isclose(np.array(mc0usm.fw, dtype=np.float64), og_mc0))
        print(np.allclose(np.array(mc0usm.fw, dtype=np.float64), og_mc0))
