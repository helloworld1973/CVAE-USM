# -*- coding: utf-8 -*-
"""
Created on Thu Jul 21 15:00:00 2022

@author: Wuestney
"""
import numpy as np

class CGR:
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

    def coord_dictMake(self, alphabet, vertices):
        #creates the coord_dict from the list of alphabet and vertices in correct order
        self.coord_dict=dict(zip(alphabet, vertices))
        
def coord_dict_make(alphabet, vertices):
    #creates the coord_dict from the list of alphabet and vertices in correct order
    coord_dict=dict(zip(alphabet, vertices))
    return coord_dict

def ngon_coords(verts):
    """
    Takes number of desired vertices and outputs the x, y coordinates of a regular
    n-gon with that number of vertices.

    Parameters
    ----------
    verts : INT
        NUMBER OF DESIRED VERTICES.

    Returns
    -------

    vert_array : 2D NUMPY ARRAY
        N x 2 NUMPY ARRAY, WHERE N=verts. EACH ROW IS AN X,Y COORDINATE FOR ONE
        THE VERTICES OF THE N-GON.

    """
    radians=[]
    for k in range(verts):
        rad = (2*np.pi*k)/verts
        radians.append(rad)
    x_vals = np.cos(radians)
    y_vals =np.sin(radians)
    vert_array = np.column_stack((x_vals, y_vals))
    return vert_array

def check_alphabet(uu, A):
    # This function takes a user-defined list of the symbols in the alphabet, A, and compares with the set of unique symbols found in seq, uu
    # raises exceptions if the length of uu is greater than A or if a symbol occurs in seq that is not in A.

    # sort A for consistency across data structures
    A.sort()
    #d is dimension of alphabet
    d=len(A)
    assert d >= len(uu), "Unique sequence units greater than alphabet. List of unique sequence units: {}".format(uu)
    # ix is list of the index in A of each symbol in uu
    ix = []
    for i in range(len(uu)):
        assert uu[i] in A, "Unrecognized symbol in sequence. List of unique sequence units: {}".format(uu)
        j = A.index(uu[i])
        ix.append(j)
    return ix

def get_alphabet_coords(seq, alphabet=None, form='USM'):
    #determine number of unique symbols in seq
    #uu is an ndarray of the unique values in seq, sorted.
    #J is an ndarray size = len(seq) containing the index values of uu that could recreate seq
    uu, J = np.unique(seq, return_inverse=True)
    # get A, a sorted list of the alphabet of the sequence
    if alphabet is not None:
        if isinstance(alphabet, dict):
            coord_dict = alphabet
            A = list(coord_dict.keys())
        else:
            coord_dict = None
            # assumes alphabet is an array-like object or set like object
            A = list(alphabet)
        A.sort()
        # ix is list of the index in A of each symbol in uu
        ix = check_alphabet(uu, A)
    else:
        coord_dict = None
        A = uu
        ix = None
    # get dimension, d, of the alphabet, which equals the number of vertices in the map
    d = len(A)
    # get Y, a numpy array of vertex coordinates, where each row is a map coordinate
    if coord_dict:
        # if user provided coord_dict, Y is constructed from the 1D arrays stored in the values of the dict
        # get list of dict values in the order of the sorted keys in A
        vrts = [coord_dict[alph] for alph in A]
        Y = np.array(vrts)
    elif not coord_dict:
        # if no coord_dict provided, Y is constructed from the default definitions
        if form == 'USM':
            Y = np.identity(d)
        elif form == 'CGR':
            Y = ngon_coords(d)
        # create a coord_dict with the newly constructed coord array
        coord_dict = coord_dict_make(A, Y)
    if ix:
        # if some of the symbols in the user-defined alphabet are missing in seq
        # then we remove the coordinate rows in Y for the symbols that are missing
        # so that the index of uu used in J is associated with the correct coordinate in Y.
        Y = Y[ix]
    # X is an N x d array where the ith row is the vertex coordinate for to the symbol at the index i in seq
    X = Y[J]
    return X, coord_dict

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
#     if alphabet:
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