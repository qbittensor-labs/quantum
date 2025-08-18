#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb 23 13:00:12 2024

@author: yz23558
"""

#%config InlineBackend.figure_formats = ['svg']
import numpy as np
import quimb as qu
import quimb.tensor as qtn
from quimb.tensor.optimize import TNOptimizer
from scipy.stats import ortho_group
import matplotlib.pyplot as plt
def range_unitary_pollmann(psi, i_start, n_apply, list_u3, depth, n_Qbit,data_type,seed_val,Qubit_ara,uni_list,rand =False):
    gate_round=None
    if n_Qbit==0: depth=1
    if n_Qbit==1: depth=1
    c_val=0
    for r in range(depth):
      for i in range(i_start, i_start+n_Qbit, 1):
         #print("U_e", i, i + 1, n_apply)
         if uni_list !=None:
            if n_apply < len(uni_list):
             G = uni_list[n_apply]
            else:
             #G = qu.rand_uni(4, dtype=complex)
             G = qu.identity(4,dtype='complex128')#+qu.randn((4,4))*val_iden
         else:
            #G = qu.rand_uni(4, dtype=complex) 
             G = qu.identity(4,dtype='complex128')
         psi.gate_(G, (i, i + 1), tags={'U',f'G{n_apply}',f'lay{Qubit_ara}',f'P{Qubit_ara}L{i}D{r}'})
         list_u3.append(f'G{n_apply}')
         n_apply+=1
         c_val+=1

    return n_apply, list_u3

def range_unitary(psi, i_start, n_apply, list_u3, depth, n_Qbit,data_type,seed_val, Qubit_ara,uni_list,rand = True):
    gate_round=None
    if n_Qbit==0: depth=1
    if n_Qbit==1: depth=1

    c_val=0
    for r in range(depth):

     if r%2==0:
      for i in range(i_start, i_start+n_Qbit, 2):
         #print("U_e", i, i + 1, n_apply)
         if uni_list !=None:
            if n_apply < len(uni_list):
             G = uni_list[n_apply]
            else:
             #G = qu.rand_uni(4, dtype=complex)
             G = qu.identity(4,dtype='complex128')#+qu.randn((4,4))*val_iden
 
         else:
            if rand == True:
                G = qu.rand_uni(4, dtype=complex)
            else:
                G = qu.identity(4,dtype='complex128')
         psi.gate_(G, (i, i + 1), tags={'U',f'G{n_apply}', f'lay{Qubit_ara}',f'P{Qubit_ara}L{i}D{r}'})
         list_u3.append(f'G{n_apply}')
         n_apply+=1
         c_val+=1

     elif r%2!=0:
      for i in range(i_start, i_start+n_Qbit-1, 2):
         #print("U_o", i+1, i + 2, n_apply)
         if uni_list!=None:
            if n_apply<len(uni_list):
             G = uni_list[n_apply]
            else:
             #G = qu.rand_uni(4, dtype=complex)#
             G = qu.identity(4,dtype='complex128')#+qu.randn((4,4))*val_iden
         else:
            if rand == True:
                G = qu.rand_uni(4, dtype=complex)
            else:
                G = qu.identity(4,dtype='complex128')
         psi.gate_(G, (i+1, i + 2), tags={'U',f'G{n_apply}',f'lay{Qubit_ara}',f'P{Qubit_ara}L{i}D{r}'})
         list_u3.append(f'G{n_apply}')
         n_apply+=1
         c_val+=1

    return n_apply, list_u3

def qmps_f(L=16, in_depth=2, n_Qbit=3, data_type='float64', qmps_structure="brickwall", canon="left",  n_q_mera=2, seed_init=0, internal_mera="brickwall", uni_list = None,rand = True):

   seed_val=seed_init
   list_u3=[]
   n_apply=0
   psi = qtn.MPS_computational_state('0' * L)
   for i in range(L):
     t = psi[i]
     indx = 'k'+str(i)
     t.modify(left_inds=[indx])

   for t in  range(L):
     psi[t].modify(tags=[f"I{t}", "MPS"])


   if canon=="left":

    for i in range(0,L-n_Qbit,1):
     #print ("quibit", i+n_Qbit, n_Qbit)
     Qubit_ara=i+n_Qbit
     if qmps_structure=="brickwall":
      n_apply, list_u3=range_unitary(psi, i, n_apply, list_u3, in_depth, n_Qbit,data_type,seed_val, Qubit_ara,uni_list = uni_list,rand =rand)
     elif qmps_structure=="pollmann":
      n_apply, list_u3=range_unitary_pollmann(psi, i, n_apply, list_u3, in_depth, n_Qbit,data_type,seed_val, Qubit_ara,uni_list= uni_list,rand =rand)


   return psi.astype_('complex128')#, list_u3

def save_para(qmps_old): #transfer parameters between 2 qmps;
    tag_list=list(qmps_old.tags)
    tag_final=[]
    for i_index in tag_list:
        if i_index.startswith('G'): tag_final.append(i_index)
    dic_mps={}
    for i in tag_final:
        t = qmps_old[i]
        t = t if isinstance(t, tuple) else [t]
        dic_mps[i] = t[0].data
    return dic_mps

def uni_list(dic,val_iden=0.,val_dic = 0.): #create the unitary list 
    uni_list = {}
    opt_tags = list(dic.keys())
    #for i in (opt_tags):
    #    uni_list[i] = qu.identity(4,dtype='complex128')+qu.randn((4,4))*val_iden
    if dic != None:
        for j in dic:
            uni_list[j] = dic[j].reshape(4,4).T + qu.randn((4,4))*val_dic
    return list(uni_list.values())

def norm_f(psi):
    # method='qr' is the default but the gradient seems very unstable
    # 'mgs' is a manual modified gram-schmidt orthog routine
    return psi.isometrize(method='mgs',allow_no_left_inds=True)

def average_peak_weight(L =10,depth = 100, shots=100):
    peak = []
    for i in range (shots):
        psi_2 = qmps_f(L, in_depth=depth, n_Qbit=L-1, qmps_structure="brickwall", canon="left",  n_q_mera=2, seed_init=10, internal_mera="brickwall")
        peak.append(max(abs((psi_2^all).data.reshape(2**L))**2))
    return np.mean(peak), np.std(peak)/np.sqrt(shots), np.max(peak)

def negative_overlap(psi, target):
    return - abs((target.H & psi)^all) ** 2  # minus so as to minimize


