# -*- coding: utf-8 -*-
"""
Created on Fri Nov 13 13:01:55 2020

@author: lucas
"""
import pandas as pd
para_dct={"blade_ksize": [7, 11, 19, 23],
		"lap_ksize": [ 3, 5, 7, 9, 11],
		"thin_th":  [-3500, -3000 ,-2500, -2000, -1500],
		"fat_th":	[-200, -100, 0, 100, 200]}

b= 15
l= 7
t= -2500
f= 0

para_df = pd.DataFrame(columns=para_dct.keys())

i = 0
for k,v in para_dct.items():
    for para in v:
        para_df.loc[i,:] = [b,l,t,f]
        para_df.loc[i, k] = para
        i += 1
        
para_df.to_csv("data/sensitivity_paras.csv")
        
    
# %%

   
    