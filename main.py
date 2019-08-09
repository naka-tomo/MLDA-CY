#!/usr/bin/env python2
# -*- coding: utf-8 -*-
import mldapy
import numpy as np
import time
import pyximport
pyximport.install(setup_args={'include_dirs':[np.get_include()]}, inplace=True)
import mlda

def main():
    data = []
    data.append( np.loadtxt( "histogram_v.txt" , dtype=np.int32) )
    data.append( np.loadtxt( "histogram_w.txt" , dtype=np.int32)*5 )
    
    start = time.clock()
    mldapy.mlda( data, 3, 100, "learn_result" )
    print( "py: ", time.clock()-start , "msec" )

    start = time.clock()
    mlda.mlda( data, 3, 100, "learn_result" )
    print( "cy: ", time.clock()-start , "msec" )


if __name__ == '__main__':
    main()
