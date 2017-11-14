#!/usr/bin/env python2

import sys
import shutil
import os
import os.path
import glob
from sys import platform
from sympy import *
from sympy.parsing.sympy_parser import parse_expr


# collecting convergence tables and merging them into one big table, including their testnames

def main(args=None):
    """The main routine."""

    testnames = ["rp",      #rot patch
                 "rpt",     #rot patch with time
                 "rs",      #rot sine
                 "rst",     #rot sine with time
                 "sc",      #sine cosine
                 "sct"]     #sine cosine with time
    output = ["rotating_patch",
              "time_dep_rotating_patch",
              "rotating_sine",
              "time_dep_rotating_sine",
              "sine_cosine",
              "time_dep_sine_cosine"]   

    prm = open("results.txt",'w')
    prm.write("Convergence in space: Results\n\n")
    prm.close()
    
    # loop over all test cases
    for k in range(0,len(testnames)):
        for i in range(0,len(testnames)):
            
            name = testnames[k]+"-"+testnames[i]
            
            #append to result file
            res = open("results.txt",'a+')
            res.write(name+"-test:\n")
            if os.path.exists("../build/"+name+"-error.txt"):
                file = open("../build/"+name+"-error.txt")
                res.write(file.read())
                file.close()
            else:
                res.write("not converged (yet)\n")
            res.close()
            
if __name__ == "__main__":
    main()
