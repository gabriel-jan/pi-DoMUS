#!/usr/bin/env python2

import sys
import shutil
import os
import glob
from sys import platform
from sympy import *
from sympy.parsing.sympy_parser import parse_expr

def main(args=None):
    """The main routine."""
    os.system("rm -rf *-test")
# To add a test you need to adjust 
#   -testnames
#   -pre_d
#   -pre_v
#   -output

    testnames = ["z",       #trivial solution
                 "p",       #patch
                 "pt",      #patch with time
                 "rp",      #rot patch
                 "rpt",     #rot patch with time
                 "rs",      #rot sine
                 "rst",     #rot sine with time
                 "sc",      #sine cosine
                 "sct"]     #sine cosine with time
    pre_d = ["0",
             "-0.2*((y**2)/2-y)", 
             "-0.2*t*((y**2)/2-y)", 
             "-0.2*y**2*(y/3 - 1/2)", 
             "-0.2*t*y**2*(y/3 - 1/2)",
             "0.2*(1/pi)*cos(pi*y)", 
             "0.2*sin(2*pi*t)*(1/pi)*cos(pi*y)",
             "0.06*(1/(2*pi))*sin(2*pi*x)*sin(2*pi*y)", 
             "0.06*(1/(2*pi))*sin(2*pi*t)*sin(2*pi*x)*sin(2*pi*y)"]
    pre_v = ["0",
             "-((y**2)/2-y)", 
             "-t*((y**2)/2-y)", 
             "-y**2*(y/3 - 1/2)", 
             "-t*y**2*(y/3 - 1/2)",
             "(1/pi)*cos(pi*y)", 
             "sin(2*pi*t)*(1/pi)*cos(pi*y)",
             "(1/(2*pi))*sin(2*pi*x)*sin(2*pi*y)", 
             "(1/(2*pi))*sin(2*pi*t)*sin(2*pi*x)*sin(2*pi*y)"] 
    output = ["zero",
              "patch",
              "time_dep_patch",
              "rotating_patch",
              "time_dep_rotating_patch",
              "rotating_sine",
              "time_dep_rotating_sine",
              "sine_cosine",
              "time_dep_sine_cosine"]      

    # tests to test the test script
   
    press = "0"
    
    inv = 0

    # mute output
    mute = 1

    #input handling
    if args is None:
        args = sys.argv[1:]
    if len(args) == 0:
        err = ("you need to specify your test for d and for u:\n" 
               +"candidate: python run_tests.py #d #u [inv]\n"
               +"ALL: perform all tests\n")
        for it in range(0,len(output)):
            err += str(it) + "  : " + output[it] + " test ("+ testnames[it] +")\n"
        sys.exit(err)
    elif (len(args)==1) & (args[0]!="ALL"):
        err = "you need to specify your test for u:\n"
        for it in range(0,len(output)):
            err += str(it) + "  : " + output[it] + " test ("+ testnames[it] +")\n"
        sys.exit(err)
    elif (len(args)==1) & (args[0]=="ALL"):
        start_d = 0
        end_d = len(pre_d)
        start_v = 0
        end_v = len(pre_v)
    else:
        in1 = args[0]   #d
        in2 = args[1]   #v
        if in1 == "ALL":
            start_d = 0
            end_d = len(pre_d)
            start_v = int(in2)
            end_v = int(in2)+1
        elif in2 == "ALL":
            start_d = int(in1)
            end_d = int(in1)+1
            start_v = 0
            end_v = len(pre_v)
        else:
            start_d = int(in1)
            end_d = int(in1)+1
            start_v = int(in2)
            end_v = int(in2)+1
        if len(args) == 3:
            if args[2] == "inv":
                inv = 1

    x, y, z, t = var('x,y,z,t')
    local_dict = {"x": x, "y": y, "z": z, "t": t}
    inv_dict = {"x": y, "y": x, "z": z, "t": t}
    for k in range(start_d,end_d):
        for i in range(start_v,end_v):
            if inv:
                pre_d_ = parse_expr(pre_d[k], local_dict=inv_dict)
            else:
                pre_d_ = parse_expr(pre_d[k], local_dict=local_dict)

            pre_v_ = parse_expr(pre_v[i], local_dict=local_dict)
            p = parse_expr(press, local_dict=local_dict)
            pt = p.diff(t)

            d = [-pre_d_.diff(y), pre_d_.diff(x)]
            dt = [di.diff(t) for di in d]

            v = [-pre_v_.diff(y), pre_v_.diff(x)]
            vt = [vi.diff(t) for vi in v]

            if mute == 0:
                print("\nPerforming "+ output[k] +"-"+ output[i] + " test" + 
                      "\nUsing pattern d, d, u, u, p in \n"+
                      "             (" + str(d[0]) + ", " + str(d[1]) + ", " + str(v[0]) + ", " + str(v[1]) + ", " + str(p)+" )\n")
    
            f_d      = [-d[0].diff(x, 2) - d[0].diff(y, 2), -d[1].diff(x, 2) - d[1].diff(y, 2), 0, 0]
            f_navier = [0, 0, v[0].diff(t) + (v[0]-dt[0])*v[0].diff(x) + (v[1]-dt[1])*v[0].diff(y), v[1].diff(t) + (v[0]-dt[0])*v[1].diff(x) + (v[1]-dt[1])*v[1].diff(y)]
            f_stokes = [0, 0, -v[0].diff(x, 2) - v[0].diff(y, 2) + p.diff(x), -v[1].diff(x, 2) - v[1].diff(y, 2) + p.diff(y)]
    
            f_ale = []
            for j in range(len(f_navier)):
                f_ale += [f_navier[j] + f_stokes[j] + f_d[j] ] 
            
            # specify prm file
            v_str = ("subsection Dirichlet boundary conditions\n"   
                    +"  set IDs and component masks = 0=d;u\n"   
                    +"  set IDs and expressions     = 0=" 
                    + str(d[0]) + "; " + str(d[1]) + "; " + str(v[0]) + "; " + str(v[1]) + "; " + str(p) + "\n" 
                    +"  set Known component names   = d,d,u,u,p\n"
                    +"  set Used constants          = \nend\n" +
                    "subsection Time derivative of Dirichlet boundary conditions\n"   
                    +"  set IDs and component masks = 0=d;u\n"   
                    +"  set IDs and expressions     = 0=" 
                    + str(dt[0]) + "; " + str(dt[1]) + "; " + str(vt[0]) + "; " + str(vt[1]) + "; " + str(pt) + "\n"
                    +"  set Known component names   = d,d,u,u,p\n"
                    +"  set Used constants          = \nend\n" +
                    "subsection Exact solution\n"
                    +"  set Function expression = " 
                    + str(d[0]) + "; " + str(d[1]) + "; " + str(v[0]) + "; " + str(v[1]) + "; " + str(p) + "\nend\n" +
                    "subsection Forcing terms\n"   
                    +"  set IDs and component masks = 0=ALL\n"   
                    +"  set IDs and expressions     = 0=" 
                    + str(f_ale[0]) + "; " + str(f_ale[1]) + "; " + str(f_ale[2]) + "; " + str(f_ale[3]) + "; " + "0\n"
                    +"  set Known component names   = d,d,u,u,p\n  set Used constants          = \nend\n" +
                    "subsection Initial solution\n"   
                    +"  set Function expression ="   
                    + str(d[0].subs(t,0)) + "; " + str(d[1].subs(t,0)) + "; " 
                    + str(v[0].subs(t,0)) + "; " + str(v[1].subs(t,0)) + "; " + str(p.subs(t,0)) + "\nend\n" +
                    "subsection Initial solution_dot\n"   
                    +"  set Function expression ="   
                    + str(dt[0].subs(t,0)) + "; " + str(dt[1].subs(t,0)) + "; " 
                    + str(vt[0].subs(t,0)) + "; " + str(vt[1].subs(t,0)) + "; " + str(pt.subs(t,0)) +  
                    "\nend")
    
            name = testnames[k]+"-"+testnames[i]
            
            #create test directory
            os.system("mkdir " + name + "-test")
            
            # create prm file from template
            shutil.copy2("./template.prm", name+"-test/ALE_"+ name + ".prm" )

            #append to prm file
            prm = open(name+"-test/ALE_"+ name + ".prm",'w')
            prm.write(v_str.replace("**", "^"))
            prm.close()
            if platform == "darwin":
                os.system("sed -i '' 's/xxx/" + name +"/g' "+name+"-test/ALE_"+ name + ".prm" )   #output
                #os.system("sed -i '' 's/yyy/" + name +"/g' ALE_"+ name + '.prm' )   #error
            else:
                os.system("sed -i 's/xxx/" + name +"/g' "+name+"-test/ALE_"+ name + ".prm" )  #output
                #os.system("sed -i 's/yyy/" + name +"/g' ALE_"+ name + '.prm' )  #error
            
            # create and run batch file
            pbs = open(name+"-test/batch_"+ name + ".pbs",'a+')
            pbs.write('#PBS -l nodes=2:ppn=10\n')
            pbs.write('#PBS -l walltime= 6:00:00\n')
            pbs.write('#PBS -q regular\n')
            pbs.write('#PBS -m abe\n')
            pbs.write('#PBS -N '+name+'-test\n')
            pbs.write('#PBS -k oe\n\n')
            pbs.write('cd /home/jgabriel/programs/sources/pi-DoMUS/examples/heart/testing/'+name+'-test\n\n')
            pbs.write('../../build/heart --prm=ALE_"+ name +".prm --dealii >/dev/null')
            pbs.close()
            #os.system("qsub batch_"+ name +".pbs")

if __name__ == "__main__":
    main()
