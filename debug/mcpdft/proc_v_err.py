import numpy as np
import argparse
from subprocess import Popen, PIPE
from shlex import split
from io import StringIO
from itertools import product

parser = argparse.ArgumentParser (description="tabulate the output of v_err_report for a logfile")
parser.add_argument ("-l", "--logfile", default='lih_ftpbe22_sto3g.log', help='log file (default: %(default)s)')
args = parser.parse_args ()

logfile=str(args.logfile)
fnals=['xc','ot']
cases_fnals=[["rhoa","rhob","rhoa'","rhob'"],
             ["rho","Pi","rho'","Pi'"]]

for fnal, cases in zip (fnals, cases_fnals):
    for case in cases:
        p1 = Popen (split ('grep "gradient debug {}:" {}'.format (case, logfile)),
                    stdout=PIPE)
        p2 = Popen (split ('grep "^eval_{}"'.format (fnal, logfile)),
                    stdin=p1.stdout, stdout=PIPE)
        p3 = Popen (split ("awk '{ print $2,$6,$12,$13 }'"),
                    stdin=p2.stdout, stdout=PIPE)
        stdout, stderr = p3.communicate ()
        rawtab = stdout.decode ("utf-8").replace (")", "").replace ("(","")[:-1]
        rawtab = np.loadtxt (StringIO(rawtab))
        if not rawtab.size: continue
        rawtab = rawtab.reshape (-1,20,4).transpose (1,0,2)
        idx = rawtab[:,0,0].astype (int)
        val = rawtab[:,:,1]
        err1 = rawtab[:,:,2]
        err2 = rawtab[:,:,3]
        fmt_str = ' '.join (['{}',]*(val.shape[1]+1))
        #relerr = np.abs (err1/val)
        #print (fnal, case, 'grad')
        #for ix, row in zip (idx, relerr):
        #    print (fmt_str.format (2.0**(-ix), *row))
        relerr = np.abs (err2/val)
        print (fnal, case, 'grad+hess')
        for ix, row in zip (idx, relerr):
            print (fmt_str.format (2.0**(-ix), *row))
        relerr = relerr[1:] / relerr[:-1]
        print (fnal, case, 'grad+hess conv')
        for ix, row in zip (idx, relerr):
            print (fmt_str.format (2.0**(-ix), *row))
        relerr = relerr[1:] / relerr[:-1]
    for c1, c2 in product (cases, repeat=2):
        p1 = Popen (split ('grep "Hessian debug (H.x_{})_{}:" {}'.format (c1, c2, logfile)),
                    stdout=PIPE)
        p2 = Popen (split ('grep "^eval_{}"'.format (fnal, logfile)),
                    stdin=p1.stdout, stdout=PIPE)
        p3 = Popen (split ("awk '{ print $2,$6,$10 }'"),
                    stdin=p2.stdout, stdout=PIPE)
        stdout, stderr = p3.communicate ()
        rawtab = np.loadtxt (StringIO(stdout.decode ("utf-8")[:-1]))
        if not rawtab.size: continue
        rawtab = rawtab.reshape (-1,20,3).transpose (1,0,2)
        idx = rawtab[:,0,0].astype (int)
        val = rawtab[:,:,1]
        err = rawtab[:,:,2]
        val[(val==0)&(err==0)] = 1
        relerr = np.abs (err/val)
        fmt_str = ' '.join (['{}',]*(relerr.shape[1]+1))
        print (fnal, c1, c2)
        for ix, row in zip (idx, relerr):
            print (fmt_str.format (2.0**(-ix), *row))


