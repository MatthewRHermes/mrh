import sys
import cProfile, pstats
import time 

pr = cProfile.Profile()
pr.enable()

print("Executing file= ", sys.argv[1])

t1 = time.time()
with open(sys.argv[1]) as f:
    exec(f.read())
t2 = time.time()
print("File finished in ", t2-t1, " seconds")

pr.disable()
#ps = pstats.Stats(pr).sort_stats('tottime')
ps = pstats.Stats(pr).sort_stats('cumulative')
ps.print_stats()
