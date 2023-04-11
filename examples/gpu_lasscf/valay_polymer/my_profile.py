import sys
import cProfile, pstats

pr = cProfile.Profile()
pr.enable()

print("Executing file= ", sys.argv[1])

with open(sys.argv[1]) as f:
    exec(f.read())

pr.disable()
#ps = pstats.Stats(pr).sort_stats('tottime')
ps = pstats.Stats(pr).sort_stats('cumulative')
ps.print_stats()
