import numpy

print("Reading Polaris")
polaris = numpy.loadtxt("tdm2_polaris.dat")
print("Reading Aurora")
aurora = numpy.loadtxt("tdm2.dat")
print("correct= ", numpy.allclose(polaris, aurora))
