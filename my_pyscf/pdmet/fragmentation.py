import numpy as np
from mrh.my_pyscf.dmet.fragmentation import Fragmentation as molFragmentation
# Author: Bhavnesh Jangid <jangidbhavnesh@uchicago.edu>

class Fragmentation(molFragmentation):
    '''
    Fragmentation class for DMET
    '''
    def __init__(self, mf, atmlst, atmlabel=None, **kwargs):
        super().__init__(mf, atmlst, atmlabel=atmlabel, **kwargs)
        self.mol = mf.cell
