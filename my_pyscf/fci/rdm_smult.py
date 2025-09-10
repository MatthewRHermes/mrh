import numpy as np

_scale_h = [
    [lambda s,m: ((1/2)*np.sqrt(2)*np.sqrt(-m + s)*np.sqrt(m + s)*np.sqrt(m + s - 1)/(np.sqrt(s - 1)*np.sqrt(2*s - 1))),
     lambda s,m: ((1/2)*np.sqrt(-m + s)*np.sqrt(m + s)*np.sqrt(-m + s - 1)/np.sqrt(s - 1))],
    [lambda s,m: ((1/2)*np.sqrt(2*m + 2*s)/np.sqrt(s)),
     lambda s,m: (np.sqrt(-m + s))],
    [lambda s,m: (np.sqrt(-m + s + 1)),
     lambda s,m: (np.sqrt(m + s + 1)/np.sqrt(2*s + 1))],
    [lambda s,m: ((1/2)*np.sqrt(2)*np.sqrt(-m + s + 1)*np.sqrt(-m + s + 2)*np.sqrt(m + s + 1)/np.sqrt(2*s + 1)),
     lambda s,m: ((1/2)*np.sqrt(2)*np.sqrt(-m + s + 1)*np.sqrt(m + s + 1)*np.sqrt(m + s + 2)/np.sqrt(2*s**2 + 3*s + 1))]
    ]

def scale_h (smult_bra, spin_op, smult_ket, spin_ket):
    d2s_idx = (smult_bra - smult_ket + 3)//2
    if (d2s_idx < 0) or (d2s_idx >= 4): return 0
    s = (smult_ket-1)/2
    m = spin_ket/2
    return _scale_h[d2s_idx][spin_op] (s, m)

_scale_hh = [
    [lambda s,m: ((1/2)*np.sqrt(2)*np.sqrt((m + s - 1)/(2*s - 1))*np.sqrt(m + s)/np.sqrt(s)),
     lambda s,m: (np.sqrt((-m + s)/(2*s - 1))*np.sqrt(m + s)/(np.sqrt(2*s - 1)*np.sqrt(1/(2*s - 1)))),
     lambda s,m: ((1/2)*np.sqrt(2)*np.sqrt((-m + s - 1)/(2*s - 1))*np.sqrt(-m + s)/np.sqrt(1/(2*s - 1)))],
    [lambda s,m: ((1/2)*np.sqrt(2)*np.sqrt(m + s)*np.sqrt(-m + s + 1)/np.sqrt(s)),
     lambda s,m: (1),
     lambda s,m: ((1/2)*np.sqrt(2)*np.sqrt(-m + s)*np.sqrt(m + s + 1)/np.sqrt(s))],
    [lambda s,m: ((1/2)*np.sqrt(2)*np.sqrt(-m + s + 1)*np.sqrt(-m + s + 2)),
     lambda s,m: (np.sqrt(-m + s + 1)*np.sqrt(m + s + 1)*np.sqrt(2*s**2 + 3*s + 1)/np.sqrt(4*s**3 + 8*s**2 + 5*s + 1)),
     lambda s,m: ((1/2)*np.sqrt(2)*np.sqrt(m + s + 1)*np.sqrt(m + s + 2)/np.sqrt(2*s**2 + 3*s + 1))]
    ]

def scale_hh (smult_bra, spin_op, smult_ket, spin_ket):
    d2s_idx = (smult_bra - smult_ket + 2)//2
    if (d2s_idx < 0) or (d2s_idx >= 3): return 0
    s = (smult_ket-1)/2
    m = spin_ket/2
    return _scale_hh[d2s_idx][spin_op] (s, m)

_scale_sm = [
    lambda s,m: ((1/2)*np.sqrt(2)*np.sqrt(m + s)*np.sqrt(m + s - 1)/(np.sqrt(s)*np.sqrt(2*s - 1))),
    lambda s,m: ((1/2)*np.sqrt(2)*np.sqrt(m + s)*np.sqrt(-m + s + 1)/np.sqrt(s)),
    lambda s,m: ((1/2)*np.sqrt(2)*np.sqrt(-m + s + 1)*np.sqrt(-m + s + 2))
    ]

def scale_sm (smult_bra, smult_ket, spin_ket):
    d2s_idx = (smult_bra - smult_ket + 2)//2
    if (d2s_idx < 0) or (d2s_idx >= 3): return 0
    s = (smult_ket-1)/2
    m = spin_ket/2
    return _scale_sm[d2s_idx] (s, m)

_scale_dm = [
    lambda s,m: ((1/2)*np.sqrt(-m + s)*np.sqrt(m + s)*np.sqrt(-m + s - 1)*np.sqrt(m + s - 1)/np.sqrt(2*s**2 - 5*s + 3)),
    lambda s,m: (np.sqrt(-m + s)*np.sqrt(m + s)/np.sqrt(2*s - 1)),
    lambda s,m: (1)
    ]

def scale_dm (smult_bra, smult_ket, spin_ket):
    d2s_idx = (smult_bra - smult_ket + 4)//2
    if (d2s_idx < 0) or (d2s_idx >= 3): return 0
    if smult_bra > smult_ket:
        return scale_dm (smult_ket, smult_bra, spin_ket)
    s = (smult_ket-1)/2
    m = spin_ket/2
    return _scale_dm[d2s_idx] (s, m)

_transpose_mup_h = {}
_transpose_mdown_h = {}

def mup_h (dm_0, smult_bra, spin_op, smult_ket, spin_ket):
    dm_1 = scale_h (smult_bra, spin_op, smult_ket, spin_ket) * dm_0
    old_shape = dm_1.shape
    temp_shape = (-1,) + old_shape[-1:]
    dm_1 = dm_1.reshape (*temp_shape)
    d2s_idx = smult_bra-smult_ket 
    key = (d2s_idx, spin_op)
    transpose = _transpose_mup_h.get (key, lambda x, s, m: x)
    s = (smult_ket - 1) / 2
    m = spin_ket / 2
    return transpose (dm_1, s, m).reshape (*old_shape)

def mdown_h (dm_0, smult_bra, spin_op, smult_ket, spin_ket):
    dm_1 = dm_0 / scale_h (smult_bra, spin_op, smult_ket, spin_ket)
    old_shape = dm_1.shape
    temp_shape = (-1,) + old_shape[-1:]
    dm_1 = dm_1.reshape (*temp_shape)
    d2s_idx = smult_bra-smult_ket 
    key = (d2s_idx, spin_op)
    transpose = _transpose_mdown_h.get (key, lambda x, s, m: x)
    s = (smult_ket - 1) / 2
    m = spin_ket / 2
    return transpose (dm_1, s, m).reshape (*old_shape)

def _transpose_mup_phh_0(dm_0, s, m):
    dm_1[:,0] = (((m + s - 1)/(2*s - 1)) * dm_0[:,0]
                 + ((-m + s)/(2*s - 1)) * dm_0[:,1]
                 + ((m - s)/(2*s - 1)) * dm_0[:,1].transpose (0,1,3,2))
    dm_1[:,1] = (((-m + s)/(2*s - 1)) * dm_0[:,0]
                 + ((m + s - 1)/(2*s - 1)) * dm_0[:,1]
                 + ((-m + s)/(2*s - 1)) * dm_0[:,1].transpose (0,1,3,2))
    return dm_1
def _transpose_mdown_phh_0(dm_0, s, m):
    dm_1[:,0] = (((1 - 2*m)/(-3*m + s + 1)) * dm_0[:,0]
                 + ((-m + s)/(-3*m + s + 1)) * dm_0[:,1]
                 + ((m - s)/(-3*m + s + 1)) * dm_0[:,1].transpose (0,1,3,2))
    dm_1[:,1] = (((-m + s)/(-3*m + s + 1)) * dm_0[:,0]
                 + ((1 - 2*m)/(-3*m + s + 1)) * dm_0[:,1]
                 + ((-m + s)/(-3*m + s + 1)) * dm_0[:,1].transpose (0,1,3,2))
    return dm_1
def _transpose_mup_phh_1(dm_0, s, m):
    dm_1[:,0] = (((1/2)*(m + 3*s - 1)/(2*s - 1)) * dm_0[:,0]
                 + ((1/2)*(-m + s - 1)/(2*s - 1)) * dm_0[:,1]
                 + ((1/2)*(-m + s - 1)/(2*s - 1)) * dm_0[:,0].transpose (0,1,3,2))
    dm_1[:,1] = (((1/2)*(-m + s - 1)/(2*s - 1)) * dm_0[:,0]
                 + ((1/2)*(m + 3*s - 1)/(2*s - 1)) * dm_0[:,1]
                 + ((1/2)*(m - s + 1)/(2*s - 1)) * dm_0[:,0].transpose (0,1,3,2))
    return dm_1
def _transpose_mdown_phh_1(dm_0, s, m):
    dm_1[:,0] = ((2*(m + s)/(3*m + s + 1)) * dm_0[:,0]
                 + ((m - s + 1)/(3*m + s + 1)) * dm_0[:,0].transpose (0,1,3,2)
                 + ((m - s + 1)/(3*m + s + 1)) * dm_0[:,1])
    dm_1[:,1] = (((m - s + 1)/(3*m + s + 1)) * dm_0[:,0]
                 + ((-m + s - 1)/(3*m + s + 1)) * dm_0[:,0].transpose (0,1,3,2)
                 + (2*(m + s)/(3*m + s + 1)) * dm_0[:,1])
    return dm_1
def _transpose_mup_phh_2(dm_0, s, m):
    dm_1[:,0] = (((1/4)*(m - s)/s) * dm_0[:,1].transpose (0,1,3,2)
                 + ((1/4)*(-m + s)/s) * dm_0[:,1]
                 + ((1/4)*(m + 3*s)/s) * dm_0[:,0])
    dm_1[:,1] = (((1/4)*(-m + s)/s) * dm_0[:,1].transpose (0,1,3,2)
                 + ((1/4)*(m + 3*s)/s) * dm_0[:,1]
                 + ((1/4)*(-m + s)/s) * dm_0[:,0])
    return dm_1
def _transpose_mdown_phh_2(dm_0, s, m):
    dm_1[:,0] = ((2*(m + s)/(3*m + s)) * dm_0[:,0]
                 + ((m - s)/(3*m + s)) * dm_0[:,1]
                 + ((-m + s)/(3*m + s)) * dm_0[:,1].transpose (0,1,3,2))
    dm_1[:,1] = (((m - s)/(3*m + s)) * dm_0[:,0]
                 + (2*(m + s)/(3*m + s)) * dm_0[:,1]
                 + ((m - s)/(3*m + s)) * dm_0[:,1].transpose (0,1,3,2))
    return dm_1
def _transpose_mup_phh_3(dm_0, s, m):
    dm_1[:,0] = (((1/2)*(-m + s)/s) * dm_0[:,0].transpose (0,1,3,2)
                 + ((1/2)*(m + s)/s) * dm_0[:,0]
                 + ((1/2)*(-m + s)/s) * dm_0[:,1])
    dm_1[:,1] = (((1/2)*(m - s)/s) * dm_0[:,0].transpose (0,1,3,2)
                 + ((1/2)*(-m + s)/s) * dm_0[:,0]
                 + ((1/2)*(m + s)/s) * dm_0[:,1])
    return dm_1
def _transpose_mdown_phh_3(dm_0, s, m):
    dm_1[:,0] = ((2*m/(3*m - s)) * dm_0[:,0]
                 + ((m - s)/(3*m - s)) * dm_0[:,0].transpose (0,1,3,2)
                 + ((m - s)/(3*m - s)) * dm_0[:,1])
    dm_1[:,1] = (((m - s)/(3*m - s)) * dm_0[:,0]
                 + ((-m + s)/(3*m - s)) * dm_0[:,0].transpose (0,1,3,2)
                 + (2*m/(3*m - s)) * dm_0[:,1])
    return dm_1
_transpose_mup_phh = {(-1, 0): _transpose_mup_phh_0,
                      (-1, 1): _transpose_mup_phh_1,
                      (1, 0): _transpose_mup_phh_2,
                      (1, 1): _transpose_mup_phh_3}
_transpose_mdown_phh = {(-1, 0): _transpose_mdown_phh_0,
                        (-1, 1): _transpose_mdown_phh_1,
                        (1, 0): _transpose_mdown_phh_2,
                        (1, 1): _transpose_mdown_phh_3}

def mup_phh (dm_0, smult_bra, spin_op, smult_ket, spin_ket):
    dm_1 = scale_h (smult_bra, spin_op, smult_ket, spin_ket) * dm_0
    old_shape = dm_1.shape
    temp_shape = (-1,) + old_shape[-4:]
    dm_1 = dm_1.reshape (*temp_shape)
    d2s_idx = smult_bra-smult_ket 
    key = (d2s_idx, spin_op)
    transpose = _transpose_mup_phh.get (key, lambda x, s, m: x)
    s = (smult_ket - 1) / 2
    m = spin_ket / 2
    return transpose (dm_1, s, m).reshape (*old_shape)

def mdown_phh (dm_0, smult_bra, spin_op, smult_ket, spin_ket):
    dm_1 = dm_0 / scale_h (smult_bra, spin_op, smult_ket, spin_ket)
    old_shape = dm_1.shape
    temp_shape = (-1,) + old_shape[-4:]
    dm_1 = dm_1.reshape (*temp_shape)
    d2s_idx = smult_bra-smult_ket 
    key = (d2s_idx, spin_op)
    transpose = _transpose_mdown_phh.get (key, lambda x, s, m: x)
    s = (smult_ket - 1) / 2
    m = spin_ket / 2
    return transpose (dm_1, s, m).reshape (*old_shape)

def _transpose_mup_hh_0(dm_0, s, m):
    dm_1[:] = (((1/2)*(-m + s)/s) * dm_0[:].transpose (0,2,1)
               + ((1/2)*(m + s)/s) * dm_0[:])
    return dm_1
def _transpose_mdown_hh_0(dm_0, s, m):
    dm_1[:] = (((1/2)*(m - s)/m) * dm_0[:].transpose (0,2,1)
               + ((1/2)*(m + s)/m) * dm_0[:])
    return dm_1
_transpose_mup_hh = {(0, 1): _transpose_mup_hh_0}
_transpose_mdown_hh = {(0, 1): _transpose_mdown_hh_0}

def mup_hh (dm_0, smult_bra, spin_op, smult_ket, spin_ket):
    dm_1 = scale_hh (smult_bra, spin_op, smult_ket, spin_ket) * dm_0
    old_shape = dm_1.shape
    temp_shape = (-1,) + old_shape[-2:]
    dm_1 = dm_1.reshape (*temp_shape)
    d2s_idx = smult_bra-smult_ket 
    key = (d2s_idx, spin_op)
    transpose = _transpose_mup_hh.get (key, lambda x, s, m: x)
    s = (smult_ket - 1) / 2
    m = spin_ket / 2
    return transpose (dm_1, s, m).reshape (*old_shape)

def mdown_hh (dm_0, smult_bra, spin_op, smult_ket, spin_ket):
    dm_1 = dm_0 / scale_hh (smult_bra, spin_op, smult_ket, spin_ket)
    old_shape = dm_1.shape
    temp_shape = (-1,) + old_shape[-2:]
    dm_1 = dm_1.reshape (*temp_shape)
    d2s_idx = smult_bra-smult_ket 
    key = (d2s_idx, spin_op)
    transpose = _transpose_mdown_hh.get (key, lambda x, s, m: x)
    s = (smult_ket - 1) / 2
    m = spin_ket / 2
    return transpose (dm_1, s, m).reshape (*old_shape)

_transpose_mup_sm = {}
_transpose_mdown_sm = {}

def mup_sm (dm_0, smult_bra, smult_ket, spin_ket):
    dm_1 = scale_sm (smult_bra, smult_ket, spin_ket) * dm_0
    old_shape = dm_1.shape
    temp_shape = (-1,) + old_shape[-2:]
    dm_1 = dm_1.reshape (*temp_shape)
    d2s_idx = smult_bra-smult_ket 
    key = (d2s_idx, 0)
    transpose = _transpose_mup_sm.get (key, lambda x, s, m: x)
    s = (smult_ket - 1) / 2
    m = spin_ket / 2
    return transpose (dm_1, s, m).reshape (*old_shape)

def mdown_sm (dm_0, smult_bra, smult_ket, spin_ket):
    dm_1 = dm_0 / scale_sm (smult_bra, smult_ket, spin_ket)
    old_shape = dm_1.shape
    temp_shape = (-1,) + old_shape[-2:]
    dm_1 = dm_1.reshape (*temp_shape)
    d2s_idx = smult_bra-smult_ket 
    key = (d2s_idx, 0)
    transpose = _transpose_mdown_sm.get (key, lambda x, s, m: x)
    s = (smult_ket - 1) / 2
    m = spin_ket / 2
    return transpose (dm_1, s, m).reshape (*old_shape)

def _transpose_mup_dm1_0(dm_0, s, m):
    dm_1[:,0] = (((1/2)*(m + s)/s) * dm_0[:,0]
                 + ((1/2)*(-m + s)/s) * dm_0[:,1])
    dm_1[:,1] = (((1/2)*(-m + s)/s) * dm_0[:,0]
                 + ((1/2)*(m + s)/s) * dm_0[:,1])
    return dm_1
def _transpose_mdown_dm1_0(dm_0, s, m):
    dm_1[:,0] = (((1/2)*(m + s)/m) * dm_0[:,0]
                 + ((1/2)*(m - s)/m) * dm_0[:,1])
    dm_1[:,1] = (((1/2)*(m - s)/m) * dm_0[:,0]
                 + ((1/2)*(m + s)/m) * dm_0[:,1])
    return dm_1
_transpose_mup_dm1 = {(0, 0): _transpose_mup_dm1_0}
_transpose_mdown_dm1 = {(0, 0): _transpose_mdown_dm1_0}

def mup_dm1 (dm_0, smult_bra, smult_ket, spin_ket):
    dm_1 = scale_dm (smult_bra, smult_ket, spin_ket) * dm_0
    old_shape = dm_1.shape
    temp_shape = (-1,) + old_shape[-3:]
    dm_1 = dm_1.reshape (*temp_shape)
    d2s_idx = -abs(smult_bra-smult_ket) 
    key = (d2s_idx, 0)
    transpose = _transpose_mup_dm1.get (key, lambda x, s, m: x)
    s = (max (smult_bra, smult_ket) - 1) / 2
    m = spin_ket / 2
    return transpose (dm_1, s, m).reshape (*old_shape)

def mdown_dm1 (dm_0, smult_bra, smult_ket, spin_ket):
    dm_1 = dm_0 / scale_dm (smult_bra, smult_ket, spin_ket)
    old_shape = dm_1.shape
    temp_shape = (-1,) + old_shape[-3:]
    dm_1 = dm_1.reshape (*temp_shape)
    d2s_idx = -abs(smult_bra-smult_ket) 
    key = (d2s_idx, 0)
    transpose = _transpose_mdown_dm1.get (key, lambda x, s, m: x)
    s = (max (smult_bra, smult_ket) - 1) / 2
    m = spin_ket / 2
    return transpose (dm_1, s, m).reshape (*old_shape)

def _transpose_mup_dm2_0(dm_0, s, m):
    dm_1[:,0] = (((1/2)*(m - s + 1)/(s - 1)) * dm_0[:,2]
                 + ((1/2)*(m + s - 1)/(s - 1)) * dm_0[:,0])
    dm_1[:,1] = ((1) * dm_0[:,1]
                 + ((1/2)*(-m + s - 1)/(s - 1)) * dm_0[:,2]
                 + ((1/2)*(-m + s - 1)/(s - 1)) * dm_0[:,0])
    dm_1[:,2] = (((1/2)*(m + s - 1)/(s - 1)) * dm_0[:,2]
                 + ((1/2)*(m - s + 1)/(s - 1)) * dm_0[:,0])
    return dm_1
def _transpose_mdown_dm2_0(dm_0, s, m):
    dm_1[:,0] = (((1/2)*(m + s - 1)/m) * dm_0[:,0]
                 + ((1/2)*(-m + s - 1)/m) * dm_0[:,2])
    dm_1[:,1] = (((1/2)*(m - s + 1)/m) * dm_0[:,0]
                 + (1) * dm_0[:,1]
                 + ((1/2)*(m - s + 1)/m) * dm_0[:,2])
    dm_1[:,2] = (((1/2)*(-m + s - 1)/m) * dm_0[:,0]
                 + ((1/2)*(m + s - 1)/m) * dm_0[:,2])
    return dm_1
def _transpose_mup_dm2_1(dm_0, s, m):
    dm_1[:,0] = (((1/2)*(-m**2 + s**2)/(s*(2*s - 1))) * dm_0[:,1]
                 + ((1/2)*(m**2 - 2*m*s + m + s**2 - s)/(s*(2*s - 1))) * dm_0[:,2]
                 + ((1/2)*(m**2 + 2*m*s - m + s**2 - s)/(s*(2*s - 1))) * dm_0[:,0]
                 + ((1/2)*(m**2 - s**2)/(s*(2*s - 1))) * dm_0[:,1].transpose (0,1,4,3,2)
                 + ((1/2)*(-m**2 + s**2)/(s*(2*s - 1))) * dm_0[:,1].transpose (0,3,4,1,2)
                 + ((1/2)*(m**2 - s**2)/(s*(2*s - 1))) * dm_0[:,1].transpose (0,3,2,1,4))
    dm_1[:,1] = (((1/2)*(m**2 + 2*m*s - m + s**2 - s)/(s*(2*s - 1))) * dm_0[:,1]
                 + ((1/2)*(-m**2 + s**2)/(s*(2*s - 1))) * dm_0[:,2]
                 + ((1/2)*(-m**2 + s**2)/(s*(2*s - 1))) * dm_0[:,0]
                 + ((1/2)*(-m**2 + s**2)/(s*(2*s - 1))) * dm_0[:,1].transpose (0,1,4,3,2)
                 + ((1/2)*(m**2 - 2*m*s + m + s**2 - s)/(s*(2*s - 1))) * dm_0[:,1].transpose (0,3,4,1,2)
                 + ((1/2)*(-m**2 + s**2)/(s*(2*s - 1))) * dm_0[:,1].transpose (0,3,2,1,4))
    dm_1[:,2] = (((1/2)*(-m**2 + s**2)/(s*(2*s - 1))) * dm_0[:,1]
                 + ((1/2)*(m**2 + 2*m*s - m + s**2 - s)/(s*(2*s - 1))) * dm_0[:,2]
                 + ((1/2)*(m**2 - 2*m*s + m + s**2 - s)/(s*(2*s - 1))) * dm_0[:,0]
                 + ((1/2)*(m**2 - s**2)/(s*(2*s - 1))) * dm_0[:,1].transpose (0,1,4,3,2)
                 + ((1/2)*(-m**2 + s**2)/(s*(2*s - 1))) * dm_0[:,1].transpose (0,3,4,1,2)
                 + ((1/2)*(m**2 - s**2)/(s*(2*s - 1))) * dm_0[:,1].transpose (0,3,2,1,4))
    return dm_1
def _transpose_mdown_dm2_1(dm_0, s, m):
    dm_1[:,0] = (((1/2)*(-2*m**3 - 3*m**2*s + m*s + s**3 + s**2)/(m*(-3*m**2 + s**2 + s))) * dm_0[:,0]
                 + ((1/2)*(-m**2 + s**2)/(-3*m**2 + s**2 + s)) * dm_0[:,1]
                 + ((1/2)*(-m**2 + s**2)/(-3*m**2 + s**2 + s)) * dm_0[:,1].transpose (0,3,4,1,2)
                 + ((1/2)*(-2*m**3 + 3*m**2*s + m*s - s**3 - s**2)/(m*(-3*m**2 + s**2 + s))) * dm_0[:,2]
                 + ((1/2)*(m**2 - s**2)/(-3*m**2 + s**2 + s)) * dm_0[:,1].transpose (0,1,4,3,2)
                 + ((1/2)*(m**2 - s**2)/(-3*m**2 + s**2 + s)) * dm_0[:,1].transpose (0,3,2,1,4))
    dm_1[:,1] = (((1/2)*(-m**2 + s**2)/(-3*m**2 + s**2 + s)) * dm_0[:,0]
                 + ((1/2)*(-2*m**3 - 3*m**2*s + m*s + s**3 + s**2)/(m*(-3*m**2 + s**2 + s))) * dm_0[:,1]
                 + ((1/2)*(-2*m**3 + 3*m**2*s + m*s - s**3 - s**2)/(m*(-3*m**2 + s**2 + s))) * dm_0[:,1].transpose (0,3,4,1,2)
                 + ((1/2)*(-m**2 + s**2)/(-3*m**2 + s**2 + s)) * dm_0[:,2]
                 + ((1/2)*(-m**2 + s**2)/(-3*m**2 + s**2 + s)) * dm_0[:,1].transpose (0,1,4,3,2)
                 + ((1/2)*(-m**2 + s**2)/(-3*m**2 + s**2 + s)) * dm_0[:,1].transpose (0,3,2,1,4))
    dm_1[:,2] = (((1/2)*(-2*m**3 + 3*m**2*s + m*s - s**3 - s**2)/(m*(-3*m**2 + s**2 + s))) * dm_0[:,0]
                 + ((1/2)*(-m**2 + s**2)/(-3*m**2 + s**2 + s)) * dm_0[:,1]
                 + ((1/2)*(-m**2 + s**2)/(-3*m**2 + s**2 + s)) * dm_0[:,1].transpose (0,3,4,1,2)
                 + ((1/2)*(-2*m**3 - 3*m**2*s + m*s + s**3 + s**2)/(m*(-3*m**2 + s**2 + s))) * dm_0[:,2]
                 + ((1/2)*(m**2 - s**2)/(-3*m**2 + s**2 + s)) * dm_0[:,1].transpose (0,1,4,3,2)
                 + ((1/2)*(m**2 - s**2)/(-3*m**2 + s**2 + s)) * dm_0[:,1].transpose (0,3,2,1,4))
    return dm_1
_transpose_mup_dm2 = {(-2, 0): _transpose_mup_dm2_0,
                      (0, 0): _transpose_mup_dm2_1}
_transpose_mdown_dm2 = {(-2, 0): _transpose_mdown_dm2_0,
                        (0, 0): _transpose_mdown_dm2_1}

def mup_dm2 (dm_0, smult_bra, smult_ket, spin_ket):
    dm_1 = scale_dm (smult_bra, smult_ket, spin_ket) * dm_0
    old_shape = dm_1.shape
    temp_shape = (-1,) + old_shape[-5:]
    dm_1 = dm_1.reshape (*temp_shape)
    d2s_idx = -abs(smult_bra-smult_ket) 
    key = (d2s_idx, 0)
    transpose = _transpose_mup_dm2.get (key, lambda x, s, m: x)
    s = (max (smult_bra, smult_ket) - 1) / 2
    m = spin_ket / 2
    return transpose (dm_1, s, m).reshape (*old_shape)

def mdown_dm2 (dm_0, smult_bra, smult_ket, spin_ket):
    dm_1 = dm_0 / scale_dm (smult_bra, smult_ket, spin_ket)
    old_shape = dm_1.shape
    temp_shape = (-1,) + old_shape[-5:]
    dm_1 = dm_1.reshape (*temp_shape)
    d2s_idx = -abs(smult_bra-smult_ket) 
    key = (d2s_idx, 0)
    transpose = _transpose_mdown_dm2.get (key, lambda x, s, m: x)
    s = (max (smult_bra, smult_ket) - 1) / 2
    m = spin_ket / 2
    return transpose (dm_1, s, m).reshape (*old_shape)

