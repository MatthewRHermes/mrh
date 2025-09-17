latex_header = r'''\documentclass[prb,amsmath,amssymb,floatfix,nofootinbib,superscriptaddress,reprint,onecolumn]{revtex4-1}
\usepackage{rotating} 
\usepackage{txfonts}
\usepackage{array}
\usepackage{bm}
\usepackage{dcolumn}
\usepackage{amsmath}
\usepackage{braket}
\usepackage{xfrac}
\DeclareMathOperator*{\argmin}{argmin}
\DeclareMathOperator*{\argmax}{argmax}
\newcommand{\crop}[1]{\ensuremath{\hat{c}_{#1}^\dagger}}
\newcommand{\anop}[1]{\ensuremath{\hat{c}_{#1}}}
\newcommand{\craop}[1]{\ensuremath{\hat{a}_{#1}^\dagger}}
\newcommand{\anaop}[1]{\ensuremath{\hat{a}_{#1}}}
\newcommand{\crbop}[1]{\ensuremath{\hat{b}_{#1}^\dagger}}
\newcommand{\anbop}[1]{\ensuremath{\hat{b}_{#1}}}
\newcommand{\crsop}[1]{\ensuremath{\hat{\sigma}_{#1}^\dagger}}
\newcommand{\ansop}[1]{\ensuremath{\hat{\sigma}_{#1}}}
\newcommand{\myapprox}[1]{\mathrel{\overset{\makebox[0pt]{\mbox{\normalfont\tiny\sffamily #1}}}{\approx}}}
%\newcommand{\redsout}[1]{\textcolor{red}{\sout{#1}}}
\newcommand{\pystrlit}{\textquotesingle\textquotesingle\textquotesingle}
\newcommand{\spforall}{\ensuremath{\hspace{2mm}\forall\hspace{2mm}}}
        
        
\begin{document}
                                    
'''                                 

_file_docstring = '''This file was generated from the execution of
{executable} {filepath}
using mrh
{mrh_git}
installed at
{mrh_path}
on {datetime}'''

_docstring_scale = '''Compute the scale factor A(s',s,m) for the transition density matrices

    <s',m{pdm}|{ops}|s,m> = A(s',s,m) <s',m'{pdm}|{ops}|s,m'>

    where m' = min (s,s'{mdm}){mdoubprime}
    not accounting for any transposition of spin sectors among the operators if present.'''

args = {'smult_bra': r'''integer
            spin multiplicity of the bra
''',
        'smult_ket': r'''integer
            spin multiplicity of the ket
''',
        'spin_op': ['', r'''0 or 1
            identify spin sector of operator: alpha (0) or beta (1)
''',
                    r'''0, 1, or 2
            identify spin sector of operator: aa (0), ba (1), or bb (2)
'''],
        'spin_ket': r'''integer
            2*spin polarization (= na - nb) in the ket
''',
        'cibra': r'''ndarray or list of ndarrays
            CI vectors of the bra
''',
        'ciket': r'''ndarray or list of ndarrays
            CI vectors of the ket
''',
        'norb': r'''integer
            Number of orbitals in the CI vectors
''',
        'nelec': r'''list of length 2 of integers
            Number of electrons of each spin in the ket CI vectors
''',
        'dm_0': r'''ndarray of shape (*,{dmshape})
            Transition density matrix or matrices in the "high-m" representation
''',
        'dm_1': r'''ndarray of shape (*,{dmshape})
            Transition density matrix or matrices with 2m = spin_ket
'''}

for key, val in args.items ():
    if isinstance (val, str):
        val = ' '*8 + key + ': ' + val
        args[key] = val
    elif isinstance (val, list):
        for i in range (len (val)):
            if len (val[i]) == 0: continue
            val[i] = ' '*8 + key + ': ' + val[i]

def _get_pdm_mdm_mdoubprime (col_indices):
    col_indices = list (set ([idx[0] - idx[1] for idx in col_indices]))
    if len (col_indices) == 1:
        mdoubprime = ''
        if col_indices[0] == 0:
            pdm = mdm = ''
        else:
            pdm = '+{}'.format (abs (col_indices[0]))
            mdm = '-{}'.format (abs (col_indices[0]))
            if col_indices[0] < 0:
                pdm, mdm = mdm, pdm
    else:
        pdm = '+m"'
        mdm = '-m"'
        mdoubprime = ' and m" is the spin sector of the operator'
    return pdm, mdm, mdoubprime

def get_docstring_scale (ops, col_indices):
    pdm, mdm, mdoubprime = _get_pdm_mdm_mdoubprime (col_indices)
    my_docstring = _docstring_scale.format (ops=ops, pdm=pdm, mdm=mdm, mdoubprime=mdoubprime)
    my_docstring += '\n\n    Args:\n'
    my_docstring += args['smult_bra']
    my_docstring += args['spin_op'][len(col_indices)-1]
    my_docstring += args['smult_ket']
    my_docstring += args['spin_ket']
    my_docstring += '\n    Returns:\n'
    my_docstring += '        A: float\n'
    my_docstring += '            scale factor'
    my_docstring = "    '''" + my_docstring + "'''\n"
    return my_docstring

_docstring_highm = r'''Maximize the spin polarization quantum number m = 1/2 (na - nb) for a pair of CI vectors
    corresponding to the bra and ket vectors of a transition density matrix of the type

    {dmtypes}

    and return the corresponding CI vectors. To say the same thing a different way: rotate the
    laboratory Z-axis parallel to either the bra or ket CI vectors, depending on which one leaves
    the spin vector of the operator string above unaltered. If either spin multiplicity quantum
    is omitted, it defaults to returning the input vectors unaltered.'''

def get_docstring_highm (dmtypes, col_indices):
    my_docstring = _docstring_highm.format (dmtypes=dmtypes)
    my_docstring += '\n\n    Args:\n'
    my_docstring += args['cibra']
    my_docstring += args['ciket']
    my_docstring += args['norb']
    my_docstring += args['nelec']
    my_docstring += args['spin_op'][len(col_indices)-1]
    my_docstring += '\n    Kwargs:\n'
    my_docstring += args['smult_bra']
    my_docstring += args['smult_ket']
    my_docstring += '\n    Returns:\n'
    my_docstring += args['cibra'].replace ('the bra', 'the rotated bra')
    my_docstring += args['ciket'].replace ('the ket', 'the rotated ket')
    my_docstring += args['nelec'].replace ('the ket', 'the rotated ket')
    my_docstring = "    '''" + my_docstring[:-1] + "'''\n"
    return my_docstring

_docstring_mdown = r'''Obtain the transition density matrix

    <s',m{pdm}|{ops}|s,m>

    from its "high-m" representation in which at least one of the bra or ket spin vectors is
    parallel to the laboratory Z-axis:

    <s',m'{pdm}|{ops}|s,m'>

    where m' = min (s,s'{mdm}){mdoubprime}.
    Note that this operation is not guaranteed to be invertible; for instance, it is impossible
    for a triplet to obtain the m=+1 case of the <a'a-b'b> matrix from the m=0 case, since the
    latter is zero by construction.'''


def get_docstring_mdown (ops, col_indices, ncomp):
    pdm, mdm, mdoubprime = _get_pdm_mdm_mdoubprime (col_indices)
    dmshape = ','.join (['nmo',]*len (ops.split ()))
    if ncomp > 1: dmshape = '{},'.format (ncomp) + dmshape
    my_docstring = _docstring_mdown.format (pdm=pdm, mdm=mdm, mdoubprime=mdoubprime, ops=ops)
    my_docstring += '\n\n    Args:\n'
    my_docstring += args['dm_0'].format (dmshape=dmshape)
    my_docstring += args['smult_bra']
    my_docstring += args['spin_op'][len(col_indices)-1]
    my_docstring += args['smult_ket']
    my_docstring += args['spin_ket']
    my_docstring += '\n    Returns:\n'
    my_docstring += args['dm_1'].format (dmshape=dmshape)
    my_docstring = "    '''" + my_docstring[:-1] + "'''\n"
    return my_docstring

