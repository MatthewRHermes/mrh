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


_docstring_scale = '''Compute the scale factor A(s',s,m) for the transition density matrices

    <s',m'{pdm}|{ops}|s,m'> = A(s',s,m) <s',m{pdm}|{ops}|s,m>

    where m' = max (s,s'{mdm}){mdoubprime}
    not accounting for any transposition of spin sectors among the operators if present.'''

args = {'smult_bra': r'''        smult_bra: integer
            spin multiplicity of the bra
''',
        'smult_ket': r'''        smult_ket: integer
            spin multiplicity of the ket
''',
        'spin_op': ['', r'''        spin_op: 0 or 1
            identify spin sector of operator: alpha (0) or beta (1)
''',
                    r'''        spin op: 0, 1, or 2
            identify spin sector of operator: aa (0), ba (1), or bb (2)
'''],
        'spin_ket': r'''        spin_ket: integer
            2*spin polarization (= na - nb) in the ket
'''}

def get_docstring_scale (ops, col_indices):
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
        mdm = '-m:'
        mdoubprime = ' and m" is the spin sector of the operator'
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

