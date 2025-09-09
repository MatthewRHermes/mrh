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

    <s',s"{dm}|{{ops}}|s,s"> = A(s',s,m) <s',m{dm}|{{ops}}|s,m>

    where {cond_mmax} = max (s,s'){cond_dm}
    not accounting for any transposition of spin sectors among the operators if present.'''

#def get_docstring_scale (

