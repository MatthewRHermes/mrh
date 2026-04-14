import mrh
from pyscf.lib.misc import repo_info
import os, re

def get_branch ():
    branch = repo_info (os.path.join (mrh.__file__, '..'))
    branch = branch.get ('git', '')
    branch = branch.split ('\n')
    for line in branch:
        if line.startswith ('GIT HEAD (branch'):
            branch = line
            break
    if not branch.startswith ('GIT HEAD (branch'): return ''
    branch = branch.split ()[3][:-1]
    return branch

