'''
Author: Bhavnesh Jangid <jangidbhavnesh@uchicago.edu>

This file creates the orbital energy dictionary from
the open molcas basis file.

Depandacies: 
1. Pickel
'''

import pickle

element_symbols = [
    "H", "He", "Li", "Be", "B", "C", "N", "O", "F", "Ne",
    "Na", "Mg", "Al", "Si", "P", "S", "Cl", "Ar", "K", "Ca",
    "Sc", "Ti", "V", "Cr", "Mn", "Fe", "Co", "Ni", "Cu", "Zn",
    "Ga", "Ge", "As", "Se", "Br", "Kr", "Rb", "Sr", "Y", "Zr",
    "Nb", "Mo", "Tc", "Ru", "Rh", "Pd", "Ag", "Cd", "In", "Sn",
    "Sb", "I", "Te", "Xe", "Cs", "Ba", "La", "Ce", "Pr", "Nd",
    "Pm", "Sm", "Eu", "Gd", "Tb", "Dy", "Ho", "Er", "Tm", "Yb",
    "Lu", "Hf", "Ta", "W", "Re", "Os", "Ir", "Pt", "Au", "Hg",
    "Tl", "Pb", "Bi", "Po", "At", "Rn", "Fr", "Ra", "Ac", "Th",
    "Pa", "U", "Np", "Pu", "Am", "Cm", "Bk", "Cf", "Es", "Fm",
    "Md", "No", "Lr", "Rf", "Db", "Sg", "Bh", "Hs", "Mt", "Ds",
    "Rg", "Cn", "Nh", "Fl", "Mc", "Lv", "Ts", "Og"]

# Only using uptill Z-96 for the ANO-RCC
element_symbols = element_symbols[:96]

def extractBasisData(filepath, element):
    data = []
    with open(filepath, 'r', encoding='utf-8') as file:
        start = False
        searchword = "/" + element + "."
        for line in file:
            line = line.strip()

            if line.startswith(searchword) and not start:
                start = True

            if start:
                if line == "":
                    break
                data.append(line)
                
    return data

def check_orbital_energy_option(data):
    return any("OrbitalEnergies" in line for line in data)

def getallangularshell(data):
    possible_ml = []
    for line in data:
        if ("type functions") in line:
            possible_ml.append(line.split()[1][0])
    return possible_ml

def extractonlyangularshell(data, shell='s'):
    extracted_data = []
    start_extracting = False

    for line in data:
        line = line.strip()
        if (shell+"-type functions") in line:
            start_extracting = True
            extracted_data.append(line)
            continue 
        if ("type functions") in line or (line == ""):
            if start_extracting:
                break
        if start_extracting:
            extracted_data.append(line)
    return extracted_data

def extractOrbitalEnergy(data):
    energy = []
    if data[-1] == '0':
        energy.append(0.)
        return energy
    elif len(data[-2]) == 1:
        energy = [float(x) for x in data[-1].split()]
        return energy
    else:
        return None

def getorbitalenergy(basis):
    '''
    args:
        basis: filepath
    return:
        orbital energy: dict
    '''

    orbitalEnergy = {}

    for element in element_symbols:
        data = extractBasisData(basis, element)
        if not check_orbital_energy_option(data):
            print(f"Orbital energies are not there in the \
            basis set file for {element}")
        possible_ml = getallangularshell(data)
        orbenergyele = {}
        for ml in possible_ml:
            data_ml = extractonlyangularshell(data, shell=ml)
            energy = extractOrbitalEnergy(data_ml)
            lvalue = angularmomentum.index(ml)
            orbenergyele[lvalue] = energy
        orbitalEnergy[element] = orbenergyele

    return orbitalEnergy

if __name__ == "__main__":
    angularmomentum = ['s', 'p', 'd', 'f', 'g', 'h', 'j']
    orbitalEnergy = getorbitalenergy('ANO-RCC')
    with open("orbitalenergy.pkl", 'wb') as f:
        pickle.dump(orbitalEnergy, f)



