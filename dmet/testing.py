#import sys
#sys.path.insert (1, "/panfs/roc/groups/6/gagliard/phamx494/pyscf-1.3/pyscf/")
#import localintegrals, dmet, qcdmet_paths
#from pyscf import gto, scf, symm, future
#from pyscf import mcscf
#import numpy as np
#import HeH2_struct

import numpy as np
import pyscf_mp2, pyscf_rhf

def schmidt_decompose_1RDM (the_1RDM, frag_states, num_zero_atol=1.0e-8):
	fn_head = "schmidt_decompose_1RDM ::"
	norbs_tot = mrh.util.la.assert_matrix_square (the_1RDM)
	norbs_frag = mrh.util.basis.assert_vector_statelist (frag_states, max_element=norbs_tot, max_length=norbs_tot)
	norbs_env = norbs_tot - norbs_frag
	nelec_tot = np.trace (the_1RDM)

	# We need to SVD the fragment-environment block of the 1RDM
	# The bath states are from the right-singular vectors corresponding to nonzero singular value
	loc2frag = np.diag (frag_states, dtype=the_1RDM.dtype)
	loc2env = np.eye (norbs_tot, dtype=the_1RDM.dtype) - c2f

	loc2bath = mrh.util.basis.get_overlapping_states (loc2env, loc2frag, across_operator=the_1RDM)
	norbs_bath = loc2bath.shape[1]
	loc2imp = np.append (loc2frag, loc2bath, axis=1)
	assert (mrh.util.la.is_matrix_orthonormal (loc2imp))

	loc2core = mrh.util.basis.basis_complement (loc2imp)
	norbs_core = loc2.core.shape[1]
	assert (norbs_tot == (norbs_frag + norbs_bath + norbs_core))

	loc2emb = np.append (loc2imp, loc2core, axis=1)
	assert (mrh.util.basis.is_basis_orthonormal_and_complete (loc2emb))

	# Calculate the number of electrons in the would-be impurity model
	nelec_imp = mrh.util.basis.compute_nelec_in_subspace (the_1RDM, loc2imp)
	report_str1 = "{0} of a maximum of {1} bath orbitals found leaving {2} core orbitals".format (norbs_bath, norbs_frag, norbs_core)
	report_str2 = "{0:.3f} electrons of {1:.3f} total located on the impurity".format (nelec_imp, nelec_tot)
	print ("{0} {1}; {2}".format (fn_head, report_str1, report_str2))

	return loc2emb, norbs_bath, nelec_imp


def deform_embedding_basis (imp_case_str, frag_states, loc2fragbanned, loc2envbanned):
	fn_head = "deforming_coeffs ({0}) ::".format (imp_case_str)
	norbs_tot = loc2fragbanned.shape[0]
	norbs_frag = mrh.util.basis.assert_vector_statelist (rfrag_states, max_element=norbs_tot, max_length=norbs_tot)
	norbs_env = norbs_tot - norbs_frag
	assert (loc2envbanned.shape[0] == norbs_tot)

	# Build raw fragment coeffs and complement frag_deforming_coeffs basis
	loc2rfrag = np.eye (norbs_tot, dtype=float)[:,frag_states]

	# Expand the fragment states in the fragallowed basis	
	loc2fragallowed = mrh.util.basis.basis_complement (loc2fragbanned)
	loc2dfrag = mrh.util.basis.get_overlapping_states (loc2fragallowed, loc2rfrag, nlvecs=norbs_frag)
	assert (loc2dfrag.shape[1] == norbs_frag)
	loc2env = mrh.util.basis.basis_complement (loc2dfrag)

	# The basis states MUST span loc2env
	# so "loc2envbanned" doesn't really mean "banned", just "encouraged"
	# This is probably why there's not much effect from just deforming the environment basis 
	# unless loc2fragbanned and loc2envbanned happen to be the same
	loc2envallowed = mrh.util.basis.basis_complement (loc2envbanned)
	loc2denv = mrh.util.basis.get_overlapping_states (loc2env, loc2envallowed)
	norbs_denv = loc2denv.shape[1]

	# Complete the basis if necessary
	loc2def = np.append (loc2dfrag, loc2denv, axis=1)
	assert (mrh.util.basis.is_basis_orthonormal (loc2def))
	norbs_froz = 0
	if not mrh.util.basis.is_basis_orthonormal_and_complete (loc2def):
		loc2froz = mrh.util.basis.basis_complement (loc2def)
		norbs_froz = loc2froz.shape[1]
		loc2def = np.append (loc2def, loc2froz)
	assert (mrh.util.basis.is_basis_orthonormal_and_complete (loc2def))

	# Report
	print ("{0} {1} fragment and {2} environment orbitals have been transformed into {3} fragment, {4} environment, and {5} frozen orbitals".format(
		fn_head, norbs_frag, norbs_env, norbs_dfrag, norbs_denv, norbs_froz))

	return loc2def, norbs_dfrag, norbs_denv, norbs_froz

def deform_1RDM (imp_case, Schmidt_1RDM, frozen_1RDM, loc2def, norbs_emb):
	fn_head = "deform_1RDM ({0})::".format (imp_case)
	norbs_tot = mrh.util.la.assert_matrix_square (Schmidt_1RDM)
	mrh.util.la.assert_matrix_square (frozen_1RDM, matdim=norbs_tot)
	assert (norbs_tot >= norbs_emb)
	loc2emb = loc2def[:,:norbs_emb]

	nelec_raw_Schmidt = np.trace (Schmidt_1RDM)
	nelec_raw_frozen = np.trace (frozen_1RDM)
	nelec_raw_tot = nelec_raw_Schmidt + nelec_raw_frozen
	nelec_raw_bleed = mrh.util.basis.compute_nelec_in_subspace (frozen_1RDM, loc2emb)
	print ("{0} before projection: {1:.3f} electrons out of {2:.3f} total frozen; {3:.3f} electrons bleeding onto embedding states".format(
		fn_head, nelec_raw_frozen, nelec_raw_tot, nelec_raw_bleed))

	deformed_Schmidt_1RDM = mrh.util.basis.project_operator_into_subspace (Schmidt_1RDM, loc2emb)
	deformed_frozen_1RDM = frozen_1RDM + (Schmidt_1RDM - deformed_Schmidt_1RDM)

	nelec_deformed_Schmidt = np.trace (deformed_Schmidt_1RDM)
	nelec_deformed_frozen = np.trace (deformed_frozen_1RDM)
	nelec_deformed_bleed = mrh.util.basis.calculate_nelec (deformed_frozen_1RDM, loc2dloc)
	nelec_deformed_tot = nelec_deformed_Schmidt + nelec_deformed_frozen
	print ("{0} after projection: {1:.3f} electrons out of {2:.3f} total frozen; {3:.3f} electrons bleeding onto embedding states".format(
		fn_head, nelec_raw_frozen, nelec_raw_tot, nelec_raw_bleed))

	for RDM, nelec in (("raw total", nelec_raw_tot), ("deformed total", nelec_deformed_tot), ("deformed Schmidt", nelec_deformed_Schmidt)):
		err_str = "{0} {1} number of electrons not an even integer ({2}, mod = {3}, err_thresh = {4})".format (fn_head, RDM, nelec,
			abs (round (nelec / 2) - (nelec / 2)), num_zero_atol)
		assert (abs (round (nelec / 2) - (nelec / 2)) < num_zero_atol), err_str

	return deformed_Schmidt_1RDM, deformed_frozen_1RDM

def calc_mp2_ecorr (imp_case, DMET_object, idempotent_1RDM, correlation_1RDM, loc2def, norbs_frag, norbs_emb, num_zero_atol=1.0e-8):
	fn_head = "calc_mp2_ecorr ({0}) ::".format (imp_case)
	norbs_tot = mrh.util.la.assert_matrix_square (deformed_Schmidt_1RDM)
	mrh.util.la.assert_matrix_square (deformed_frozen_1RDM, matdim=norbs_tot)
	mrh.util.la.assert_matrix_square (loc2def, matdim=norbs_tot)
	assert (norbs_tot >= norbs_frag)
	assert (norbs_tot >= norbs_emb)
	loc2dloc = loc2def[:,:norbs_emb]
	loc2froz = loc2def[:,norbs_emb:]
	norbs_froz = norbs_tot - norbs_emb
	correlated_1RDM = idempotent_1RDM + correlation_1RDM

	# Do the Schmidt decomposition
	isfrag = np.append (np.ones (norbs_frag, dtype=int), np.zeros (norbs_emb-norbs_frag, dtype=int))
	idem_1RDM_dloc_basis = mrh.util.basis.represent_operator_in_basis (idempotent_1RDM, loc2dloc)
	corr_1RDM_dloc_basis = mrh.util.basis.represent_operator_in_basis (correlated_1RDM, loc2dloc)
	dloc2emb_corr, norbs_bath_corr, nelec_imp_corr = schmidt_decompose_1RDM (corr_1RDM_dloc_basis, isfrag)
	dloc2emb,      norbs_bath,      nelec_imp      = schmidt_decompose_1RDM (idem_1RDM_dloc_basis, isfrag)

	# Count orbitals and arrange coefficients
        loc2emb = np.dot (loc2dloc, dloc2emb)
        loc2dmet = np.append (loc2emb,      loc2froz, axis=1)
        assert (mrh.util.basis.is_basis_orthonormal_and_complete (loc2dmet))
	assert (norbs_frag + norbs_bath      <= norbs_emb)
	norbs_imp = norbs_frag + norbs_bath
	norbs_imp_corr = norbs_frag + norbs_bath_corr
	norbs_core = norbs_emb - norbs_imp
	norbs_fcor = norbs_core + norbs_froz
	assert (norbs_imp + norbs_fcor == norbs_tot)
	loc2imp = loc2dmet[:,:norbs_imp]
	loc2fcor = loc2dmet[:,norbs_imp:]

	# Partition up 1RDMs
	core_1RDM      = mrh.util.basis.project_operator_into_subspace (idempotent_1RDM, loc2fcor) + correlation_1RDM
	imp_1RDM       = correlated_1RDM - core_1RDM

	# Count electrons; compare results for schmidt-decomposing the whole thing to schmidt-decomposing only the idempotent 1RDM
	nelec_tot = np.trace (correlated_1RDM)
	nelec_bleed = mrh.util.basis.compute_nelec_in_subspace (core_1RDM, loc2imp)
        report_str1 = "Schmidt-decomposition report:"
        report_str2 = "Decomposing the correlated 1RDM leads to a {0:.3f}-electron in {1} orbital impurity problem".format (nelec_imp_corr, norbs_imp_corr)
        report_str3 = "Decomposing the idempotent 1RDM leads to a {0:.3f}-electron in {1} orbital impurity problem".format (nelec_imp, norbs_imp)
        report_str4 = "in which {0} electrons from the correlation 1RDM were found bleeding on to the impurity space".format (nelec_bleed)
	print ("{0} {1}\n{2}\n{3} {4}".format (fn_head, report_str1, report_str2, report_str3, report_str4))
	for space, nelec in (("impurity", nelec_imp), ("total", nelec_tot)):
		err_str = "{0} number of {1} electrons not an even integer ({2})".format (fn_head, space, nelec)
		err_measure = abs (round (nelec/2) - (nelec/2))
		assert (err_measure < num_zero_atol), err_str
	nelec_imp = int (round (nelec_imp))

	# Perform the solver calculation and report the energy
	# All I want to do is read off the extra correlation energy, so I'll use pyscf_rhf and pyscf_mp2 together
	# The chemical potential shouldn't matter because this is a post-facto one-off correction, so there's no breaking the number
	# (As long as I passed the assertions a few lines above!)
	imp_OEI  = DMET_object.ints.dmet_oei  (loc2dmet, norbs_imp)
	imp_FOCK = DMET_object.ints.dmet_fock (loc2dmet, norbs_imp, core_1RDM)
	imp_TEI  = DMET_object.ints.dmet_tei  (loc2dmet, norbs_imp)
	chempot = 0.0
	DMguessRHF = DMET_object.ints.dmet_init_guess_rhf( loc2dmet, norbs_imp, nelec_imp//2, norbs_frag, chempot)
	imp_erhf, imp_rhf_1RDM = pyscf_rhf.solve( 0.0, imp_OEI, imp_FOCK, imp_TEI, norbs_imp, nelec_imp, norbs_frag, DMguessRHF, chempot )
	imp_emp2, imp_mp2_1RDM = pyscf_mp2.solve( 0.0, imp_OEI, imp_FOCK, imp_TEI, norbs_imp, nelec_imp, norbs_frag, DMguessRHF, chempot )

	return (imp_emp2 - imp_erhf)

