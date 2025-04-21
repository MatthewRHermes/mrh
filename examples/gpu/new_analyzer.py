import pandas as pd
import os
import numpy as np
from glob import glob
analytic_position_dict={'wall':-2}#,'CPU':-6}

## Basic script to isolate and analyze wall times for different kernels based on where they are

def p_counts(file_names, parameter, analytic):
	position=analytic_position_dict[analytic]
	total=0
	if parameter=='LASSCF energy':position=-1
	try:
		for file_name in file_names:
			total+=sum([float(line.split()[position]) for line in open(file_name).readlines() if parameter in line])
		return round(float(total),8)
	except:
		return 0
def p_counts_combined(file_names, parameter,combined_parameters, analytic):
	position=analytic_position_dict[analytic]
	total=0
	try:
		for combined_parameter in combined_parameters:
			for file_name in file_names:
				print(file_name)
				lines = open(file_name).readlines()
				total+=sum([float(line.split()[position]) for i,line in enumerate(lines) if (parameter in line) and ((combined_parameter in lines[i+1]) and (i+1<len(lines)))])
				#[print(line) for i,line in enumerate(lines) if (parameter in line) and ((excluded_parameter in lines[i+1]) and (i+1<len(lines)))]
		return round(float(total),8)
	except:
		return 0


def fetch_fragment_files(logfile,fragments):
	all_files = glob(logfile+"*")
	fragment_files=[logfile+'.'+str(i) for i in range(fragments)]
	flas_files=[ x for x in all_files if x not in fragment_files+[logfile]]
	#print(fragment_files, flas_files)
	return fragment_files,flas_files
def fetch_flas_file(logfile):
	return logfile+'.flas'
def provide_combinations():
	combine_dict = {'df vj and vk ': ['LASSCF macro','init E','cycle= '],
                        'df vj and vk': ['density fitting ao2mo  ']
                         }
	return combine_dict
def create_parameter_file_dict(logfile,fragments):
	fragment_files, flas_files=fetch_fragment_files(logfile,fragments)
	parameter_file_dict={	#'cholesky_eri':[logfile],
				#'scf_cycle':[logfile],
				'LASSCF energy':[logfile],
                 		'LASSCF kernel': [logfile],
                                #'contract1': [logfile],
                 		'LASCI kernel': flas_files,
                                'las_ao2mo': flas_files,
                 		#'df vj and vk ':[logfile],
                 		#'df vj and vk':fragment_files,
                 		'df vj and vk':flas_files + fragment_files + [logfile],
				#'vj_mo in microcycle':flas_files,#[flas_file],     
                                #'vk_mo vPpj in microcycle':flas_files,#[flas_file],   
                                #'vk_mo (bb|jj) in microcycle':flas_files,#[flas_file],
                                #'vk_mo (bi|aj) in microcycle':flas_files,#[flas_file],
                                'update_h2eff_sub':flas_files,#[flas_file],
                                
                 		'LASSCF setup':[logfile],
                 		#'Pull keyframe for fragment':[logfile],
                 		#'Push keyframe for fragment':[logfile],
                 		'keyframe for fragment':[logfile],
                 		'Fragment CASSCF':[logfile],
                 		#'Push keyframe for fragment':[logfile],
                 		'Energy and gradient calculation':[logfile],
                 		#'for 1-step CASSCF':fragment_files,
			        #'density fitting ao2mo pass1 ': fragment_files,                                  
                 	        #'density fitting papa pass2': fragment_files,                                    
			        #'density fitting ppaa pass2': fragment_files,                                    
                 	        'density fitting ao2mo  ': fragment_files,                                       
                 	        'FCI solver':fragment_files,                                                      
				#'integral transformation to LAS space':flas_files,
				#'Two-electron integrals':fragment_files,
				#'initial get_veff':flas_files,
				#'FCI box for subspace':flas_files,
				#'Hessian constructor':flas_files,#[flas_file],
				#'Hessian operator 1':flas_files,#[flas_file],
				#'Hessian operator 2':flas_files,#[flas_file],
                                #'LASCI get_veff_Heff 1':flas_files,#[flas_file],
                                #'LASCI get_veff_Heff 2':flas_files,#[flas_file],
                                #'LASCI get_veff_Heff 3':flas_files,#[flas_file],     
                                #'LASCI get_veff_Heff 4':flas_files,#[flas_file],      
                                #'LASCI get_veff_Heff 5':flas_files,#[flas_file],
				#'Hessian operator 3':flas_files,#[flas_file],
				#'Hessian operator 4':flas_files,#[flas_file],
				#'Hessian operator 5':flas_files,#[flas_file],
				#'Hessian operator 6':flas_files,#[flas_file],
				#'Hessian operator 7':flas_files,#[flas_file],
				#'Hessian operator 8':flas_files,#[flas_file],
				#'Hessian operator total':flas_files,#[flas_file],
				#'update_h2eff_sub':flas_files,#[flas_file],
				#'microcycles':flas_files,
				#'get_veff after secondorder':flas_files,
				#'ci_cycle':flas_files,#[flas_file],
				#'get_veff after ci':flas_files,#[flas_file],
								#'get_jk setup':fragment_files+[logfile,flas_file],
	}
	return parameter_file_dict

def do_system_and_create_row(sys_desc,active_space,processor_type,analytic,parameter_list,logfile,fragments,ngpu):
	row=[sys_desc,active_space,ngpu,processor_type,analytic]
	parameter_file_dict=create_parameter_file_dict(logfile,fragments)
	combine_dict = provide_combinations()
	for parameter,file_names in parameter_file_dict.items():
		results=p_counts(file_names,parameter,analytic)
		for parameter_2,combined_parameters in combine_dict.items():
			if parameter_2 == parameter:
				print('combined parameter found, subtracting ', parameter, 'from', combined_parameters)
				#results -=p_counts_combined(file_names, parameter, combined_parameters, analytic)
		row.append(results)
	if row[-1]==0:
		print(row,' did not finish - check errors')
	return row

def get_row (sys_desc, active_space, processor, ngpu, logfile,fragments):
    parameter_file_dict=create_parameter_file_dict(logfile,fragments)
    parameter_list=list(parameter_file_dict.keys())
    return do_system_and_create_row(sys_desc,active_space=active_space,processor_type=processor,analytic='wall',parameter_list=parameter_list,logfile=logfile,fragments=fragments,ngpu=ngpu)
    

def initialize_data (logfile,fragments):
    parameter_file_dict=create_parameter_file_dict(logfile,fragments)
    parameter_list=list(parameter_file_dict.keys())
    return np.hstack((np.array(['System','AS','ngpu','Type','Analytic']),parameter_list))

def analyze_files(logfile, fragments, sys_desc, active_space, processor, ngpu, data):
    row = get_row(sys_desc, active_space, processor, ngpu, logfile, fragments)
    data=np.vstack((data,row))
    return data

def generate_df(data):
    df=pd.DataFrame(data[1:],columns=data[0])
    print(df.T)
    return df

