#Script to show how to perform profiling 
from mrh.examples.gpu import new_analyzer

logfile = 'polymer_async/1_6-31g_out_gpu.log'
fragments = 1
sys_desc = 'polyene'
active_space = '2_2'
processor = 'gpu'
ngpu = 4
#Initialize with any logfile
data = new_analyzer.initialize_data(logfile,fragments)
#run this as many times as there are files, changing parameters each time as required
data = new_analyzer.analyze_files (logfile, fragments, sys_desc, active_space, processor, ngpu, data)
new_analyzer.generate_df(data) #returns dataframe with rows of different runs being analyzed and columns of various parameters being analyzed 

