# D2AGE

This is the cource codes for our paper : \
	Distance-aware DAG Embedding for Proximity Search on Heterogeneous Graphs. 

============================================================

Files list:\
	1). D2AGE : the main dir for the source code \
	2). readme : this file

The codes are written in python-2.7, and we use theano for model development. You should generate the subpaths by yourselves, that is: \
	1) random walk in the given graph \
	2) truncate the subpaths from the sampled paths then save to file.

After this step, you can use these codes to first generate the DAGs, and then to model them by D2AGE.

============================================================

D2AGE directory

There are two directories in /D2AGE/, symmetric and asymmetric.
The symmetric dir is the source codes for symmetric relation; while the asymmetric dir is the source codes for asymmetric relation. Next we only use the symmetric to explain the details.

In /D2AGE/symmetric, \
	1)pythonParamsConfig : this file is to set all the parameters used in the model. We explain these parameters in this file. \
	2)prepareSubgraphsWithAllSubpaths.py : this file is to generate the DAGs between (q,v) by the given sampled subpaths. \
	3)experimentForOneFileByParams.py : after DAG generation, you could use this file to train the model, and then test the model.
	
For methods in other files, they would be called in the above three files.

============================================================

If you use the code, please cite our paper:

@inproceedings{liu2018distance, \
  title={Distance-aware DAG Embedding for Proximity Search on Heterogeneous Graphs}, \
  author={Liu, Zemin and Zheng, Vincent W and Zhao, Zhou and Zhu, Fanwei and Chang, Kevin Chen-Chuan and Wu, Minghui and Ying, Jing}, \
  year={2018}, \
  organization={AAAI} \
}
