
                                      #####################################################################################################################################
				       #                                                                                                                                   #
				       # Please see the Sec.1 of Supplementary Materials .pdf document for better illustration of the instructions for executing the code. # 
				       #                                                                                                                                   #
				       #####################################################################################################################################
	


Instructions for executing the code

Using the attached PyTorch implementation of our approach, this section presents how to reproduce the results of SetFeat4-64 and SetFeat12$^*$ (presented in table 3 of main paper) with our best set sum-min metric on the CUB dataset[10]. Since this dataset contains the fewest number of images across all tested datasets (table 1), it provides a good testbed for our approach. The backbones are detailed in table~\ref{tab:backboneparameters}. 



Please follow these step-by-step instructions to download the dataset and execute the code in Ubuntu: 

	1. Unzip and copy the Code_PaperID_2919_Matching_feature_set
	2. Go to thecubbenchmark folder at Code_PaperID_2919_Matching_feature_set/benchmarks/cub/and run cubdownload.sh. 
		If for any reason, this fails, follow the next steps	
		I) Download the CUB dataset from http://www.vision.caltech.edu/visipedia-data/CUB-200-2011/CUB_200_2011.tgz
		II) Copy the dataset to Code_PaperID_2919_Matching_feature_set/benchmarks/cub/ ,and unzip the dataset using tar -zxvf CUB_200_2011.tgz .
		III) In the same directory, run cub traintestval.py: python cub_traintestval.py .
		
	3. Go to Code_PaperID_2919_Matching_feature_set/ and run main.py: python main.py
	4. The code will be run with SetFeat12∗ by default. Feel free to change it to SetFeat4-64 in -backbone to SetFeat4 in Code_PaperID_2919_Matching_feature_set/args.py



In the evaluations, we used Cuda 11.0 with the following list of dependencies:
	- Python 3.8.10; 
	- Numpy 1.21.2; 
	- PyTorch 1.9.1+cu111; 
	- Torchvision 0.10.1; 
	- PIL 7.0.0; 
	- Einops 0.3.0. 
