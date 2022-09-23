EconvProPy

This project presents a benchmark for proteomic deconvolution. Deconvolution algorithms take heterogeneous tissues (known as "bulk"), which contain a mixture of cells, and output a prediction for the cell's proportions. As far as we know, our project is the first attempt to deconvolve heterogeneous MassSpac data. Most deconvolution methods utilize a "signature" matrix which contains the expression of each cell. In our case, the signatures are from [ref].

The main part of this project is a pipeline executing *most of the known deconvolution methods*. Also, we use a novel EM-based technique for dropouts imputation using mRNA data. Usually, deconvolution methods require a set of predefined DE genes, unfortunately, those methods are known to be very sensitive to this set of genes (or proteins in our case). Therefore we run each method on many sets of genes, and each one is automatically picked according to some sensible heuristic. 

The pipeline allows us to run all the possible combinations of set / methods / matrices with all the possible steps and Params : 

Data sets  ::############
Data set : Artificial  mixtures
	Options : 
	Pre label propagation for Cytof - yes/no 
	With copy numbers - yes/no
Data set : IBD mixtures
	Options : 
	Aggregate versions  - yes/no
	Pre label propagation for Cytof - yes/no 
	With copy numbers - yes / no
Data set : simulator 
	Options : 
	All simulator Params (look at simulation algorithm)

Computational steps :############ 
Step name : filter by Intra Variance
	Possible steps : 
	Filter out based on normalized std. Params : cutoff std
	No filtering

Step name: Aggregate Intra Variance(Aggregate  the 4 measures for each cell )
	Possible steps : 
	Aggregate Intra Variance by mean 
	Aggregate Intra Variance by median
	Take first

Step name : clean irrelevant proteins(remove all zero genes)

Step name : preprocess (finding signature)
	Possible steps : 
	Entropy based. Params : number of genes per cell, entropy to Zscore ratio, with/out normalization 
	Entropy based only largest(pick only genes that are the highest for the specific cell) . Params : number of genes per cell, entropy to Zscore ratio, with/out normalization 
	Top quantile DE(instead of Anova). Params : quantile
	Iterative feature selection .Params :  auto selected         
	No preprocess.

Step name : deconvolution 
	Possible steps : 
	basic deconvolution (none negative optimization ).Params : normalize / total counts
	CellMix .Params : normalize / total counts
  Robust linear reggresion. Params : [Todo: update readme]
  Weak learners. Params : [Todo: update readme]
  Xcell. Params : None
  Generalized Estimating Equations. Params : [Todo: update readme]
  RANSAC. Params : [Todo: update readme]
	Advanced  method : using correlation and mRNA data to impute missing values 

Matrices ::############ 
Metric : mean correlation  between expected mixtures(with deconvolution) and real mixtures (goodness of fitting ) 
	Options : 
	Correlation between the mixtures across cells
	Correlation between the cells across mixtures

Metric : mean correlation between deconvolution and Cytof 
	Options : 
	Correlation between the mixtures across cells. Params - with/out label propagation  
	Correlation between the cells across mixtures. Params - with/out label propagation  
	Per group correlation  - split the cells to two groups(high/low abundance) and measure separately. Params - with/out label propagation
	
Metric : distance conservation between Cytof and deconvolution (the deconvolution results should be the closest to the Cytof per patient across patients  )
	Options :
	Euclidean distance, Params - high dimension / over TSNE / over PCA
	 


















