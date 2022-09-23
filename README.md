EconvProPy
This project is for internal usage of the Shan-Orr lab at the Technion.

*Notice :<br/>
1) This project doesn't cotein the raw data given the GIT storage limitations.<br/>
2) This project was built to be generic, each one can add any algorithms to the pipe, just try to inherit from the right classes<br/>
3) The pipeline is *highly" cached. We have - 1) Tree cash - each "leaf" in the execution DAG is cached based on its predecessor and their params (that is, a small change in the higher leaves will make everything run again ). 2) E2E cache. <br/>
4) Some methods require an external process for running R code, so you need R on your pc.<br/>
5) The "Archive" folder contains all the detailed results, the "dashboards" module can read those results and plot lots of beautiful figures.<br/>
6) I will add an explanation for the "simulated data".<br/>
7) This project contains 9399 lines of python code:) (as of today)  So if you have a question, better talk to me first. <br/>




This project presents a benchmark for proteomic deconvolution. Deconvolution algorithms take heterogeneous tissues (known as "bulk"), which contain a mixture of cells, and output a prediction for the cell's proportions. As far as we know, our project is the first attempt to deconvolve heterogeneous MassSpac data. Most deconvolution methods utilize a "signature" matrix which contains the expression of each cell. In our case, the signatures are from [ref].

The main part of this project is a pipeline executing *most of the known deconvolution methods*. Also, we use a novel EM-based technique for dropouts imputation using mRNA data. Usually, deconvolution methods require a set of predefined DE genes, unfortunately, those methods are known to be very sensitive to this set of genes (or proteins in our case). Therefore we run each method on many sets of genes, and each one is automatically picked according to some sensible heuristic. 

The pipeline allows us to run all the possible combinations of set / methods / matrices with all the possible steps and Params.

This is a *partial* list of the possible steps implemented in the project :

Data sets  ::############<br/>
Data set : Artificial  mixtures<br/>
	Options : <br/>
	Pre label propagation for Cytof - yes/no <br/>
	With copy numbers - yes/no <br/>
Data set : IBD mixtures <br/>
	Options :  <br/>
	Aggregate versions  - yes/no <br/>
	Pre label propagation for Cytof - yes/no  <br/>
	With copy numbers - yes / no <br/>
Data set : simulator  <br/>
	Options :  <br/>
	All simulator Params (look at simulation algorithm) <br/>

Computational steps :############  <br/>
Step name : filter by Intra Variance <br/>
	Possible steps :  <br/>
	Filter out based on normalized std. Params : cutoff std <br/>
	No filtering <br/>

Step name: Aggregate Intra Variance(Aggregate  the 4 measures for each cell ) <br/>
	Possible steps :  <br/>
	Aggregate Intra Variance by mean  <br/>
	Aggregate Intra Variance by median <br/>
	Take first <br/>

Step name : clean irrelevant proteins(remove all zero genes) <br/>

Step name : preprocess (finding signature) <br/>
	Possible steps :  <br/>
	Entropy based. Params : number of genes per cell, entropy to Zscore ratio, with/out normalization <br/> 
	Entropy based only largest(pick only genes that are the highest for the specific cell) . Params : number of genes per cell, entropy to Zscore  <br/>ratio, with/out normalization  <br/>
	Top quantile DE(instead of Anova). Params : quantile <br/>
	Iterative feature selection .Params :  auto selected    <br/>      
	No preprocess. <br/>

Step name : deconvolution  <br/>
	Possible steps :  <br/>
	basic deconvolution (none negative optimization ).Params : normalize / total counts <br/>
	CellMix .Params : normalize / total counts <br/>
  Robust linear reggresion. Params : [Todo: update readme] <br/>
  Weak learners. Params : [Todo: update readme] <br/>
  Xcell. Params : None <br/>
  Generalized Estimating Equations. Params : [Todo: update readme] <br/>
  RANSAC. Params : [Todo: update readme] <br/>
	Advanced  method : using correlation and mRNA data to impute missing values  <br/>

Matrices ::############  <br/>
Metric : mean correlation  between expected mixtures(with deconvolution) and real mixtures (goodness of fitting )  <br/>
	Options :  <br/>
	Correlation between the mixtures across cells <br/>
	Correlation between the cells across mixtures <br/>

Metric : mean correlation between deconvolution and Cytof  <br/>
	Options :  <br/>
	Correlation between the mixtures across cells. Params - with/out label propagation   <br/>
	Correlation between the cells across mixtures. Params - with/out label propagation   <br/>
	Per group correlation  - split the cells to two groups(high/low abundance) and measure separately. Params - with/out label propagation <br/>
	
Metric : distance conservation between Cytof and deconvolution (the deconvolution results should be the closest to the Cytof per patient across patients  ) <br/>
	Options : <br/>
	Euclidean distance, Params - high dimension / over TSNE / over PCA <br/>
	 


















