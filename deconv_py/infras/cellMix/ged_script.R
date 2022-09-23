print('start Rscript')
options(warn=-1)
.libPaths(c("C:/Users/Shenorr/Documents/R/win-library/3.5", .libPaths()))
print(.libPaths())
#library('CellMix',lib.loc="C:\\Users\\Shenorr\\Documents\\R\\win-library\\3.5\\")
library('CellMix')

library('feather')

print_log <- function(message,verbose) {
  if (verbose) {
    print(message)
  }
}

args = commandArgs(trailingOnly=TRUE)

dir_path <- args[1]

verbose <- FALSE

print_log("dir path is :" ,verbose)
print_log( dir_path,verbose)

calculate_result <- function(dir_path){
  mixture_path <- file.path(dir_path,"\\mixture")
  sign_path <-  file.path(dir_path,"\\sign")
  
  mixture <- t(data.matrix(data.frame(t(sapply(read_feather(mixture_path),c)))))
  sig <- t(data.matrix(data.frame(t(sapply(read_feather(sign_path),c)))))
  
  mixture.non0 = mixture[rowSums(mixture)!=0,]
  
  print_log("start ged",verbose)
  ged_res <- ged(mixture.non0,sig)
  final_res <- coef(ged_res)
  
  print_log("write_results to csv",verbose)
  write.csv(final_res,file.path(dir_path,"\\ged_res"))
  
  print_log("write finish indc",verbose)
  fileConn<-file(file.path(dir_path,"\\run_finished.txt"))
  writeLines("finish", fileConn)
  close(fileConn)
}

on_failer <- function(e) {
  print_log("write fail indc",verbose)
  fileConn<-file(file.path(dir_path,"\\run_failed.txt"))
  writeLines(toString(e), fileConn)
  close(fileConn)
} 

tryCatch(calculate_result(dir_path = dir_path),error = on_failer)
  

