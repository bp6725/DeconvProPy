print('start Rscript')
options(warn=-1)

library('CellMix')
library('feather')

print_log <- function(message,verbose) {
  if (verbose) {
    print(message)
  }
}

args = commandArgs(trailingOnly=TRUE)

mixture_feather <- args[1]
sign_feather <-  args[2]

verbose <- TRUE

mixture <- t(data.matrix(data.frame(t(sapply(mixture_feather,c)))))
sig <- t(data.matrix(data.frame(t(sapply(sign_feather,c)))))

print(dim(mixture))
print(dim(sign))

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