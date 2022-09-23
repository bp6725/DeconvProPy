options(warn=-1)

library('CellMix')
library('feather')

curr_dir <- getSrcDirectory(function(x) {x})
setwd(curr_dir)

a_path <- "./A.feather"
b_path <- "./B.feather"

a_df <- data.frame(t(sapply(read_feather(a_path),c)))
b_df <- data.frame(t(sapply(read_feather(b_path),c)))

mixture <- t(data.matrix(b_df))
sig <- t(data.matrix(a_df))

mixture.non0 = mixture[rowSums(mixture)!=0,]

ged_res <- ged(mixture.non0,sig,method = 'qprog')
final_res <- coef(ged_res)
write.csv(final_res,'ged_res')
print()


