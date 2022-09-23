## ---- echo=FALSE, results="hide", message=FALSE-------------------------------
knitr::opts_chunk$set(error=FALSE, message=FALSE, warning=FALSE)
library(BiocStyle)

## -----------------------------------------------------------------------------
library(celldex)
ref_tabulate <- HumanPrimaryCellAtlasData()

## ----tabulate, echo=FALSE-----------------------------------------------------
samples_tabulate <- colData(ref_tabulate)[,c("label.main", "label.fine","label.ont")]
samples <- as.data.frame(samples_tabulate)
DT::datatable(unique(samples_tabulate))

## -----------------------------------------------------------------------------
ref_Blueprint <- BlueprintEncodeData()

## ---- echo=FALSE, ref.label="tabulate"----------------------------------------
samples_Blueprint <- colData(ref_Blueprint)[,c("label.main", "label.fine","label.ont")]
samples_Blueprint <- as.data.frame(samples_Blueprint)
DT::datatable(unique(samples_Blueprint))



## -----------------------------------------------------------------------------
ref_ImmuneCellExpression <- DatabaseImmuneCellExpressionData()

## ---- echo=FALSE, ref.label="tabulate"----------------------------------------
samples_ImmuneCellExpression <- colData(ref_ImmuneCellExpression)[,c("label.main", "label.fine","label.ont")]
samples_ImmuneCellExpression <- as.data.frame(samples_ImmuneCellExpression)
DT::datatable(unique(samples_ImmuneCellExpression))


## -----------------------------------------------------------------------------
ref_Monaco <- MonacoImmuneData()

## ---- echo=FALSE, ref.label="tabulate"----------------------------------------
samples_Monaco <- colData(ref_Monaco)[,c("label.main", "label.fine","label.ont")]
samples_Monaco <- as.data.frame(samples_Monaco)
DT::datatable(unique(sample_Monacos))

## -----------------------------------------------------------------------------
sessionInfo()
