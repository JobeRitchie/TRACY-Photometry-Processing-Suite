args <- commandArgs(trailingOnly=TRUE)
data_csv <- args[1]; out_csv <- args[2]; qn_csv <- args[3]
formula_str <- args[4]; refkey <- args[5]
suppressMessages(library(fastFMM))
dat <- read.csv(data_csv, stringsAsFactors=FALSE)
ycols <- grep("^Y\\.", names(dat), value=TRUE)
L <- length(ycols)
Y <- as.matrix(dat[, ycols])
d <- data.frame(id=factor(dat$id))
if ("group" %in% names(dat)) d$group <- factor(dat$group)
d$bout_order <- as.numeric(dat$bout_order)
d$Y <- Y
fit <- fui(as.formula(formula_str), data=d, family="gaussian",
           var=TRUE, analytic=TRUE, silent=TRUE)
beta <- fit$betaHat
bv <- fit$betaHat_var
nm <- rownames(beta)
if (is.null(nm)) nm <- as.character(seq_len(nrow(beta)))
idx <- switch(refkey,
  "zero"          = which(grepl("Intercept", nm)),
  "across_bouts"  = which(nm == "bout_order"),
  "between_groups"= which(grepl("^group", nm) & !grepl(":", nm)),
  "interaction"   = which(grepl(":", nm)))
if (length(idx) == 0) {
  idx <- switch(refkey, "zero"=1, "across_bouts"=2,
                "between_groups"=2, "interaction"=nrow(beta))
}
idx <- idx[1]
b <- beta[idx, ]
se <- sqrt(diag(bv[, , idx]))
qn <- fit$qn[idx]
write.csv(data.frame(t=seq_len(L), beta=b, se=se), out_csv, row.names=FALSE)
write.csv(data.frame(qn=qn), qn_csv, row.names=FALSE)
cat("rowname idx=", idx, " name=", nm[idx], " qn=", round(qn,5), "\n")
cat("OK\n")
