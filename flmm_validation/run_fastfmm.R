# Run fastFMM::fui on the synthetic dataset and export ground-truth results.
suppressMessages(library(fastFMM))

dat <- read.csv("flmm_validation/flmm_data.csv", stringsAsFactors = FALSE)
ycols <- grep("^Y\\.", names(dat), value = TRUE)
L <- length(ycols)
Y <- as.matrix(dat[, ycols])

d <- data.frame(id = factor(dat$id), group = factor(dat$group),
                bout_order = as.numeric(dat$bout_order))
d$Y <- Y

cat("Running fui: Y ~ group + (1 | id)\n")
fit <- fui(Y ~ group + (1 | id), data = d, family = "gaussian",
           var = TRUE, analytic = TRUE, silent = TRUE)

cat("names(fit):", paste(names(fit), collapse = ", "), "\n")
cat("dim(betaHat):", paste(dim(fit$betaHat), collapse = " x "), "\n")
cat("qn:", paste(round(fit$qn, 5), collapse = ", "), "\n")

# betaHat: p x L. Row 2 = groupB coefficient (the between-group effect).
beta <- fit$betaHat
bv <- fit$betaHat_var
cat("class(betaHat_var):", class(bv), " dim:",
    paste(dim(bv), collapse = " x "), "\n")
cat("dim(se_mat):", paste(dim(fit$se_mat), collapse = " x "), "\n")

# Extract pointwise SE + full covariance for the groupB coefficient (index 2).
coef_idx <- 2
covmat <- NULL
if (!is.null(bv) && length(dim(bv)) == 3) {
  # betaHat_var is L x L x p -> coef's full covariance surface
  covmat <- bv[, , coef_idx]
  se <- sqrt(diag(covmat))
} else if (!is.null(fit$se_mat)) {
  # se_mat is p x L pointwise standard errors
  se <- as.numeric(fit$se_mat[coef_idx, ])
} else {
  stop("could not find pointwise variance/SE in fui output")
}

betaTilde <- fit$betaTilde   # raw (unsmoothed) per-location estimates, p x L
out <- data.frame(t = seq_len(L),
                  beta_intercept = beta[1, ],
                  beta_group = beta[coef_idx, ],
                  betaTilde_group = betaTilde[coef_idx, ],
                  se_group = se)
write.csv(out, "flmm_validation/r_results.csv", row.names = FALSE)
write.csv(data.frame(coef = colnames(beta), qn = fit$qn),
          "flmm_validation/r_qn.csv", row.names = FALSE)

# Also export the full coefficient covariance matrix for groupB if available.
if (!is.null(covmat)) {
  write.csv(covmat, "flmm_validation/r_cov_group.csv", row.names = FALSE)
  cat("wrote r_cov_group.csv (", nrow(covmat), "x", ncol(covmat), ")\n")
}
cat("qn(group) =", round(fit$qn[coef_idx], 5), "\n")
cat("DONE\n")
