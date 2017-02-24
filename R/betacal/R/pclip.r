pclip <- function(vec, LB=1e-16, UB=1-1e-16) pmax(LB, pmin( vec, UB))
