beta_predict <- function(p, calib){
  p <- pmax(1e-16, pmin(p, 1-1e-16))
  d <- data.frame(p)
  if (calib$parameters == "abm"){
    d$lp <- log(p)
    d$l1p <- -log(1-p)
  }else if (calib$parameters == "am"){
    d$lp <- log(p / (1 - p))
  }else if (calib$parameters == "ab"){
    d$lp <- log(2 * p)
    d$l1p <- log(2*(1-p))
  }
  return(predict(calib$model, newdata=d, type="response"))
}
