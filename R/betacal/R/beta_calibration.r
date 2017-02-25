beta_calibration <- function(p, y, parameters="abm"){
  p <- pmax(1e-16, pmin(p, 1-1e-16))
  if (parameters == "abm"){
    d <- data.frame(y)
    d$lp <- log(p)
    d$l1p <- -log(1-p)

    fit <- glm(y~lp+l1p,family=binomial(link='logit'),data=d)

    a <- as.numeric(fit$coefficients['lp'])
    b <- as.numeric(fit$coefficients['l1p'])
    if (a < 0){
      fit <- glm(y~l1p,family=binomial(link='logit'),data=d)
      a <- 0
      b <- as.numeric(fit$coefficients['l1p'])
    }else if (b < 0){
      fit <- glm(y~lp,family=binomial(link='logit'),data=d)
      a <- as.numeric(fit$coefficients['lp'])
      b <- 0
    }
    inter <- as.numeric(fit$coefficients['(Intercept)'])
    m <- uniroot(function(mh) b*log(1-mh)-a*log(mh)-inter,c(1e-16,1-1e-16))$root

    calibration <- list("map" = c(a,b,m), "model" = fit, "parameters" = parameters)
    return(calibration)

  }else if (parameters == "am"){
    d <- data.frame(y)
    d$lp <- log(p / (1 - p))

    fit <- glm(y~lp,family=binomial(link='logit'), data=d)

    inter = as.numeric(fit$coefficients['(Intercept)'])
    a <- as.numeric(fit$coefficients['lp'])
    b <- a
    m <- 1.0 / (1.0 + exp(inter / a))

    calibration <- list("map" = c(a,b,m), "model" = fit, "parameters" = parameters)
    return(calibration)

  }else if (parameters == "ab"){
    d <- data.frame(y)
    d$lp <- log(2 * p)
    d$l1p <- log(2*(1-p))

    fit = glm(y~lp+l1p-1,family=binomial(link='logit'), data=d)

    a <- as.numeric(fit$coefficients['lp'])
    b <- -as.numeric(fit$coefficients['l1p'])
    m = 0.5

    calibration <- list("map" = c(a,b,m), "model" = fit, "parameters" = parameters)
    return(calibration)
  }else{
    stop("Unknown parameters. Expected abm, am or ab.")
  }
}
