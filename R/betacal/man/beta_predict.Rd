\name{beta_predict}
\alias{beta_predict}
\title{Predict Calibrated Probabilities}
\usage{
beta_predict(p, calib)
}
\arguments{
  \item{p}{A vector of probabilities that the model will calibrate.}
  \item{calib}{A list containing a calibration map, a calibration model and the fitted parameters, obtained by calling the \code{beta_calibration} function.}
}
\seealso{
  \code{\link{beta_predict}}.
}
\description{
Returns calibrated probabilities from calib$model, where calib is obtained by calling the \code{beta_calibration} function.
}
\examples{
## Creating a vector of probabilities
p <- seq(0.01,0.99,0.01)

## Creating a label vector based on the probability vector
y <- rbinom(99,1,p)

## Fitting beta calibration with three parameters
calib <- beta_calibration(p, y, "abm")

## Obtaining calibrated probabilities
probas <- beta_predict(p, calib)
}
