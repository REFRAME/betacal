\name{beta_calibration}
\alias{beta_calibration}
\title{Beta Calibration}
\usage{
beta_calibration(p, y, parameters="abm")
}
\arguments{
  \item{p}{A vector of probabilities that will be used to train the calibration model.}
  \item{y}{A vector of labels that will be used to train the calibration model.}
  \item{parameters}{The parameters that will be fitted by the model.}
}
\seealso{
  \code{\link{beta_predict}}.
}
\description{
Builds a beta calibration model on probability vector p and label vector y, fitting the parameters chosen by the user, with possible values being "abm", "ab" and "am". Returns the calibration model, the calibration map and the chosen parameters.
}
\examples{
## Creating a vector of probabilities
p <- seq(0.01,0.99,0.01)

## Creating a label vector based on the probability vector
y <- rbinom(99,1,p)

## Fitting beta calibration with three parameters
calib <- beta_calibration(p, y, "abm")

## Fitting beta calibration with two shape parameters
calib <- beta_calibration(p, y, "ab")

## Fitting beta calibration with one shape parameter and one location parameter
calib <- beta_calibration(p, y, "am")
}
