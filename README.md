# Beta calibration: a well-founded and easily implemented improvement on logistic calibration for binary classifiers

[Meelis Kull], [Telmo de Menezes e Silva Filho] and [Peter Flach]

For optimal decision making under variable class distributions and misclassification costs a classifier needs to produce well-calibrated estimates of the posterior probability. Isotonic calibration is a powerful non-parametric method that is however prone to overfitting on smaller datasets; hence a parametric method based on the logistic curve is commonly used. While logistic calibration is designed to correct for a specific kind of distortion where classifiers tend to score on too narrow a scale, we demonstrate experimentally that many classifiers including naive Bayes and Adaboost suffer from the opposite distortion where scores tend too much to the extremes. In such cases logistic calibration can easily yield probability estimates that are worse than the original scores. Moreover, the logistic curve family does not include the identity function, and hence logistic calibration can easily uncalibrate a perfectly calibrated classifier. 



In this paper we solve all these problems with a richer class of calibration maps based on the Beta distribution. We derive the method from first principles and show that fitting it is as easy as fitting a logistic curve. Extensive experiments show that beta calibration is superior to logistic calibration for naive Bayes and Adaboost.

# Packages

To make it easier for practitioners to experiment with our method, we have developed packages for [Python] and [R].

* [Python package] 
* [R package]

# Tutorials

We provide usage tutorials for beta calibration in Python and R.

* [Python tutorial] 
* [R tutorial]

# Citing Beta Calibration

If you want to cite this work, please use the following citation format: 

_Kull, M., Silva Filho, T.M. and Flach, P., Beta calibration: a well-founded and easily implemented improvement on logistic calibration for binary classifiers. AISTATS 2017. (in press)_.

# Support or Contact

If you are having problems executing the experiments or the tutorials, do not hesitate to [open an issue] or contact us.

[//]: # (References)
   [Meelis Kull]: <http://www.bris.ac.uk/engineering/people/meelis-kull/>
   [Telmo de Menezes e Silva Filho]: <https://www.researchgate.net/profile/Telmo_Silva_Filho>
   [Peter Flach]: <https://www.cs.bris.ac.uk/~flach/>
   [Python]: <https://www.python.org/>
   [R]: <https://www.r-project.org/>
   [open an issue]: <https://github.com/REFRAME/betacal/issues>
   [Python tutorial]: <https://github.com/REFRAME/betacal/blob/master/python/tutorial/Python%20tutorial.ipynb>
[R tutorial]: <https://github.com/REFRAME/betacal/blob/master/R/tutorial/Rtutorial.pdf>
[Python package]: <https://pypi.python.org/pypi/betacal>
[R package]: <https://cran.r-project.org/web/packages/betacal/index.html>