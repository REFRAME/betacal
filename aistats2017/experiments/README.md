# Beta calibration - Experiments

This is the code we used for all the paper's experiments.
The code follows scikit-learn's calibration structure, using the CalibratedClassifierCV class, though we handle the cross-validation ourselves, to ensure that all calibration methods have access to exactly the same calibration set.

There's a main file for each one of the base classifiers:

* Logistic Regression: main_lr.py
* Adaboost: main_boosting.py
* Naive Bayes: main_nb.py

All experiments can be parallelized using scoop, with instructions at the beginning of each main file.

## Dependencies

* [Numpy] - NumPy is the fundamental package for scientific computing with
  Python.
* [Scikit-learn] - Machine Learning in Python.
* [Scoop] - Scalable COncurrent Operations in Python (optional).

## License

MIT

[//]: # (References)
   [Numpy]: <http://www.numpy.org/>
   [Scikit-learn]: <http://scikit-learn.org/>
   [Scoop]: <https://github.com/soravux/scoop/>