# Contributing

When contributing to this repository, please first discuss the change you wish to make via a GitHub issue, email, or any other method with the owners of this repository before making a change. 

Please note we have a [code of conduct](CODE_OF_CONDUCT.md), please follow it in all your interactions with the project.


## Scope of this package

The scope of `sklearn-pmml-model` is to import functionality to all major estimator classes of the popular machine learning library [scikit-learn](https://scikit-learn.org) using [PMML](http://dmg.org/pmml/v4-4/GeneralStructure.html).

The API is designed to closely resemble the `scikit-learn` API. The same directory and component structure is used, and each estimator is a sub-class of a corresponding estimator. Note that some models may not have a `scikit-learn` implementation (e.g., Bayesian networks) and hence cannot currently be represented.

We intend for the library to remain as light-weight as possible, and stick with the minimum number of additions to enable PMML import functionality without affecting the outward facing API of estimators.


## Reporting bugs

We use GitHub issues to track all bugs and feature requests; feel free to open an issue if you have found a bug or wish to see a feature implemented.

It is recommended to check that your issue complies with the  following rules before submitting:

- Verify that your issue is not being currently addressed by other [issues](https://github.com/iamDecode/sklearn-pmml-model/issues) or [pull requests](https://github.com/iamDecode/sklearn-pmml-model/pulls).
- Please include code snippets or error messages when reporting issues. When doing so, please make sure to format them using code blocks. See [Creating and highlighting code blocks](https://help.github.com/articles/creating-and-highlighting-code-blocks).
- It can often be helpful to include your operating system type and version number, as well as your Python, sklearn-pmml-model, scikit-learn, numpy, and scipy versions. This information can be found by running the following code snippet:
```python
import platform; print(platform.platform())
import sys; print("Python", sys.version)
import numpy; print("NumPy", numpy.__version__)
import scipy; print("SciPy", scipy.__version__)
import sklearn; print("Scikit-Learn", sklearn.__version__)
import sklearn_pmml_model; print("sklearn-pmml-model", sklearn_pmml_model.__version__)
```


## Get a local copy

These are the steps you need to take to create a copy of the `sklearn-pmml-model` repository on your computer.

1. [Create an account](https://github.com/join) on GitHub if you do not already have one.

2. [Fork](https://help.github.com/en/github/getting-started-with-github/fork-a-repo) the [`sklearn-pmml-model` repository](https://github.com/iamDecode/sklearn-pmml-model).

3. Clone your fork of the `sklearn-pmml-model` repository from your GitHub account. Use a git GUI application (e.g., Sourcetree, GitKraken) or from command line, run:

   ```
   $ git clone git@github.com:iamDecode/sklearn-pmml-model.git
   $ cd sklearn-pmml-model
   ```

4. Create a feature branch to hold your development changes:

   ```
   $ git checkout -b <username>/<feature description>
   ```

   (For example: `decode/regression-trees`)


## Setting up a development environment

After you created a copy of our main repository on GitHub, your need to setup a local development environment.  We recommend creating a virtual environment and activating it:
```
$ python3 -m venv venv
$ source venv/bin/activate
```

and install the dependencies within the virtual environment:

```
$ pip install -r requirements.txt
```

The final step is to build the Cython extensions (you need to rebuilt once you make changes to the Cython code):

```
$ python setup.py build_ext --inplace
```

## Making changes to the code

For pull requests to be accepted, your changes must at least meet the following requirements:

1. All changes related to *one feature* must belong to *one branch*. Each branch must be self-contained, with a single new feature or bugfix.
2. Commit messages should be formulated according to [Conventional Commits](https://www.conventionalcommits.org/en/v1.0.0/).
3. If your pull request addresses an issue, please make sure to [link back](https://github.blog/changelog/2020-12-15-reference-issues-discussions-and-pull-requests-faster-with-multi-word-suggestions/) to the original issue.
4. Follow the [PEP8 style guide](https://www.python.org/dev/peps/pep-0008/). With the following exceptions or additions:
   - The max line length is 120 characters instead of 80.
   - Indents with double spaces, not 4 spaces or tabs.
  
   You can check for compliance locally by running:
   ```
   $ flake8 sklearn_pmml_model
   ```
5. Each function, class, method, and attribute needs to be documented using docstrings. `sklearn-pmml-model` conforms to the [numpy docstring standard](https://numpydoc.readthedocs.io/en/latest/format.html#docstring-standard). 
6. Finally, ensure all the test cases still pass after you have made your changes. To test locally, you can run:
   ```
   $ python setup.py pytest
   ```

In addition to these requirements, we strongly prefer you to consider the following guidelines. However, they are not strictly required to not be overly prohibitive to new contributors.

7. Your change should include test cases for all new functionality being introduced.
8. No additional code style issues should be reported by [LGTM](https://lgtm.com).

Continuous integration will automatically verify compliance with all of the discussed requirements.



## Submitting a Pull Request

1. When you are done coding in your feature branch, [add changed or new files](https://git-scm.com/book/en/v2/Git-Basics-Recording-Changes-to-the-Repository#_tracking_files>):
   ```
   $ git add path/to/modified_file
   ```
2. Create a [commit](https://git-scm.com/book/en/v2/Git-Basics-Recording-Changes-to-the-Repository#_committing_changes) with a message describing what you changed. Commit messages should be formulated according to [Conventional Commits](https://www.conventionalcommits.org/en/v1.0.0/) standard: 
   ```
   $ git commit
   ```
3. Push the changes to GitHub:
   ```
   $ git push -u origin my_feature
   ```
4. [Create a pull request](https://help.github.com/en/github/collaborating-with-issues-and-pull-requests/creating-a-pull-request).