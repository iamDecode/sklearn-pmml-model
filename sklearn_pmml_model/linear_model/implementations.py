from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn_pmml_model.linear_model.base import PMMLLinearModel, PMMLGeneralRegression
from itertools import chain
import numpy as np


class PMMLLinearRegression(PMMLLinearModel, LinearRegression):
    """
    Ordinary least squares Linear Regression.

    The PMML model consists out of a <RegressionModel> element, containing at
    least one <RegressionTable> element. Every table element contains a
    <NumericPredictor> element for numerical fields and <CategoricalPredictor>
    per value of a categorical field, describing the coefficients.

    Parameters
    ----------
    pmml : str, object
      Filename or file object containing PMML data.

    See more
    --------
    http://dmg.org/pmml/v4-3/Regression.html

    """
    def __init__(self, pmml):
        super().__init__(pmml)

        # Import coefficients and intercepts
        model = self.root.find('RegressionModel')

        if model is None:
            raise Exception('PMML model does not contain RegressionModel.')

        tables = model.findall('RegressionTable')

        self.coef_ = np.array([
            self._get_coefficients(table)
            for table in tables
        ])
        self.intercept_ = np.array([
            float(table.get('intercept'))
            for table in tables
        ])

        if self.coef_.shape[0] == 1:
            self.coef_ = self.coef_[0]

        if self.intercept_.shape[0] == 1:
            self.intercept_ = self.intercept_[0]

    def _get_coefficients(self, table):
        def coefficient_for_category(predictors, category):
            predictor = [p for p in predictors if p.get('value') == category]

            if not predictor:
                return 0

            return float(predictor[0].get('coefficient'))

        def coefficients_for_field(name, field):
            predictors = table.findall(f"*[@name='{name}']")

            if field.get('optype') != 'categorical':
                if len(predictors) > 1:
                    raise Exception('PMML model is not linear.')

                return [float(predictors[0].get('coefficient'))]

            return [
                coefficient_for_category(predictors, c)
                for c in self.field_mapping[name][1].categories
            ]

        return list(chain.from_iterable([
            coefficients_for_field(name, field)
            for name, field in self.fields.items()
            if table.find(f"*[@name='{name}']") is not None
        ]))


'''
NOTE: Many of these variants only differ in the training part, not the 
classification part. Hence they are equavalent in terms of parsing.
'''


class PMMLRidge(PMMLGeneralRegression, Ridge):
    pass


class PMMLLasso(PMMLGeneralRegression, Lasso):
    pass


class PMMLElasticNet(PMMLGeneralRegression, ElasticNet):
    pass