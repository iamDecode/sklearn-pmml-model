from sklearn_pmml_model.base import PMMLBaseRegressor
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
import numpy as np


class PMMLLinearModel(PMMLBaseRegressor):
    """
    Abstract class for linear models.

    """
    def __init__(self, pmml):
        super().__init__(pmml)

        # Setup a column transformer to deal with categorical variables
        target = self.target_field.get('name')
        fields = [field for name, field in self.fields.items() if name != target]

        def encoder_for(field):
            if field.get('optype') != 'categorical':
                return 'passthrough'

            encoder = OneHotEncoder()
            encoder.categories_ = np.array([self.field_mapping[field.get('name')][1].categories])
            encoder._legacy_mode = False
            return encoder

        transformer = ColumnTransformer(
            transformers=[
                (field.get('name'), encoder_for(field), [self.field_mapping[field.get('name')][0]])
                for field in fields
                if field.tag == 'DataField'
            ]
        )
        transformer.transformers_ = transformer.transformers
        transformer.sparse_output_ = False
        self.transformer = transformer

    def _prepare_data(self, X):
        """
        Overrides the default data preparation operation by one-hot encoding
        categorical variables.
        """
        return self.transformer.transform(X)
