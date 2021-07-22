from unittest import TestCase
import sklearn_pmml_model
from sklearn.svm import LinearSVC, LinearSVR, NuSVC, NuSVR, SVC, SVR
from sklearn_pmml_model.svm import PMMLLinearSVC, PMMLLinearSVR, PMMLNuSVC, PMMLNuSVR, PMMLSVC, PMMLSVR
import pandas as pd
import numpy as np
from os import path, remove
from io import StringIO
from sklearn2pmml.pipeline import PMMLPipeline
from sklearn2pmml import sklearn2pmml

BASE_DIR = path.dirname(sklearn_pmml_model.__file__)


class TestBaseSVR(TestCase):
    def test_invalid_model(self):
        with self.assertRaises(Exception) as cm:
            PMMLSVC(pmml=StringIO("""
              <PMML xmlns="http://www.dmg.org/PMML-4_3" version="4.3">
                <DataDictionary>
                  <DataField name="Class" optype="categorical" dataType="string">
                    <Value value="setosa"/>
                    <Value value="versicolor"/>
                    <Value value="virginica"/>
                  </DataField>
                </DataDictionary>
                <MiningSchema>
                  <MiningField name="Class" usageType="target"/>
                </MiningSchema>
              </PMML>
              """))

        assert str(cm.exception) == 'PMML model does not contain SupportVectorMachineModel.'


class TestLinearSVRIntegration(TestCase):
    def setUp(self):
        df = pd.read_csv(path.join(BASE_DIR, '../models/categorical-test.csv'))
        Xte = df.iloc[:, 1:]
        Xenc = pd.get_dummies(Xte, prefix_sep='')
        yte = df.iloc[:, 0]
        self.test = (Xte, yte)
        self.enc = (Xenc, yte)

        pmml = path.join(BASE_DIR, '../models/linear-model-lm.pmml')
        self.clf = PMMLLinearSVR(pmml)

        self.ref = LinearSVR()
        self.ref.fit(Xenc, yte == 'Yes')

    def test_fit_exception(self):
        with self.assertRaises(Exception) as cm:
            self.clf.fit(np.array([[]]), np.array([]))

        assert str(cm.exception) == 'Not supported.'

    def test_more_tags(self):
        assert self.clf._more_tags() == LinearSVR()._more_tags()

    def test_sklearn2pmml(self):
        # Export to PMML
        pipeline = PMMLPipeline([
            ("classifier", self.ref)
        ])
        pipeline.fit(self.enc[0], self.enc[1] == 'Yes')
        sklearn2pmml(pipeline, "svm-sklearn2pmml.pmml", with_repr=True)

        try:
            # Import PMML
            model = PMMLLinearSVR(pmml='svm-sklearn2pmml.pmml')

            # Verify classification
            Xenc, _ = self.enc
            assert np.allclose(
                self.ref.predict(Xenc),
                model.predict(Xenc)
            )

        finally:
            remove("svm-sklearn2pmml.pmml")


class TestLinearSVCIntegration(TestCase):
    def setUp(self):
        df = pd.read_csv(path.join(BASE_DIR, '../models/categorical-test.csv'))
        Xte = df.iloc[:, 1:]
        Xenc = pd.get_dummies(Xte, prefix_sep='')
        yte = df.iloc[:, 0]
        self.test = (Xte, yte)
        self.enc = (Xenc, yte)

        pmml = path.join(BASE_DIR, '../models/linear-model-lmc.pmml')
        self.clf = PMMLLinearSVC(pmml)

        self.ref = LinearSVC()
        self.ref.fit(Xenc, yte)

    def test_fit_exception(self):
        with self.assertRaises(Exception) as cm:
            self.clf.fit(np.array([[]]), np.array([]))

        assert str(cm.exception) == 'Not supported.'

    def test_more_tags(self):
        assert self.clf._more_tags() == LinearSVC()._more_tags()

    def test_sklearn2pmml(self):
        # Export to PMML
        pipeline = PMMLPipeline([
            ("classifier", self.ref)
        ])
        pipeline.fit(self.enc[0], self.enc[1] == 'Yes')
        sklearn2pmml(pipeline, "svm-sklearn2pmml.pmml", with_repr=True)

        try:
            # Import PMML
            model = PMMLLinearSVC(pmml='svm-sklearn2pmml.pmml')

            # Verify classification
            Xenc, _ = self.enc
            assert np.allclose(
                self.ref.decision_function(Xenc),
                model.decision_function(Xenc)
            )

        finally:
            remove("svm-sklearn2pmml.pmml")


class TestSVRIntegration(TestCase):
    def setUp(self):
        df = pd.read_csv(path.join(BASE_DIR, '../models/categorical-test.csv'))
        Xte = df.iloc[:, 1:]
        Xenc = pd.get_dummies(Xte, prefix_sep='')
        yte = df.iloc[:, 0]
        self.test = (Xte, yte)
        self.enc = (Xenc, yte)

        pmml = path.join(BASE_DIR, '../models/svr-cat-pima.pmml')
        self.clf = PMMLSVR(pmml)

        self.ref_rbf = SVR(kernel='rbf')
        self.ref_rbf.fit(Xenc, yte == 'Yes')
        self.ref_linear = SVR(kernel='linear')
        self.ref_linear.fit(Xenc, yte == 'Yes')
        self.ref_poly = SVR(kernel='poly')
        self.ref_poly.fit(Xenc, yte == 'Yes')
        self.ref_sigmoid = SVR(kernel='sigmoid')
        self.ref_sigmoid.fit(Xenc, yte == 'Yes')

    def test_fit_exception(self):
        with self.assertRaises(Exception) as cm:
            self.clf.fit(np.array([[]]), np.array([]))

        assert str(cm.exception) == 'Not supported.'

    def test_more_tags(self):
        assert self.clf._more_tags() == SVR()._more_tags()

    def test_sklearn2pmml_rbf(self):
        # Export to PMML
        pipeline = PMMLPipeline([
            ("regressor", self.ref_rbf)
        ])
        pipeline.fit(self.enc[0], self.enc[1] == 'Yes')
        sklearn2pmml(pipeline, "svr-sklearn2pmml.pmml", with_repr=True)

        try:
            # Import PMML
            model = PMMLSVR(pmml='svr-sklearn2pmml.pmml')

            # Verify classification
            Xenc, _ = self.enc
            assert np.allclose(
                self.ref_rbf.predict(Xenc),
                model.predict(Xenc)
            )

        finally:
            remove("svr-sklearn2pmml.pmml")

    def test_sklearn2pmml_linear(self):
        # Export to PMML
        pipeline = PMMLPipeline([
            ("regressor", self.ref_linear)
        ])
        pipeline.fit(self.enc[0], self.enc[1] == 'Yes')
        sklearn2pmml(pipeline, "svr-sklearn2pmml.pmml", with_repr=True)

        try:
            # Import PMML
            model = PMMLSVR(pmml='svr-sklearn2pmml.pmml')

            # Verify classification
            Xenc, _ = self.enc
            assert np.allclose(
                self.ref_linear.predict(Xenc),
                model.predict(Xenc)
            )

        finally:
            remove("svr-sklearn2pmml.pmml")

    def test_sklearn2pmml_poly(self):
        # Export to PMML
        pipeline = PMMLPipeline([
            ("regressor", self.ref_poly)
        ])
        pipeline.fit(self.enc[0], self.enc[1] == 'Yes')
        sklearn2pmml(pipeline, "svr-sklearn2pmml.pmml", with_repr=True)

        try:
            # Import PMML
            model = PMMLSVR(pmml='svr-sklearn2pmml.pmml')

            # Verify classification
            Xenc, _ = self.enc
            assert np.allclose(
                self.ref_poly.predict(Xenc),
                model.predict(Xenc)
            )

        finally:
            remove("svr-sklearn2pmml.pmml")

    def test_sklearn2pmml_sigmoid(self):
        # Export to PMML
        pipeline = PMMLPipeline([
            ("regressor", self.ref_sigmoid)
        ])
        pipeline.fit(self.enc[0], self.enc[1] == 'Yes')
        sklearn2pmml(pipeline, "svr-sklearn2pmml.pmml", with_repr=True)

        try:
            # Import PMML
            model = PMMLSVR(pmml='svr-sklearn2pmml.pmml')

            # Verify classification
            Xenc, _ = self.enc
            assert np.allclose(
                self.ref_sigmoid.predict(Xenc),
                model.predict(Xenc)
            )

        finally:
            remove("svr-sklearn2pmml.pmml")


class TestSVCIntegration(TestCase):
    def setUp(self):
        df = pd.read_csv(path.join(BASE_DIR, '../models/categorical-test.csv'))
        Xte = df.iloc[:, 1:]
        Xenc = pd.get_dummies(Xte, prefix_sep='')
        yte = df.iloc[:, 0]
        self.test = (Xte, yte)
        self.enc = (Xenc, yte)

        pmml = path.join(BASE_DIR, '../models/svc-cat-pima.pmml')
        self.clf = PMMLSVC(pmml)

        self.ref_rbf = SVC(kernel='rbf')
        self.ref_rbf.fit(Xenc, yte)
        self.ref_linear = SVC(kernel='linear')
        self.ref_linear.fit(Xenc, yte)
        self.ref_poly = SVC(kernel='poly')
        self.ref_poly.fit(Xenc, yte)
        self.ref_sigmoid = SVC(kernel='sigmoid')
        self.ref_sigmoid.fit(Xenc, yte)

    def test_fit_exception(self):
        with self.assertRaises(Exception) as cm:
            self.clf.fit(np.array([[]]), np.array([]))

        assert str(cm.exception) == 'Not supported.'

    def test_more_tags(self):
        assert self.clf._more_tags() == SVC()._more_tags()

    def test_sklearn2pmml_rbf(self):
        # Export to PMML
        pipeline = PMMLPipeline([
            ("classifier", self.ref_rbf)
        ])
        pipeline.fit(self.enc[0], self.enc[1])
        sklearn2pmml(pipeline, "svc-sklearn2pmml.pmml", with_repr=True)

        try:
            # Import PMML
            model = PMMLSVC(pmml='svc-sklearn2pmml.pmml')

            # Verify classification
            Xenc, _ = self.enc
            assert np.allclose(
                self.ref_rbf.decision_function(Xenc),
                model.decision_function(Xenc)
            )

        finally:
            remove("svc-sklearn2pmml.pmml")

    def test_sklearn2pmml_linear(self):
        # Export to PMML
        pipeline = PMMLPipeline([
            ("classifier", self.ref_linear)
        ])
        pipeline.fit(self.enc[0], self.enc[1])
        sklearn2pmml(pipeline, "svc-sklearn2pmml.pmml", with_repr=True)

        try:
            # Import PMML
            model = PMMLSVC(pmml='svc-sklearn2pmml.pmml')

            # Verify classification
            Xenc, _ = self.enc
            assert np.allclose(
                self.ref_linear.decision_function(Xenc),
                model.decision_function(Xenc)
            )

        finally:
            remove("svc-sklearn2pmml.pmml")

    def test_sklearn2pmml_poly(self):
        # Export to PMML
        pipeline = PMMLPipeline([
            ("classifier", self.ref_poly)
        ])
        pipeline.fit(self.enc[0], self.enc[1])
        sklearn2pmml(pipeline, "svc-sklearn2pmml.pmml", with_repr=True)

        try:
            # Import PMML
            model = PMMLSVC(pmml='svc-sklearn2pmml.pmml')

            # Verify classification
            Xenc, _ = self.enc
            assert np.allclose(
                self.ref_poly.decision_function(Xenc),
                model.decision_function(Xenc)
            )

        finally:
            remove("svc-sklearn2pmml.pmml")

    def test_sklearn2pmml_sigmoid(self):
        # Export to PMML
        pipeline = PMMLPipeline([
            ("classifier", self.ref_sigmoid)
        ])
        pipeline.fit(self.enc[0], self.enc[1])
        sklearn2pmml(pipeline, "svc-sklearn2pmml.pmml", with_repr=True)

        try:
            # Import PMML
            model = PMMLSVC(pmml='svc-sklearn2pmml.pmml')

            # Verify classification
            Xenc, _ = self.enc
            assert np.allclose(
                self.ref_sigmoid.decision_function(Xenc),
                model.decision_function(Xenc)
            )

        finally:
            remove("svc-sklearn2pmml.pmml")


class TestNuSVRIntegration(TestCase):
    def setUp(self):
        df = pd.read_csv(path.join(BASE_DIR, '../models/categorical-test.csv'))
        Xte = df.iloc[:, 1:]
        Xenc = pd.get_dummies(Xte, prefix_sep='')
        yte = df.iloc[:, 0]
        self.test = (Xte, yte)
        self.enc = (Xenc, yte)

        pmml = path.join(BASE_DIR, '../models/svr-cat-pima.pmml')
        self.clf = PMMLNuSVR(pmml)

        self.ref = NuSVR()
        self.ref.fit(Xenc, yte == 'Yes')

    def test_fit_exception(self):
        with self.assertRaises(Exception) as cm:
            self.clf.fit(np.array([[]]), np.array([]))

        assert str(cm.exception) == 'Not supported.'

    def test_more_tags(self):
        assert self.clf._more_tags() == NuSVR()._more_tags()

    def test_sklearn2pmml(self):
        # Export to PMML
        pipeline = PMMLPipeline([
            ("regressor", self.ref)
        ])
        pipeline.fit(self.enc[0], self.enc[1] == 'Yes')
        sklearn2pmml(pipeline, "svr-sklearn2pmml.pmml", with_repr=True)

        try:
            # Import PMML
            model = PMMLNuSVR(pmml='svr-sklearn2pmml.pmml')

            # Verify classification
            Xenc, _ = self.enc
            assert np.allclose(
                self.ref.predict(Xenc),
                model.predict(Xenc)
            )

        finally:
            remove("svr-sklearn2pmml.pmml")


class TestNuSVCIntegration(TestCase):
    def setUp(self):
        df = pd.read_csv(path.join(BASE_DIR, '../models/categorical-test.csv'))
        Xte = df.iloc[:, 1:]
        Xenc = pd.get_dummies(Xte, prefix_sep='')
        yte = df.iloc[:, 0]
        self.test = (Xte, yte)
        self.enc = (Xenc, yte)

        pmml = path.join(BASE_DIR, '../models/svc-cat-pima.pmml')
        self.clf = PMMLNuSVC(pmml)

        self.ref = NuSVC()
        self.ref.fit(Xenc, yte)

    def test_fit_exception(self):
        with self.assertRaises(Exception) as cm:
            self.clf.fit(np.array([[]]), np.array([]))

        assert str(cm.exception) == 'Not supported.'

    def test_more_tags(self):
        assert self.clf._more_tags() == NuSVC()._more_tags()

    def test_sklearn2pmml(self):
        # Export to PMML
        pipeline = PMMLPipeline([
            ("classifier", self.ref)
        ])
        pipeline.fit(self.enc[0], self.enc[1])
        sklearn2pmml(pipeline, "svc-sklearn2pmml.pmml", with_repr=True)

        try:
            # Import PMML
            model = PMMLNuSVC(pmml='svc-sklearn2pmml.pmml')

            # Verify classification
            Xenc, _ = self.enc
            assert np.allclose(
                self.ref.decision_function(Xenc),
                model.decision_function(Xenc)
            )

        finally:
            remove("svc-sklearn2pmml.pmml")