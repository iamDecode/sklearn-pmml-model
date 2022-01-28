from unittest import TestCase

from sklearn.datasets import load_iris
import sklearn_pmml_model
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn_pmml_model.neural_network import PMMLMLPClassifier, PMMLMLPRegressor
import pandas as pd
import numpy as np
from os import path, remove
from io import StringIO
from sklearn2pmml.pipeline import PMMLPipeline
from sklearn2pmml import sklearn2pmml


BASE_DIR = path.dirname(sklearn_pmml_model.__file__)


class TestNeuralNetwork(TestCase):
  def test_invalid_model(self):
    with self.assertRaises(Exception) as cm:
      PMMLMLPClassifier(pmml=StringIO("""
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

    assert str(cm.exception) == 'PMML model does not contain NeuralNetwork.'

  def test_no_inputs(self):
    with self.assertRaises(Exception) as cm:
      PMMLMLPClassifier(pmml=StringIO("""
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
                <NeuralNetwork/>
              </PMML>
              """))

    assert str(cm.exception) == 'PMML model does not contain NeuralInputs.'

  def test_mismatching_inputs(self):
    with self.assertRaises(Exception) as cm:
      PMMLMLPClassifier(pmml=StringIO("""
              <PMML xmlns="http://www.dmg.org/PMML-4_3" version="4.3">
                <DataDictionary>
                  <DataField name="Class" optype="categorical" dataType="string">
                    <Value value="setosa"/>
                    <Value value="versicolor"/>
                    <Value value="virginica"/>
                  </DataField>
                  <DataField name="sepal length (cm)" optype="continuous" dataType="float"/>
                  <DataField name="sepal width (cm)" optype="continuous" dataType="float"/>
                </DataDictionary>
                <MiningSchema>
                  <MiningField name="Class" usageType="target"/>
                </MiningSchema>
                <NeuralNetwork>
                  <NeuralInputs>
                    <NeuralInput id="1">
                      <DerivedField name="test1" optype="continuous" dataType="double">
                        <FieldRef field="sepal length (cm)"/>
                      </DerivedField>
                    </NeuralInput>
                  </NeuralInputs>
                </NeuralNetwork>
              </PMML>
              """))

    assert str(cm.exception) == 'PMML model preprocesses the data which currently unsupported.'

  def test_no_layers(self):
    with self.assertRaises(Exception) as cm:
      PMMLMLPClassifier(pmml=StringIO("""
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
                <NeuralNetwork>
                  <NeuralInputs/>
                </NeuralNetwork>
              </PMML>
              """))

    assert str(cm.exception) == 'PMML model does not contain any NeuralLayer elements.'

  def test_unsupported_activation(self):
    with self.assertRaises(Exception) as cm:
      PMMLMLPClassifier(pmml=StringIO("""
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
                <NeuralNetwork>
                  <NeuralInputs/>
                  <NeuralLayer activationFunction="Elliott" />
                </NeuralNetwork>
              </PMML>
              """))

    assert str(cm.exception) == 'PMML model uses unsupported activationFunction.'

  def test_not_matching_activation(self):
    with self.assertRaises(Exception) as cm:
      PMMLMLPClassifier(pmml=StringIO("""
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
                <NeuralNetwork activationFunction="tanh">
                  <NeuralInputs/>
                  <NeuralLayer activationFunction="rectifier" />
                  <NeuralLayer activationFunction="identity" />
                </NeuralNetwork>
              </PMML>
              """))

    assert str(cm.exception) == 'Neural networks with different activation functions per ' \
                                'layer are not currently supported by scikit-learn.'


class TestNeuralNetworkIntegrationIris(TestCase):
  def setUp(self):
    data = load_iris(as_frame=True)

    X = data.data
    y = pd.Series(np.array(data.target_names)[data.target])
    y.name = "Class"
    self.test = (X, y)

    pmml = path.join(BASE_DIR, '../models/nn-iris.pmml')
    self.clf = PMMLMLPClassifier(pmml=pmml)
    self.ref = MLPClassifier(random_state=1).fit(X, y)

  def test_more_tags(self):
    assert self.clf._more_tags() == MLPClassifier()._more_tags()

  def test_more_tags_regressor(self):
    pmml = path.join(BASE_DIR, '../models/nn-iris.pmml')
    clf = PMMLMLPRegressor(pmml=pmml)
    assert clf._more_tags() == MLPClassifier()._more_tags()

  def test_predict_proba(self):
    Xte, _ = self.test
    ref = np.array([
      [0.0003538189393654, 0.9996358135096475, 0.0000103675509872],
      [0.0005894724920352, 0.9993997448119392, 0.0000107826960258],
      [0.0004380193229982, 0.9995515433793557, 0.0000104372976461],
      [0.0005354681246794, 0.9994535413297830, 0.0000109905455375],
      [0.0003284058882994, 0.9996612813044171, 0.0000103128072834],
      [0.0003020804540654, 0.9996875196439093, 0.0000103999020252],
      [0.0003873149374128, 0.9996022893407164, 0.0000103957218707],
      [0.0003926221353402, 0.9995968731614151, 0.0000105047032448],
      [0.0006577474414988, 0.9993310528345251, 0.0000111997239761],
      [0.0005271002159970, 0.9994621096838615, 0.0000107901001416],
      [0.0003181410252599, 0.9996714818133867, 0.0000103771613533],
      [0.0004074841456608, 0.9995818361335840, 0.0000106797207552],
      [0.0005681536533975, 0.9994211064677111, 0.0000107398788914],
      [0.0004832893129287, 0.9995063365060303, 0.0000103741810410],
      [0.0002673404024783, 0.9997222664524931, 0.0000103931450285],
      [0.0002392673660676, 0.9997504559289266, 0.0000102767050058],
      [0.0002814645071093, 0.9997082789314528, 0.0000102565614380],
      [0.0003587005281854, 0.9996309361523789, 0.0000103633194358],
      [0.0003175754508584, 0.9996719522472274, 0.0000104723019144],
      [0.0003020601426267, 0.9996876599424405, 0.0000102799149328],
      [0.0004323222423873, 0.9995569111656204, 0.0000107665919923],
      [0.0003250320017879, 0.9996646445589406, 0.0000103234392717],
      [0.0003064497385245, 0.9996833831880103, 0.0000101670734650],
      [0.0005348843669774, 0.9994535983488416, 0.0000115172841810],
      [0.0005019589868546, 0.9994861980156874, 0.0000118429974581],
      [0.0006713524272075, 0.9993172831209979, 0.0000113644517946],
      [0.0004297255740071, 0.9995594797779345, 0.0000107946480585],
      [0.0003642109447433, 0.9996253473936240, 0.0000104416616328],
      [0.0003875847510213, 0.9996019938181319, 0.0000104214308469],
      [0.0005043906571372, 0.9994845417330437, 0.0000110676098190],
      [0.0005730997917240, 0.9994156515290933, 0.0000112486791825],
      [0.0004180676467817, 0.9995713813317555, 0.0000105510214629],
      [0.0002587884598955, 0.9997309374557840, 0.0000102740843204],
      [0.0002487336575223, 0.9997409576084451, 0.0000103087340327],
      [0.0005410895574644, 0.9994480377492715, 0.0000108726932642],
      [0.0004395115158191, 0.9995501138553344, 0.0000103746288464],
      [0.0003615332441064, 0.9996280862676278, 0.0000103804882658],
      [0.0003235906492552, 0.9996660965602384, 0.0000103127905063],
      [0.0005392896580896, 0.9994500237624566, 0.0000106865794538],
      [0.0003946673129527, 0.9995948287452184, 0.0000105039418288],
      [0.0003492462716720, 0.9996404596386512, 0.0000102940896769],
      [0.0022671995029275, 0.9977194647136336, 0.0000133357834387],
      [0.0004348712924043, 0.9995546927266189, 0.0000104359809767],
      [0.0004170397807762, 0.9995721414812122, 0.0000108187380116],
      [0.0003563168016465, 0.9996327984206748, 0.0000108847776787],
      [0.0005997076306096, 0.9993893851327539, 0.0000109072366367],
      [0.0003041129271737, 0.9996855459085943, 0.0000103411642320],
      [0.0004532072562403, 0.9995362105015143, 0.0000105822422452],
      [0.0003166692829531, 0.9996729673991938, 0.0000103633178532],
      [0.0004180777989678, 0.9995714510665723, 0.0000104711344599],
      [0.9996147920831117, 0.0003283255080990, 0.0000568824087893],
      [0.9995382472385207, 0.0003967804823661, 0.0000649722791131],
      [0.9996818636544671, 0.0002683793672473, 0.0000497569782856],
      [0.9998470267882932, 0.0001206073760030, 0.0000323658357038],
      [0.9997795290801949, 0.0001842101115850, 0.0000362608082202],
      [0.9993490949459823, 0.0004101867997476, 0.0002407182542701],
      [0.9994779579547285, 0.0004445678220837, 0.0000774742231876],
      [0.9997084532046492, 0.0002590684177314, 0.0000324783776195],
      [0.9997486729416101, 0.0002140023948379, 0.0000373246635520],
      [0.9996966795708531, 0.0002604872337402, 0.0000428331954067],
      [0.9999137734858133, 0.0000739934624902, 0.0000122330516964],
      [0.9996070187969389, 0.0003397204533438, 0.0000532607497173],
      [0.9999238280285531, 0.0000651092855464, 0.0000110626859007],
      [0.9994351758117057, 0.0003815709473772, 0.0001832532409171],
      [0.9994394107726446, 0.0004965297854387, 0.0000640594419166],
      [0.9996152830588168, 0.0003297581224312, 0.0000549588187521],
      [0.9991153760104478, 0.0005625243820633, 0.0003220996074889],
      [0.9997330511692665, 0.0002301414295430, 0.0000368074011905],
      [0.9860217501245976, 0.0004632051797487, 0.0135150446956537],
      [0.9998087666819834, 0.0001651559576367, 0.0000260773603800],
      [0.9834548860163552, 0.0019575144677700, 0.0145875995158747],
      [0.9997025881343822, 0.0002571953346928, 0.0000402165309251],
      [0.0197053774766285, 0.0001744054936725, 0.9801202170296991],
      [0.9995557873247197, 0.0003084308743270, 0.0001357818009533],
      [0.9997057284009238, 0.0002526181392658, 0.0000416534598105],
      [0.9996779602474649, 0.0002759744922847, 0.0000460652602505],
      [0.9998073854292886, 0.0001592608769184, 0.0000333536937930],
      [0.9991434793877859, 0.0004535705025655, 0.0004029501096486],
      [0.9996349901640699, 0.0002960206768722, 0.0000689891590580],
      [0.9996193133306069, 0.0003393071054507, 0.0000413795639423],
      [0.9998371532405628, 0.0001407942010761, 0.0000220525583610],
      [0.9998213115909798, 0.0001553351276255, 0.0000233532813947],
      [0.9997207686168851, 0.0002421765874080, 0.0000370547957070],
      [0.0000986445984258, 0.0000063834638488, 0.9998949719377254],
      [0.9956524413015063, 0.0011491374249326, 0.0031984212735612],
      [0.9993871250337886, 0.0005295248788529, 0.0000833500873587],
      [0.9996553700587613, 0.0002938111799143, 0.0000508187613245],
      [0.9998991433269051, 0.0000779077247297, 0.0000229489483651],
      [0.9995518064255887, 0.0003889398745664, 0.0000592536998449],
      [0.9998057740143796, 0.0001641314909898, 0.0000300944946305],
      [0.9983758073486436, 0.0005290787914473, 0.0010951138599091],
      [0.9996067333282338, 0.0003258006122714, 0.0000674660594949],
      [0.9997891605250822, 0.0001818588366596, 0.0000289806382583],
      [0.9997800960414337, 0.0001946210284330, 0.0000252829301333],
      [0.9997195102367209, 0.0002345233019324, 0.0000459664613467],
      [0.9995599110286824, 0.0003805446839442, 0.0000595442873733],
      [0.9996376383471243, 0.0003120191740825, 0.0000503424787932],
      [0.9996886793687706, 0.0002675466694490, 0.0000437739617805],
      [0.9992764904176524, 0.0006704076524906, 0.0000531019298570],
      [0.9996898333821236, 0.0002676524771599, 0.0000425141407167],
      [0.0000406578988893, 0.0000052312099494, 0.9999541108911612],
      [0.0000657946450301, 0.0000048282637630, 0.9999293770912069],
      [0.0000519568474286, 0.0000041718504916, 0.9999438713020797],
      [0.0000478928721790, 0.0000043679104917, 0.9999477392173293],
      [0.0000497631605019, 0.0000045765629487, 0.9999456602765494],
      [0.0000692865293424, 0.0000039915285743, 0.9999267219420833],
      [0.0000854987778829, 0.0000056970199588, 0.9999088042021583],
      [0.0000671725133298, 0.0000038782844203, 0.9999289492022500],
      [0.0001340157571678, 0.0000041452954376, 0.9998618389473946],
      [0.0000233990285554, 0.0000045658726664, 0.9999720350987783],
      [0.7477865890712819, 0.0034238475687576, 0.2487895633599605],
      [0.0000831833070238, 0.0000048590329631, 0.9999119576600131],
      [0.0000757789125798, 0.0000061802538279, 0.9999180408335924],
      [0.0000972781080372, 0.0000048105304136, 0.9998979113615493],
      [0.0000660897632021, 0.0000051652019406, 0.9999287450348572],
      [0.0000762674155948, 0.0000085230835878, 0.9999152095008174],
      [0.0000552669146608, 0.0000054246151263, 0.9999393084702128],
      [0.0000165043619836, 0.0000039619974622, 0.9999795336405543],
      [0.0002395733917633, 0.0000043592123268, 0.9997560673959099],
      [0.0001852481103993, 0.0000044677814170, 0.9998102841081837],
      [0.0000412827079753, 0.0000049326037351, 0.9999537846882897],
      [0.0000650023901456, 0.0000057468470008, 0.9999292507628537],
      [0.0001169928862901, 0.0000039463131428, 0.9998790608005671],
      [0.0210392652978838, 0.0002462892652972, 0.9787144454368190],
      [0.0000353742817678, 0.0000052223142558, 0.9999594034039764],
      [0.0000413207001657, 0.0000046702658933, 0.9999540090339409],
      [0.7023304046641650, 0.0023357791032961, 0.2953338162325388],
      [0.2943901054135695, 0.0021777584132102, 0.7034321361732202],
      [0.0000665020536947, 0.0000045033275268, 0.9999289946187785],
      [0.0001422005580342, 0.0000085507362832, 0.9998492487056826],
      [0.0000855949707283, 0.0000038570080200, 0.9999105480212517],
      [0.0001655938889145, 0.0000202357346634, 0.9998141703764221],
      [0.0000696159738725, 0.0000045932217270, 0.9999257908044005],
      [0.0053232504872226, 0.0001103462728265, 0.9945664032399509],
      [0.0000757589849895, 0.0000041646769109, 0.9999200763380995],
      [0.0000668819965700, 0.0000040219079480, 0.9999290960954821],
      [0.0000295867142981, 0.0000050304479930, 0.9999653828377089],
      [0.0000479208692597, 0.0000056725319031, 0.9999464065988373],
      [0.8122021842511724, 0.0030032082616502, 0.1847946074871774],
      [0.0012209977902577, 0.0000483342278216, 0.9987306679819207],
      [0.0000441323917942, 0.0000047113973578, 0.9999511562108481],
      [0.9363192966928233, 0.0020971368907008, 0.0615835664164759],
      [0.0000657946450301, 0.0000048282637630, 0.9999293770912069],
      [0.0000365806628551, 0.0000044484017774, 0.9999589709353676],
      [0.0000336919173149, 0.0000047916758028, 0.9999615164068824],
      [0.0006686654325515, 0.0000293091186820, 0.9993020254487663],
      [0.0001991689985446, 0.0000069489525690, 0.9997938820488865],
      [0.0007413876012268, 0.0000336734973524, 0.9992249389014208],
      [0.0000389710766809, 0.0000063978648403, 0.9999546310584788],
      [0.0000933631817535, 0.0000090568697852, 0.9998975799484613]
    ])
    assert np.allclose(ref, self.clf.predict_proba(Xte))

  def test_predict(self):
    Xte, _ = self.test
    ref = np.array(['setosa', 'setosa', 'setosa', 'setosa', 'setosa', 'setosa', 'setosa', 'setosa', 'setosa', 'setosa', 'setosa', 'setosa', 'setosa', 'setosa', 'setosa', 'setosa', 'setosa', 'setosa', 'setosa', 'setosa', 'setosa', 'setosa', 'setosa', 'setosa', 'setosa', 'setosa', 'setosa', 'setosa', 'setosa', 'setosa', 'setosa', 'setosa', 'setosa', 'setosa', 'setosa', 'setosa', 'setosa', 'setosa', 'setosa', 'setosa', 'setosa', 'setosa', 'setosa', 'setosa', 'setosa', 'setosa', 'setosa', 'setosa', 'setosa', 'setosa', 'versicolor', 'versicolor', 'versicolor', 'versicolor', 'versicolor', 'versicolor', 'versicolor', 'versicolor', 'versicolor', 'versicolor', 'versicolor', 'versicolor', 'versicolor', 'versicolor', 'versicolor', 'versicolor', 'versicolor', 'versicolor', 'versicolor', 'versicolor', 'versicolor', 'versicolor', 'virginica', 'versicolor', 'versicolor', 'versicolor', 'versicolor', 'versicolor', 'versicolor', 'versicolor', 'versicolor', 'versicolor', 'versicolor', 'virginica', 'versicolor', 'versicolor', 'versicolor', 'versicolor', 'versicolor', 'versicolor', 'versicolor', 'versicolor', 'versicolor', 'versicolor', 'versicolor', 'versicolor', 'versicolor', 'versicolor', 'versicolor', 'versicolor', 'virginica', 'virginica', 'virginica', 'virginica', 'virginica', 'virginica', 'virginica', 'virginica', 'virginica', 'virginica', 'versicolor', 'virginica', 'virginica', 'virginica', 'virginica', 'virginica', 'virginica', 'virginica', 'virginica', 'virginica', 'virginica', 'virginica', 'virginica', 'virginica', 'virginica', 'virginica', 'versicolor', 'virginica', 'virginica', 'virginica', 'virginica', 'virginica', 'virginica', 'virginica', 'virginica', 'virginica', 'virginica', 'virginica', 'versicolor', 'virginica', 'virginica', 'versicolor', 'virginica', 'virginica', 'virginica', 'virginica', 'virginica', 'virginica', 'virginica', 'virginica'])
    assert all(ref == self.clf.predict(Xte))

  def test_score(self):
    Xte, yte = self.test
    ref = 0.96
    assert np.allclose(ref, self.clf.score(Xte, yte))

  def test_fit_exception(self):
    with self.assertRaises(Exception) as cm:
      self.clf.fit(np.array([[]]), np.array([]))

    assert str(cm.exception) == 'Not supported.'

  def test_fit_exception_regressor(self):
    with self.assertRaises(Exception) as cm:
      pmml = path.join(BASE_DIR, '../models/nn-iris.pmml')
      clf = PMMLMLPRegressor(pmml=pmml)
      clf.fit(np.array([[]]), np.array([]))

    assert str(cm.exception) == 'Not supported.'


class TestNeuralNetworkIntegrationPima(TestCase):
  def setUp(self):
    df = pd.read_csv(path.join(BASE_DIR, '../models/categorical-test.csv'))
    Xte = df.iloc[:, 1:]
    Xenc = pd.get_dummies(Xte, prefix_sep='')
    yte = df.iloc[:, 0]
    self.test = (Xte, yte)
    self.enc = (Xenc, yte)

    self.ref = MLPClassifier()
    self.ref.fit(Xenc, yte)

  def test_sklearn2pmml(self):
    # Export to PMML
    pipeline = PMMLPipeline([
      ("classifier", self.ref)
    ])
    pipeline.fit(self.enc[0], self.enc[1])
    sklearn2pmml(pipeline, "mlp-sklearn2pmml.pmml", with_repr = True)

    try:
      # Import PMML
      model = PMMLMLPClassifier(pmml='mlp-sklearn2pmml.pmml')

      # Verify classification
      Xenc, _ = self.enc
      assert np.allclose(
        self.ref.predict_proba(Xenc),
        model.predict_proba(Xenc)
      )

    finally:
      remove("mlp-sklearn2pmml.pmml")


class TestNeuralNetworkRegressionIntegrationPima(TestCase):
  def setUp(self):
    df = pd.read_csv(path.join(BASE_DIR, '../models/categorical-test.csv'))
    Xte = df.iloc[:, 1:]
    Xenc = pd.get_dummies(Xte, prefix_sep='')
    yte = df.iloc[:, 0]
    self.test = (Xte, yte)
    self.enc = (Xenc, yte)

    self.ref = MLPRegressor()
    self.ref.fit(Xenc, yte == 'Yes')

  def test_sklearn2pmml(self):
    # Export to PMML
    pipeline = PMMLPipeline([
      ("regressor", self.ref)
    ])
    pipeline.fit(self.enc[0], self.enc[1] == 'Yes')
    sklearn2pmml(pipeline, "mlp-sklearn2pmml.pmml", with_repr = True)

    try:
      # Import PMML
      model = PMMLMLPRegressor(pmml='mlp-sklearn2pmml.pmml')

      # Verify classification
      Xenc, _ = self.enc
      assert np.allclose(
        self.ref.predict(Xenc),
        model.predict(Xenc)
      )

    finally:
      remove("mlp-sklearn2pmml.pmml")

