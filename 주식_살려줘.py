
!pip install pandas_datareader
!pip install --upgrade pandas_datareader
!pip install yfinance
!pip install qiskit qiskit-machine-learning pylatexenc --user

# Importing standard Qiskit libraries
from qiskit import QuantumCircuit, transpile
from qiskit.tools.jupyter import *
from qiskit.visualization import *
from ibm_quantum_widgets import *

# qiskit-ibmq-provider has been deprecated.
# Please see the Migration Guides in https://ibm.biz/provider_migration_guide for more detail.
from qiskit_ibm_runtime import QiskitRuntimeService, Sampler, Estimator, Session, Options

# Loading your IBM Quantum account(s)
service = QiskitRuntimeService(channel="ibm_quantum")

# Invoke a primitive. For more details see https://qiskit.org/documentation/partners/qiskit_ibm_runtime/tutorials.html
# result = Sampler("ibmq_qasm_simulator").run(circuits).result()

import pandas_datareader.data as web
import datetime
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from qiskit import QuantumCircuit, Aer
from sklearn.preprocessing import MinMaxScaler
from qiskit.opflow import PauliSumOp
from qiskit_machine_learning.neural_networks import CircuitQNN
from qiskit.circuit.library import ZZFeatureMap, PauliFeatureMap
from qiskit.utils import QuantumInstance, algorithm_globals
from qiskit_machine_learning.algorithms.classifiers import NeuralNetworkClassifier
from qiskit.algorithms.optimizers import SPSA, COBYLA
from sklearn.metrics import accuracy_score

from pandas_datareader import data as data_reader
import yfinance as yf

# 데이터 수집 및 전처리
symbol = "AAPL"
start_date = datetime.datetime(2010, 1, 4)
end_date = datetime.datetime(2020, 1 ,1)
stock_data = yf.download(symbol, start=start_date, end=end_date)

# 데이터 전처리
stock_data['Returns'] = (stock_data['Close'] / stock_data['Close'].shift(1)) - 1
stock_data.dropna(inplace=True)
stock_data['Returns_label'] = np.where(stock_data['Returns'] > 0, 1, 0)

# 데이터 전처리
X_data_temp = stock_data[['Open', 'High', 'Low', 'Close']].values  # 이 부분이 수정됨
Y_data_temp = stock_data['Returns_label'].values


# 데이터 정규화
scaler = MinMaxScaler()
X_data_temp = scaler.fit_transform(X_data_temp)

# 파라미터 설정 및 변분 회로 생성
features = X_data_temp.shape[1]
num_qubits = features
algorithm_globals.random_seed = 42

feature_map = PauliFeatureMap(feature_dimension=features, reps=2, paulis=['Z', 'X', 'ZY'])
variational_circ = QuantumCircuit(num_qubits)
for _ in range(4):
    variational_circ.h(range(num_qubits))
    variational_circ.barrier()
    for i in range(num_qubits):
        variational_circ.crz(np.pi, i, (i + 1) % num_qubits)
        variational_circ.barrier()

backend = Aer.get_backend('statevector_simulator')

# Observable 지정
observable = PauliSumOp.from_list([("ZZ" * num_qubits, 1)])

quantum_instance = QuantumInstance(backend,
                                   shots=1024,
                                   seed_simulator=algorithm_globals.random_seed,
                                   seed_transpiler=algorithm_globals.random_seed)

# 양자 회로 생성
num_inputs = features
qc = QuantumCircuit(num_inputs)
qc.append(feature_map, range(num_inputs))
qc.append(variational_circ, range(num_inputs))

# 양자 회로 이미지로 저장
qc.decompose().draw(output="mpl", filename="quantum_circuit.png")

def parity(x):
    return "{:b}".format(x).count("1") % 2

output_shape = len(np.unique(Y_data_temp))

qnn_architecture = CircuitQNN(
    circuit=qc,
    input_params=feature_map.parameters,
    weight_params=variational_circ.parameters,
    interpret=parity,
    output_shape=output_shape,
    quantum_instance=quantum_instance,
)

# 데이터셋 분할
X_train, X_test, y_train, y_test = train_test_split(X_data_temp, Y_data_temp, test_size=0.3, random_state=algorithm_globals.random_seed)

# Optimizer 설정
max_iters = 600
optimizer = COBYLA(maxiter=max_iters)

qnnc = NeuralNetworkClassifier(qnn_architecture, optimizer=optimizer)

# 모델 학습
qnnc.fit(X_train, y_train)

# 모델 예측
y_pred = qnnc.predict(X_test)

# 정확도 계산
accuracy = accuracy_score(y_test, y_pred)

print("모델 정확도: {:.2f}%".format(accuracy * 100))

# 실제값과 예측값의 선 그래프 표시 및 이미지로 저장
plt.figure(figsize=(12, 6))
plt.plot(y_test[:50], label="real")
plt.plot(y_pred[:50], label="predict")  # 이 부분이 수정됨
plt.ylabel("value")
plt.xlabel("data index")
plt.legend()
plt.title("real and predict compare graph")
plt.savefig("price_prediction.png")
plt.show()
