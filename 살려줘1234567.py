!pip install qiskit qiskit-machine-learning pylatexenc --user
!pip install tensorflow

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from qiskit import QuantumCircuit, Aer
from qiskit.opflow import PauliSumOp
from qiskit_machine_learning.neural_networks import CircuitQNN
from qiskit.circuit.library import ZZFeatureMap
from qiskit.utils import QuantumInstance, algorithm_globals
from qiskit_machine_learning.algorithms.classifiers import NeuralNetworkClassifier
from qiskit.algorithms.optimizers import SPSA
from sklearn.metrics import accuracy_score

# iris 데이터셋 로드
data = load_iris(as_frame=False)
X_data_temp = data.data
Y_data_temp = data.target

features = X_data_temp.shape[1]
num_qubits = features
algorithm_globals.random_seed = 42

# Feature 맵과 변분 회로 생성
feature_map = ZZFeatureMap(feature_dimension=features, reps=1, entanglement="full")
variational_circ = QuantumCircuit(num_qubits)
for _ in range(2):
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

num_inputs = features
qc = QuantumCircuit(num_inputs)
qc.append(feature_map, range(num_inputs))
qc.append(variational_circ, range(num_inputs))
qc.decompose().draw(output="mpl")

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

X_train, X_test, y_train, y_test = train_test_split(X_data_temp, Y_data_temp, test_size=0.2, random_state=algorithm_globals.random_seed)

# Optimizer 설정
max_iters = 400
optimizer = SPSA(maxiter=max_iters)

qnnc = NeuralNetworkClassifier(qnn_architecture, optimizer=optimizer)

# 훈련
qnnc.fit(X_train, y_train)

# 모델 예측
y_pred = qnnc.predict(X_test)

# 정확도 계산
accuracy = accuracy_score(y_test, y_pred)

print("모델 정확도: {:.2f}%".format(accuracy * 100))

# 실제값과 예측값의 선 그래프 표시
plt.figure(figsize=(12, 6))
plt.plot(y_test, label="real")
plt.plot(y_pred, label="predict")
plt.ylabel("value")
plt.xlabel("data index")
plt.legend()
plt.title("real and predict compare graph")
plt.show()
