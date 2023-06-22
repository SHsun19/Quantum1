
# !pip install qiskit qiskit-machine-learning pylatexenc --user

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


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
import qiskit
from qiskit import QuantumCircuit, transpile, Aer
from qiskit.opflow import PauliSumOp
from qiskit_machine_learning.neural_networks import TwoLayerQNN, CircuitQNN
from qiskit.circuit.library import ZZFeatureMap
from qiskit.utils import QuantumInstance, algorithm_globals
from qiskit.algorithms.optimizers import COBYLA
from qiskit_machine_learning.algorithms.classifiers import NeuralNetworkClassifier
import tensorflow as tf
import time
from IPython.display import clear_output

# Load the iris dataset
data = load_iris(as_frame=False)
X_data_temp = data.data
Y_data_temp = data.target

features = X_data_temp.shape[1]
num_qubits = features
algorithm_globals.random_seed = 42

# Create feature map and variational circuit
feature_map = ZZFeatureMap(feature_dimension=features, reps=1, entanglement="full")
variational_circ = QuantumCircuit(num_qubits)
for _ in range(2):
    variational_circ.h(range(num_qubits))
    variational_circ.barrier()
    for i in range(num_qubits):
        variational_circ.crz(np.pi, i, (i + 1) % num_qubits)
        variational_circ.barrier()
variational_circ.draw('mpl')

backend = Aer.get_backend('statevector_simulator')

# Specify the observable
observable = PauliSumOp.from_list([("ZZ" * num_qubits, 1)])

quantum_instance = QuantumInstance(backend, 
                                   shots=1024, 
                                   seed_simulator = algorithm_globals.random_seed, 
                                   seed_transpiler = algorithm_globals.random_seed)

two_layers = False
if two_layers:
    qnn_architecture = TwoLayerQNN(
        num_qubits, 
        feature_map=feature_map, 
        ansatz=variational_circ, 
        observable=observable, 
        quantum_instance=quantum_instance
    )
else:
    num_inputs = features
    qc = QuantumCircuit(num_inputs)
    qc.append(feature_map, range(num_inputs))
    qc.append(variational_circ, range(num_inputs))
    qc.decompose().draw(output="mpl")
    
    def parity(x):
        return "{:b}".format(x).count("1") % 2

    output_shape = len(np.unique(Y_data_temp))

    # gradient 인수를 제거하였습니다.
    qnn_architecture = CircuitQNN(
        circuit=qc,
        input_params=feature_map.parameters,
        weight_params=variational_circ.parameters,
        interpret=parity,
        output_shape=output_shape,
        quantum_instance=quantum_instance,
    )

# Define (random) input and weights
input3 = algorithm_globals.random.random(qnn_architecture.num_inputs)
weights3 = algorithm_globals.random.random(qnn_architecture.num_weights)

# 'qnn_architecture.backward(input3, weights3)' 부분이 삭제되었습니다.

X_data_temp.shape
Y_data_temp.shape

X_train, X_test, y_train, y_test = train_test_split(X_data_temp, Y_data_temp, test_size=0.2, random_state=algorithm_globals.random_seed)

optimizer = COBYLA(maxiter=400, tol=0.001)
def callback_graph(weights, obj_func_eval):
    objective_func_vals.append(obj_func_eval)
    plt.cla()  # 매 반복 때마다 현재 그래프를 지웁니다.
    plt.title("Objective function value against iteration")
    plt.xlabel("Iteration")
    plt.ylabel("Objective function value")
    plt.plot(range(len(objective_func_vals)), objective_func_vals)
    plt.pause(0.01)  # 그래프를 갱신하고 일시 정지합니다.

    
qnnc = NeuralNetworkClassifier(qnn_architecture, optimizer=optimizer, callback=callback_graph)

start = time.time()

plt.rcParams['figure.figsize'] = [10, 7]
objective_func_vals = []



y_train_fit = y_train.reshape(-1)
qnnc.fit(X_train, y_train_fit)
plt.close()  # 플롯을 닫습니다.

qnnc.fit(X_train, y_train_fit)

elapsed = time.time() - start
print("Time elapsed: ", elapsed)
plt.show()

from sklearn.metrics import accuracy_score

# 모델 예측
y_pred = qnnc.predict(X_test)

# 정확도 계산
accuracy = accuracy_score(y_test, y_pred)

print("모델 정확도: {:.2f}%".format(accuracy * 100))

# 혼동 행렬 계산
cm = confusion_matrix(y_test, y_pred)
print('Confusion matrix:')
print(cm)

# 분류 리포트 출력
print('Classification report:')
print(classification_report(y_test, y_pred))
