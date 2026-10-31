"""Optional simulated quantum kernels (qiskit-machine-learning). Import is guarded."""
try:
    from qiskit.circuit.library import ZZFeatureMap
    from qiskit_machine_learning.kernels import FidelityQuantumKernel
    HAS_QISKIT = True
except Exception:  # pragma: no cover
    HAS_QISKIT = False


def quantum_kernel(n_features: int, reps: int = 2):
    """Return a FidelityQuantumKernel with a ZZ feature map (statevector simulation)."""
    if not HAS_QISKIT:
        raise ImportError("install extras: pip install -e .[quantum]")
    return FidelityQuantumKernel(feature_map=ZZFeatureMap(n_features, reps=reps))


def quantum_gram(X, Y=None, reps=2):
    qk = quantum_kernel(X.shape[1], reps)
    return qk.evaluate(X, Y)
