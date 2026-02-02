import numpy as np
from qiskit import QuantumCircuit
from qiskit.quantum_info import Statevector
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge

import QRC_construction.DataEncoding as DE

"""
FEATURE VECTOR SLIDING WINDOW
"""
def x_sliding_window(input_states, 
                    res, 
                    nb_input_qubits, 
                    nb_extra_qubits, 
                    x_meas_bases, 
                    x_qubits_measured,
                    sliding_window_size,
                    is_dataset_classical,
                    Rz=False, 
                    classical_min_val=0, 
                    classical_max_val=1, 
                    ):
    """
    Parameters:
    input_states (list): a list of input states
    res (QuantumCircuit): the quantum circuit to be composed
    nb_input_qubits (int): the number of input qubits
    nb_extra_qubits (int): the number of extra qubits
    x_meas_bases (list): the bases to measure the qubits
    x_qubits_measured (list): the qubits to measure
    sliding_window_size (int): the size of the sliding window
    is_dataset_classical (bool): whether the input states are classical or not
    Rz (bool): whether to use Rz gates for encoding instead of Ry gates
    classical_min_val (float): the minimum value of the classical input states
    classical_max_val (float): the maximum value of the classical input states

    Returns:
    x_feature (numpy array): the feature vector
    """
    n = sliding_window_size
    nb_qubits = n*nb_input_qubits + nb_extra_qubits
    qc = QuantumCircuit(nb_qubits)

    for j in range(n):
        if is_dataset_classical:
            real_values = input_states[j]
            if nb_input_qubits == 1:
                real_values = [real_values]
            qc_encoded_state = DE.angle_encoding(real_values, classical_min_val, classical_max_val, Rz=Rz)
            qc.compose(qc_encoded_state, qubits=range(j*nb_input_qubits, (j+1)*nb_input_qubits), inplace=True)      
        else:
            qc.initialize(input_states[j], range(j*nb_input_qubits, (j+1)*nb_input_qubits))
    
    qc.compose(res, qubits=range(nb_qubits), inplace=True)

    state = Statevector.from_instruction(qc)

    x_feature = DE.features(state, nb_qubits, x_meas_bases, x_qubits_measured)

    return x_feature

"""
TRAIN SLIDING WINDOW
"""
def train_sliding_window(y_true, 
                        T_train, 
                        res, 
                        nb_input_qubits, 
                        nb_extra_qubits, 
                        x_meas_bases, 
                        x_qubits_measured,
                        sliding_window_size, 
                        is_dataset_classical, 
                        Rz=False,
                        classical_min_val=0,
                        classical_max_val=1,
                        y_states=None,
                        ridge=False,
                        ridge_alpha=1e-2
                        ):
    """
    Parameters:
    y_true (list): the true output states
    T_train (int): the number of training samples
    res (QuantumCircuit): the quantum circuit to be composed
    nb_input_qubits (int): the number of input qubits
    nb_extra_qubits (int): the number of extra qubits
    x_meas_bases (list): the bases to measure the qubits
    x_qubits_measured (list): the qubits to measure
    slinding_window_size (int): the size of the sliding window
    is_dataset_classical (bool): whether the input states are classical or not
    Rz (bool): whether to use Rz gates for encoding instead of Ry gates
    classical_min_val (float): the minimum value of the classical input states
    classical_max_val (float): the maximum value of the classical input states
    y_states (list): the input states, if not None
    ridge (bool): whether to use ridge regression or not
    ridge_alpha (float): the regularization parameter

    Returns:
    W (numpy array): the weights of the linear regression model
    model (LinearRegression): the linear regression model
    X_train (numpy array): the feature vector
    Y_train (numpy array): the output states
    """
    n = sliding_window_size
    X_train= []

    for i in range(n, T_train+1):

        if is_dataset_classical:
            input_states = y_true[i-n:i]
        else: 
            input_states = y_states[i-n:i]

        x_feat = x_sliding_window(input_states, 
                                res, 
                                nb_input_qubits, 
                                nb_extra_qubits, 
                                x_meas_bases, 
                                x_qubits_measured,
                                n, 
                                is_dataset_classical,
                                Rz, 
                                classical_min_val, 
                                classical_max_val,
                                )

        X_train.append(x_feat)

    X_train = np.asarray(X_train, float)

    Y_train = y_true.copy()
    Y_train = Y_train[n:T_train+1]
    Y_train = np.asarray(Y_train, float)

    if ridge:
        # Ridge regression
        """
        Smally comment about regularization (alpha) choice: 
        alpha -> 0 (too small): overfitting (high variance)
        alpha -> inf (too large): penalizes coeffs too much; underfitting (high bias)

        """
        model = Ridge(alpha=ridge_alpha, fit_intercept=False)
        model.fit(X_train, Y_train)
    else:
        # Linear regression
        model = LinearRegression(fit_intercept=False)
        model.fit(X_train, Y_train)
    
    # Weight matrix
    W = model.coef_

    return W, model, X_train, Y_train

"""
PREDICT SLIDING WINDOW
"""
def predict_sliding_window(y, 
                        T_train, 
                        T_test, 
                        res, 
                        nb_input_qubits, 
                        nb_extra_qubits, 
                        sliding_window_size, 
                        is_dataset_classical=False,
                        x_meas_bases=None, 
                        x_qubits_measured=None,
                        y_meas_bases=None, 
                        y_qubits_measured=None,
                        Rz=False,
                        classical_min_val=0,
                        classical_max_val=1,
                        ridge=False,
                        ridge_alpha=1e-2
                        ):
    """
    Parameters:
    y (list): the input sequence
    T_train (int): the number of time steps for training
    T_test (int): the number of time steps for testing
    res (QuantumCircuit): the quantum circuit to be composed
    nb_input_qubits (int): the number of input qubits
    nb_extra_qubits (int): the number of extra qubits
    sliding_window_size (int): the size of the sliding window
    is_dataset_classical (bool): whether the input states are classical or not
    x_meas_bases (list): the bases to measure the qubits
    x_qubits_measured (list): the qubits to measure
    y_meas_bases (list): the bases to measure the qubits
    y_qubits_measured (list): the qubits to measure
    Rz (bool): whether to use Rz gates for encoding instead of Ry gates
    classical_min_val (float): the minimum value of the classical input states
    classical_max_val (float): the maximum value of the classical input states
    ridge (bool): whether to use ridge regression or not
    ridge_alpha (float): the regularization parameter

    Returns:
    Y_hat (numpy array): the predicted output states
    y_true (numpy array): the true output states
    W (numpy array): the weights of the linear regression model
    X_train (numpy array): the feature vector
    Y_train (numpy array): the output states
    """
    
    n = sliding_window_size
    
    if is_dataset_classical:
        y_true = y.copy()
        W, model, X_train, Y_train = train_sliding_window(y, 
                                                        T_train, 
                                                        res, 
                                                        nb_input_qubits, 
                                                        nb_extra_qubits, 
                                                        x_meas_bases, 
                                                        x_qubits_measured, 
                                                        n, 
                                                        is_dataset_classical, 
                                                        Rz, 
                                                        classical_min_val, 
                                                        classical_max_val,
                                                        ridge = ridge,
                                                        ridge_alpha=ridge_alpha
                                                        )
        Y_hat = y_true.copy() 

        for i in range(T_train+1, T_train+T_test):
            states = Y_hat[i-n:i]
            x_feat = x_sliding_window(states, 
                                    res, 
                                    nb_input_qubits, 
                                    nb_extra_qubits, 
                                    x_meas_bases, 
                                    x_qubits_measured, 
                                    n, 
                                    is_dataset_classical, 
                                    Rz, 
                                    classical_min_val, 
                                    classical_max_val)
            
            y_hat = model.predict(x_feat.reshape(1, -1)).ravel()[0]
            Y_hat[i] = y_hat
    
    else:
        y_true = DE.construct_y_true_from_states(y, nb_input_qubits, y_meas_bases)
        W, model, X_train, Y_train = train_sliding_window(y_true, 
                                                        T_train, 
                                                        res, 
                                                        nb_input_qubits, 
                                                        nb_extra_qubits, 
                                                        x_meas_bases, 
                                                        x_qubits_measured, 
                                                        n, 
                                                        is_dataset_classical, 
                                                        y_states=y,
                                                        ridge = ridge,
                                                        ridge_alpha=ridge_alpha
                                                        )
        Y_hat = y_true.copy()
        y_states_hat = y.copy()
        labels = DE.build_pauli_labels(nb_input_qubits, y_meas_bases, y_qubits_measured)

        for i in range(T_train+1, T_train+T_test):
            states = y_states_hat[i-n:i]
            x_feat = x_sliding_window(states, 
                                    res, 
                                    nb_input_qubits, 
                                    nb_extra_qubits, 
                                    x_meas_bases, 
                                    x_qubits_measured, 
                                    n, 
                                    is_dataset_classical)
            
            y_hat = model.predict(x_feat.reshape(1, -1))

            Y_hat[i] = y_hat

            exp_dict = DE.yhat_to_exp_dict(y_hat, labels)
            rho_lin  = DE.pauli_expectations_to_rho(exp_dict, nb_input_qubits)
            
            rho_phys = DE.project_to_physical_state(rho_lin)
            psi = DE.rho_to_pure_statevector(rho_phys)
            y_states_hat[i] = psi

    return Y_hat, y_true, W, X_train, Y_train

"""
FEATURE VECTOR PARTIAL TRACE
"""
def x_partial_trace(input_state, 
                    res, 
                    nb_input_qubits, 
                    nb_memory_qubits, 
                    x_meas_bases, 
                    x_qubits_measured, 
                    circuit_with_pt, 
                    is_dataset_classical, 
                    Rz=False,
                    classical_min_val=0, 
                    classical_max_val=1,
                    perfect_recovery=True):
    """
    Parameters:
    input_state: the input states from the last step
    res (QuantumCircuit): the quantum circuit to be composed
    nb_input_qubits (int): the number of input qubits
    nb_extra_qubits (int): the number of extra qubits
    x_meas_bases (list): the bases to measure the qubits
    x_qubits_measured (list): the qubits to measure
    circuit_with_pt: the quantum circuit from previous steps
    is_dataset_classical (bool): whether the input states are classical or not
    Rz (bool): whether to use Rz gates for encoding instead of Ry gates
    classical_min_val (float): the minimum value of the classical input states
    classical_max_val (float): the maximum value of the classical input states
    perfect_recovery (bool): whether to use perfect recovery (perfect reconstruction of the measured state) or not

    Returns:
    x_feature (numpy array): the feature vector
    circuit_with_pt (QuantumCircuit): the quantum circuit after application of the reservoir (and measurements if not perfect recovery)
    """
    nb_qubits = nb_memory_qubits + nb_input_qubits

    if x_qubits_measured is None:
        x_qubits_measured = list(range(nb_qubits))
    
    circuit_with_pt.reset(range(nb_input_qubits))
    
    if is_dataset_classical:
        real_value = input_state
        if nb_input_qubits == 1:
            real_value = [input_state]
        qc_encoded_state = DE.angle_encoding(real_value, classical_min_val, classical_max_val, Rz=Rz)
        circuit_with_pt.compose(qc_encoded_state, qubits=range(nb_input_qubits), inplace=True)      
    else:
        circuit_with_pt.initialize(input_state, range(nb_input_qubits)) 

    circuit_with_pt.compose(res, qubits=range(nb_qubits), inplace=True)

    state = Statevector.from_instruction(circuit_with_pt)
    
    x_feature = DE.features(state, nb_qubits, x_meas_bases, x_qubits_measured)

    # If perfect recovery is not used, we need to reconstruct the measured state.
    if not perfect_recovery:
        labels = DE.build_pauli_labels(nb_qubits, x_meas_bases, x_qubits_measured)
        exp_dict = DE.yhat_to_exp_dict(x_feature, labels)
        rho_lin = DE.pauli_expectations_to_rho(exp_dict, len(x_qubits_measured))
        rho_phys = DE.project_to_physical_state(rho_lin)
        psi = DE.rho_to_pure_statevector(rho_phys)

        circuit_with_pt.reset(x_qubits_measured)
        circuit_with_pt.initialize(psi, x_qubits_measured)

    return x_feature, circuit_with_pt


"""
TRAIN PARTIAL TRACE
"""
def train_partial_trace(y_true, 
                        T_train, 
                        res, 
                        nb_input_qubits, 
                        nb_memory_qubits, 
                        x_meas_bases, 
                        x_qubits_measured, 
                        is_dataset_classical, 
                        Rz=False,
                        classical_min_val=0, 
                        classical_max_val=1,
                        perfect_recovery=True, 
                        y_states=None):
    """
    Parameters:
    y_true (list): the true output states
    T_train (int): the number of training samples
    res (QuantumCircuit): the quantum circuit to be composed
    nb_input_qubits (int): the number of input qubits
    nb_memory_qubits (int): the number of memory qubits
    x_meas_bases (list): the bases to measure the qubits
    x_qubits_measured (list): the qubits to measure
    is_dataset_classical (bool): whether the input states are classical or not
    Rz (bool): whether to use Rz gates for encoding instead of Ry gates
    classical_min_val (float): the minimum value of the classical input states
    classical_max_val (float): the maximum value of the classical input states
    perfect_recovery (bool): whether to use perfect recovery or not
    y_states (list): the input states, if not None

    Returns:
    W (numpy array): the weights of the linear regression model
    model (LinearRegression): the linear regression model
    X_train (numpy array): the input features
    Y_train (numpy array): the output states
    qc (QuantumCircuit): the quantum circuit after application of the reservoir (and measurements if not perfect recovery)
    """
    nb_qubits = nb_input_qubits + nb_memory_qubits
    qc = QuantumCircuit(nb_qubits)

    X_train = []
    for i in range(1, T_train+1):
        if is_dataset_classical:
            input_state = y_true[i-1]
        else: 
            input_state = y_states[i-1]

        x_feat, qc = x_partial_trace(input_state, 
                                    res, 
                                    nb_input_qubits, 
                                    nb_memory_qubits, 
                                    x_meas_bases, 
                                    x_qubits_measured, 
                                    qc, 
                                    is_dataset_classical, 
                                    Rz,
                                    classical_min_val, 
                                    classical_max_val,
                                    perfect_recovery=perfect_recovery)
        
        X_train.append(x_feat)

    X_train = np.asarray(X_train, float)
    Y_train = y_true.copy()
    Y_train = Y_train[1:T_train+1]
    Y_train = np.asarray(Y_train, float)

    model = LinearRegression(fit_intercept=False) 
    model.fit(X_train, Y_train)
    W = model.coef_

    return W, model, X_train, Y_train, qc


"""
PREDICT PARTIAL TRACE
"""
def predict_partial_trace(y, 
                        T_train, 
                        T_test, 
                        res,
                        nb_input_qubits, 
                        nb_memory_qubits, 
                        is_dataset_classical=False,
                        x_meas_bases=None, 
                        x_qubits_measured=None,
                        y_meas_bases=None, 
                        y_qubits_measured=None,
                        Rz=False,
                        classical_min_val=0,
                        classical_max_val=1, 
                        perfect_recovery=True): 
    """
    Parameters:
    y (list): the input sequence
    T_train (int): the number of time steps for training
    T_test (int): the number of time steps for testing
    res (QuantumCircuit): the quantum circuit to be composed
    nb_input_qubits (int): the number of input qubits
    nb_memory_qubits (int): the number of memory qubits
    is_dataset_classical (bool): whether the input states are classical or not
    x_meas_bases (list): the bases to measure the qubits
    x_qubits_measured (list): the qubits to measure
    y_meas_bases (list): the bases to measure the qubits
    y_qubits_measured (list): the qubits to measure
    Rz (bool): whether to use Rz gates for encoding instead of Ry gates
    classical_min_val (float): the minimum value of the classical input states
    classical_max_val (float): the maximum value of the classical input states
    perfect_recovery (bool): whether to use perfect recovery (perfect reconstruction of the measured state) or not

    Returns:
    Y_hat (numpy array): the predicted output states
    y_true (numpy array): the true output states
    W (numpy array): the weights of the model
    X_train (numpy array): the training data
    Y_train (numpy array): the training labels
    qc (QuantumCircuit): the quantum circuit after application of the reservoir (and measurements if not perfect recovery)
    """
    if is_dataset_classical:
        y_true = y.copy()
        
        W, model, X_train, Y_train, qc = train_partial_trace(y_true, 
                                                            T_train, 
                                                            res, 
                                                            nb_input_qubits, 
                                                            nb_memory_qubits, 
                                                            x_meas_bases, 
                                                            x_qubits_measured, 
                                                            is_dataset_classical,
                                                            Rz, 
                                                            classical_min_val, 
                                                            classical_max_val, 
                                                            perfect_recovery=perfect_recovery)
        Y_hat = y_true.copy() 

        for i in range(T_train+1, T_train+T_test):
            input_state = Y_hat[i-1]
            x_feat, qc = x_partial_trace(input_state, 
                                        res, 
                                        nb_input_qubits, 
                                        nb_memory_qubits, 
                                        x_meas_bases, 
                                        x_qubits_measured, 
                                        qc, 
                                        is_dataset_classical,
                                        Rz, 
                                        classical_min_val, 
                                        classical_max_val, 
                                        perfect_recovery=perfect_recovery)
            
            y_hat = model.predict(x_feat.reshape(1, -1)).ravel()[0]

            Y_hat[i] = y_hat
    else:
        y_true = DE.construct_y_true_from_states(y, nb_input_qubits, y_meas_bases)
        W, model, X_train, Y_train, qc = train_partial_trace(y_true, 
                                                            T_train, 
                                                            res, 
                                                            nb_input_qubits, 
                                                            nb_memory_qubits, 
                                                            x_meas_bases, 
                                                            x_qubits_measured, 
                                                            is_dataset_classical, 
                                                            perfect_recovery=perfect_recovery,
                                                            y_states=y)
        Y_hat = y_true.copy()
        y_states_hat = y.copy()

        labels = DE.build_pauli_labels(nb_input_qubits, y_meas_bases, y_qubits_measured)

        for i in range(T_train+1, T_train+T_test):
            
            input_state = y_states_hat[i-1]
            x_feat, qc = x_partial_trace(input_state, 
                                        res, 
                                        nb_input_qubits, 
                                        nb_memory_qubits, 
                                        x_meas_bases, 
                                        x_qubits_measured, 
                                        qc, 
                                        is_dataset_classical, 
                                        perfect_recovery=perfect_recovery)
            
            y_hat = model.predict(x_feat.reshape(1, -1))

            Y_hat[i] = y_hat

            exp_dict = DE.yhat_to_exp_dict(y_hat, labels)
            rho_lin  = DE.pauli_expectations_to_rho(exp_dict, nb_input_qubits)
            
            rho_phys = DE.project_to_physical_state(rho_lin)
            psi = DE.rho_to_pure_statevector(rho_phys)
            y_states_hat[i] = psi

    return Y_hat, y_true, W, X_train, Y_train, qc


"""
PREDICT LINEAR REGRESSION (OR RIDGE)
"""
def predict_linear_regression(y_true, T_train, T_test, n, ridge=False, ridge_alpha=1e-2):
    """
    Parameters:
    y_true (numpy array): the true output states
    T_train (int): the number of training points
    T_test (int): the number of test points
    n (int): the number of previous points to use as features
    ridge (bool): whether to use ridge regression or not
    ridge_alpha (float): the regularization parameter

    Returns:
    Y_hat (numpy array): the predicted output states
    y_true (numpy array): the true output states
    W (numpy array): the weights of the model
    X_train (numpy array): the training data
    Y_train (numpy array): the training labels
    """
    X_train = []
    for i in range(n, T_train+1):
        x_feat = y_true[i-n:i]
        X_train.append(x_feat)

    X_train = np.asarray(X_train, float)
    Y_train = y_true.copy()
    Y_train = Y_train[n:T_train+1]
    Y_train = np.asarray(Y_train, float)

    if ridge:
        model = Ridge(alpha=ridge_alpha, fit_intercept=False)
        model.fit(X_train, Y_train)
    else:
        model = LinearRegression(fit_intercept=False)
        model.fit(X_train, Y_train)

    Y_hat = y_true.copy()
    for i in range(T_train+1, T_train+T_test):
        x_feat = y_true[i-n:i]
        y_hat = model.predict(x_feat.reshape(1, -1))
        Y_hat[i] = y_hat
    
    W = model.coef_

    return Y_hat, y_true, W, X_train, Y_train
