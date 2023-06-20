import numpy as np
import random
from qiskit import transpile
import numpy as np
import random
from qiskit import QuantumCircuit
from qiskit.circuit import Parameter, Measure
from qiskit.transpiler import Target, InstructionProperties
from qiskit.circuit.library import CXGate, UGate
import networkx as nx
from qiskit.circuit import ClassicalRegister, QuantumCircuit, CircuitInstruction
from qiskit.circuit import Reset
from qiskit.circuit.library import standard_gates
from qiskit.circuit.exceptions import CircuitError
from tf_agents.environments import py_environment
from tf_agents.environments import utils
from tf_agents.specs import array_spec
from tf_agents.trajectories import time_step as ts

class QubitAssignmentEnv(py_environment.PyEnvironment):

    def __init__(self, num_qubits, depth):
        self._action_spec = array_spec.BoundedArraySpec(
            shape=(), dtype=np.int32, minimum=0, maximum=num_qubits*(num_qubits-1)//2, name='action')
        self._observation_spec = array_spec.BoundedArraySpec(
            shape=(num_qubits,num_qubits+depth-1), dtype=np.int32, minimum=-1, name='observation')
        self.target = None
        self.qc = None
        self.num_qubits = num_qubits
        self.depth = depth
        self._state = None
        self.current_mapping = None
        self._episode_ended = False
        self.swaps = [(i,j) for i in range(num_qubits) for j in range(i+1,num_qubits)]
        self.prev_depth = None
        self.moves = None
        self._initial_state = None

    def action_spec(self):
        return self._action_spec

    def observation_spec(self):
        return self._observation_spec

    def _reset(self):
        self._episode_ended = False
        self.current_mapping = list(range(self.num_qubits))
        self._initial_state = self.get_initial_state(self.num_qubits, self.depth)
        self._state = np.copy(self._initial_state)
        self.moves = 0
        self.prev_depth = min([transpile(self.qc, target=self.target,initial_layout=self.current_mapping,routing_method='sabre',optimization_level=0).depth() for _ in range(8)])
        #self._last_action = None
        return ts.restart(self._state)

    def _step(self, action):
        self.moves += 1
        #print(f'At move number {self.moves} taking action {action}')
        if self._episode_ended:
            # The last action ended the episode. Ignore the current action and start a new episode.
            return self.reset()
        # Make sure episodes don't go on forever.
        if action == self.num_qubits*(self.num_qubits-1)//2:
            self._episode_ended = True
            return ts.termination(self._state, reward=0.0)
        elif action >= 0 and action < self.num_qubits*(self.num_qubits-1)//2:
            qubit1 = self.swaps[action][0]
            qubit2 = self.swaps[action][1]
            self.current_mapping[qubit1], self.current_mapping[qubit2] = self.current_mapping[qubit2], self.current_mapping[qubit1]
            for i in range(self.num_qubits):
                for j in range(self.depth):
                    if self._initial_state[i][j] >= 0 and self._initial_state[i][j] < self.num_qubits:
                        self._state[self.current_mapping[i]][j] = self.current_mapping[self._initial_state[i][j]]
                    else:
                        self._state[self.current_mapping[i]][j] = self._initial_state[i][j]
        else:
            raise ValueError(f'`action` should be between 0 and {self.num_qubits*(self.num_qubits-1)//2}')

        new_depth = min([transpile(self.qc,target=self.target,initial_layout=self.current_mapping,routing_method='sabre',optimization_level=0).depth() for _ in range(8)])
        reward = self.prev_depth - new_depth - 0.5
        self.prev_depth = new_depth
        if self.moves >= 3*self.num_qubits-1:
            self._episode_ended = True
            return ts.termination(self._state, reward=reward)
        return ts.transition(self._state, reward=reward, discount=1)

    def get_initial_state(self, num_qubits, depth):
        self.target , adj_qubits = self.generate_fake_athens_target()
        self.qc = self.random_circuit(num_qubits, depth, measure=False)
        gates_per_qubit = self.get_lists_of_gates(self.qc)
        return np.array([gates_per_qubit[i] + adj_qubits[i] for i in range(self.num_qubits)], dtype=np.int32)

    @staticmethod
    def generate_random_non_directional_target(n_qubits):
        '''
        Returns a target with n_qubits that can execute any single qubit and CNOT gate in both directions
        '''
        target = Target(num_qubits=n_qubits)

        #List of all possible edges between qubits
        two_qubit_connections = []
        for i in range(n_qubits):
            for j in range(i+1, n_qubits):
                two_qubit_connections.append((i, j))
        

        # Create an empty graph with n nodes
        G = nx.Graph()
        G.add_nodes_from(range(n_qubits))

        # Add random edges until the graph is connected
        edges = []
        while not nx.is_connected(G):
            # Choose a random connection and remove it from the list
            u, v = random.choice(two_qubit_connections)
            two_qubit_connections.remove((u, v))
            edges.append((u, v))
            edges.append((v, u))
            # Add the edge between the two nodes
            G.add_edge(u, v)
        #Obtain list of adjacent nodes for each node
        adj_nodes = [list(G.adj[node]) for node in range(n_qubits)]
        #Fill all the lists with -1 values to make them all the same length
        adj_nodes = [x + [-1]*(n_qubits-1-len(x)) for x in adj_nodes]
        #Get connections of the target
        two_qubit_connections= edges

        #Add the rotation gates and the C-NOT gate to the target
        target.add_instruction(
            UGate(Parameter('theta'), Parameter('phi'), Parameter('lam')),
            {
                (x,): InstructionProperties(error=.00001, duration=5e-8) for x in range(n_qubits)
            }
        )
        target.add_instruction(CXGate(), {c : None for c in two_qubit_connections})
        target.add_instruction(
            Measure(),
            {
                (x,): InstructionProperties(error=.001, duration=5e-5) for x in range(n_qubits)
            }
        )
        
        #display(target.build_coupling_map().draw())

        return target, adj_nodes
    
    @staticmethod
    def generate_fake_athens_target():
        '''
        Returns a target with n_qubits that can execute any single qubit and CNOT gate in both directions
        '''
        target = Target(num_qubits=5)
        # Add random edges until the graph is connected
        two_qubit_connections = [(0,1),(1,0),(1,2),(2,1),(2,3),(3,2),(3,4),(4,3)]

        #Add the rotation gates and the C-NOT gate to the target
        target.add_instruction(
            UGate(Parameter('theta'), Parameter('phi'), Parameter('lam')),
            {
                (x,): InstructionProperties(error=.00001, duration=5e-8) for x in range(5)
            }
        )
        target.add_instruction(CXGate(), {c : None for c in two_qubit_connections})
        target.add_instruction(
            Measure(),
            {
                (x,): InstructionProperties(error=.001, duration=5e-5) for x in range(5)
            }
        )
        
        #display(target.build_coupling_map().draw())

        return target, [[1,-1,-1,-1],[0,2,-1,-1], [1,3,-1,-1], [2,4,-1,-1], [3,-1,-1,-1]]
        
    @staticmethod
    def random_circuit(num_qubits, depth, max_operands=4, measure=False, conditional=False, reset=False, seed=None):
        """Generate random circuit of arbitrary size and form.
        This function will generate a random circuit by randomly selecting gates
        from the set of standard gates in :mod:`qiskit.extensions`. For example:
        .. plot::
        :include-source:
        from qiskit.circuit.random import random_circuit
        circ = random_circuit(2, 2, measure=True)
        circ.draw(output='mpl')
        Args:
            num_qubits (int): number of quantum wires
            depth (int): layers of operations (i.e. critical path length)
            max_operands (int): maximum qubit operands of each gate (between 1 and 4)
            measure (bool): if True, measure all qubits at the end
            conditional (bool): if True, insert middle measurements and conditionals
            reset (bool): if True, insert middle resets
            seed (int): sets random seed (optional)
        Returns:
            QuantumCircuit: constructed circuit
        Raises:
            CircuitError: when invalid options given
        """
        if num_qubits == 0:
            return QuantumCircuit()
        if max_operands < 1 or max_operands > 4:
            raise CircuitError("max_operands must be between 1 and 4")
        max_operands = max_operands if num_qubits > max_operands else num_qubits

        gates_1q = [ (standard_gates.UGate, 1, 3) ]
        if reset:
            gates_1q.append((Reset, 1, 0))
        gates_2q = [(standard_gates.CXGate, 2, 0)]

        gates = gates_1q.copy()
        if max_operands >= 2:
            gates.extend(gates_2q)
        gates = np.array(
            gates, dtype=[("class", object), ("num_qubits", np.int64), ("num_params", np.int64)]
        )
        gates_1q = np.array(gates_1q, dtype=gates.dtype)

        qc = QuantumCircuit(num_qubits)

        if measure or conditional:
            cr = ClassicalRegister(num_qubits, "c")
            qc.add_register(cr)

        if seed is None:
            seed = np.random.randint(0, np.iinfo(np.int32).max)
        rng = np.random.default_rng(seed)

        qubits = np.array(qc.qubits, dtype=object, copy=True)

        # Apply arbitrary random operations in layers across all qubits.
        for _ in range(depth):
            # We generate all the randomness for the layer in one go, to avoid many separate calls to
            # the randomisation routines, which can be fairly slow.

            # This reliably draws too much randomness, but it's less expensive than looping over more
            # calls to the rng. After, trim it down by finding the point when we've used all the qubits.
            gate_specs = rng.choice(gates, size=len(qubits))
            cumulative_qubits = np.cumsum(gate_specs["num_qubits"], dtype=np.int64)
            # Efficiently find the point in the list where the total gates would use as many as
            # possible of, but not more than, the number of qubits in the layer.  If there's slack, fill
            # it with 1q gates.
            max_index = np.searchsorted(cumulative_qubits, num_qubits, side="right")
            gate_specs = gate_specs[:max_index]
            slack = num_qubits - cumulative_qubits[max_index - 1]
            if slack:
                gate_specs = np.hstack((gate_specs, rng.choice(gates_1q, size=slack)))

            # For efficiency in the Python loop, this uses Numpy vectorisation to pre-calculate the
            # indices into the lists of qubits and parameters for every gate, and then suitably
            # randomises those lists.
            q_indices = np.empty(len(gate_specs) + 1, dtype=np.int64)
            p_indices = np.empty(len(gate_specs) + 1, dtype=np.int64)
            q_indices[0] = p_indices[0] = 0
            np.cumsum(gate_specs["num_qubits"], out=q_indices[1:])
            np.cumsum(gate_specs["num_params"], out=p_indices[1:])
            parameters = rng.uniform(0, 2 * np.pi, size=p_indices[-1])
            rng.shuffle(qubits)

            # We've now generated everything we're going to need.  Now just to add everything.  The
            # conditional check is outside the two loops to make the more common case of no conditionals
            # faster, since in Python we don't have a compiler to do this for us.
            if conditional:
                is_conditional = rng.random(size=len(gate_specs)) < 0.1
                condition_values = rng.integers(
                    0, 1 << min(num_qubits, 63), size=np.count_nonzero(is_conditional)
                )
                c_ptr = 0
                for gate, q_start, q_end, p_start, p_end, is_cond in zip(
                    gate_specs["class"],
                    q_indices[:-1],
                    q_indices[1:],
                    p_indices[:-1],
                    p_indices[1:],
                    is_conditional,
                ):
                    operation = gate(*parameters[p_start:p_end])
                    if is_cond:
                        # The condition values are required to be bigints, not Numpy's fixed-width type.
                        operation.condition = (cr, int(condition_values[c_ptr]))
                        c_ptr += 1
                    qc._append(CircuitInstruction(operation=operation, qubits=qubits[q_start:q_end]))
            else:
                for gate, q_start, q_end, p_start, p_end in zip(
                    gate_specs["class"], q_indices[:-1], q_indices[1:], p_indices[:-1], p_indices[1:]
                ):
                    operation = gate(*parameters[p_start:p_end])
                    qc._append(CircuitInstruction(operation=operation, qubits=qubits[q_start:q_end]))

        if measure:
            qc.measure(qc.qubits, cr)

        return qc

    @staticmethod
    def get_lists_of_gates(qc):
        lists = [[] for _ in range(qc.num_qubits)]
        #Iterate over all the gates in the circuit
        for gate in qc.data:
            if gate[0].name == 'measure':
                #Get qubit index
                qubit = qc.find_bit(gate[1][0]).index
                #Append gate to list of qubit
                lists[qubit].append(qc.num_qubits+1)
            elif gate[0].name == 'cx':
                #Get qubit indices
                qubit1 = qc.find_bit(gate[1][0]).index
                qubit2 = qc.find_bit(gate[1][1]).index
                #Append gate to the lists of both qubits
                lists[qubit1].append(qubit2)
                lists[qubit2].append(qubit1)
            elif gate[0].name == 'u':
                #Get qubit index
                qubit = qc.find_bit(gate[1][0]).index
                #Append gate to list of qubit
                lists[qubit].append(qc.num_qubits)
            elif gate[0].name == 'barrier':
                pass
            else:
                raise ValueError(f'Gate {gate[0].name} not supported')
        return lists


