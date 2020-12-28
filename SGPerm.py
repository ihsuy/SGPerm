import math
import collections
import numpy as np
import tensorflow as tf
from sklearn.utils import shuffle as train_shuffler
from tqdm.notebook import trange as nested_progress_bar


class SGPerm:
    """Stochastic gradient permutation optimizer.

    An algorithm that trains deep neural networks by looking for permutation 
    of weights based on gradients (optionally).
    
    # Arguments
        Refer to class SGPermTrainLoop's docstring.

    # References
        - [Permute to Train: A New Dimension to Training Deep Neural Networks](
        https://arxiv.org/abs/2003.02570/v4)
    """

    def __init__(self,
                 alpha0,
                 lamda0,
                 beta1,
                 beta2,
                 alpha_min,
                 lamda_max,
                 momentum_lr=0.01,
                 momentum=0.9,
                 partition_size_min=10,
                 include_partition_remainder=True):

        self.alpha = alpha0
        self.lamda = lamda0
        self.beta1 = beta1
        self.beta2 = beta2
        self.lamda_max = lamda_max
        self.alpha_min = alpha_min

        self.momentum_lr = momentum_lr
        self.momentum = momentum

        self.partition_size_min = partition_size_min
        self.include_partition_remainder = include_partition_remainder

        # Keep track of total iterations
        self.iterations = 0

    def set_model(self, model, trainer):
        self.model = model
        self.trainer = trainer

    def create_slots(self):
        # create momentum slots
        self.moments = [np.zeros(variable.shape)
                        for variable in self.model.trainable_variables]

    def permute(self, grads, variables):
        """ Main algorithm loop.

        Using momentum accelerated gradients as recommendations,
        find cycles which result in permutations that satisfies these
        recommendations. And finally performs the permutations.

        # References
        - Algorithm 1 SGPerm.
        - Section 2 Stochastic Gradient Permutation: 
        Training DNNs Like Solving Picture Puzzles
        """

        if self.iterations == 0:
            # collect feature extractor's value range for each variable
            # for once at the begining of training
            self._collect_value_ranges(variables)

        self.alpha = max(self.beta1*self.alpha, self.alpha_min)
        self.lamda = min(self.beta2*self.lamda, self.lamda_max)

        for layer_index, (variable, g, m) in enumerate(zip(variables, grads, self.moments)):
            recommendations = self._get_recommendations(layer_index, g, m)

            # Create vertices for graph building
            weight_vectors, variable_type, \
                oldshape, vertices = self._make_vertices(
                    variable, recommendations)

            partition_size = max(self.partition_size_min,
                                 math.floor(self.lamda*vertices.shape[1]))

            # Graph building -> Cycle Finding -> Permutation
            # For each weight vector, build graphs and find cycles.
            # Then performs permutation.
            for vrow_index, vrow in enumerate(vertices):
                # Optional shortcut: If recommendations are all
                # of the same signs, no cycle can be formed.
                num_positive = np.count_nonzero(vrow[:, 1] >= 0)
                if not num_positive or (num_positive == len(vrow)):
                    continue

                # Vertices partitioning
                subsets = self._random_partition(vrow, partition_size,
                                                 self.include_partition_remainder)

                # Create a permissibility subgraph for each partitioned subset
                # then find cycles in this subgraph
                for subset in subsets:
                    # Optional shortcut: Use a threshold to remove excessively small recommendations
                    # default set to 0, i.e., parameters associated with zero recommendations are
                    # ignored.
                    subset = [vertex for vertex in subset
                              if abs(vertex[1]) > 0]

                    if len(subset):
                        # Reorder vertices w.r.t. weights.
                        subset = np.array(sorted(subset, key=lambda x: x[0]))

                        # [NOTE] Define epsilon to be u_j-\ell_j
                        # Reference: Section 2.1.3 Aggressiveness
                        epsilon = self.A[layer_index][vrow_index] * \
                            (self.alpha+(1/len(subset)))

                        # Graph building
                        permissibility_graph = self._graph_builder(layer_index=layer_index,
                                                                   vrow_index=vrow_index,
                                                                   vertices=subset,
                                                                   epsilon=epsilon)
                        # Cycle finding
                        if len(permissibility_graph):
                            # Memorization
                            memorized = np.full(len(subset), False)
                            recommendations = subset[:, 1]
                            indices = subset[:, 2]

                            # Begin with the vertex associated with the largest recommendation
                            for i in reversed(np.argsort(abs(recommendations))):
                                # Optional shortcut: If the recommendations of the
                                # vertices (haven't been memorized) are all of the same signs,
                                # no more cycles can be formed.
                                recommendations_remained = recommendations[np.invert(
                                    memorized)]
                                num_positive = np.count_nonzero(
                                    recommendations_remained >= 0)
                                if not num_positive or (num_positive == len(subset)):
                                    break

                                if not memorized[i]:
                                    cycle = self._cycle_finder(
                                        i, permissibility_graph, memorized)

                                    if len(cycle):
                                        # convert cycles to actual parameter indices
                                        true_cycle = indices[cycle].astype(
                                            np.int64)

                                        # permutation
                                        weight_vectors[vrow_index][true_cycle] = \
                                            weight_vectors[vrow_index][np.roll(
                                                true_cycle, -1)]

            # Recover the shapes of variables from weight vectors
            new_variable = self._make_variable(
                weight_vectors, variable_type, oldshape)

            # Update model variables
            variable.assign(new_variable)

        self.iterations += 1

    def _get_recommendations(self, i, g, m):
        """ Produce recommendations.

        Update momentum using gradients and use the current 
        momentum as the recommendations.

        # References
        - Section 2.1.2 Recommendations.
        """

        self.moments[i] = self.momentum_lr * g + self.momentum * m
        return self.moments[i]

    def _collect_value_ranges(self, variables):
        """ Collect value range A_j for each weight vector w_j.

        Since A_j is never changed during training 
        (because only permutation is performed to w_j), this method
        is only called once before the training begins.

        # References
        - Section 2.1.3 Aggressiveness.
        """

        self.A = []
        for variable in variables:
            weight_vectors, _, _ = self._make_weight_vectors(
                variable=variable)
            value_ranges = []
            for weight_vector in weight_vectors:
                value_ranges.append(np.ptp(weight_vector))
            self.A.append(value_ranges)

    def _make_vertices(self, variable, recommendations):
        """ Create vertices based on variable and their recommendations.

        Produce weight vectors where each entry is a tuple of three values:
        [weight, recommendation, index]

        This method does the opposite of self._make_variable.
        """

        # Reshape variable to 2d matrix (details refer to the definition of self._make_weight_vectors)
        weight_vectors, variable_type, oldshape = self._make_weight_vectors(
            variable)
        # Reshape the variable's corresponding recommendations
        recommendations, _, _ = self._make_weight_vectors(recommendations)

        # Create a zero-based index vector for each weight vector
        indices = np.asarray(
            [np.arange(weight_vectors.shape[1])]*weight_vectors.shape[0])

        # Produce a tensor of shape (weight_vector.shape[0], weight_vector.shape[1] , 3)
        # which can be considered as a 2d matrix of shape "weight_vector.shape" where
        # each entry (a vertex) of this matrix is a tuple of three values:
        # weight, recommendation, and index
        wri = [weight_vectors, recommendations, indices]
        vertices = np.transpose(wri, axes=[1, 2, 0])
        return weight_vectors, variable_type, oldshape, vertices

    def _make_weight_vectors(self, variable):
        """ Convert variable into weight matrix

        Given input "variable" representing all connections 
        between two neighboring layers, return three items:
        1. a weight matrix (2d) where each row is a weight vector.
        2. type of the original variable 
        3. shape of the original variable

        The input "variable" could be a bias vector, or
        a tensor representing all neuron connections between two
        neighboring layers. The tensor could be either 2d (Fully connected layer) where
        each column is a weight vector, or 4d (Convolutional kernel) where the 
        four dimensions are: [filter_height, filter_width, in_channel, out_channel].

        # References
        - Section 1.1 Neuron Connections and Weight Matrices
        """

        # Converts EagerTensor to ndarray
        try:
            variable = variable.numpy()
        except:
            pass

        # Judge variable type and reshape based on this information
        if variable.ndim == 1:
            variable_type = 0
            oldshape = variable.shape
            weight_vectors = np.expand_dims(variable, axis=0)
        elif variable.ndim == 2:
            variable_type = 1
            oldshape = variable.shape
            weight_vectors = variable.T
        elif variable.ndim == 4:
            variable_type = 2
            # first, move the out_channel dimension to the first dimension, such that
            # variable becomes [out_channel, filter_height, filter_width, in_channel]
            variable = np.transpose(variable, axes=[3, 0, 1, 2])
            # flatten variable while keeping the first dimension unchanged
            oldshape, newshape = variable.shape, (variable.shape[0], np.prod(
                variable.shape[1:]))
            weight_vectors = np.reshape(variable, newshape=newshape)
        else:
            # [NOTE] Current implementation only support the above 3 cases.
            raise ValueError(
                "Unknown variable shape: {}".format(variable.shape))

        return weight_vectors, variable_type, oldshape

    def _make_variable(self, weight_vectors, variable_type, oldshape):
        """  Convert weight_matrix into variable.

        This method does the opposite of self._make_weight_vectors.
        """

        if variable_type == 0:
            variable = weight_vectors[0]
        elif variable_type == 1:
            variable = weight_vectors.T
        elif variable_type == 2:
            variable = np.reshape(weight_vectors, newshape=oldshape)
            variable = np.transpose(variable, axes=[1, 2, 3, 0])
        else:
            ValueError("Unknown variable type: {}".format(variable_type))
        return variable

    def _graph_builder(self, layer_index, vrow_index, vertices, epsilon):
        """ Build permissibility graph given vertices.

        # References
        - Section 2.1 Permissibility Graph
        - Section 2.2 Finding the Permutations
        """

        # Optional shortcut: at least 2 vertices must present to build a graph
        if len(vertices) < 2:
            return []

        # Optional shortcut: since the largest vertex can't increase and the
        # smallest vertex can't decrease, exclude them from graph (using offset)
        # if necessary
        left_offset = 0 if vertices[0][1] < 0 else 1
        right_offset = 0 if vertices[-1][1] > 0 else -1

        permissibility_graph = []
        for i, vertex in enumerate(vertices):
            weight, recommendations, _ = vertex

            if recommendations < 0:  # wants to increase
                lower_bound = weight
                upper_bound = weight + epsilon
                # Temporary permissible movement candidates.
                permissible_movements = range(i+1, len(vertices)+right_offset)
            else:  # wants to decrease
                lower_bound = weight - epsilon
                upper_bound = weight
                # Temporary permissible movement candidates.
                permissible_movements = range(0+left_offset, i)

            if not len(permissible_movements):
                permissibility_graph.append([])
                continue

            # Weight candidates for constructing the permissible movements
            permissible_weights = vertices[permissible_movements][:, 0]
            # Binary search to find weights that are within the
            # lower bound(\ell) and upper bounds (u).
            # These weights are the permissible movements we are looking for
            begin_candidate, end_candidate = np.searchsorted(
                permissible_weights, [lower_bound, upper_bound])
            permissible_movements = list(
                permissible_movements[begin_candidate:end_candidate])

            # Apply the "closer priority" to the permissble movements
            # Reference Section 2.2.1 Priority of Movements.
            if recommendations > 0:
                permissible_movements = list(reversed(permissible_movements))

            permissibility_graph.append(permissible_movements)

        return permissibility_graph

    def _cycle_finder(self, src, permissibility_graph, taken):
        """ Given permissibility graph, find cycles.

        DFS based algorithm for cycle finding.
        [NOTE] This method doesn't have to be a class method.

        # References
        - Section 2.2 Finding the Permutations.
        """

        taken_copy = taken.copy()
        path = []
        stk = collections.deque([(None, src)])

        while stk:
            frm, dest = stk.popleft()
            if dest == src and frm is not None:
                taken[path] = True
                return path

            path.append(dest)
            taken_copy[dest] = True
            next_dests = [(dest, next_dest) for next_dest in permissibility_graph[dest]
                          if (not taken_copy[next_dest] or next_dest == src)]

            if len(next_dests) != 0:
                stk.extendleft(reversed(next_dests))
            else:
                while stk and path and path[-1] != stk[0][0]:
                    path.pop()

        # if no cycle is found, only the source is memorized
        taken[src] = True
        return []

    def _random_partition(self, vertices, partition_size, include_remainder):
        """Partition 1d vector into disjoint random subsets of 
        sizes approximately equal to partition_size.

        Despite the shape of the input, it is treated as a 1d vector, 
        and thus will only be shuffled along its first axis.

        If include_remainder is set to True and partition_size is less than 
        and doesn't divide len(vertices), the last the subtset will be of 
        the size len(data)%partition_size.

        If partition_size is larger than len(vertices), [vertices] 
        is returned.

        [NOTE] This method doesn't have to be a class method.

        # Reference: 
        # - Section 2.1.4 Graph Partitioning
        """

        assert partition_size > 0, "partition_size must be larger than 0, but received {}"\
            .format(partition_size)

        num_vertices = len(vertices)
        if partition_size >= num_vertices:
            return [vertices]

        np.random.shuffle(vertices)
        num_partition = -(-num_vertices//partition_size)
        partitions = []
        begin = 0
        for _ in range(num_partition):
            end = begin+partition_size
            if end >= len(vertices) and not include_remainder:
                break
            partitions.append(vertices[begin:end])
            begin = end

        return partitions


class SGPermTrainLoop:
    """ Train loop which runs SGPerm on a deep neural network model.

    # Arguments
            model: A model instance which supports access/modification
                    of its trainable variables. Here assumes 
                    tensorflow.python.keras.engine.sequential.
            loss: Str of name of a tensorflow.losses or 
                    a tensorflow.losses instance. The loss function used to compare 
                    the model output and result.
            alpha0: float>=0 && <= 1. Initial aggressiveness.
            lamda0: float>=0 && <= 1. Initial partition ratio.
            beta1: float>=0. Exponential decay rate for aggressiveness.
            beta2: float>=0. Exponential growth rate for partition ratio.
            alpha_min: float>=0 && <= 1. Minimum aggressiveness.
            lamda_max: float>=0 && <= 1. Maximum partition ratio.
            momentum_lr: float >= 0. Learning rate for momentum.
            momentum: float >= 0. Parameter that accelerates gradients in the 
                    relevant direction and dampens oscillations.
    """

    def __init__(self,
                 model,
                 loss,
                 alpha0,
                 lamda0,
                 beta1,
                 beta2,
                 alpha_min,
                 lamda_max,
                 momentum_lr=0.01,
                 momentum=0.9):
        # model must be tensorflor.keras layers or model
        # with trainable variables
        self.model = model
        self.iterations = 0

        if isinstance(loss, str):
            try:
                self._loss = getattr(tf.losses, loss)
            except:
                raise Exception("""undefined loss function: {}""".format(
                    self._loss))

        self.SGPerm = SGPerm(alpha0=alpha0,
                             lamda0=lamda0,
                             beta1=beta1,
                             beta2=beta2,
                             alpha_min=alpha_min,
                             lamda_max=lamda_max,
                             momentum_lr=momentum_lr,
                             momentum=momentum)

        self.SGPerm.set_model(self.model, self)
        self.SGPerm.create_slots()

    def grad(self, x, y):
        with tf.GradientTape() as tape:
            tape.watch(self.model.trainable_variables)
            y_pred = self.model(x)
            loss_value = self._loss(y_true=y, y_pred=y_pred)
        grads = tape.gradient(loss_value, self.model.trainable_variables)

        return loss_value, grads

    def evaluate(self, x, y, batch=32):
        if not batch:
            batch = len(x)

        accuracy_fn = tf.metrics.Accuracy()
        total = len(x)
        loc = 0
        last_loc = total-batch
        while loc < total:
            if loc > last_loc:
                batch = total-loc

            x_ = x[loc:loc+batch]
            y_ = y[loc:loc+batch]
            y_pred = self.model(x_)

            loss = self._loss(y_, y_pred)

            pred = tf.argmax(y_pred, axis=1)
            accuracy_fn.update_state(y_true=y_, y_pred=pred)

            loc += batch

        acc = accuracy_fn.result().numpy()

        return tf.reduce_mean(loss).numpy(), acc

    def fit(self,
            x=None,
            y=None,
            batch_size=None,
            epochs=1,
            validation_data=None,
            shuffle=True,
            continuous=None):
        """ Trains the model for a given number of epochs.

        x: Input data. 
        y: Target data. 
        batch_size: Integer or None. Number of samples per gradient update. 
        epochs: Integer. Number of epochs to train the model. 
        validation_data: Data on which to evaluate the loss and accuracy at 
                the end of each batch. 
        shuffle: boolean. Whether to shuffle the data.
        continuous: None or a tuple of following two values:
                1. switch_iteration: the number of iteration to switch to another optimizer and
                                                continue training.
                2. alternative_optimizer: another optimizer which to switch to after self.iterations 
                                                exceeded switch_iteration.
                If is None, only SGPerm will be used throughout training.
        """

        self.iterations = 0

        val_x, val_y = validation_data

        if continuous is not None:
            self.switch_iteration, self.alternative_optimizer = continuous
        else:
            self.switch_iteration = math.inf

        # Optional: collect validation data before training begins.
        loss, acc = self.evaluate(val_x, val_y)
        self._history = {'val accuracy': [acc], 'val loss': [loss]}

        epoch_progress_bar = nested_progress_bar(epochs, desc='Epoch')
        for epoch in epoch_progress_bar:
            if shuffle:
                x, y = train_shuffler(x, y)

            batch_num = -(-len(x)//batch_size)
            batch_begin = 0

            batch_progress_bar = nested_progress_bar(batch_num, desc='Batch')
            for batch in batch_progress_bar:

                batch_end = batch_begin+batch_size
                x_, y_ = x[batch_begin:batch_end], y[batch_begin:batch_end]
                batch_begin = batch_end

                variables = self.model.trainable_variables
                losses, grads = self.grad(x_, y_)

                if self.iterations < self.switch_iteration:
                    # Optional: Update inner states of the alternative optimizer (sub-training)
                    if continuous is not None:
                        self.alternative_optimizer.apply_gradients(
                            zip(grads, []))
                    # Convert grads to numpy arrays
                    grads_numpy = np.array([item.numpy() for item in grads])
                    self.SGPerm.permute(grads, variables)
                else:
                    self.alternative_optimizer.apply_gradients(
                        zip(grads, variables))

                # Optional: Record validation data at the end of each batch.
                loss, acc = self.evaluate(val_x, val_y)
                self._history['val loss'].append(loss)
                self._history['val accuracy'].append(acc)

                batch_progress_bar.set_description(
                    "Loss:{:.3f} Acc:{:.3f}".format(loss, acc))
                batch_progress_bar.refresh()

                self.iterations += 1

            # # Optional: Record validation data at the end of each epoch.
            # loss, acc = self.evaluate(val_x, val_y)
            # epoch_progress_bar.set_description("Loss:{:.3f} Acc:{:.3f}".format(loss, acc))

        batch_progress_bar.close()
        epoch_progress_bar.close()

    def history(self):
        return self._history.copy()
