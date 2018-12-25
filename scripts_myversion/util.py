import tensorflow as tf
import numpy as np
import copy
import collections
import future_builtins
from mpi4py import MPI

# ================================================================
# simple_math
# ================================================================

def switch(condition, then_expression, else_expression):
    """

    :param condition: scalar tensor
    :param then_expression: tensorflow operations
    :param else_expression: tensorflow operations
    :return: switches between two operations depending on a scalar value
    """

    x_shape = copy.copy(then_expression.get_shape())
    x = tf.cond(tf.cast(condition, 'bool'), lambda: then_expression, lambda: else_expression)
    x.set_shape(x_shape)
    return x

# ================================================================

# ================================================================
# placeholder
# ================================================================

_PLACEHOLDER_CACHE = {} # name -> (placeholder, dtype, shape)

def get_placeholder(name, dtype, shape):
    if name in _PLACEHOLDER_CACHE:
        out, dtype1, shape1 = _PLACEHOLDER_CACHE[name]
        assert dtype1 == dtype and shape1 == shape
        return out
    else:
        out = tf.placeholder(dtype=dtype, shape=shape, name=name)
        _PLACEHOLDER_CACHE[name] = (out, dtype, shape)
        return out

def get_placeholder_cached(name):
    return _PLACEHOLDER_CACHE[name][0]

def reset():
    global _PLACEHOLDER_CACHE
    _PLACEHOLDER_CACHE = {}
    tf.reset_default_graph()

# ================================================================

# ================================================================
# model operations
# ================================================================

def normc_initializer(std=1.0):
    def _initializer(shape, dtype=None, partition_info=None):
        out = np.random.randn(*shape).astype(np.float32)
        out *= std / np.sqrt(np.square(out).sum(axis=0, keepdims=True))
        return tf.constant(out)
    return _initializer

def conv2d(x, num_filters, name, filter_size=(3, 3), stride=(1, 1), pad="SAME", dtype=tf.float32, collections=None,
           summary_tag=None):
    with tf.variable_scope(name):
        stride_shape = [1, stride[0], stride[1], 1]
        filter_shape = [filter_size[0], filter_size[1], int(x.get_shape()[3], num_filters)]

        fan_in = intprod(filter_shape[:3])
        fan_out = intprod(filter_shape[:2]) * num_filters

        w_bound = np.sqrt(6. / (fan_in + fan_out))

        w = tf.get_variable("W", filter_shape, dtype, tf.random_uniform_initializer(-w_bound, w_bound),
                            collections=collections)
        b = tf.get_variable("b", [1, 1, 1, num_filters], dtype, tf.zeros_initializer(), collections=collections)

        if summary_tag is not None:
            tf.summary.image(summary_tag, tf.transpose(tf.reshape(w, [filter_size[0], filter_size[1], -1, 1]), [2, 0, 1, 3]),
                             max_outputs=10)

        return tf.nn.conv2d(x, w, stride_shape, pad) + b

def dense(x, size, name, weight_init=None, bias=True):
    w = tf.get_variable(name + "/w", [x.get_shape()[1], size], initializer=weight_init)
    ret = tf.matmul(x, w)
    if bias:
        b = tf.get_variable(name + "/b", [size], initializer=tf.zeros_initializer())
        return ret + b
    else:
        return ret

def var_shape(x):
    out = x.get_shape().as_list()
    assert all(isinstance(a, int) for a in out), \
    "shape function assumes that shape is fully known"
    return out

def numel(x):
    return intprod(var_shape(x))

def intprod(x):
    return int(np.prod(x))

def flatgrad(loss, var_list, clip_norm=None):
    grads = tf.gradients(loss, var_list)
    if clip_norm is not None:
        grads = [tf.clip_by_norm(grad, clip_norm=clip_norm) for grad in grads]
    return tf.concat(axis=0, values=[
        tf.reshape(grad if grad is not None else tf.zero_like(v), [numel(v)])
        for (v, grad) in zip(var_list, grads)
    ])

class SetFromFlat(object):
    def __init__(self, var_list, dtype=tf.float32):
        assigns = []
        shapes = list(map(var_shape, var_list))
        total_size = np.sum([intprod(shape) for shape in shapes])

        self.theta = theta = tf.placeholder(dtype, [total_size])
        start = 0
        assigns = []
        for (shape, v) in zip(shapes, var_list):
            size = intprod(shape)
            assigns.append(tf.assign(v, tf.reshape(theta[start:start + size], shape)))
            start += size
        self.op = tf.group(*assigns)

    def __call__(self, theta):
        get_session().run(self.op, feed_dict={self.theta: theta})


class GetFlat(object):
    def __init__(self, var_list):
        self.op = tf.concat(axis=0, values=[tf.reshape(v, [numel(v)]) for v in var_list])

    def __call__(self):
        return get_session().run(self.op)

def zipsame(*seqs):
    L = len(seqs[0])
    assert all(len(seq) == L for seq in seqs[1:])

    return zip(*seqs)
# ================================================================


# ================================================================
# running mean and std
# ================================================================

class RunningMeanStd(object):
    def __init__(self, epsilon=1e-2, shape=()):
        self._sum = tf.get_variable(dtype=tf.float64, shape=shape,
                                    initializer=tf.constant_initializer(0.0),
                                    name="runningsum", trainable=False)
        self._sumsq = tf.get_variable(dtype=tf.float64, shape=shape,
                                      initializer=tf.constant_initializer(epsilon),
                                      name="runningsumsq", trainable=False)
        self._count = tf.get_variable(dtype=tf.float64, shape=(),
                                      initializer=tf.constant_initializer(epsilon),
                                      name="count", trainable=False)
        self.shape = shape

        self.mean = tf.to_float(self._sum / self._count)
        self.std = tf.sqrt(tf.maximum(tf.to_float(self._sumsq / self._count) - tf.square(self.mean), 1e-2))

        newsum = tf.placeholder(dtype=tf.float64, shape=shape, name='sum')
        newsumsq = tf.placeholder(dtype=tf.float64, shape=shape, name='var')
        newcount = tf.placeholder(dtype=tf.float64, shape=shape, name='count')
        self.incfiltparams = function([newsum, newsumsq, newcount], [],
                                      updates=[tf.assign_add(self._sum, newsum),
                                               tf.assign_add(self._sumsq, newsumsq),
                                               tf.assign_add(self.count, newcount)])

    def update(self, x):
        x = x.astype('float64')
        n = int(np.prod(self.shape))
        totalvec = np.zeros(n*2+1, 'float64')
        addvec = np.concatenate([x.sum(axis=0).ravel(), np.square(x).sum(axis=0).ravel(), np.array([len(x)], dtype='float64')])
        MPI.COMM_WORLD.Allreduce(addvec, totalvec, op=MPI.SUM)
        self.incfiltparams(totalvec[0:n].reshape(self.shape), totalvec[n:2*n].reshape(self.shape), totalvec[2*n])


# ================================================================


# ================================================================
# Theano-like Function
# ================================================================



def function(inputs, outputs, updates=None, givens=None):
    """Just like Theano function. Take a bunch of tensorflow placeholders and expressions
    computed based on those placeholders and produces f(inputs) -> outputs. Function f takes
    values to be fed to the input's placeholders and produces the values of the expressions
    in outputs.
    Input values can be passed in the same order as inputs or can be provided as kwargs based
    on placeholder name (passed to constructor or accessible via placeholder.op.name).
    Example:
        x = tf.placeholder(tf.int32, (), name="x")
        y = tf.placeholder(tf.int32, (), name="y")
        z = 3 * x + 2 * y
        lin = function([x, y], z, givens={y: 0})
        with single_threaded_session():
            initialize()
            assert lin(2) == 6
            assert lin(x=3) == 9
            assert lin(2, 2) == 10
            assert lin(x=2, y=3) == 12
    Parameters
    ----------
    inputs: [tf.placeholder or TfInput]
        list of input arguments
    outputs: [tf.Variable] or tf.Variable
        list of outputs or a single output to be returned from function. Returned
        value will also have the same shape.
    """
    if isinstance(outputs, list):
        return _Function(inputs, outputs, updates, givens=givens)
    elif isinstance(outputs, (dict, collections.OrderedDict)):
        f = _Function(inputs, outputs.values(), updates, givens=givens)
        return lambda *args, **kwargs: type(outputs)(zip(outputs.keys(), f(*args, **kwargs)))
    else:
        f = _Function(inputs, [outputs], updates, givens=givens)
        return lambda *args, **kwargs: f(*args, **kwargs)[0]


class _Function(object):
    def __init__(self, inputs, outputs, updates, givens, check_nan=False):
        for inpt in inputs:
            if not issubclass(type(inpt), TfInput):
                assert len(inpt.op.inputs) == 0, "inputs should all be placeholders of baselines.common.TfInput"
        self.inputs = inputs
        updates = updates or []
        self.update_group = tf.group(*updates)
        self.outputs_update = list(outputs) + [self.update_group]
        self.givens = {} if givens is None else givens
        self.check_nan = check_nan

    def _feed_input(self, feed_dict, inpt, value):
        if issubclass(type(inpt), TfInput):
            feed_dict.update(inpt.make_feed_dict(value))
        elif is_placeholder(inpt):
            feed_dict[inpt] = value

    def __call__(self, *args, **kwargs):
        assert len(args) <= len(self.inputs), "Too many arguments provided"
        feed_dict = {}
        # Update the args
        for inpt, value in zip(self.inputs, args):
            self._feed_input(feed_dict, inpt, value)
        # Update the kwargs
        kwargs_passed_inpt_names = set()
        for inpt in self.inputs[len(args):]:
            inpt_name = inpt.name.split(':')[0]
            inpt_name = inpt_name.split('/')[-1]
            assert inpt_name not in kwargs_passed_inpt_names, \
                "this function has two arguments with the same name \"{}\", so kwargs cannot be used.".format(inpt_name)
            if inpt_name in kwargs:
                kwargs_passed_inpt_names.add(inpt_name)
                self._feed_input(feed_dict, inpt, kwargs.pop(inpt_name))
            else:
                assert inpt in self.givens, "Missing argument " + inpt_name
        assert len(kwargs) == 0, "Function got extra arguments " + str(list(kwargs.keys()))
        # Update feed dict with givens.
        for inpt in self.givens:
            feed_dict[inpt] = feed_dict.get(inpt, self.givens[inpt])
        results = get_session().run(self.outputs_update, feed_dict=feed_dict)[:-1]
        if self.check_nan:
            if any(np.isnan(r).any() for r in results):
                raise RuntimeError("Nan detected")
        return results


def mem_friendly_function(nondata_inputs, data_inputs, outputs, batch_size):
    if isinstance(outputs, list):
        return _MemFriendlyFunction(nondata_inputs, data_inputs, outputs, batch_size)
    else:
        f = _MemFriendlyFunction(nondata_inputs, data_inputs, [outputs], batch_size)
        return lambda *inputs: f(*inputs)[0]


class _MemFriendlyFunction(object):
    def __init__(self, nondata_inputs, data_inputs, outputs, batch_size):
        self.nondata_inputs = nondata_inputs
        self.data_inputs = data_inputs
        self.outputs = list(outputs)
        self.batch_size = batch_size

    def __call__(self, *inputvals):
        assert len(inputvals) == len(self.nondata_inputs) + len(self.data_inputs)
        nondata_vals = inputvals[0:len(self.nondata_inputs)]
        data_vals = inputvals[len(self.nondata_inputs):]
        feed_dict = dict(zip(self.nondata_inputs, nondata_vals))
        n = data_vals[0].shape[0]
        for v in data_vals[1:]:
            assert v.shape[0] == n
        for i_start in range(0, n, self.batch_size):
            slice_vals = [v[i_start:future_builtins.min(i_start + self.batch_size, n)] for v in data_vals]
            for (var, val) in zip(self.data_inputs, slice_vals):
                feed_dict[var] = val
            results = tf.get_default_session().run(self.outputs, feed_dict=feed_dict)
            if i_start == 0:
                sum_results = results
            else:
                for i in range(len(results)):
                    sum_results[i] = sum_results[i] + results[i]
        for i in range(len(results)):
            sum_results[i] = sum_results[i] / n
        return sum_results


# ================================================================

# ================================================================
# Inputs
# ================================================================


def is_placeholder(x):
    return type(x) is tf.Tensor and len(x.op.inputs) == 0


class TfInput(object):
    def __init__(self, name="(unnamed)"):
        """Generalized Tensorflow placeholder. The main differences are:
            - possibly uses multiple placeholders internally and returns multiple values
            - can apply light postprocessing to the value feed to placeholder.
        """
        self.name = name

    def get(self):
        """Return the tf variable(s) representing the possibly postprocessed value
        of placeholder(s).
        """
        raise NotImplemented()

    def make_feed_dict(data):
        """Given data input it to the placeholder(s)."""
        raise NotImplemented()

# ================================================================

# ================================================================
# Global session
# ================================================================

def get_session():
    """Returns recently made Tensorflow session"""
    return tf.get_default_session()

def make_session(num_cpu):
    tf_config = tf.ConfigProto(
        inter_op_parallelism_threads=num_cpu,
        intra_op_parallelism_threads=num_cpu
    )
    return tf.Session(config=tf_config)

ALREADY_INITIALIZED = set()

def initialize():
    new_variables = set(tf.global_variables()) - ALREADY_INITIALIZED
    get_session().run(tf.variables_initializer(new_variables))
    ALREADY_INITIALIZED.update(new_variables)

