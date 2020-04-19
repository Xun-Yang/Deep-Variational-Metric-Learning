from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from tensorflow.python.framework import dtypes
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import logging_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import nn
import lib.nn_Ops as nn_Ops
from FLAGS import *
from lib.nn_Ops import distance, weight_variable, bias_variable

def cross_entropy(embedding, label, size=1024):
    with tf.variable_scope("Softmax_classifier"):
        W_fc = weight_variable([size, FLAGS.num_class], "softmax_w", wd=False)
        b_fc = bias_variable([FLAGS.num_class], "softmax_b")
    Logits = tf.matmul(embedding, W_fc) + b_fc
    cross_entropy = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=label, logits=Logits))
    return cross_entropy, W_fc, b_fc


def pairwise_distance(feature, squared=False):
    """Computes the pairwise distance matrix with numerical stability.
    output[i, j] = || feature[i, :] - feature[j, :] ||_2
    Args:
        feature: 2-D Tensor of size [number of data, feature dimension].
        squared: Boolean, whether or not to square the pairwise distances.
    Returns:
        pairwise_distances: 2-D Tensor of size [number of data, number of data].
    """
    pairwise_distances_squared = math_ops.add(
        math_ops.reduce_sum(
            math_ops.square(feature),
            axis=[1],
            keepdims=True),
        math_ops.reduce_sum(
            math_ops.square(
                array_ops.transpose(feature)),
            axis=[0],
            keepdims=True)) - 2.0 * math_ops.matmul(
        feature, array_ops.transpose(feature))

    # Deal with numerical inaccuracies. Set small negatives to zero.
    pairwise_distances_squared = math_ops.maximum(pairwise_distances_squared, 0.0)
    # Get the mask where the zero distances are at.
    error_mask = math_ops.less_equal(pairwise_distances_squared, 0.0)

    # Optionally take the sqrt.
    if squared:
        pairwise_distances = pairwise_distances_squared
    else:
        pairwise_distances = math_ops.sqrt(
            pairwise_distances_squared + math_ops.to_float(error_mask) * 1e-16)

    # Undo conditionally adding 1e-16.
    pairwise_distances = math_ops.multiply(
        pairwise_distances, math_ops.to_float(math_ops.logical_not(error_mask)))

    num_data = array_ops.shape(feature)[0]
    # Explicitly set diagonals to zero.
    mask_offdiagonals = array_ops.ones_like(pairwise_distances) - array_ops.diag(
        array_ops.ones([num_data]))
    pairwise_distances = math_ops.multiply(pairwise_distances, mask_offdiagonals)
    return pairwise_distances


def contrastive_loss(labels, embeddings_anchor, embeddings_positive,
                     margin=1.0):
    """Computes the contrastive loss.
    This loss encourages the embedding to be close to each other for
        the samples of the same label and the embedding to be far apart at least
        by the margin constant for the samples of different labels.
    See: http://yann.lecun.com/exdb/publis/pdf/hadsell-chopra-lecun-06.pdf
    Args:
        labels: 1-D tf.int32 `Tensor` with shape [batch_size] of
            binary labels indicating positive vs negative pair.
        embeddings_anchor: 2-D float `Tensor` of embedding vectors for the anchor
            images. Embeddings should be l2 normalized.
        embeddings_positive: 2-D float `Tensor` of embedding vectors for the
            positive images. Embeddings should be l2 normalized.
        margin: margin term in the loss definition.
    Returns:
        contrastive_loss: tf.float32 scalar.
    """
    # Get per pair distances
    distances = math_ops.sqrt(
        math_ops.reduce_sum(
            math_ops.square(embeddings_anchor - embeddings_positive), 1))

    # Add contrastive loss for the siamese network.
    #   label here is {0,1} for neg, pos.
    return math_ops.reduce_mean(
        math_ops.to_float(labels) * math_ops.square(distances) +
        (1. - math_ops.to_float(labels)) *
        math_ops.square(math_ops.maximum(margin - distances, 0.)),
        name='contrastive_loss')



def triplet_loss(anchor, positive, negative, margin=1.0):
    d_ap = tf.reduce_sum(tf.square(anchor - positive), axis=1)
    d_an = tf.reduce_sum(tf.square(anchor - negative), axis=1)
    loss = tf.maximum(d_ap - d_an + margin, 0.)
    return tf.reduce_sum(loss)

def masked_maximum(data, mask, dim=1):
    """Computes the axis wise maximum over chosen elements.
      Args:
        data: 2-D float `Tensor` of size [n, m].
        mask: 2-D Boolean `Tensor` of size [n, m].
        dim: The dimension over which to compute the maximum.
      Returns:
        masked_maximums: N-D `Tensor`.
          The maximized dimension is of size 1 after the operation.
    """
    axis_minimums = math_ops.reduce_min(data, dim, keepdims=True)
    masked_maximums = math_ops.reduce_max(
        math_ops.multiply(
            data - axis_minimums, mask), dim, keepdims=True) + axis_minimums
    return masked_maximums


def masked_minimum(data, mask, dim=1):
    """Computes the axis wise minimum over chosen elements.
  Args:
    data: 2-D float `Tensor` of size [n, m].
    mask: 2-D Boolean `Tensor` of size [n, m].
    dim: The dimension over which to compute the minimum.
  Returns:
    masked_minimums: N-D `Tensor`.
      The minimized dimension is of size 1 after the operation.
  """
    axis_maximums = math_ops.reduce_max(data, dim, keepdims=True)
    masked_minimums = math_ops.reduce_min(
        math_ops.multiply(
            data - axis_maximums, mask), dim, keepdims=True) + axis_maximums
    return masked_minimums


def triplet_semihard_loss(labels, embeddings, margin=1.0):
    """Computes the triplet loss with semi-hard negative mining.
      The loss encourages the positive distances (between a pair of embeddings with
      the same labels) to be smaller than the minimum negative distance among
      which are at least greater than the positive distance plus the margin constant
      (called semi-hard negative) in the mini-batch. If no such negative exists,
      uses the largest negative distance instead.
      See: https://arxiv.org/abs/1503.03832.
      Args:
        labels: 1-D tf.int32 `Tensor` with shape [batch_size] of
          multiclass integer labels.
        embeddings: 2-D float `Tensor` of embedding vectors. Embeddings should
          be l2 normalized.
        margin: Float, margin term in the loss definition.
      Returns:
        triplet_loss: tf.float32 scalar.
    """
    # Reshape [batch_size] label tensor to a [batch_size, 1] label tensor.
    lshape = array_ops.shape(labels)
    assert lshape.shape == 1
    labels = array_ops.reshape(labels, [lshape[0], 1])

    # Build pairwise squared distance matrix.
    pdist_matrix = pairwise_distance(embeddings, squared=True)
    # Build pairwise binary adjacency matrix.
    adjacency = math_ops.equal(labels, array_ops.transpose(labels))
    # Invert so we can select negatives only.
    adjacency_not = math_ops.logical_not(adjacency)

    batch_size = array_ops.size(labels)

    # Compute the mask.
    pdist_matrix_tile = array_ops.tile(pdist_matrix, [batch_size, 1])
    mask = math_ops.logical_and(
        array_ops.tile(adjacency_not, [batch_size, 1]),
        math_ops.greater(
            pdist_matrix_tile, array_ops.reshape(
                array_ops.transpose(pdist_matrix), [-1, 1])))
    mask_final = array_ops.reshape(
        math_ops.greater(
            math_ops.reduce_sum(
                math_ops.cast(
                    mask, dtype=dtypes.float32), 1, keepdims=True),
            0.0), [batch_size, batch_size])
    mask_final = array_ops.transpose(mask_final)

    adjacency_not = math_ops.cast(adjacency_not, dtype=dtypes.float32)
    mask = math_ops.cast(mask, dtype=dtypes.float32)

    # negatives_outside: smallest D_an where D_an > D_ap.
    negatives_outside = array_ops.reshape(
        masked_minimum(pdist_matrix_tile, mask), [batch_size, batch_size])
    negatives_outside = array_ops.transpose(negatives_outside)

    # negatives_inside: largest D_an.
    negatives_inside = array_ops.tile(
        masked_maximum(pdist_matrix, adjacency_not), [1, batch_size])
    semi_hard_negatives = array_ops.where(
        mask_final, negatives_outside, negatives_inside)

    loss_mat = math_ops.add(margin, pdist_matrix - semi_hard_negatives)

    mask_positives = math_ops.cast(
        adjacency, dtype=dtypes.float32) - array_ops.diag(
        array_ops.ones([batch_size]))

    # In lifted-struct, the authors multiply 0.5 for upper triangular
    #   in semihard, they take all positive pairs except the diagonal.
    num_positives = math_ops.reduce_sum(mask_positives)

    _triplet_loss = math_ops.truediv(
        math_ops.reduce_sum(
            math_ops.maximum(
                math_ops.multiply(loss_mat, mask_positives), 0.0)),
        num_positives,
        name='triplet_semihard_loss')
    return _triplet_loss




def npairs_loss(labels, embeddings_anchor, embeddings_positive,
                reg_lambda=3e-3, print_losses=False):
    """Computes the npairs loss.
          Npairs loss expects paired data where a pair is composed of samples from the
          same labels and each pairs in the minibatch have different labels. The loss
          has two components. The first component is the L2 regularizer on the
          embedding vectors. The second component is the sum of cross entropy loss
          which takes each row of the pair-wise similarity matrix as logits and
          the remapped one-hot labels as labels.
          See:
          http://www.nec-labs.com/uploads/images/Department-Images/MediaAnalytics/papers/nips16_npairmetriclearning.pdf
          Args:
            labels: 1-D tf.int32 `Tensor` of shape [batch_size/2].
            embeddings_anchor: 2-D Tensor of shape [batch_size/2, embedding_dim] for the
              embedding vectors for the anchor images. Embeddings should not be
              l2 normalized.
            embeddings_positive: 2-D Tensor of shape [batch_size/2, embedding_dim] for the
              embedding vectors for the positive images. Embeddings should not be
              l2 normalized.
            reg_lambda: Float. L2 regularization term on the embedding vectors.
            print_losses: Boolean. Option to print the xent and l2loss.
          Returns:
            npairs_loss: tf.float32 scalar.
      """
    # pylint: enable=line-too-long
    # Add the regularizer on the embedding.
    reg_anchor = math_ops.reduce_mean(
        math_ops.reduce_sum(math_ops.square(embeddings_anchor), 1))
    reg_positive = math_ops.reduce_mean(
        math_ops.reduce_sum(math_ops.square(embeddings_positive), 1))
    l2loss = math_ops.multiply(
        0.25 * reg_lambda, reg_anchor + reg_positive, name='l2loss')

    # Get per pair similarities.
    similarity_matrix = math_ops.matmul(
        embeddings_anchor, embeddings_positive, transpose_a=False,
        transpose_b=True)

    # Reshape [batch_size] label tensor to a [batch_size, 1] label tensor.
    lshape = array_ops.shape(labels)
    assert lshape.shape == 1
    labels = array_ops.reshape(labels, [lshape[0], 1])

    labels_remapped = math_ops.to_float(
        math_ops.equal(labels, array_ops.transpose(labels)))
    labels_remapped /= math_ops.reduce_sum(labels_remapped, 1, keepdims=True)

    # Add the softmax loss.
    xent_loss = nn.softmax_cross_entropy_with_logits(
        logits=similarity_matrix, labels=labels_remapped)
    xent_loss = math_ops.reduce_mean(xent_loss, name='xentropy')

    if print_losses:
        xent_loss = logging_ops.Print(
            xent_loss, ['cross entropy:', xent_loss, 'l2loss:', l2loss])

    return l2loss + xent_loss




