import numpy as np

import chainer
import chainerrl
import chainer.links as L
import chainer.functions as F


class QFunction(chainer.Chain):

    def __init__(self, n_target, n_finger_loc, n_sat_desired, n_sat_true, n_action_type, embed_size, n_actions,
                 n_hidden_channels=50, dropout_ratio=0.2):
        super().__init__()
        self.dropout = dropout_ratio
        with self.init_scope():
            self.target = L.EmbedID(n_target, embed_size)
            self.finger = L.EmbedID(n_finger_loc, embed_size)
            self.sat_d = L.EmbedID(n_sat_desired, int(embed_size/10))
            self.sat_t = L.EmbedID(n_sat_true, int(embed_size/10))
            self.action_type = L.EmbedID(n_action_type, int(embed_size/10))

            self.l0 = L.Linear((embed_size * 2 + int(embed_size/10) * 3 + 1), n_hidden_channels)
            self.l1 = L.Linear(n_hidden_channels, n_hidden_channels)
            self.l2 = L.Linear(n_hidden_channels, n_actions)

    def __call__(self, x, test=False):
        """
        Args:
            x (ndarray or chainer.Variable): An observation
            test (bool): a flag indicating whether it is in test mode
        """
        target = x[:, 0]
        finger_loc = x[:, 1]
        sat_desired = x[:, 2]
        sat_true = x[:, 3]
        action_type = x[:, 4]
        entropy = x[:, 5].reshape((-1, 1))

        embed_target = self.target(target.astype(np.int32))
        embed_finger = self.finger(finger_loc.astype(np.int32))
        embed_sat_d = self.sat_d(sat_desired.astype(np.int32))
        embed_sat_t = self.sat_t(sat_true.astype(np.int32))
        embed_action_type = self.action_type(action_type.astype(np.int32))

        h = F.concat((embed_target, embed_finger, embed_sat_d, embed_sat_t, embed_action_type, entropy))

        h = F.tanh(self.l0(h))
        h = F.dropout(h, self.dropout)
        h = F.tanh(self.l1(h))
        h = F.dropout(h, self.dropout)
        return chainerrl.action_value.DiscreteActionValue(self.l2(h))
