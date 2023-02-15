import torch

from torch_geometric.utils import negative_sampling
from typing import Any
from sklearn.metrics import average_precision_score, roc_auc_score

MAX_LOGSTD = 10


def reset(value: Any):
    if hasattr(value, 'reset_parameters'):
        value.reset_parameters()
    else:
        for child in value.children() if hasattr(value, 'children') else []:
            reset(child)


class InnerProductDecoderJittable(torch.nn.Module):
    r"""The inner product decoder from the `"Variational Graph Auto-Encoders"
    <https://arxiv.org/abs/1611.07308>`_ paper

    .. math::
        \sigma(\mathbf{Z}\mathbf{Z}^{\top})

    where :math:`\mathbf{Z} \in \mathbb{R}^{N \times d}` denotes the latent
    space produced by the encoder."""
    def forward(self, z, edge_index, sigmoid : bool=True):
        r"""Decodes the latent variables :obj:`z` into edge probabilities for
        the given node-pairs :obj:`edge_index`.

        Args:
            z (Tensor): The latent space :math:`\mathbf{Z}`.
            sigmoid (bool, optional): If set to :obj:`False`, does not apply
                the logistic sigmoid function to the output.
                (default: :obj:`True`)
        """
        value = (z[edge_index[0]] * z[edge_index[1]]).sum(dim=1)
        return torch.sigmoid(value) if sigmoid else value

    def forward_all(self, z, sigmoid=True):
        r"""Decodes the latent variables :obj:`z` into a probabilistic dense
        adjacency matrix.

        Args:
            z (Tensor): The latent space :math:`\mathbf{Z}`.
            sigmoid (bool, optional): If set to :obj:`False`, does not apply
                the logistic sigmoid function to the output.
                (default: :obj:`True`)
        """
        adj = torch.matmul(z, z.t())
        return torch.sigmoid(adj) if sigmoid else adj



class GAEJittable(torch.nn.Module):
    r"""The Graph Auto-Encoder model from the
    `"Variational Graph Auto-Encoders" <https://arxiv.org/abs/1611.07308>`_
    paper based on user-defined encoder and decoder models.

    Args:
        encoder (Module): The encoder module.
        decoder (Module, optional): The decoder module. If set to :obj:`None`,
            will default to the
            :class:`torch_geometric.nn.models.InnerProductDecoderJittable`.
            (default: :obj:`None`)
    """
    def __init__(self, encoder, decoder=None):
        super().__init__()
        self.encoder = encoder
        self.decoder = InnerProductDecoderJittable() if decoder is None else decoder
        GAEJittable.reset_parameters(self)

    def reset_parameters(self):
        reset(self.encoder)
        reset(self.decoder)

    @torch.jit.export
    def encode(self, x, edge_index):
        r"""Runs the encoder and computes node-wise latent variables."""
        return self.encoder(x, edge_index)
    
    @torch.jit.export
    def decode(self, z: torch.Tensor,edge_index: torch.Tensor,sigmoid: bool = True):
        r"""Runs the decoder and computes edge probabilities."""
        return self.decoder(z,edge_index,sigmoid)

    @torch.jit.export
    def recon_loss(self, z, pos_edge_index):
        r"""Given latent variables :obj:`z`, computes the binary cross
        entropy loss for positive edges :obj:`pos_edge_index` and negative
        sampled edges.

        Args:
            z (Tensor): The latent space :math:`\mathbf{Z}`.
            pos_edge_index (LongTensor): The positive edges to train against.
            neg_edge_index (LongTensor, optional): The negative edges to train
                against. If not given, uses negative sampling to calculate
                negative edges. (default: :obj:`None`)
        """
        EPS = 1e-15
        pos_loss = -torch.log(
            self.decoder(z, pos_edge_index, True) + EPS).mean()

        neg_edge_index = negative_sampling(pos_edge_index, z.size(0), None, "sparse", False)

        neg_loss = -torch.log(1 -
                              self.decoder(z, neg_edge_index, sigmoid=True) +
                              EPS).mean()

        return pos_loss + neg_loss
    @torch.jit.export
    def test(self, z, pos_edge_index, neg_edge_index):
        r"""Given latent variables :obj:`z`, positive edges
        :obj:`pos_edge_index` and negative edges :obj:`neg_edge_index`,
        computes area under the ROC curve (AUC) and average precision (AP)
        scores.

        Args:
            z (Tensor): The latent space :math:`\mathbf{Z}`.
            pos_edge_index (LongTensor): The positive edges to evaluate
                against.
            neg_edge_index (LongTensor): The negative edges to evaluate
                against.
        """

        pos_y = z.new_ones(pos_edge_index.size(1))
        neg_y = z.new_zeros(neg_edge_index.size(1))
        y = torch.cat([pos_y, neg_y], dim=0)

        pos_pred = self.decoder(z, pos_edge_index, sigmoid=True)
        neg_pred = self.decoder(z, neg_edge_index, sigmoid=True)
        pred = torch.cat([pos_pred, neg_pred], dim=0)
        return y, pred
    

class VGAEJittable(GAEJittable):
    r"""The Variational Graph Auto-Encoder model from the
    `"Variational Graph Auto-Encoders" <https://arxiv.org/abs/1611.07308>`_
    paper.

    Args:
        encoder (Module): The encoder module to compute :math:`\mu` and
            :math:`\log\sigma^2`.
        decoder (Module, optional): The decoder module. If set to :obj:`None`,
            will default to the
            :class:`torch_geometric.nn.models.InnerProductDecoderJittable`.
            (default: :obj:`None`)
    """
    def __init__(self, encoder, decoder=None):
        super().__init__(encoder, decoder)

    def reparametrize(self, mu, logstd):
        if self.training:
            return mu + torch.randn_like(logstd) * torch.exp(logstd)
        else:
            return mu

    def encode(self, *args, **kwargs):
        """"""
        self.__mu__, self.__logstd__ = self.encoder(*args, **kwargs)
        self.__logstd__ = self.__logstd__.clamp(max=MAX_LOGSTD)
        z = self.reparametrize(self.__mu__, self.__logstd__)
        return z

    def kl_loss(self, mu=None, logstd=None):
        r"""Computes the KL loss, either for the passed arguments :obj:`mu`
        and :obj:`logstd`, or based on latent variables from last encoding.

        Args:
            mu (Tensor, optional): The latent space for :math:`\mu`. If set to
                :obj:`None`, uses the last computation of :math:`mu`.
                (default: :obj:`None`)
            logstd (Tensor, optional): The latent space for
                :math:`\log\sigma`.  If set to :obj:`None`, uses the last
                computation of :math:`\log\sigma^2`.(default: :obj:`None`)
        """
        mu = self.__mu__ if mu is None else mu
        logstd = self.__logstd__ if logstd is None else logstd.clamp(
            max=MAX_LOGSTD)
        return -0.5 * torch.mean(
            torch.sum(1 + 2 * logstd - mu**2 - logstd.exp()**2, dim=1))


class ARGAJittable(GAEJittable):
    r"""The Adversarially Regularized Graph Auto-Encoder model from the
    `"Adversarially Regularized Graph Autoencoder for Graph Embedding"
    <https://arxiv.org/abs/1802.04407>`_ paper.
    paper.

    Args:
        encoder (Module): The encoder module.
        discriminator (Module): The discriminator module.
        decoder (Module, optional): The decoder module. If set to :obj:`None`,
            will default to the
            :class:`torch_geometric.nn.models.InnerProductDecoderJittable`.
            (default: :obj:`None`)
    """
    def __init__(self, encoder, discriminator, decoder=None):
        super().__init__(encoder, decoder)
        self.discriminator = discriminator
        reset(self.discriminator)

    def reset_parameters(self):
        super().reset_parameters()
        reset(self.discriminator)

    def reg_loss(self, z):
        r"""Computes the regularization loss of the encoder.

        Args:
            z (Tensor): The latent space :math:`\mathbf{Z}`.
        """
        real = torch.sigmoid(self.discriminator(z))
        real_loss = -torch.log(real + EPS).mean()
        return real_loss

    def discriminator_loss(self, z):
        r"""Computes the loss of the discriminator.

        Args:
            z (Tensor): The latent space :math:`\mathbf{Z}`.
        """
        real = torch.sigmoid(self.discriminator(torch.randn_like(z)))
        fake = torch.sigmoid(self.discriminator(z.detach()))
        real_loss = -torch.log(real + EPS).mean()
        fake_loss = -torch.log(1 - fake + EPS).mean()
        return real_loss + fake_loss


class ARGVAJittable(ARGAJittable):
    r"""The Adversarially Regularized Variational Graph Auto-Encoder model from
    the `"Adversarially Regularized Graph Autoencoder for Graph Embedding"
    <https://arxiv.org/abs/1802.04407>`_ paper.
    paper.

    Args:
        encoder (Module): The encoder module to compute :math:`\mu` and
            :math:`\log\sigma^2`.
        discriminator (Module): The discriminator module.
        decoder (Module, optional): The decoder module. If set to :obj:`None`,
            will default to the
            :class:`torch_geometric.nn.models.InnerProductDecoderJittable`.
            (default: :obj:`None`)
    """
    def __init__(self, encoder, discriminator, decoder=None):
        super().__init__(encoder, discriminator, decoder)
        self.VGAEJittableJittable = VGAEJittableJittable(encoder, decoder)

    @property
    def __mu__(self):
        return self.VGAEJittableJittable.__mu__

    @property
    def __logstd__(self):
        return self.VGAEJittableJittable.__logstd__

    def reparametrize(self, mu, logstd):
        return self.VGAEJittableJittable.reparametrize(mu, logstd)

    def encode(self, *args, **kwargs):
        """"""
        return self.VGAEJittableJittable.encode(*args, **kwargs)

    def kl_loss(self, mu=None, logstd=None):
        return self.VGAEJittableJittable.kl_loss(mu, logstd)