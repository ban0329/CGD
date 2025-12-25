import os
import torch.nn as nn
import torch


class PCA(nn.Module):
    """
    Class to  compute and apply PCA.
    """

    def __init__(self, dim1=0, dim2=0, whit=0.5):
        super().__init__()
        self.register_buffer("dim1", torch.tensor(dim1, dtype=torch.long))
        self.register_buffer("dim2", torch.tensor(dim2, dtype=torch.long))
        self.register_buffer("whit", torch.tensor(whit, dtype=torch.float32))
        self.register_buffer("d", torch.zeros(self.dim1, dtype=torch.float32))
        self.register_buffer("v", torch.zeros(self.dim1, self.dim1, dtype=torch.float32))
        self.register_buffer("n_0", torch.tensor(0, dtype=torch.long))
        self.register_buffer("mean", torch.zeros(1, self.dim1, dtype=torch.float32))
        self.register_buffer("dvt", torch.zeros(self.dim2, self.dim1, dtype=torch.float32))

    def train_pca(self, x):
        """
        Takes a covariance matrix (torch.Tensor) as input.
        """

        x = torch.tensor(x)

        x_mean = x.mean(dim=0, keepdim=True)
        self.mean = x_mean
        x -= x_mean
        cov = x.t().mm(x) / x.size(0)

        d, v = torch.linalg.eigh(cov)

        self.d.copy_(d)
        self.v.copy_(v)

        eps = d.max() * 1e-5
        n_0 = (d < eps).sum()
        if n_0 > 0:
            d[d < eps] = eps

        self.n_0 = n_0

        # total energy
        totenergy = d.sum()

        # sort eigenvectors with eigenvalues order
        idx = torch.argsort(d, descending=True)[:self.dim2]
        d = d[idx]
        v = v[:, idx]

        logger = logging.getLogger("pca")
        logger.info(f"keeping {d.sum() / totenergy * 100.0:.2f} % of the energy")

        # for the whitening
        d = torch.diag(1. / d ** self.whit)

        # principal components
        self.dvt = d @ v.T

    def forward(self, x):
        x -= self.mean
        return torch.mm(self.dvt, x.transpose(0, 1)).transpose(0, 1)


def mocov3_pca_layer(path_to_pretrained_weights, *args, **kwargs):
    pretrained_weights = os.path.join(path_to_pretrained_weights, "pca_weights/mocov3_pca_512d_svd_224x224_p1.pt")
    checkpoint = torch.load(pretrained_weights, map_location="cpu")
    pca_layer = PCA(dim1=checkpoint["dim1"].item(), dim2=checkpoint["dim2"].item())
    pca_layer.load_state_dict(checkpoint)

    return pca_layer


def barlowtwins_pca_layer(path_to_pretrained_weights, *args, **kwargs):
    pretrained_weights = os.path.join(path_to_pretrained_weights, "pca_weights/barlowtwins_pca_512d_svd_224x224_p1.pt")
    checkpoint = torch.load(pretrained_weights, map_location="cpu")
    pca_layer = PCA(dim1=checkpoint["dim1"].item(), dim2=checkpoint["dim2"].item())
    pca_layer.load_state_dict(checkpoint)

    return pca_layer


def resnet101_gem_pca_layer(path_to_pretrained_weights, embed_dim=512):
    pretrained_weights = os.path.join(path_to_pretrained_weights, f"pca_weights/resnet101_gem_pca_{embed_dim}d_gldv2_512x512_p3_randcrop.pt")
    checkpoint = torch.load(pretrained_weights, map_location="cpu")
    pca_layer = PCA(dim1=checkpoint["dim1"].item(), dim2=checkpoint["dim2"].item())
    pca_layer.load_state_dict(checkpoint)

    return pca_layer


def resnet101_ap_gem_pca_layer(path_to_pretrained_weights, embed_dim=512):
    pretrained_weights = os.path.join(path_to_pretrained_weights, f"pca_weights/resnet101_ap_gem_pca_{embed_dim}d_gldv2_512x512_p3_randcrop.pt")
    checkpoint = torch.load(pretrained_weights, map_location="cpu")
    pca_layer = PCA(dim1=checkpoint["dim1"].item(), dim2=checkpoint["dim2"].item())
    pca_layer.load_state_dict(checkpoint)

    return pca_layer


def resnet101_solar_pca_layer(path_to_pretrained_weights, embed_dim=512):
    pretrained_weights = os.path.join(path_to_pretrained_weights, f"pca_weights/resnet101_solar_pca_{embed_dim}d_gldv2_512x512_p3_randcrop.pt")
    checkpoint = torch.load(pretrained_weights, map_location="cpu")
    pca_layer = PCA(dim1=checkpoint["dim1"].item(), dim2=checkpoint["dim2"].item())
    pca_layer.load_state_dict(checkpoint)

    return pca_layer

def resnet101_delg_pca_layer(path_to_pretrained_weights, embed_dim=512):
    pretrained_weights = os.path.join(path_to_pretrained_weights, f"resnet101_delg_pca_512d_SfM120k_512x512_p3_randcrop.pt")
    pretrained_weights = r".\weights2\pca_weights\resnet101_delg_pca_512d_GLDnew3_512x512_p3_randcrop.pt"
    checkpoint = torch.load(pretrained_weights, map_location="cpu")
    pca_layer = PCA(dim1=checkpoint["dim1"].item(), dim2=checkpoint["dim2"].item())
    pca_layer.load_state_dict(checkpoint)

    return pca_layer

def resnet101_delg_pca_layer(path_to_pretrained_weights, embed_dim=512):
    pretrained_weights = os.path.join(path_to_pretrained_weights, f"resnet101_delg_pca_512d_SfM120k_512x512_p3_randcrop.pt")
    pretrained_weights = r".\weights2\pca_weights\resnet101_delg_pca_512d_SfM120k_512x512_p3_randcrop.pt"
    checkpoint = torch.load(pretrained_weights, map_location="cpu")
    pca_layer = PCA(dim1=checkpoint["dim1"].item(), dim2=checkpoint["dim2"].item())
    pca_layer.load_state_dict(checkpoint)

    return pca_layer


def resnet101_dolg_pca_layer(path_to_pretrained_weights, embed_dim=512):
    pretrained_weights = os.path.join(path_to_pretrained_weights, f"resnet101_dolg_pca_{embed_dim}d_SfM120k_512x512_p3_randcrop.pt")
    pretrained_weights = r".\weights2\pca_weights\resnet101_dolg_pca_512d_SfM120k_512x512_p3_randcrop.pt"
    checkpoint = torch.load(pretrained_weights, map_location="cpu")
    pca_layer = PCA(dim1=checkpoint["dim1"].item(), dim2=checkpoint["dim2"].item())
    pca_layer.load_state_dict(checkpoint)

    return pca_layer


pca_layers = {
    "mocov3": mocov3_pca_layer,
    "barlowtwins": barlowtwins_pca_layer,
    "resnet101_gem": resnet101_gem_pca_layer,
    "resnet101_ap_gem": resnet101_ap_gem_pca_layer,
    "resnet101_solar": resnet101_solar_pca_layer,
    "resnet101_delg": resnet101_delg_pca_layer,
    "resnet101_dolg": resnet101_dolg_pca_layer
}