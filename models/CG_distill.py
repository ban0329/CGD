import torch.nn.functional as F
import torch.nn as nn
from torch.cuda.amp.autocast_mode import autocast
from torch import Tensor

from models.teachers import teacher_models
from models.pca_layer import pca_layers
from models.lightweight import lightweight
from models.gem_pooling import GeneralizedMeanPooling as GeM
import os
from CCF import *
from utils.helpfunc import load_pickle



class MultiTeacher(nn.Module):
    def __init__(self, path_to_pretrained_weights, *teachers, p=3.,embed_dim=512):
        super().__init__()
        self.teachers = teachers
        encoders = list()
        for teacher in self.teachers:
            encoders.append(teacher_models[teacher](path_to_pretrained_weights, pretrained=True, gem_p=p))
        self.encoders = nn.ModuleList(encoders)
        self.embed_dims = [encoder.embed_dim for encoder in self.encoders]
        norm_layers = list()
        for teacher in self.teachers:
            norm_layers.append(pca_layers[teacher](path_to_pretrained_weights, embed_dim=embed_dim))
        self.norm_layers = nn.ModuleList(norm_layers)
        self.embed_dims = [norm_layer.dim2 for norm_layer in self.norm_layers]

    def forward(self, x):
        out = list()
        for teacher, encoder in zip(self.teachers, self.encoders):
            if teacher.endswith("delg") or teacher.endswith("dolg512"):
                out.append(encoder(x[:, (2, 1, 0)]))
            else:
                out.append(encoder(x))

        out = [nn.functional.normalize(o, p=2, dim=-1) for o in out]
        out = [norm_layer(o) for norm_layer, o in zip(self.norm_layers, out)]
        out = fusion(out)
        return out

class MultiTeacherDistillModel(nn.Module):
    def __init__(self, args=None, PQ_centroid_path=None):
        super().__init__()

        
        self.base_encoder = lightweight[args.student](dim=args.stu_dim)
        self.whiten = nn.Linear(args.stu_dim, 512)
        self.pooling = GeM()


        self.is_training_mode = (
            hasattr(args, "path_to_pretrained_weights")
            and hasattr(args, "teachers")
            and PQ_centroid_path is not None
            and os.path.exists(PQ_centroid_path)
        )


        if self.is_training_mode:
            print("[Info] Initializing full distillation model.")
            self.m = getattr(args, "m", None)
            self.n_bit = getattr(args, "n_bits", None)

            # load and freeze teacher models
            self.teacher_encoders = MultiTeacher(
                args.path_to_pretrained_weights, *args.teachers, p=args.p,
            )
            for param in self.teacher_encoders.parameters():
                param.requires_grad = False

            # load PQ centroids
            PQ_centroids = torch.from_numpy(load_pickle(PQ_centroid_path)).float()
            self.register_buffer("centroids", F.normalize(PQ_centroids, dim=-1))
        else:
            print("[Info] Initializing lightweight backbone only (inference mode).")
            self.teacher_encoders = None
            self.centroids = None

    def state_dict(self, destination=None, prefix='', keep_vars=False):
        """Save only student-related parameters."""
        student_state = {}
        for name, param in super().state_dict(destination, prefix, keep_vars).items():
            if name.startswith("base_encoder") or name.startswith("whiten") or name.startswith("pooling"):
                student_state[name] = param
        return student_state

    @autocast()
    def forward(self, x: Tensor):
        stu = self.base_encoder(x)
        stu = self.pooling(stu).squeeze(-1).squeeze(-1)
        stu = self.whiten(stu)
        stu = F.normalize(stu, dim=-1)

        with torch.no_grad():
            tch = self.teacher_encoders(x)
            B = tch.size(0)
            feature =F.normalize(tch.reshape(B, self.m, -1), dim=-1)
            soft_code = torch.einsum('bmc, mnc -> bmn', [feature, self.centroids])
            soft_code = F.softmax(soft_code * 10, dim=-1)

        student = F.normalize(stu.reshape(B, self.m, -1), dim=-1)
        x_distance = torch.einsum('bmc, mnc -> bmn', [student, self.centroids.detach()])
        distill = F.kl_div(F.log_softmax(x_distance * 10, dim=-1), soft_code, reduction='none')
        loss = distill.sum(dim=-1).mean()

        return loss

    @autocast()
    def forward_test(self, x: Tensor):
        x = self.base_encoder(x)
        x = self.pooling(x).squeeze(-1).squeeze(-1)
        x = self.whiten(x)
        x = F.normalize(x, dim=-1)
        return x



