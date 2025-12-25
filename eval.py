import os
import warnings
import argparse
import numpy as np
import torch
import torch.nn.functional as F
import torch.utils.data as data
import torchvision.transforms as transforms
from PIL import Image, ImageFile
from torch import cuda,Tensor
from torch.utils.data import DataLoader
from tqdm import tqdm
from Dataset import ImageFromList, RoxfordAndRparis
from models.CG_distill import MultiTeacherDistillModel,MultiTeacher
from utils import (compute_map_and_print, extract_vectors, load_pickle)
from torchvision.transforms import InterpolationMode
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
warnings.filterwarnings('ignore')





@torch.no_grad()
def evaluate_model(model, image_size=None,epoch = None, save_dir=None, device='cuda', plus1m=False, test_data_dir=None):

    model.eval()

    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    transform = transforms.Compose([
        transforms.Resize(size=(image_size, image_size)),
        transforms.ToTensor(),
        normalize
    ])

    datasets = ['roxford5k', 'rparis6k']
    ms = [1,2**(1/2),(1/2)**(1/2)]

    # Use default test data directory if not provided
    if test_data_dir is None:
        test_data_dir = "data/test"

    if plus1m:
        r1mvecs = load_pickle("effr1m.pkl")

    results = {}
    for dataset in datasets:
        cfg = RoxfordAndRparis(dataset, test_data_dir)
        db_images = cfg['im_fname']
        qimages = cfg['qim_fname']


        query_loader = DataLoader(ImageFromList(qimages, transforms=transform, imsize=image_size),
                                  batch_size=1, shuffle=False, num_workers=4, pin_memory=True)
        db_loader = DataLoader(ImageFromList(db_images, transforms=transform, imsize=image_size),
                               batch_size=1, shuffle=False, num_workers=4, pin_memory=True)

        qvecs = extract_vectors(model, query_loader, ms=ms, device=device).numpy()
        vecs = extract_vectors(model, db_loader, ms=ms, device=device)

        if plus1m:
            distractor_features = F.normalize(r1mvecs, dim=1, p=2)
            vecs = torch.cat([vecs, distractor_features], dim=0)

        vecs = vecs.numpy()
        scores = np.dot(vecs, qvecs.T)
        ranks = np.argsort(-scores, axis=0)

        _, mapM, mapH = compute_map_and_print(dataset, f"Epoch {epoch}", 'MobileNetV2', ranks, cfg['gnd'])
        results[dataset] = {'mAP_medium': mapM, 'mAP_hard': mapH}

    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        save_path = os.path.join(save_dir, f"eval_epoch{epoch}.txt")
        with open(save_path, "w") as f:
            for dataset, r in results.items():
                f.write(f"{dataset} — mAP_M: {r['mAP_medium']:.4f}, mAP_H: {r['mAP_hard']:.4f}\n")
        print(f">> Evaluation results saved to {save_path}")

    return results



def collate_tuples_topk(batch):
    batch = list(filter(lambda x: x is not None, batch))
    image, feature = zip(*batch)
    feature = [torch.from_numpy(feat) if isinstance(feat, np.ndarray) else feat for feat in feature]

    return torch.stack(image, dim=0), torch.stack(feature, dim=0)







def imthumbnail(img, imsize):
    img.thumbnail((imsize, imsize), Image.BICUBIC)
    return img

@torch.no_grad()
def extract_vectors(net, loader, ms=[1], device=torch.device('cuda')):
    net.eval()
    total = len(loader)
    vecs = torch.zeros(total, 512)
    if len(ms) == 1:
        for i, input in tqdm(enumerate(loader), total=total):
            batch_size_inner = input.shape[0]
            input=input.to(device)
            vecs[i * batch_size_inner:((i + 1) * batch_size_inner), :] = net.forward_test(input).cpu().data.squeeze()
    else:
        for i, input in tqdm(enumerate(loader), total=total):
            batch_size_inner = input.shape[0]
            input = input.to(device)
            vec = torch.zeros(batch_size_inner, 512)
            for s in ms:
                if s == 1:
                    input_ = input.clone()
                else:
                    input_ = F.interpolate(input, scale_factor=s, mode='bilinear', align_corners=False)
                vec += net.forward_test(input_.to(device)).cpu().data.squeeze()
            vec /= len(ms)
            vecs[i * batch_size_inner:((i + 1) * batch_size_inner), :] = F.normalize(vec, p=2, dim=-1)
    return vecs


@torch.no_grad()
def pil_loader(path):
    # to avoid crashing for truncated (corrupted images)
    ImageFile.LOAD_TRUNCATED_IMAGES = True
    # open path as file to avoid ResourceWarning
    # (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, 'rb') as f:
        img = Image.open(f)
        return img.convert('RGB')


class ImageFromList(data.Dataset):
    def __init__(self, Image_paths=None, transforms=None, imsize=None, bbox=None, loader=pil_loader):
        super(ImageFromList, self).__init__()
        self.Image_paths = Image_paths
        self.transforms = transforms
        self.imsize = imsize
        self.loader = loader
        self.len = len(Image_paths)

    def __getitem__(self, index):
        path = self.Image_paths[index]
        img = self.loader(path)
        img = imthumbnail(img, self.imsize)
        img = self.transforms(img)

        return img

    def __len__(self):
        return self.len

class RGB2BGR(object):
    def __call__(self, x):
        return x[(2,1,0),]

@torch.no_grad()
def test(datasets,image_size=None,query_net=None, db_net=None, device=torch.device('cuda'), ms=[1], pool='GeM asys',plus1m=None, test_data_dir=None):

    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    transform = transforms.Compose([transforms.ToTensor(),normalize])
    query_net.eval()
    db_net.eval()
    query_net.to(device)
    db_net.to(device)
    
    # Use default test data directory if not provided
    if test_data_dir is None:
        test_data_dir = "data/test"
    
    if plus1m:

        r1mvecs  = load_pickle("r1m.pkl")

    for dataset in datasets:
        # prepare config structure for the test dataset
        cfg = RoxfordAndRparis(dataset, test_data_dir)
        db_images = cfg['im_fname']
        qimages = cfg['qim_fname']

        query_loader = DataLoader(ImageFromList(Image_paths=qimages, transforms=transform, imsize=image_size, bbox=None), batch_size=1, shuffle=False, num_workers=8, pin_memory=True)
        qvecs = extract_vectors(net=query_net, loader=query_loader, ms=ms, device=device)
        qvecs = qvecs.numpy()

        db_loader = DataLoader(ImageFromList(Image_paths=db_images, transforms=transform, imsize=image_size, bbox=None), batch_size=1, shuffle=False, num_workers=8, pin_memory=True)
        vecs = extract_vectors(net=db_net, loader=db_loader, ms=ms, device=device)

        if plus1m:
            distractor_features= F.normalize(r1mvecs,dim=1, p=2)
            vecs = torch.cat([vecs, distractor_features], dim=0)
        vecs = vecs.numpy()
        scores = np.dot(vecs, qvecs.T)
        ranks = np.argsort(-scores, axis=0)

        _, mapM, mapH = compute_map_and_print(dataset, pool, ranks, cfg['gnd'])




if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--imsize', default = 1024, type=int, metavar='N',
                        help='maximum size of longer image side used for training (default: 1024)')
    parser.add_argument("-ts", "--teachers", default=["resnet101_delg", "resnet101_dolg"], nargs="+", metavar="LIST")

    parser.add_argument('-p', default=3, type=float, help='power rate')
    parser.add_argument('--stu_dim', default=2048, type=int, help='embedding dimension')
    parser.add_argument('--student', type=str, default='efficientnet_b3', help='student model')
    parser.add_argument('--rank', type=int, default=None)
    parser.add_argument("--path_to_pretrained_weights", default="./weights", type=str, metavar="DIR",
                        help="path to pretrained teacher models and whitening weights")
    parser.add_argument('--resume', default=None, type=str, metavar='FILENAME',
                        help='name of the latest checkpoint (default: None)')
    parser.add_argument('--plus1m', default=False, action='store_true')
    parser.add_argument('--test-data-dir', type=str, default=r"D:\SSP-mainS\data\test", metavar='DIR',
                        help='directory containing test datasets (default: data/test)')
    parser.add_argument('--checkpoint', type=str, default='D:/efficientnet_b3.pth', metavar='PATH',
                        help='path to model checkpoint (default: mobilenet_v2.pth)')
    args = parser.parse_args()

    device = torch.device('cuda' if cuda.is_available() else 'cpu')

    query_net = MultiTeacherDistillModel(args)
    checkpoint = torch.load(args.checkpoint, map_location="cuda:0")
    state_dict = checkpoint["state_dict"]

    query_net.load_state_dict(state_dict, strict=False)


    query_net.eval()


    result = test(['roxford5k', 'rparis6k'],image_size=1024, db_net=query_net,query_net=query_net,plus1m=args.plus1m,ms=[1,2**(1/2),1/2**(1/2)],test_data_dir=args.test_data_dir)

