import dataset
import torch
import pytorch3d.ops as ops
from torchvision import datasets, transforms, utils
import zipfile
import h5py as h5
import numpy as np
import matplotlib.pyplot as plt
from tqdm import trange, tqdm
import copy

import os
import sys


# imports the dataset python file which loads the AnTao97/PointCloudDataset
script_dir = os.path.dirname(__file__)
shape_net_part_2015_dir = os.path.join(
    script_dir, '..', 'Datasets', 'ShapeNetPart2015', 'PointCloudDatasets')
sys.path.append(shape_net_part_2015_dir)


"""

Paramètres

"""


dataset_name = 'shapenetpart'

d = 3

cuda = False

lr = 0.001

n_epoch = 1000

ifd = 0  # input feature dimension

K = 50


"""

Time taken 

"""


"""

Objets utilisés

"""


class Data_points:
    def __init__(self, coords, features):
        if coords.shape[0] != features.shape[0] or coords.shape[1] != d:
            raise Exception("Dimension issue")
        self.coords = coords  # coordonnées du point ::tensor(N,d)
        self.features = features  # features du point ::tensor(N,C)

    def __str__(self):
        return str(torch.cat((self.coords, self.features), dim=1))

    def __repr__(self):
        return str(self.__str__())


class Group:
    # data_points : ensemble des points dans le groupe
    def __init__(self, center, coords, features):
        self.center = center  # coordonnées du centre du groupe ::tensor(1,d)
        # coordonnées des points centré en 0 coords ::tensor(K,d)
        self.coords = coords - center
        self.features = features        # ::tensor(K,C)

    def __str__(self):
        v = torch.cat((self.coords+self.center, self.features), dim=1)
        return str(self.center) + str(v.shape) + str(v)

    def __repr__(self):
        v = torch.cat((self.coords+self.center, self.features), dim=1)
        return repr(self.center) + str(v.shape) + repr(v)


"""

Evaluation metric

"""

# =============================================================================
# def iou_pytorch(outputs: torch.Tensor, labels: torch.Tensor):
#     # You can comment out this line if you are passing tensors of equal shape
#     # But if you are passing output from UNet or something it will most probably
#     # be with the BATCH x 1 x H x W shape
#     outputs = outputs.squeeze(1)  # BATCH x 1 x H x W => BATCH x H x W
#
#     intersection = (outputs & labels).float().sum((1, 2))  # Will be zero if Truth=0 or Prediction=0
#     union = (outputs | labels).float().sum((1, 2))         # Will be zzero if both are 0
#
#     iou = (intersection + SMOOTH) / (union + SMOOTH)  # We smooth our devision to avoid 0/0
#
#     thresholded = torch.clamp(20 * (iou - 0.5), 0, 10).ceil() / 10  # This is equal to comparing with thresolds
#
#     return thresholded  # Or thresholded.mean() if you are interested in average across the batch
# =============================================================================


"""

Loss function

"""

loss = torch.nn.CrossEntropyLoss(weight=None, size_average=None,
                                 ignore_index=- 100, reduce=None, reduction='mean', label_smoothing=0.0)


"""

Farthest Point Sampling Layer

"""


def down_sample(point_cloud, point_number):
    N = point_cloud.size()[0]
    if N > point_number:
        new_point_cloud = torch.zeros((point_number, d), dtype=torch.float)
        for i in range(point_number):
            new_point_cloud[i] = point_cloud[np.random.randint(0, N)]
        return new_point_cloud
    return point_cloud


def update_distance_matrix(point_cloud, D, index):
    for j in range(0, point_cloud.size()[0]):

        dist = torch.norm(point_cloud[index] - point_cloud[j])
        D[index, j] = dist


def FPS(point_cloud, k):
    """
    Parameters
    ----------
    data_points : tensor (N, d) 

    Return
    ------
    list (k-length array of indices)
    """
    point_number = 2*k
    point_cloud = down_sample(point_cloud, point_number)
    N = point_cloud.size()[0]
    # An NxN distance matrix for points
    D = torch.zeros((N, N), dtype=torch.float)
    list = torch.zeros((k, d), dtype=torch.float)
    # first point in the permutation is random
    first_point_index = np.random.randint(0, N)
    update_distance_matrix(point_cloud, D, first_point_index)
    list[0] = point_cloud[first_point_index]
    ds = D[first_point_index, :]
    for i in range(1, k):
        idx = torch.argmax(ds)
        update_distance_matrix(point_cloud, D, idx)
        list[i] = point_cloud[idx]
        ds = torch.minimum(ds, D[idx, :])
    return list


"""

Grouping layer

"""


def ball_grouping(center, data_points, r):
    distances = torch.linalg.vector_norm(
        data_points.coords[:, :3] - center[:, :3], dim=1)
    group = Group(
        center, data_points.coords[distances < r], data_points.features[distances < r])
    return group


# centers :  tensor (N,d), points_data : all the data in one tensor Nx(d+C)
def grouping_layer(centers, data_points, grouping_method, parameter):
    grouping = []
    for i in range(centers.shape[0]):
        grouping.append(grouping_method(
            centers[i:i+1, :], data_points, parameter))
    # liste de N' tensor de chacun Kx(d+C)  (K variable selon les groupes)
    return grouping


"""

PointNet Layer

"""


class Pointnet(torch.nn.Module):  # N’xKx(d+C) -> N’x(d+C’)
    def __init__(self, ifd, l1, l2, l3):  # ifd : input feature dimension ()
        super(Pointnet, self).__init__()
        self.fc1 = torch.nn.Sequential(
            torch.nn.Linear(ifd+d, l1),
            # torch.nn.BatchNorm1d(l1),
            torch.nn.ReLU(0)
        )
        self.fc2 = torch.nn.Sequential(
            torch.nn.Linear(l1, l2),
            # torch.nn.BatchNorm1d(l2),
            torch.nn.ReLU(0)
        )
        self.fc3 = torch.nn.Sequential(
            torch.nn.Linear(l2, l3),
            # torch.nn.BatchNorm1d(l3),
            torch.nn.ReLU(0)
        )

    def forward(self, X):   # X au format Kx(d+C)
        X = self.fc1(X)
        X = self.fc2(X)
        X = self.fc3(X)   # X au format KxC'
        X = torch.max(X, 0, keepdim=True)[0]  # X au format C'
        return X

    def process(self, groups):     # groups : liste de N' de groupe de Kx(d+C)
        centers = []
        features = []
        for j in range(len(groups)):
            group = groups[j]
            # print(group)
            centers.append(group.center)
            # print(group.coords,group.features,group.coords.shape,group.features.shape)
            features.append(
                self(torch.cat((group.coords, group.features), dim=1)))
        data_points = Data_points(torch.cat(centers), torch.cat(features))
        return data_points       # return :  data_points N'x(d+C')


"""

SA

"""


class SA(torch.nn.Module):
    def __init__(self, K, r, ifd, L):
        super(SA, self).__init__()
        self.K = K
        self.r = r
        [l1, l2, l3] = L
        self.pointnet = Pointnet(ifd, l1, l2, l3)

    def abstract(self, data_points):
        # centers = FPS(data_points.coords,self.K)        #tensor (Nxd) des centres des groupes
        coords = data_points.coords.to('cpu')
        point_cloud_size = torch.tensor([coords.shape[0]])
        centers = ops.sample_farthest_points(
            coords[None, :], point_cloud_size, self.K)[0].squeeze()
        if cuda:
            centers = centers.to('cuda').type(torch.cuda.FloatTensor)
        groups = grouping_layer(centers, data_points, ball_grouping, self.r)
        next_data_points = self.pointnet.process(groups)
        return next_data_points


class SA_end(torch.nn.Module):
    def __init__(self, ifd, L):
        super(SA_end, self).__init__()
        [l1, l2, l3] = L
        self.pointnet = Pointnet(ifd, l1, l2, l3)

    def abstract(self, data_points):
        # groupe englobant tous les points restants
        final_group = Group(
            data_points.coords[0:1, :], data_points.coords, data_points.features)
        groups = [final_group]
        next_data_points = self.pointnet.process(groups)
        return next_data_points


"""

Abstraction Network

"""


class Abstract_net(torch.nn.Module):
    def __init__(self, ifd, SA_configs, SA_end_config):
        super(Abstract_net, self).__init__()
        # data_points successifs
        self.sdp = [0 for i in range(len(SA_configs)+2)]
        self.SA = []
        for SA_config in SA_configs:
            [K, r, L] = SA_config
            self.SA.append(SA(K, r, ifd, L))
            ifd = L[-1]
        self.SA = torch.nn.ModuleList(self.SA)
        self.SA_end = SA_end(ifd, SA_end_config)
        self.out_dim = SA_end_config[-1]

    def forward(self, data_points):
        for i, SA in enumerate(self.SA):
            self.sdp[i] = data_points
            data_points = SA.abstract(data_points)
        self.sdp[-2] = data_points
        data_points = self.SA_end.abstract(data_points)
        self.sdp[-1] = data_points
        features = data_points.features
        return self.sdp


"""

FP

"""


class FP(torch.nn.Module):
    def __init__(self, inp_dim, L, last):
        super(FP, self).__init__()
        fc = []
        n_layer = len(L)
        for i in range(n_layer):
            NN = []
            NN.append(torch.nn.Linear(inp_dim, L[i]))

            if not (last and i == n_layer-1):
                NN.append(torch.nn.BatchNorm1d(L[i]))
                NN.append(torch.nn.ReLU())

            fc.append(torch.nn.Sequential(*NN))
            inp_dim = L[i]
        if not last:
            fc.append(torch.nn.Dropout(0.5))
        self.fc = torch.nn.Sequential(*fc)

    def forward(self, X):  # X ::tensor(N,C)
        X = self.fc(X)
        return X


# =============================================================================
# """
#
# Classification Network
#
# """
#
#
#
# class Class_net(torch.nn.Module):
#     def __init__(self,inp_dim,FP_configs):
#         self.fp = []
#         for FP_config in FP_configs:
#             fp = FP(inp_dim,FP_config)
#             self.fp.append(fp)
#             inp_dim = FP_config[-1]
#
#     def forward(self,X):        # X :: tensor(1,C)
#         for fp in self.fp:
#             X = fp(X)
#         return X
# =============================================================================


"""

Segmentation layer

"""

one_tensor = torch.tensor([1], dtype=torch.float)

if cuda:
    one_tensor = one_tensor.type(torch.cuda.FloatTensor)


# points_coords : tensor(N,d) , data_points : N'x(C+d)
def k_means(k, p, data_points, points_coords):
    Np = data_points.coords.shape[0]
    k = min(k, Np)
    N = points_coords.shape[0]
    data_points_coords = data_points.coords[None, :, :]  # 1xN'xd
    data_points_coords.expand(N, -1, -1)  # NxN'xd
    distances = torch.linalg.vector_norm(
        data_points_coords-points_coords[:, None, :], dim=2)  # NxN'
    sorted, indices = torch.sort(distances)  # NxN'
    weights_raw = sorted[:, :k].pow(-p)
    weights_sum = torch.sum(weights_raw, 1, keepdim=True)
    weights_nan = weights_raw/weights_sum
    weights = torch.where(torch.isnan(weights_nan), one_tensor, weights_nan)
    data_points_features = data_points.features
    C = data_points_features.shape[1]
    res = torch.zeros((N, C))
    if cuda:
        res = res.type(torch.cuda.FloatTensor)
    for i in range(k):
        selected = torch.index_select(data_points_features, 0, indices[:, i])
        res += weights[:, i:i+1]*selected
    return res


class Seg_lay(torch.nn.Module):
    def __init__(self, k, p, inp_dim, FP_config, last):
        super(Seg_lay, self).__init__()
        self.FP = FP(inp_dim, FP_config, last)
        self.k = k
        self.p = p

    # from_points : data_points interpolé, to_points : data_points vers lesquels on interpole
    def forward(self, from_points, to_points):
        res = torch.cat((to_points.features, k_means(
            self.k, self.p, from_points, to_points.coords)), dim=1)
        to_points.features = self.FP(res)
        return to_points  # res :: data_points


"""

Segmentation Network

"""


class Seg_net(torch.nn.Module):
    def __init__(self, k, p, inp_dims, FP_configs):
        super(Seg_net, self).__init__()
        self.seg_lays = []
        for i, FP_config in enumerate(FP_configs):
            self.seg_lays.append(
                Seg_lay(k, p, inp_dims[i], FP_config, i == len(FP_configs)-1))
        self.seg_lays = torch.nn.ModuleList(self.seg_lays)

    def forward(self, sdp):
        for i, seg_lay in enumerate(self.seg_lays):
            sdp[-2-i] = seg_lay.forward(sdp[-1-i], sdp[-2-i])
        return sdp[0]


"""

Network complet

"""


class PointNetpp(torch.nn.Module):
    def __init__(self, ifd, SA_configs, SA_end_config, k_seg, p_seg, seg_inp_dims, FP_configs):
        super(PointNetpp, self).__init__()
        self.abstract_net = Abstract_net(ifd, SA_configs, SA_end_config)
        self.seg_net = Seg_net(k_seg, p_seg, seg_inp_dims, FP_configs)

    def forward(self, data_points):
        # successive data points calculated by abstract_net
        sdp = self.abstract_net(data_points)
        per_point_scores = self.seg_net.forward(sdp)      # N(C+d)
        return per_point_scores


"""

Plot Loss

"""


class Plot():  # plots everything

    colors = {
        'train_accuracy': '--b',
        'valid_accuracy': '-b',
        'train_loss': '--o',
        'valid_loss': '-o'
    }

    def __init__(self, labels, n_x):
        self.x_values = [i for i in range(n_x)]
        self.values = {}
        for v_lbl in ['train_accuracy', 'valid_accuracy', 'train_loss', 'valid_loss']:
            self.values[v_lbl] = [[] for i in labels]
        self.labels = labels

    def add_value(self, value, v_lbl, i=0):
        self.values[v_lbl][i].append(value)

    def draw(self):
        fig = plt.figure(figsize=(16, 10))
        plt.style.use('fivethirtyeight')
        for i, label in enumerate(self.labels):
            for v_lbl in self.values:
                n_values = len(self.values[v_lbl][i])
                plt.plot(self.x_values[:n_values], self.values[v_lbl][i],
                         Plot.colors[v_lbl], label=self.labels[i] + '_' + v_lbl)
        plt.legend()
        plt.title("Accuracy and Loss")

        plt.draw()
        plt.pause(0.0001)
        plt.clf()

    def save_data(self):
        for i, label in enumerate(self.labels):
            for v_lbl in self.values:
                np.save(label + '_' + v_lbl, self.values[v_lbl][i])


"""

Evaluation Functions

"""


def net_accuracy_and_loss(loader, net, n_max=64):
    acc = []
    l = []
    i = 0
    for data in loader:
        if i > n_max:
            break
        i += 1
        data_points, categories = data
        used_data_points = copy.deepcopy(data_points)
        res_data_points = net(used_data_points)

        Loss = loss(res_data_points.features, categories)
        l.append(Loss.detach().numpy())

        res_data_points_features = torch.argmax(
            res_data_points.features, dim=1)
        accuracy = torch.mean(
            torch.eq(res_data_points_features, categories).float())
        acc.append(accuracy.detach().numpy())
    return np.mean(acc), np.mean(l)


"""

Training Function

"""


def train(net, optimizer,  train_loader, valid_loader, n_epoch, plot, loss=torch.nn.CrossEntropyLoss()):  # trains one network

    net.eval()
    train_acc, train_l = net_accuracy_and_loss(train_loader, net)
    valid_acc, valid_l = net_accuracy_and_loss(valid_loader, net)
    plot.add_value(train_acc, 'train_accuracy')
    plot.add_value(valid_acc, 'valid_accuracy')
    plot.add_value(train_l, 'train_loss')
    plot.add_value(valid_l, 'valid_loss')
    net.train()

    for epoch in range(n_epoch):

        print('epoch : ', epoch+1)
        for i, data in tqdm(enumerate(train_loader)):
            data_points, categories = data
            used_data_points = copy.deepcopy(data_points)

            optimizer.zero_grad()
            res_data_points = net(used_data_points)

            # print(res_data_points.features)

            Loss = loss(res_data_points.features, categories)
            # print(Loss)
            Loss.backward()  # Calcul du gradient de Loss

            optimizer.step()  # Mise à jour des poids

        net.eval()
        train_acc, train_l = net_accuracy_and_loss(train_loader, net)
        valid_acc, valid_l = net_accuracy_and_loss(valid_loader, net)
        plot.add_value(train_acc, 'train_accuracy')
        plot.add_value(valid_acc, 'valid_accuracy')
        plot.add_value(train_l, 'train_loss')
        plot.add_value(valid_l, 'valid_loss')
        net.train()
        plot.draw()

        torch.save(net, 'point_net_pp_epoch_'+str(epoch+1))


# =============================================================================
#
# """
#
# Données de test
#
# """
#
#
#
# X1 = torch.tensor(
#     [[0,1,2,3],
#      [0,1,1,5],
#      [1,0,2,7],
#      [4,5,3,15],
#      [4,4,4,8],
#      [4,4,5,-54],
#      [5,4,4,0.7],
#      [8,5,6,4.5]], dtype=torch.float)
#
# X2 = torch.tensor(
#     [[0,1,2,5,8,9,0],
#      [4,4,4,15,41,3.5,1],
#      [5,4,4,14,5,-5,4],
#      [8,5,6,4.7,-57,-0.52,1.2]],dtype=torch.float)
#
# dp1 = Data_points(X1[:,:3], X1[:,3:])
#
# dp2 = Data_points(X2[:,:3], X2[:,3:])
#
# Xr1 = torch.randn(10000, 4)
#
# Xr2 = torch.randn(1, 128)
#
# dpr1 = Data_points(Xr1[:,:3], Xr1[:,3:])
#
# dpr2 = Data_points(Xr2[:,:3], Xr2[:,3:])
#
#
# centers = torch.cat((dp1.coords[0:1],dp1.coords[4:5]), dim = 0 )
#
# groups = grouping_layer(centers,dp1,ball_grouping,1.1)
#
# # print(groups)
#
# testnet = Pointnet(ifd,8,8,4)
#
#
# print(testnet.process(groups))
#
# =============================================================================


"""

Data loading

"""


def load_shape_net_2015_data(cuda=False):
    class_list = [6]

    shape_net_2015_dataset = dataset.Dataset(
        root=shape_net_part_2015_dir, dataset_name=dataset_name, num_points=2048, split='train', segmentation=True)
    train_loader = []
    for data in shape_net_2015_dataset:
        if data[1][0].numpy() in class_list:
            coords, features, lbl = data[0], torch.empty(
                (data[0].shape[0], 0)), data[2]
            if cuda:
                coords, features, lbl = coords.type(torch.cuda.FloatTensor), features.type(
                    torch.cuda.FloatTensor), lbl.type(torch.cuda.LongTensor)
            data_points = Data_points(coords, features)
            data = [data_points, lbl]
            train_loader.append(data)

    shape_net_2015_dataset = dataset.Dataset(
        root=shape_net_part_2015_dir, dataset_name=dataset_name, num_points=2048, split='val', segmentation=True)
    valid_loader = []
    for data in shape_net_2015_dataset:
        if data[1][0].numpy() in class_list:
            coords, features, lbl = data[0], torch.empty(
                (data[0].shape[0], 0)), data[2]
            if cuda:
                coords, features, lbl = coords.type(torch.cuda.FloatTensor), features.type(
                    torch.cuda.FloatTensor), lbl.type(torch.cuda.LongTensor)
            data_points = Data_points(coords, features)
            data = [data_points, lbl]
            valid_loader.append(data)

    shape_net_2015_dataset = dataset.Dataset(
        root=shape_net_part_2015_dir, dataset_name=dataset_name, num_points=2048, split='test', segmentation=True)
    test_loader = []
    for data in shape_net_2015_dataset:
        if data[1][0].numpy() in class_list:
            coords, features, lbl = data[0], torch.empty(
                (data[0].shape[0], 0)), data[2]
            if cuda:
                coords, features, lbl = coords.type(torch.cuda.FloatTensor), features.type(
                    torch.cuda.FloatTensor), lbl.type(torch.cuda.LongTensor)
            data_points = Data_points(coords, features)
            data = [data_points, lbl]
            test_loader.append(data)

    return train_loader, valid_loader, test_loader


"""

Main

"""


point_net_pp = PointNetpp(ifd,
                          [[512, 0.2, [64, 64, 128]],
                           [128, 0.4, [128, 128, 256]]],
                          [256, 512, 1024],
                          3,
                          2,
                          [1280, 384, 128+ifd],
                          [[256, 256],
                              [256, 128],
                              [128, 128, 128, 128, K]])


plot = Plot(['PointNet++'], n_epoch+1)

train_loader, valid_loader, test_loader = load_shape_net_2015_data(cuda)

print(len(train_loader))


# =============================================================================
# if cuda:
#     point_net_pp.cuda()
#
# optimizer = torch.optim.Adam(point_net_pp.parameters(),lr)
#
# train(point_net_pp, optimizer, train_loader, valid_loader, n_epoch, plot)
# =============================================================================


pts, lb = valid_loader[34]


# defining axes
z = pts.coords[:, 2].numpy()
x = pts.coords[:, 0].numpy()
y = pts.coords[:, 1].numpy()
c = lb.numpy()


epochs = [1, 5, 10, 20, 35, 42, 45]


fig = plt.figure(figsize=(30, 30))

ax = fig.add_subplot(3, 3, 1, projection='3d')
ax.set_xlim([-0.5, 0.5])
ax.set_ylim([-0.5, 0.5])
ax.set_zlim([-0.5, 0.5])
ax.set_title('Guitar ground truth')
ax.scatter(x, y, z, c=c)


used_data_points = copy.deepcopy(pts)
res_data_points_features = torch.argmax(
    point_net_pp(used_data_points).features, dim=1)
cbis = res_data_points_features.numpy()
ax = fig.add_subplot(3, 3, 2, projection='3d')
ax.set_xlim([-0.5, 0.5])
ax.set_ylim([-0.5, 0.5])
ax.set_zlim([-0.5, 0.5])
ax.set_title('Guitar predicted no training')
ax.scatter(x, y, z, c=cbis)

for i, epoch in enumerate(epochs):
    point_net_pp = torch.load('point_net_pp_epoch_'+str(epoch))
    used_data_points = copy.deepcopy(pts)
    res_data_points_features = torch.argmax(
        point_net_pp(used_data_points).features, dim=1)
    cbis = res_data_points_features.numpy()
    ax = fig.add_subplot(3, 3, 3+i, projection='3d')
    ax.set_xlim([-0.5, 0.5])
    ax.set_ylim([-0.5, 0.5])
    ax.set_zlim([-0.5, 0.5])
    ax.set_title('Guitar predicted epoch ' + str(epoch))
    ax.scatter(x, y, z, c=cbis)


plt.show()


# =============================================================================
# epochs = [1,5,11,20,39,42,45]
#
# for epoch in epochs:
#     net = torch.load('point_net_pp_epoch_'+str(epoch))
#     net.cpu()
#     net.eval()
#     train_acc,train_l = net_accuracy_and_loss(train_loader,net)
#     print(train_acc,train_l)
#
#     valid_acc,valid_l = net_accuracy_and_loss(valid_loader,net)
#     print(valid_acc,valid_l)
#
#     plot.add_value(train_acc,'train_accuracy')
#     plot.add_value(valid_acc,'valid_accuracy')
#     plot.add_value(train_l,'train_loss')
#     plot.add_value(valid_l,'valid_loss')
#     plot.draw()
#     net.train()
#
#
# =============================================================================
