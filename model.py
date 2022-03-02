from torchvision import models as torchvision_models
from torchvision import transforms
import time
import torch.nn as nn
import torch
from pytorch3d.utils import ico_sphere
import pytorch3d

from torch.nn import Conv3d, LeakyReLU

class SingleViewto3D(nn.Module):
    def __init__(self, args):
        super(SingleViewto3D, self).__init__()
        self.device = "cuda"
        vision_model = torchvision_models.__dict__[args.arch](pretrained=True)
        self.encoder = torch.nn.Sequential(*(list(vision_model.children())[:-1]))
        self.normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])

        # define decoder
        if args.type == "vox":
            # TODO:
            self.decoder = VoxDecoder()            
        elif args.type == "point":
            self.n_point = args.n_points
            # TODO:
            self.decoder = MLPDecoder(size_list = [512, 1024, 1024, 1024, self.n_point*3])

            # self.decoder = PointDecoder()   
        elif args.type == "mesh":
            # try different mesh initializations
            mesh_pred = ico_sphere(4,args.device)
            self.n_point = mesh_pred.verts_list()[0].shape[0]
            print(self.n_point)
            self.mesh_pred = pytorch3d.structures.Meshes(mesh_pred.verts_list()*args.batch_size, mesh_pred.faces_list()*args.batch_size)
            # TODO:
            # self.decoder = _MLP(size_list = [512, 1024, 1024, 1024, self.n_point*3])
            self.decoder = MLPDecoder(size_list = [512, 1024, 1024, 1024, self.n_point*3])

    def forward(self, images, args):
        results = dict()

        total_loss = 0.0
        start_time = time.time()

        B = images.shape[0]

        images_normalize = self.normalize(images.permute(0,3,1,2))
        encoded_feat = self.encoder(images_normalize).squeeze(-1).squeeze(-1)

        # call decoder
        if args.type == "vox":
            # TODO:
            voxels_pred = self.decoder(encoded_feat)      
            return voxels_pred

        elif args.type == "point":
            # TODO:
            pointclouds_pred = self.decoder(encoded_feat)  
            pointclouds_pred = pointclouds_pred.reshape([-1, self.n_point,3])
            return pointclouds_pred

        elif args.type == "mesh":
            # TODO:
            deform_vertices_pred = self.decoder(encoded_feat)
            mesh_pred = self.mesh_pred.offset_verts(deform_vertices_pred.reshape([-1,3]))
            return  mesh_pred          


class PointDecoder(nn.Module):
    def __init__(self, n_deconvfilter = [64, 64, 32, 32, 3], debug = False):
        print("\ninitializing \"decoder\"")
        super(PointDecoder, self).__init__()
        self.debug = debug

        self.conv1a = Conv3d(n_deconvfilter[0], n_deconvfilter[1], 3, padding=1)
        self.conv1b = Conv3d(n_deconvfilter[1], n_deconvfilter[1], 3, padding=1)
        
        self.conv2a = Conv3d(n_deconvfilter[1], n_deconvfilter[2], 3, padding=1)
        self.conv2b = Conv3d(n_deconvfilter[2], n_deconvfilter[2], 3, padding=1)
        self.conv2c = Conv3d(n_deconvfilter[1], n_deconvfilter[2], 1)
        
        self.conv3a = Conv3d(n_deconvfilter[2], n_deconvfilter[3], 3, padding=1)
        self.conv3b = Conv3d(n_deconvfilter[3], n_deconvfilter[3], 3, padding=1)
        self.conv3c = Conv3d(n_deconvfilter[3], n_deconvfilter[3], 3, padding=1)

        self.conv4 = Conv3d(n_deconvfilter[3], n_deconvfilter[4], 3, padding=1)

        self.leaky_relu = LeakyReLU(negative_slope= 0.01)
        
    def forward(self, feature):
        feature = feature.view(-1,64,2,2,2)
        #(B, 64, 2, 2, 2)
        unpool_feature = torch.nn.functional.interpolate(feature, scale_factor=2)
        conv1 = self.conv1a(unpool_feature)
        rect1 = self.leaky_relu(conv1)
        conv1 = self.conv1b(rect1)
        rect1 = self.leaky_relu(conv1)
        res1 = unpool_feature + rect1

        #(B, 64, 4, 4, 4)
        if self.debug:
            print("res1", res1.shape)
        
        feature = torch.nn.functional.interpolate(res1, scale_factor=2)
        conv2 = self.conv2a(feature)
        rect2 = self.leaky_relu(conv2)
        conv2 = self.conv2b(rect2)
        rect2 = self.leaky_relu(conv2)
        conv_feature = self.conv2c(feature)

        res2 = conv_feature + rect2

        #(B, 32, 8, 8, 8)
        if self.debug:
            print("res2", res2.shape)
        
        feature = torch.nn.functional.interpolate(res2, scale_factor=2)
        conv3 = self.conv3a(feature)
        rect3 = self.leaky_relu(conv3)
        conv3 = self.conv3b(rect3)
        rect3 = self.leaky_relu(conv3)
        conv3 = self.conv3c(feature)
        
        res3 = feature + conv3
        #(B, 32, 16, 16， 16)
        if self.debug:
            print("res3", res3.shape)

        conv4 = self.conv4(res3)

        #(B, 3, 16, 16， 16)
        if self.debug:
            print("conv4", conv4.shape)
        conv4 = conv4.reshape(-1, 3, 16*16*16)
        return conv4.permute(0,2,1)


class _MLP(nn.Module):
    def __init__(self, size_list):
        super(_MLP, self).__init__()
        layers = []
        self.size_list = size_list
        for i in range(len(size_list) - 2):
            layers.append(nn.Linear(size_list[i],size_list[i+1]))
            layers.append(nn.BatchNorm1d(size_list[i+1]))
            layers.append(nn.LeakyReLU(negative_slope= 0.01))
            # layers.append(nn.Dropout(p=0.2))
        layers.append(nn.Linear(size_list[-2], size_list[-1]))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


class MLPDecoder(nn.Module):
    def __init__(self, size_list):
        super(MLPDecoder, self).__init__()
        layers = []
        self.size_list = size_list
        for i in range(len(size_list) - 2):
            layers.append(nn.Linear(size_list[i],size_list[i+1]))
            # layers.append(nn.BatchNorm1d(size_list[i+1]))
            layers.append(nn.LeakyReLU(negative_slope= 0.01))
            # layers.append(nn.Dropout(p=0.2))
        layers.append(nn.Linear(size_list[-2], size_list[-1]))
        layers.append(nn.Tanh())
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)
    

class VoxDecoder(nn.Module):
    def __init__(self, n_deconvfilter = [64, 64, 64, 32, 32, 1], debug = False):
        print("\ninitializing \"decoder\"")
        super(VoxDecoder, self).__init__()
        self.debug = debug
        self.n_deconvfilter = n_deconvfilter

        self.conv1a = Conv3d(n_deconvfilter[0], n_deconvfilter[1], 3, padding=1)
        self.conv1b = Conv3d(n_deconvfilter[1], n_deconvfilter[1], 3, padding=1)
        
        self.conv2a = Conv3d(n_deconvfilter[1], n_deconvfilter[2], 3, padding=1)
        self.conv2b = Conv3d(n_deconvfilter[2], n_deconvfilter[2], 3, padding=1)
        
        self.conv3a = Conv3d(n_deconvfilter[2], n_deconvfilter[3], 3, padding=1)
        self.conv3b = Conv3d(n_deconvfilter[3], n_deconvfilter[3], 3, padding=1)
        self.conv3c = Conv3d(n_deconvfilter[2], n_deconvfilter[3], 1)
        
        self.conv4a = Conv3d(n_deconvfilter[3], n_deconvfilter[4], 3, padding=1)
        self.conv4b = Conv3d(n_deconvfilter[4], n_deconvfilter[4], 3, padding=1)
        self.conv4c = Conv3d(n_deconvfilter[4], n_deconvfilter[4], 3, padding=1)
        
        self.conv5 = Conv3d(n_deconvfilter[4], n_deconvfilter[5], 3, padding=1)

        self.leaky_relu = LeakyReLU(negative_slope= 0.01)
        
    def forward(self, feature):
        feature = feature.view(-1,self.n_deconvfilter[0],2,2,2)
        #(B, 64, 2, 2, 2)
        unpool_feature = torch.nn.functional.interpolate(feature, scale_factor=2)
        conv1 = self.conv1a(unpool_feature)
        rect1 = self.leaky_relu(conv1)
        conv1 = self.conv1b(rect1)
        rect1 = self.leaky_relu(conv1)
        res1 = unpool_feature + rect1

        #(B, 64, 4, 4, 4)
        if self.debug:
            print("res1", res1.shape)
        
        feature = torch.nn.functional.interpolate(res1, scale_factor=2)
        conv2 = self.conv2a(feature)
        rect2 = self.leaky_relu(conv2)
        conv2 = self.conv2b(rect2)
        rect2 = self.leaky_relu(conv2)
        res2 = feature + rect2

        #(B, 64, 8, 8, 8)
        if self.debug:
            print("res2", res2.shape)
        
        feature = torch.nn.functional.interpolate(res2, scale_factor=2)
        conv3 = self.conv3a(feature)
        rect3 = self.leaky_relu(conv3)
        conv3 = self.conv3b(rect3)
        rect3 = self.leaky_relu(conv3)
        conv_feature = self.conv3c(feature)
        
        res3 = conv_feature + rect3
        #(B, 32, 16, 16， 16)
        if self.debug:
            print("res3", res3.shape)
        
        feature = torch.nn.functional.interpolate(res3, scale_factor=2)
        conv4 = self.conv4a(feature)
        rect4 = self.leaky_relu(conv4)
        conv4 = self.conv4b(rect4)
        rect4 = self.leaky_relu(conv4)
        
        conv4 = self.conv4c(rect4)
        res4 = feature + conv4

        #(B, 32, 32, 32， 32)
        if self.debug:
            print("res4", res4.shape)
        
        conv5 = self.conv5(res4)

        #(B, 1, 32, 32， 32)
        if self.debug:
            print("conv5", conv5.shape)
        return conv5