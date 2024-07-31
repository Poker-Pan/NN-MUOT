from scipy import stats
import math, os, time, torch, copy, sys, random, inspect, psutil, gc

import pprint as pp
import matplotlib.pyplot as plt
import numpy as np
import torch.nn as nn
from matplotlib import cm
import scipy.io as scio
from mpl_toolkits.mplot3d import Axes3D
from torch.linalg import cholesky
import torch.distributions as td
from scipy.spatial import Delaunay




class Utilize(object):
    def __init__(self, Key_Para): 
        super(Utilize, self).__init__()
        self.Key_Para = Key_Para
        self.Dim_time = Key_Para['Dim_time']
        self.Dim_space = Key_Para['Dim_space']

    def make_file(self):
        root = os.getcwd()
        path = root + '/' + self.Key_Para['File_name']
        if not os.path.exists(path):
            os.makedirs(path)
            if self.Key_Para['type_print'] == 'True':
                pass
            elif self.Key_Para['type_print'] == 'False':
                sys.stdout = open(self.Key_Para['File_name'] + '/' + str(self.Key_Para['File_name']) + '-Code-Print.txt', 'w')
            else:
                print('There need code!')
            print('************' + str(self.Key_Para['File_name']) + '************')

    def print_key(self, keyword):
        print('************Key-Word************')
        pp.pprint(keyword)
        print('************************************')

    def setup_seed(self, seed):
        torch.manual_seed(seed)
        random.seed(seed)
        torch.cuda.manual_seed_all(seed)

    def mu_0_mu_1(self):
        mu_0, mu_1 = np.array([self.Key_Para['mu_0']]), np.array([self.Key_Para['mu_1']])
        self.Key_Para['mu_0'], self.Key_Para['mu_1'] = mu_0, mu_1

    def sigma_0_sigma_1(self):
        type_surface = self.Key_Para['type_surface']
        if type_surface == 'Cylinder' or type_surface == 'Sphere' or type_surface == 'Opener'  or type_surface == 'Cube'\
            or type_surface == 'Ellipsoid' or type_surface == 'Torus' or type_surface == 'Peanut' or type_surface == 'Hand' or type_surface == 'Airplane':
            sigma_0 = self.Key_Para['sigma_0']
            sigma_1 = self.Key_Para['sigma_1']
        elif type_surface == 'Mazes':
            sigma_0 = self.Key_Para['sigma_0'] * np.identity(self.Dim_space)
            sigma_1 = self.Key_Para['sigma_1'] * np.identity(self.Dim_space)

        self.Key_Para['sigma_0'], self.Key_Para['sigma_1'] = sigma_0, sigma_1

    def Time_Space(self):
        Time = np.array(self.Key_Para['Time']).reshape(1,-1)
        Space = np.array(self.Key_Para['Space']).reshape(1,-1).repeat(self.Dim_space, axis=0)
        
        self.Key_Para['Time'], self.Key_Para['Space'] = Time, Space

    def element(self):
        cur_element = np.prod(self.Key_Para['Space'][:, 1] - self.Key_Para['Space'][:, 0])
        self.Key_Para['element'] = cur_element


class Gnerate_node(object):
    def __init__(self, Key_Para):
        super(Gnerate_node, self).__init__()
        self.Key_Para = Key_Para
        self.Dim_time = Key_Para['Dim_time']
        self.Dim_space = Key_Para['Dim_space']
        self.Time = Key_Para['Time']
        self.Space = Key_Para['Space']
        self.Num_Nodes_t = Key_Para['Num_Nodes_t']
        self.Num_Nodes_all_space = Key_Para['Num_Nodes_all_space']
        self.type_node = Key_Para['type_node']

    def forward(self, type_train='True'):
        if self.type_node == 'Load':
            type_surface = self.Key_Para['type_surface']
            if type_surface == 'Cylinder':
                r = 0.25
                num_element = 1220
                file_name = type_surface + '_r_' + str(r) + '_n_' + str(num_element)

                load_data = scio.loadmat('./Moving-Surface-UOT/Data_Set/surface/' + file_name + '.mat')
                nodes_space = load_data['nodes']
                elements = load_data['triangles'].astype(int)
                normal = load_data['normal']
                self.Key_Para['r'] = r
                self.Key_Para['nodes_center'] = np.array([0.5, 0.5, 0.5])

            elif type_surface == 'Sphere':
                r = 0.5
                num_element = 1158
                file_name = type_surface + '_r_' + str(r) + '_n_' + str(num_element)

                load_data = scio.loadmat('./PINNs-UOT/Data_Set/surface/' + file_name + '.mat')
                nodes_space = load_data['nodes']
                elements = (load_data['triangles'] - 1).astype(int)
                normal = load_data['normal']
                self.Key_Para['r'] = r
                self.Key_Para['nodes_center'] = np.array([0.5, 0.5, 0.5])

            elif type_surface == 'Ellipsoid':
                r = 0.04
                num_element = 1222
                file_name = type_surface + '_r_' + str(r) + '_n_' + str(num_element)

                load_data = scio.loadmat('./Moving-Surface-UOT/Data_Set/surface/' + file_name + '.mat')
                nodes_space = load_data['nodes']
                elements = (load_data['triangles'] - 1).astype(int)
                normal = load_data['normal']
                self.Key_Para['r'] = r
                self.Key_Para['nodes_center'] = np.array([0.5, 0.5, 0.5])

            elif type_surface == 'Torus':
                r = 0.03
                num_element = 2120
                file_name = type_surface + '_r_' + str(r) + '_n_' + str(num_element)

                load_data = scio.loadmat('./Moving-Surface-UOT/Data_Set/surface/' + file_name + '.mat')
                nodes_space = load_data['nodes']
                elements = (load_data['triangles'] - 1).astype(int)
                normal = load_data['normal']
                self.Key_Para['r'] = r
                self.Key_Para['nodes_center'] = np.array([0.5, 0.5, 0.5])

            elif type_surface == 'Peanut':
                r = 0.03
                num_element = 1430
                file_name = type_surface + '_r_' + str(r) + '_n_' + str(num_element)

                load_data = scio.loadmat('./Moving-Surface-UOT/Data_Set/surface/' + file_name + '.mat')
                nodes_space = load_data['nodes']
                elements = (load_data['triangles'] - 1).astype(int)
                normal = load_data['normal']
                self.Key_Para['r'] = r
                self.Key_Para['nodes_center'] = np.array([0.5, 0.5, 0.5])

            elif type_surface == 'Opener':
                r = 0.03
                num_element = 1410 #1454
                file_name = type_surface + '_r_' + str(r) + '_n_' + str(num_element)

                load_data = scio.loadmat('./Moving-Surface-UOT/Data_Set/surface/' + file_name + '.mat')
                nodes_space = load_data['nodes']
                elements = (load_data['triangles'] - 1).astype(int)
                normal = load_data['normal']
                self.Key_Para['r'] = r
                self.Key_Para['nodes_center'] = np.array([0.5, 0.5, 0.5])

            elif type_surface == 'Cube':
                r = 0.075
                num_element = 1594
                file_name = type_surface + '_r_' + str(r) + '_n_' + str(num_element)

                load_data = scio.loadmat('./Moving-Surface-UOT/Data_Set/surface/' + file_name + '.mat')
                nodes_space = load_data['nodes']
                elements = (load_data['triangles'] - 1).astype(int)
                normal = load_data['normal']
                nodes_space = nodes_space
                self.Key_Para['r'] = r
                self.Key_Para['nodes_center'] = np.array([0.5, 0.5, 0.5])

            elif type_surface == 'Hand':
                num_element = 1515
                file_name = type_surface + '_n_' + str(num_element)

                load_data = scio.loadmat('./Moving-Surface-UOT/Data_Set/surface/' + file_name + '.mat')
                nodes_space = load_data['nodes']
                elements = load_data['triangles']
                normal = load_data['normal']

            elif type_surface == 'Airplane':
                num_element = 3772
                file_name = type_surface + '_n_' + str(num_element)

                load_data = scio.loadmat('./Moving-Surface-UOT/Data_Set/surface/' + file_name + '.mat')
                nodes_space = load_data['nodes']
                elements = load_data['triangles']
                normal = load_data['normal']

            elif type_surface == 'Mazes':
                num_element = 985
                file_name = type_surface + '_n_' + str(num_element)

                load_data = scio.loadmat('./Moving-Surface-UOT/Data_Set/surface/' + file_name + '.mat')
                nodes_space = load_data['nodes']
                elements = load_data['triangles']
                normal = load_data['normal']


            self.Key_Para['Num_Nodes_all_space'] = nodes_space.shape[0]          
            if type_train == 'True':
                # t = np.linspace(self.Time[0, 0], self.Time[0, 1], self.Num_Nodes_t + 15)
                # id_t = np.random.randint(1, self.Num_Nodes_t + 15 - 1, 8)
                # t = np.hstack((self.Time[0, 0], t[id_t], self.Time[0, 1]))
                # nodes = np.hstack((t.reshape(-1, 1).repeat(self.Key_Para['Num_Nodes_all_space'], axis=1).T.reshape(-1, 1), nodes_space.repeat(self.Num_Nodes_t, axis=0)))
                t = np.linspace(self.Time[0, 0], self.Time[0, 1], self.Num_Nodes_t)
                nodes = np.hstack((t.reshape(-1, 1).repeat(self.Key_Para['Num_Nodes_all_space'], axis=1).T.reshape(-1, 1), nodes_space.repeat(self.Num_Nodes_t, axis=0)))
            else:
                t = np.linspace(self.Time[0, 0], self.Time[0, 1], 10)
                nodes = np.hstack((t.reshape(-1, 1).repeat(self.Key_Para['Num_Nodes_all_space'], axis=1).T.reshape(-1, 1), nodes_space.repeat(10, axis=0)))


            nodes = torch.tensor(nodes, requires_grad=True, dtype=torch.float64).cuda()
            elements = torch.tensor(elements, requires_grad=True, dtype=torch.float64).cuda().type(torch.int64)
            normal = torch.tensor(normal, requires_grad=True, dtype=torch.float64).cuda()

            normal_x = (2*(nodes[:, 1] - self.Key_Para['nodes_center'][0])).reshape((-1, 1))
            normal_y = (2*(nodes[:, 2] - self.Key_Para['nodes_center'][1])).reshape((-1, 1))
            normal_z = (2*(nodes[:, 3] - self.Key_Para['nodes_center'][2])).reshape((-1, 1))
            normal = torch.cat((normal_x, normal_y, normal_z), dim=1)[::self.Key_Para['Num_Nodes_t'], :]

            def Add_noise(x):
                basic_var = torch.var(x, dim=0)
                add_var = (self.Key_Para['noise'] * basic_var).cpu().detach().numpy()
                for i in range(x.shape[1]):
                    noise = torch.normal(0, add_var[i], size=(x.shape[0], 1)).cuda()
                    x[:, i] = x[:, i] + noise.reshape(-1)
                return x

            nodes = Add_noise(nodes)
            normal = Add_noise(normal)
            nodes = torch.tensor(nodes, requires_grad=True, dtype=torch.float64).cuda()
            normal = torch.tensor(normal, requires_grad=True, dtype=torch.float64).cuda()
            self.Key_Para['nodes_normal'] = normal



            # nodes = nodes.detach().cpu().numpy()
            # nodes = nodes[::self.Key_Para['Num_Nodes_t'], :]
            # normal = normal.detach().cpu().numpy()
            # fig = plt.figure(figsize=(6, 6))
            # ax = fig.add_subplot(1, 1, 1, projection='3d')
            # ax.scatter(nodes[:, 1], nodes[:, 2], nodes[:, 3], c='r', s=0.5)
            # ax.quiver(nodes[:, 1], nodes[:, 2], nodes[:, 3], normal[:, 0], normal[:, 1], normal[:, 2], length=0.05, linewidths=0.5, normalize=True, color='b')
            # ax.view_init(elev=25, azim=45)
            # ax.set_xlim3d(0, 1), ax.set_ylim3d(0, 1), ax.set_zlim3d(0, 1)
            # ax.set_xticks([0.0, 0.5, 1.0]),  ax.set_yticks([0.0, 0.5, 1.0]), ax.set_zticks([0.0, 0.5, 1.0])
            # ax.set_xlabel('x'), ax.set_ylabel('y'), ax.set_zlabel('z')
            # ax.set_box_aspect([1, 1, 1])

            # ax.grid(None)
            # ax.xaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))
            # ax.yaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))
            # ax.zaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))
            # # ax.axis('off')

            # plt.savefig('nodes_normal_' + str(self.Key_Para['noise'])  + '.png', dpi=300)
            # plt.close()
            # w = 1








        elif self.type_node == 'Generate':
            print('There need code!')

        return nodes, elements

    def oprater_nodes(self, nodes, elements):
        pt = nodes[::self.Key_Para['Num_Nodes_t'], 1:]
        trg = elements

        npt = pt.shape[0]
        ntrg = trg.shape[0]
        normalVec = torch.zeros(ntrg, 3).cuda()
        trgCenter = torch.zeros(ntrg, 3).cuda()
        trgArea = torch.zeros(ntrg, 1).cuda()
        ptArea = torch.zeros(npt, 1).cuda()

        for i in range(trg.shape[0]):
            p1, p2, p3 = trg[i, 0], trg[i, 1], trg[i, 2]
            v1, v2, v3 = pt[p1, :], pt[p2, :], pt[p3, :]
            v12 = (v2 - v1).reshape(1, -1)
            v31 = (v1 - v3).reshape(1, -1)
            n = torch.cross(v12, -v31, axis=1)
            trgCenter[i, :] = torch.mean(torch.stack([v1, v2, v3]), dim=0)
            normalVec[i, :] = n / torch.norm(n)
            trgArea[i] = 1 / 2 * torch.norm(n)

            ptArea[p1] = ptArea[p1] + trgArea[i] / 3
            ptArea[p2] = ptArea[p2] + trgArea[i] / 3
            ptArea[p3] = ptArea[p3] + trgArea[i] / 3

        self.Key_Para['tri_normal'] = normalVec.detach()
        self.Key_Para['nodes_area'] = ptArea.detach()
        self.Key_Para['tri_area'] = trgArea.detach()
        self.Key_Para['tri_center'] = trgCenter.detach()


class Model_rho(nn.Module):
    def __init__(self, Key_Para): 
        super(Model_rho, self).__init__()
        self.Key_Para = Key_Para
        self.Dim_time = Key_Para['Dim_time']
        self.Dim_space = Key_Para['Dim_space']

        self.activation = nn.Tanh()
        self.num_layer = 5
        self.num_neuron = 200
        self.layer_size_rho = [self.Dim_time + self.Dim_space] + [self.num_neuron] * self.num_layer + [self.Dim_time]
        
        def set_model(layer_sizes, activation):
            net = nn.Sequential()
            net.add_module('LL_0', nn.Linear(layer_sizes[0], layer_sizes[1]))
            net.add_module('AL_0', activation)
            for i in range(1, len(layer_sizes)-2):
                net.add_module('LL_%d' % (i), nn.Linear(layer_sizes[i], layer_sizes[i + 1]))
                net.add_module('AL_%d' % (i), activation)
            net.add_module('LL_%d' % (len(layer_sizes)-2), nn.Linear(layer_sizes[-2], layer_sizes[-1]))

            return net

        self.net_rho = set_model(self.layer_size_rho, self.activation).cuda()

    def forward(self, nodes):
        cur_t = nodes[:, 0:1]
        cur_x = nodes[:, 1:]


        fun_relu = nn.Softplus()
        x1 = self.net_rho(nodes)
        out = fun_relu(x1)
        # torch.log(1 + torch.exp(x1))

        return out


class Model_phi(nn.Module):
    def __init__(self, Key_Para): 
        super(Model_phi, self).__init__()
        self.Key_Para = Key_Para
        self.Dim_time = Key_Para['Dim_time']
        self.Dim_space = Key_Para['Dim_space']

        self.activation = nn.Tanh()
        self.num_layer = 5
        self.num_neuron = 200
        self.layer_size_phi = [self.Dim_time + self.Dim_space] + [self.num_neuron] * self.num_layer + [self.Dim_time]
    
        def set_model(layer_sizes, activation):
            net = nn.Sequential()
            net.add_module('LL_0', nn.Linear(layer_sizes[0], layer_sizes[1]))
            net.add_module('AL_0', activation)
            for i in range(1, len(layer_sizes)-2):
                net.add_module('LL_%d' % (i), nn.Linear(layer_sizes[i], layer_sizes[i + 1]))
                net.add_module('AL_%d' % (i), activation)
            net.add_module('LL_%d' % (len(layer_sizes)-2), nn.Linear(layer_sizes[-2], layer_sizes[-1]))

            return net

        self.net_phi = set_model(self.layer_size_phi, self.activation).cuda()

    def forward(self, nodes):
        cur_t = nodes[:, 0:1]
        cur_x = nodes[:, 1:]
        out = self.net_phi(nodes)
        return out


class Model_g(nn.Module):
    def __init__(self, Key_Para): 
        super(Model_g, self).__init__()
        self.Key_Para = Key_Para
        self.Dim_time = Key_Para['Dim_time']
        self.Dim_space = Key_Para['Dim_space']

        self.activation = nn.Tanh()
        self.num_layer = 2
        self.num_neuron = 2
        self.layer_size_g = [self.Dim_time + self.Dim_space] + [self.num_neuron] * self.num_layer + [self.Dim_time]
    
        def set_model(layer_sizes, activation):
            net = nn.Sequential()
            net.add_module('LL_0', nn.Linear(layer_sizes[0], layer_sizes[1]))
            net.add_module('AL_0', activation)
            for i in range(1, len(layer_sizes)-2):
                net.add_module('LL_%d' % (i), nn.Linear(layer_sizes[i], layer_sizes[i + 1]))
                net.add_module('AL_%d' % (i), activation)
            net.add_module('LL_%d' % (len(layer_sizes)-2), nn.Linear(layer_sizes[-2], layer_sizes[-1]))

            for name, param in net.named_parameters():
                if 'weight' in name:
                    nn.init.xavier_normal_(param, gain=1.0)

            return net

        self.net_g = set_model(self.layer_size_g, self.activation).cuda()

    def forward(self, nodes):
        out = self.net_g(nodes)
        return self.Key_Para['type_UOT'] * out


def gradients(outputs, inputs):
    return torch.autograd.grad(outputs, inputs, grad_outputs=torch.ones_like(outputs), create_graph=True)


def Gaussian_distribution_torch(mu, sigma, x):
    mu = torch.tensor(mu, requires_grad=False, dtype=torch.float64).cuda()
    sigma = torch.tensor(sigma, requires_grad=False, dtype=torch.float64).cuda()

    x = x
    d = 2

    coefficient = (1 / torch.sqrt(((2 * np.pi)**d) * torch.det(sigma)))
    if Dim_space == 1:
        out = coefficient * (torch.diag(torch.exp(-0.5 * (x - mu) * (torch.linalg.inv(sigma) * (x - mu).T))))
    else:
        out = coefficient * (torch.diag(torch.exp(-0.5 * torch.mm((x - mu), torch.mm(torch.linalg.inv(sigma), (x - mu).T)))))
    out = out.reshape((-1, 1))
    return out


class Loss_Net(nn.Module):
    def __init__(self, model_rho, model_phi, model_g, Key_Para):
        super(Loss_Net, self).__init__()
        self.Key_Para = Key_Para
        self.model_rho = model_rho
        self.model_phi = model_phi
        self.model_g = model_g
        self.num_losses = len(Key_Para['beta'])
        self.params = torch.nn.Parameter(torch.ones(self.num_losses, requires_grad=True)) 

    def forward(self, nodes, elements):
        def get_variable_name(var):
            def retrieve_name(var):
                for fi in inspect.stack()[2:]:
                    for item in fi.frame.f_locals.items():
                        if var is item[1]:
                            return item[0]
                return ""
            return retrieve_name(var)

        def compute_change_P(self, nodes):
            nodes = nodes[:, 1:]

            type_surface = self.Key_Para['type_surface']
            # if type_surface == 'Sphere':
            #     normal_x = (2*(nodes[:, 0] - self.Key_Para['nodes_center'][0])).reshape((-1, 1))
            #     normal_y = (2*(nodes[:, 1] - self.Key_Para['nodes_center'][1])).reshape((-1, 1))
            #     normal_z = (2*(nodes[:, 2] - self.Key_Para['nodes_center'][2])).reshape((-1, 1))
            #     nodes_normal = torch.cat((normal_x, normal_y, normal_z), dim=1)
            # el
            if type_surface == 'Sphere' or type_surface == 'Cylinder' or type_surface == 'Opener' or type_surface == 'Hand' or type_surface == 'Airplane' or type_surface == 'Mazes' \
                or type_surface == 'Cube' or type_surface == 'Ellipsoid' or type_surface == 'Torus' or type_surface == 'Peanut':
                nodes_normal = self.Key_Para['nodes_normal']

            # norm = torch.divide(nodes_normal, torch.linalg.norm(nodes_normal, dim=1).repeat(3, 1).T)
            product = torch.bmm(nodes_normal.unsqueeze(2), nodes_normal.unsqueeze(1))
            Identity = torch.eye(3).unsqueeze(0).expand(nodes.shape[0], 3, 3).cuda()
            change_f_P = Identity - product
            
            return change_f_P, product

        def Compute_Gaussian_integral(self, nodes, elements, f):
            nodes = nodes[:, 1:]

            nodes_area = self.Key_Para['nodes_area'].detach()
            integral = torch.sum(nodes_area * f)
            return integral

        def W2(self, nodes, elements):
            out = 0
            for i in range(self.Key_Para['Num_Nodes_t']):
                cur_nodes = nodes[i::self.Key_Para['Num_Nodes_t'], :]
                cur_change_f_P, change_f_P_norm = compute_change_P(self, cur_nodes)

                cur_rho = self.model_rho(cur_nodes)
                cur_phi = self.model_phi(cur_nodes)

                cur_v = gradients(cur_phi, cur_nodes)[0][:, 1:]
                cur_v = torch.bmm(cur_change_f_P, cur_v.unsqueeze(-1)).squeeze(-1)
                cur_g = self.model_g(cur_nodes)

                f = torch.mul(cur_rho, torch.sum(torch.mul(cur_v, cur_v), dim=1).reshape(-1, 1)) + torch.mul(cur_rho, torch.mul(cur_g, cur_g))
                out = out + Compute_Gaussian_integral(self, cur_nodes, elements, f)

            out = 0.5 * out * ((self.Key_Para['Time'][0, 1] - self.Key_Para['Time'][0, 0]) / self.Key_Para['Num_Nodes_t']) 
            del cur_nodes, cur_change_f_P, cur_rho, cur_phi, cur_v, cur_g, f
            return out

        def Continuity_Equation(self, nodes, elements):
            out = 0
            res_c_eq = torch.zeros_like(nodes[:, 0:1]).cuda()
            # for i in range(self.Key_Para['Num_Nodes_t']):
            for i in range(1, self.Key_Para['Num_Nodes_t']-1):
                cur_nodes = nodes[i::self.Key_Para['Num_Nodes_t'], :]
                cur_change_f_P, _ = compute_change_P(self, cur_nodes)

                cur_rho = self.model_rho(cur_nodes)
                cur_grad_rho_t = gradients(cur_rho, cur_nodes)[0][:, 0:1]
                cur_phi = self.model_phi(cur_nodes)
                cur_v = gradients(cur_phi, cur_nodes)[0][:, 1:]
                cur_v = torch.bmm(cur_change_f_P, cur_v.unsqueeze(-1)).squeeze(-1)
                cur_g = self.model_g(cur_nodes)
                
                cur_rho_v = torch.mul(cur_rho.repeat(1, self.Key_Para['Dim_space']), cur_v)  
                div_cur_rho_v = 0
                for j in range(self.Key_Para['Dim_space']):
                    grad_cur_rho_v = gradients(cur_rho_v[:, j], cur_nodes)[0][:, 1:]
                    grad_cur_rho_v = torch.bmm(cur_change_f_P, grad_cur_rho_v.unsqueeze(-1)).squeeze(-1)
                    div_cur_rho_v = div_cur_rho_v + grad_cur_rho_v[:, j]
                cur_rho_g = torch.mul(cur_rho, cur_g)
                
                cur_out = cur_grad_rho_t + div_cur_rho_v.reshape(-1, 1) - cur_rho_g
                res_c_eq[i::self.Key_Para['Num_Nodes_t']] = cur_out
                out = out + torch.sum(torch.mul(cur_out, cur_out))

            out = out / (cur_nodes.shape[0]*(self.Key_Para['Num_Nodes_t'] - 2))
            # out = out / nodes.shape[0] 
            self.Key_Para['res_c_eq'] = res_c_eq
            del cur_nodes, cur_change_f_P, cur_rho, cur_grad_rho_t, cur_phi, cur_v, cur_g, cur_rho_v, div_cur_rho_v, cur_rho_g, cur_out
            return out

        def Hamilton_Jacobi_Equation(self, nodes, elements):
            out = 0
            res_HJ_eq = torch.zeros_like(nodes[:, 0:1]).cuda()
            for i in range(self.Key_Para['Num_Nodes_t']):
                cur_nodes = nodes[i::self.Key_Para['Num_Nodes_t'], :]
                cur_change_f_P, _ = compute_change_P(self, cur_nodes)
            
                cur_phi = self.model_phi(cur_nodes)
                grad_cur_phi = gradients(cur_phi, cur_nodes)[0]
                grad_cur_phi_space = torch.bmm(cur_change_f_P, grad_cur_phi[:, 1:].unsqueeze(-1)).squeeze(-1)

                cur_out = grad_cur_phi[:, 0].reshape(-1, 1) + 0.5*torch.sum(torch.mul(grad_cur_phi_space, grad_cur_phi_space), dim=1).reshape(-1, 1)
                res_HJ_eq[i::self.Key_Para['Num_Nodes_t']] = cur_out.reshape(-1, 1)
                out = out + torch.sum(torch.mul(cur_out, cur_out))

            out = out / nodes.shape[0]
            self.Key_Para['res_HJ_eq'] = res_HJ_eq 
            del cur_nodes, cur_change_f_P, cur_phi, grad_cur_phi, grad_cur_phi_space, cur_out, res_HJ_eq
            return out

        def rho_boundary(self, nodes, elements):
            def initial_end_data(nodes, elements):
                def cut_off(x, sigma):
                    x /= sigma
                    if x <= 0.:
                        return 1.
                    elif x >= 1.:
                        return 0.
                    else:
                        return (x - 1.) ** 2 * (x + 1) ** 2

                if self.Key_Para['type_node'] == 'Load':
                    type_surface = self.Key_Para['type_surface']
                    if type_surface == 'Cylinder':
                        def Cylinder_Surface_Gaussian(surface_nodes, surface_triangles, center, concentration):      
                            center = torch.tensor(center, requires_grad=False, dtype=torch.float64).cuda().reshape(-1)
                            concentration = torch.tensor(concentration, requires_grad=False, dtype=torch.float64).cuda()
                            surface_nodes = surface_nodes[:, 1:]

                            concentration = 1 / concentration
                            coefficient = 100
                            von_mises_fisher = coefficient * torch.exp(concentration * -(torch.norm(surface_nodes - center, dim=1)**2))
                            return von_mises_fisher.reshape((-1, 1))

                        rho_initial = Cylinder_Surface_Gaussian(nodes, elements, self.Key_Para['mu_0'], 1*self.Key_Para['sigma_0'])
                        rho_end = Cylinder_Surface_Gaussian(nodes, elements, self.Key_Para['mu_1'], self.Key_Para['sigma_1'])

                        # nodes = nodes.detach().cpu().numpy()
                        # rho_initial = rho_initial.detach().cpu().numpy()
                        # rho_end = rho_end.detach().cpu().numpy()
                        # fig = plt.figure(figsize=(int(6*6), 10))
                        # for i in range(1, 7):
                        #     ax = fig.add_subplot(2, 6, i, projection='3d')
                        #     im = ax.scatter(nodes[:, 1], nodes[:, 2], nodes[:,3], c=rho_initial.reshape(-1))
                        #     ax.view_init(elev=60*i, azim=60)
                        #     # ax.set_xlim3d(0, 1), ax.set_ylim3d(0, 1), ax.set_zlim3d(-0.1, 0.1)
                        #     # ax.set_xticks([]),  ax.set_yticks([]), ax.set_zticks([])
                        #     ax.set_box_aspect([1, 1, 2])
                        #     plt.colorbar(im)

                        #     ax = fig.add_subplot(2, 6, i+6, projection='3d')
                        #     im = ax.scatter(nodes[:, 1], nodes[:, 2], nodes[:, 3], c=rho_end.reshape(-1))
                        #     ax.view_init(elev=60*i, azim=60)
                        #     # ax.set_xlim3d(0, 1), ax.set_ylim3d(0, 1), ax.set_zlim3d(-0.1, 0.1)
                        #     # ax.set_xticks([]),  ax.set_yticks([]), ax.set_zticks([])
                        #     ax.set_box_aspect([1, 1, 2])
                        #     plt.colorbar(im)

                        # plt.savefig('rho.png')
                        # plt.close()

                        # w = 1

                    elif type_surface == 'Sphere':
                        def Sphere_Surface_Gaussian(surface_nodes, surface_triangles, center, concentration):      
                            center = torch.tensor(center, requires_grad=False, dtype=torch.float64).cuda().reshape(-1)
                            concentration = torch.tensor(concentration, requires_grad=False, dtype=torch.float64).cuda()
                            surface_nodes = surface_nodes[:, 1:]

                            concentration = 1 / concentration
                            coefficient = 100
                            von_mises_fisher = coefficient * torch.exp(concentration * -(torch.norm(surface_nodes - center, dim=1)**2))
                            return von_mises_fisher.reshape((-1, 1))
                        
                        
                        rho_end = Sphere_Surface_Gaussian(nodes, elements, self.Key_Para['mu_0'], self.Key_Para['sigma_0'])
                        # rho_end = rho_end + Sphere_Surface_Gaussian(nodes, elements, np.array([0.0, 0.5, 0.5]), self.Key_Para['sigma_0'])
                        # rho_end = rho_end + Sphere_Surface_Gaussian(nodes, elements, np.array([0.5, 0.0, 0.5]), self.Key_Para['sigma_0'])
                        # rho_end = rho_end + Sphere_Surface_Gaussian(nodes, elements, np.array([0.5, 1.0, 0.5]), self.Key_Para['sigma_0'])
                        rho_initial = Sphere_Surface_Gaussian(nodes, elements, self.Key_Para['mu_1'], self.Key_Para['sigma_1'])
                        
                        # rho_initial = rho_initial / torch.sum(rho_initial*self.Key_Para['nodes_area'])
                        # rho_end = rho_end / torch.sum(rho_end*self.Key_Para['nodes_area'])

                    elif type_surface == 'Ellipsoid':
                        def Ellipsoid_Surface_Gaussian(surface_nodes, surface_triangles, center, concentration):      
                            center = torch.tensor(center, requires_grad=False, dtype=torch.float64).cuda().reshape(-1)
                            concentration = torch.tensor(concentration, requires_grad=False, dtype=torch.float64).cuda()
                            surface_nodes = surface_nodes[:, 1:]

                            concentration = 1 / concentration
                            coefficient = 100
                            von_mises_fisher = coefficient * torch.exp(concentration * -(torch.norm(surface_nodes - center, dim=1)**2))
                            return von_mises_fisher.reshape((-1, 1))


                        rho_initial = Ellipsoid_Surface_Gaussian(nodes, elements, self.Key_Para['mu_0'], 1*self.Key_Para['sigma_0'])
                        rho_end = Ellipsoid_Surface_Gaussian(nodes, elements, self.Key_Para['mu_1'], self.Key_Para['sigma_1'])

                    elif type_surface == 'Torus':
                        def Torus_Surface_Gaussian(surface_nodes, surface_triangles, center, concentration):      
                            center = torch.tensor(center, requires_grad=False, dtype=torch.float64).cuda().reshape(-1)
                            concentration = torch.tensor(concentration, requires_grad=False, dtype=torch.float64).cuda()
                            surface_nodes = surface_nodes[:, 1:]

                            concentration = 1 / concentration
                            coefficient = 100
                            von_mises_fisher = coefficient * torch.exp(concentration * -(torch.norm(surface_nodes - center, dim=1)**2))
                            return von_mises_fisher.reshape((-1, 1))


                        rho_initial = Torus_Surface_Gaussian(nodes, elements, self.Key_Para['mu_0'], 1*self.Key_Para['sigma_0'])
                        rho_end = Torus_Surface_Gaussian(nodes, elements, self.Key_Para['mu_1'], self.Key_Para['sigma_1'])
                        rho_initial = rho_initial / torch.sum(rho_initial*self.Key_Para['nodes_area'])
                        rho_end = rho_end / torch.sum(rho_end*self.Key_Para['nodes_area'])

                        # nodes = nodes.detach().cpu().numpy()
                        # rho_initial = rho_initial.detach().cpu().numpy()
                        # rho_end = rho_end.detach().cpu().numpy()
                        # fig = plt.figure(figsize=(int(6*6), 10))
                        # for i in range(1, 7):
                        #     ax = fig.add_subplot(2, 6, i, projection='3d')
                        #     im = ax.scatter(nodes[:, 1], nodes[:, 2], nodes[:,3], c=rho_initial.reshape(-1))
                        #     ax.view_init(elev=60*i, azim=60)
                        #     # ax.set_xlim3d(0, 1), ax.set_ylim3d(0, 1), ax.set_zlim3d(-0.1, 0.1)
                        #     ax.set_xticks([]),  ax.set_yticks([]), ax.set_zticks([])
                        #     ax.set_box_aspect([10, 10, 2])
                        #     plt.colorbar(im)

                        #     ax = fig.add_subplot(2, 6, i+6, projection='3d')
                        #     im = ax.scatter(nodes[:, 1], nodes[:, 2], nodes[:, 3], c=rho_end.reshape(-1))
                        #     ax.view_init(elev=60*i, azim=60)
                        #     # ax.set_xlim3d(0, 1), ax.set_ylim3d(0, 1), ax.set_zlim3d(-0.1, 0.1)
                        #     ax.set_xticks([]),  ax.set_yticks([]), ax.set_zticks([])
                        #     ax.set_box_aspect([10, 10, 2])
                        #     plt.colorbar(im)

                        # plt.savefig('rho.png')
                        # plt.close()

                        # w = 1

                    elif type_surface == 'Peanut':
                        def Peanut_Surface_Gaussian(surface_nodes, surface_triangles, center, concentration):      
                            center = torch.tensor(center, requires_grad=False, dtype=torch.float64).cuda().reshape(-1)
                            concentration = torch.tensor(concentration, requires_grad=False, dtype=torch.float64).cuda()
                            surface_nodes = surface_nodes[:, 1:]

                            concentration = 1 / concentration
                            coefficient = 100
                            von_mises_fisher = coefficient * torch.exp(concentration * -(torch.norm(surface_nodes - center, dim=1)**2))
                            return von_mises_fisher.reshape((-1, 1))


                        rho_initial = Peanut_Surface_Gaussian(nodes, elements, self.Key_Para['mu_0'], 1*self.Key_Para['sigma_0'])
                        rho_end = Peanut_Surface_Gaussian(nodes, elements, self.Key_Para['mu_1'], self.Key_Para['sigma_1'])

                    elif type_surface == 'Opener':
                        def Opener_Surface_Gaussian(surface_nodes, surface_triangles, center, concentration):      
                            center = torch.tensor(center, requires_grad=False, dtype=torch.float64).cuda().reshape(-1)
                            concentration = torch.tensor(concentration, requires_grad=False, dtype=torch.float64).cuda()
                            surface_nodes = surface_nodes[:, 1:]

                            concentration = 1 / concentration
                            coefficient = 100
                            von_mises_fisher = coefficient * torch.exp(concentration * -(torch.norm(surface_nodes - center, dim=1)**2))
                            return von_mises_fisher.reshape((-1, 1))
                        

                        rho_initial = Opener_Surface_Gaussian(nodes, elements, self.Key_Para['mu_0'], 1*self.Key_Para['sigma_0'])
                        rho_end = Opener_Surface_Gaussian(nodes, elements, self.Key_Para['mu_1'], self.Key_Para['sigma_1'])

                        # nodes = nodes.detach().cpu().numpy()
                        # rho_initial = rho_initial.detach().cpu().numpy()
                        # rho_end = rho_end.detach().cpu().numpy()
                        # fig = plt.figure(figsize=(int(6*6), 10))
                        # for i in range(1, 7):
                        #     ax = fig.add_subplot(2, 6, i, projection='3d')
                        #     im = ax.scatter(nodes[:, 1], nodes[:, 2], nodes[:,3], c=rho_initial.reshape(-1))
                        #     ax.view_init(elev=0, azim=60*i)
                        #     # ax.set_xlabel('x'), ax.set_ylabel('y'), ax.set_zlabel('z')
                        #     # ax.set_xlim3d(0, 1), ax.set_ylim3d(0, 1), ax.set_zlim3d(-0.1, 0.1)
                        #     ax.set_xticks([]),  ax.set_yticks([]), ax.set_zticks([])
                        #     ax.set_box_aspect([10, 8, 2])
                        #     plt.colorbar(im)

                        #     ax = fig.add_subplot(2, 6, i+6, projection='3d')
                        #     im = ax.scatter(nodes[:, 1], nodes[:, 2], nodes[:, 3], c=rho_end.reshape(-1))
                        #     ax.view_init(elev=0, azim=60*i)
                        #     # ax.set_xlim3d(0, 1), ax.set_ylim3d(0, 1), ax.set_zlim3d(-0.1, 0.1)
                        #     ax.set_xticks([]),  ax.set_yticks([]), ax.set_zticks([])
                        #     ax.set_box_aspect([10, 8, 2])
                        #     plt.colorbar(im)

                        # plt.savefig('rho.png')
                        # plt.close()

                        # w = 1

                    elif type_surface == 'Cube':
                        def Cube_Surface_Gaussian(surface_nodes, surface_triangles, center, concentration):      
                            center = torch.tensor(center, requires_grad=False, dtype=torch.float64).cuda().reshape(-1)
                            concentration = torch.tensor(concentration, requires_grad=False, dtype=torch.float64).cuda()
                            surface_nodes = surface_nodes[:, 1:]

                            concentration = 1 / concentration
                            coefficient = 100
                            von_mises_fisher = coefficient * torch.exp(concentration * -(torch.norm(surface_nodes - center, dim=1)**2))
                            return von_mises_fisher.reshape((-1, 1))
                        
                        rho_initial = Cube_Surface_Gaussian(nodes, elements, self.Key_Para['mu_0'], 1*self.Key_Para['sigma_0'])
                        rho_end = Cube_Surface_Gaussian(nodes, elements, self.Key_Para['mu_1'], self.Key_Para['sigma_1'])

                    elif type_surface == 'Hand':
                        def Hand_Surface_Gaussian(surface_nodes, surface_triangles, center, concentration):      
                            concentration = torch.tensor(concentration, requires_grad=False, dtype=torch.float64).cuda()
                            surface_nodes = surface_nodes[:, 1:]

                            concentration = 1 / concentration
                            coefficient = 100
                            von_mises_fisher = coefficient * torch.exp(concentration * -(torch.norm(surface_nodes - center, dim=1)**2))
                            return von_mises_fisher.reshape((-1, 1))
                        

                        mu_0, mu_1 = nodes[912, 1:], nodes[656, 1:]
                        rho_initial = Hand_Surface_Gaussian(nodes, elements, mu_0, self.Key_Para['sigma_0'])
                        rho_end = Hand_Surface_Gaussian(nodes, elements, mu_1, self.Key_Para['sigma_1'])
                        rho_initial = rho_initial / torch.sum(rho_initial*self.Key_Para['nodes_area'])
                        rho_end = rho_end / torch.sum(rho_end*self.Key_Para['nodes_area'])

                        # file = open('./Moving-Surface-UOT/Data_Set/surface/hand_data_mu0.txt', 'r')
                        # rho_initial = file.read()
                        # rho_initial = rho_initial.split('\n')
                        # rho_initial = [line.strip() for line in rho_initial if line.strip()]
                        # rho_initial = np.array([float(line) for line in rho_initial])
                        # rho_initial = torch.tensor(rho_initial, requires_grad=False, dtype=torch.float64).cuda().reshape(-1, 1)

                        # file = open('./Moving-Surface-UOT/Data_Set/surface/hand_data_mu1.txt', 'r')
                        # rho_end = file.read()
                        # rho_end = rho_end.split('\n')
                        # rho_end = [line.strip() for line in rho_end if line.strip()]
                        # rho_end = np.array([float(line) for line in rho_end])
                        # rho_end = torch.tensor(rho_end, requires_grad=False, dtype=torch.float64).cuda().reshape(-1, 1)

                    elif type_surface == 'Airplane':
                        def Airplane_Surface_Gaussian(surface_nodes, surface_triangles, center, concentration):      
                            concentration = torch.tensor(concentration, requires_grad=False, dtype=torch.float64).cuda()
                            surface_nodes = surface_nodes[:, 1:]

                            concentration = 1 / concentration
                            coefficient = 100
                            von_mises_fisher = coefficient * torch.exp(concentration * -(torch.norm(surface_nodes - center, dim=1)**2))
                            return von_mises_fisher.reshape((-1, 1))
                        

                        mu_0, mu_1 = nodes[89, 1:], nodes[3526, 1:]
                        rho_initial = Airplane_Surface_Gaussian(nodes, elements, mu_0, self.Key_Para['sigma_0'])
                        rho_end = Airplane_Surface_Gaussian(nodes, elements, mu_1, self.Key_Para['sigma_1'])
                        rho_initial = rho_initial / torch.sum(rho_initial*self.Key_Para['nodes_area'])
                        rho_end = rho_end / torch.sum(rho_end*self.Key_Para['nodes_area'])

                        # rho_initial  = torch.zeros((nodes.shape[0], 1)).requires_grad_(True).cuda()
                        # rho_end  = torch.zeros((nodes.shape[0], 1)).requires_grad_(True).cuda()
                        # for i in range(nodes.shape[0]):
                        #     rho_initial[i] = cut_off(-(nodes[i, 3] - 0.5), 0.3)
                        #     rho_end[i] = cut_off(nodes[i, 3] + 0.1, 0.3)

                        # nodes = nodes.detach().cpu().numpy()
                        # rho_initial = rho_initial.detach().cpu().numpy()
                        # rho_end = rho_end.detach().cpu().numpy()
                        # fig = plt.figure(figsize=(int(6*6), 10))
                        # for i in range(1, 7):
                        #     ax = fig.add_subplot(2, 6, i, projection='3d')
                        #     im = ax.scatter(nodes[:, 1], nodes[:, 2], nodes[:,3], c=rho_initial.reshape(-1))
                        #     ax.view_init(elev=60*i, azim=60)
                        #     ax.set_xlim3d(-1, 1), ax.set_ylim3d(-0.5, 0.5), ax.set_zlim3d(-1, 1)
                        #     ax.set_xticks([]),  ax.set_yticks([]), ax.set_zticks([])
                        #     ax.set_box_aspect([2, 1, 2])
                        #     plt.colorbar(im)

                        #     ax = fig.add_subplot(2, 6, i+6, projection='3d')
                        #     im = ax.scatter(nodes[:, 1], nodes[:, 2], nodes[:, 3], c=rho_end.reshape(-1))
                        #     ax.view_init(elev=60*i, azim=60)
                        #     ax.set_xlim3d(-1, 1), ax.set_ylim3d(-0.5, 0.5), ax.set_zlim3d(-1, 1)
                        #     ax.set_xticks([]),  ax.set_yticks([]), ax.set_zticks([])
                        #     ax.set_box_aspect([2, 1, 2])
                        #     plt.colorbar(im)

                        # plt.savefig('rho.png')
                        # plt.close()

                        # w = 1

                    elif type_surface == 'Mazes':
                        def Sphere_Mazes_Gaussian(x, elements, mu, sigma):
                            mu = torch.tensor(mu, requires_grad=False, dtype=torch.float64).cuda()
                            sigma = torch.tensor(sigma, requires_grad=False, dtype=torch.float64).cuda()

                            x = x[:, 1:]
                            d = 3

                            coefficient = (1 / torch.sqrt(((2 * np.pi)**d) * torch.det(sigma)))
                            if Dim_space == 1:
                                out = coefficient * (torch.diag(torch.exp(-0.5 * (x - mu) * (torch.linalg.inv(sigma) * (x - mu).T))))
                            else:
                                out = coefficient * (torch.diag(torch.exp(-0.5 * torch.mm((x - mu), torch.mm(torch.linalg.inv(sigma), (x - mu).T)))))
                            out = out.reshape((-1, 1))
                            return out


                        rho_initial = Sphere_Mazes_Gaussian(nodes, elements, self.Key_Para['mu_0'], 1*self.Key_Para['sigma_0'])
                        rho_end = Sphere_Mazes_Gaussian(nodes, elements, self.Key_Para['mu_1'], self.Key_Para['sigma_1'])

                        # nodes = nodes.detach().cpu().numpy()
                        # rho_initial = rho_initial.detach().cpu().numpy()
                        # rho_end = rho_end.detach().cpu().numpy()
                        # fig = plt.figure(figsize=(int(6*6), 10))
                        # for i in range(1, 7):
                        #     ax = fig.add_subplot(2, 6, i, projection='3d')
                        #     im = ax.scatter(nodes[:, 1], nodes[:, 2], nodes[:,3], c=rho_initial.reshape(-1))
                        #     ax.view_init(elev=90, azim=45)
                        #     ax.set_xlim3d(0, 1), ax.set_ylim3d(0, 1), ax.set_zlim3d(-0.1, 0.1)
                        #     # ax.set_xticks([]),  ax.set_yticks([]), ax.set_zticks([])
                        #     # ax.set_box_aspect([2, 1, 2])
                        #     plt.colorbar(im)

                        #     ax = fig.add_subplot(2, 6, i+6, projection='3d')
                        #     im = ax.scatter(nodes[:, 1], nodes[:, 2], nodes[:, 3], c=rho_end.reshape(-1))
                        #     ax.view_init(elev=90, azim=45)
                        #     ax.set_xlim3d(0, 1), ax.set_ylim3d(0, 1), ax.set_zlim3d(-0.1, 0.1)
                        #     # ax.set_xticks([]),  ax.set_yticks([]), ax.set_zticks([])
                        #     # ax.set_box_aspect([2, 1, 2])
                        #     plt.colorbar(im)

                        # plt.savefig('rho.png')
                        # plt.close()

                        # w = 1

                elif self.Key_Para['type_node'] == 'Generate':
                    print('There need code!')

                return rho_initial, rho_end


            rho_initial, rho_end = initial_end_data(nodes[::self.Key_Para['Num_Nodes_t'],:], elements)
            cur_rho_initial = self.model_rho(nodes[::self.Key_Para['Num_Nodes_t'],:])
            cur_rho_end = self.model_rho(nodes[int(self.Key_Para['Num_Nodes_t']-1)::self.Key_Para['Num_Nodes_t'],:])

            # abs_rho = Compute_Gaussian_integral(self, nodes[0::self.Key_Para['Num_Nodes_t'], :], elements, rho_initial) - Compute_Gaussian_integral(self, nodes[0::self.Key_Para['Num_Nodes_t'], :], elements, rho_end)
            # change_rho = ((abs_rho/self.Key_Para['nodes_area'])/cur_rho_end.shape[0])
            # rho_end = rho_end + change_rho
            # new_abs_rho = Compute_Gaussian_integral(self, nodes[0::self.Key_Para['Num_Nodes_t'], :], elements, rho_initial) - Compute_Gaussian_integral(self, nodes[0::self.Key_Para['Num_Nodes_t'], :], elements, rho_end)

            initial = rho_initial - cur_rho_initial
            end = rho_end - cur_rho_end


            out = torch.mean(torch.mul(initial, initial)) + torch.mean(torch.mul(end, end))
            del rho_initial, rho_end, cur_rho_initial, cur_rho_end, initial, end
            return out

        def L2_v_penalty(self, nodes, elements):
            out = 0
            for i in range(self.Key_Para['Num_Nodes_t']):
                cur_nodes = nodes[i::self.Key_Para['Num_Nodes_t'], :]
                cur_change_f_P, change_f_P_norm = compute_change_P(self, cur_nodes)

                cur_phi = self.model_phi(cur_nodes)
                cur_v = gradients(cur_phi, cur_nodes)[0][:, 1:]
                cur_v = torch.bmm(change_f_P_norm, cur_v.unsqueeze(-1)).squeeze(-1)

                cur_out = torch.sum(torch.mul(cur_v, cur_v), dim=1).reshape(-1, 1)
                out = out + torch.sum(torch.mul(cur_out, cur_out))

            out = out / nodes.shape[0] 
            del cur_nodes, cur_change_f_P, cur_phi, cur_v, cur_out
            return out

        def Integral_rho(self, nodes, elements):
            out = 0
            int_rho = []
            for i in range(self.Key_Para['Num_Nodes_t']):
                cur_nodes = nodes[i::self.Key_Para['Num_Nodes_t'], :]
                cur_rho = self.model_rho(cur_nodes)

                int_rho.append(Compute_Gaussian_integral(self, cur_nodes, elements, cur_rho).cpu().detach().tolist())
                out = out + torch.pow(torch.sum(cur_rho*self.Key_Para['nodes_area']) - 1, 2)

            self.Key_Para['int_rho'] = int_rho
            del cur_nodes, cur_rho, int_rho
            return out

        loss_W2 = W2(self, nodes, elements)
        loss_C_eq = Continuity_Equation(self, nodes, elements)
        loss_HJ_eq  = Hamilton_Jacobi_Equation(self, nodes, elements)
        loss_rho_bc = rho_boundary(self, nodes, elements)
        loss_v_L2 = L2_v_penalty(self, nodes, elements)
        loss_rho_int = Integral_rho(self, nodes, elements)


        if self.Key_Para['type_pre_train'] == 'True' and self.Key_Para['ep_t'] <= 5000:
            self.Key_Para['beta'] = [0, self.Key_Para['beta'][1] + 500/5000, self.Key_Para['beta'][2] + 500/5000, self.Key_Para['beta'][3] - 500/5000, 0, 0]


        if self.Key_Para['type_loss'] == 'Res-Attention':
            self.Key_Para['lambda'] = torch.zeros((nodes.shape[0], 1)).cuda()
            
            # update lambda
            new_lambda = 0.999 * self.Key_Para['lambda'] + 0.01 * (torch.abs(self.Key_Para['res_c_eq']) / torch.max(torch.abs(self.Key_Para['res_c_eq'])))
            self.Key_Para['lambda'] = new_lambda
            loss_Res_attention = torch.sum(torch.mul(self.Key_Para['lambda']*self.Key_Para['res_c_eq'], self.Key_Para['lambda']*self.Key_Para['res_c_eq']))

            loss = self.Key_Para['beta'][0]*loss_W2 + self.Key_Para['beta'][1]*loss_C_eq + self.Key_Para['beta'][2]*loss_HJ_eq + self.Key_Para['beta'][3]*loss_rho_bc + self.Key_Para['beta'][4]*loss_v_L2 + self.Key_Para['beta'][5]*loss_rho_int + loss_Res_attention
            sub_loss = [loss_W2, loss_C_eq, loss_HJ_eq, loss_rho_bc, loss_v_L2, loss_rho_int, loss_Res_attention]
        elif self.Key_Para['type_loss'] == 'General':
            loss = self.Key_Para['beta'][0]*loss_W2 + self.Key_Para['beta'][1]*loss_C_eq + self.Key_Para['beta'][2]*loss_HJ_eq + self.Key_Para['beta'][3]*loss_rho_bc + self.Key_Para['beta'][4]*loss_v_L2 + self.Key_Para['beta'][5]*loss_rho_int
            sub_loss = [loss_W2, loss_C_eq, loss_HJ_eq, loss_rho_bc, loss_v_L2, loss_rho_int]
        
        if self.Key_Para['ep_t'] == 0:
            temp = []
            for i in range(len(sub_loss)):
                temp.append(get_variable_name(sub_loss[i]))
            self.Key_Para['loss_name'] = temp        
        return loss, sub_loss


class Train_net(object):
    def __init__(self, Key_Para, gen_Nodes, model_rho, model_phi, model_g, loss_net, Plot_Result):
        super(Train_net, self).__init__()
        self.Key_Para = Key_Para
        self.gen_Nodes = gen_Nodes
        self.model_rho = model_rho
        self.model_phi = model_phi
        self.model_g = model_g
        self.loss_net = loss_net
        self.Plot_Result = Plot_Result
        self.optimizer = torch.optim.Adam(self.loss_net.parameters(), lr=self.Key_Para['learning_rate'])


    def train(self):
        torch.cuda.empty_cache()
        all_sub_loss = []
        for ep_s in range(self.Key_Para['epochs_sample']):
            print('Sampling point: ', ep_s)
            nodes, elements = self.gen_Nodes.forward()
            self.gen_Nodes.oprater_nodes(nodes, elements)
            for ep_t in range(self.Key_Para['epochs_train']):  

                self.Key_Para['ep_t'] = ep_t + ep_s*self.Key_Para['epochs_train']
                loss, sub_loss = self.loss_net(nodes, elements)
                loss.backward()
                self.optimizer.step()
                all_sub_loss.append(sub_loss)
                # print(u'Use: %.4f GB' % (psutil.Process(os.getpid()).memory_info().rss / 1024 / 1024 / 1024) )
                w = 1


                if self.Key_Para['type_lr'] == 'Gradual-Decline':
                    if ep_t % 500 == 0:
                        for p in self.optimizer.param_groups:
                            p['lr'] = p['lr']*0.95

                if self.Key_Para['type_pre_plot'] == 'True':
                    if ep_t % 100 == 0:
                        self.Plot_Result.plot_rho(self.model_rho, self.Key_Para['ep_t'])
                        w = 1
                    if ep_t % 500 == 0:
                        # self.Plot_Result.plot_v(self.model_phi, ep_t)
                        # self.Plot_Result.plot_g(self.model_g, ep_t)
                        # self.Plot_Result.plot_c_eq(ep_t)
                        # print(self.Key_Para['int_rho'])
                        # self.Plot_Result.plot_hj_eq(ep_t)
                        w = 1   

                if self.Key_Para['type_loss'] == 'Res-Attention':
                    if ep_t % 100 == 0:
                        print('ep:', '%d' %ep_t,
                            'loss:',"%.6f " %loss,
                            '[loss_W2:',"%.6f" %sub_loss[0], 
                            'loss_C_eq:',"%.6f" %sub_loss[1],
                            'loss_HJ_eq:',"%.6f" %sub_loss[2],
                            'loss_rho_bc:',"%.6f" %sub_loss[3],
                            'loss_v_L2:',"%.6f" %sub_loss[4],
                            'loss_rho_int:',"%.6f" %sub_loss[5],
                            'loss_Res_attention:',"%.6f]" %sub_loss[6],
                            )
                else:    
                    if ep_t % 100 == 0:
                        print('ep:', '%d' %self.Key_Para['ep_t'],
                            'loss:',"%.6f " %loss,
                            '[loss_W2:',"%.6f" %sub_loss[0], 
                            'loss_C_eq:',"%.6f" %sub_loss[1],
                            'loss_HJ_eq:',"%.6f" %sub_loss[2],
                            'loss_rho_bc:',"%.6f" %sub_loss[3],
                            'loss_v_L2:',"%.6f" %sub_loss[4],
                            'loss_rho_int:',"%.6f]" %sub_loss[5],
                            )

                if sub_loss[1]<=self.Key_Para['break_delta'] and sub_loss[2]<=self.Key_Para['break_delta'] and sub_loss[3]<=self.Key_Para['break_delta']:
                    self.Plot_Result.plot_rho(self.model_rho, self.Key_Para['ep_t'])
                    break

                self.optimizer.zero_grad()
                loss.zero_()
                del loss
                torch.cuda.empty_cache()
                gc.collect()

        torch.cuda.empty_cache()
        torch.save(self.model_rho, self.Key_Para['File_name'] + '/model_rho.pth')
        torch.save(self.model_phi, self.Key_Para['File_name'] + '/model_phi.pth')
        torch.save(self.model_g, self.Key_Para['File_name'] + '/model_g.pth')
        return all_sub_loss


class Compute_Wasserstein_distance(object):
    def __init__(self, Key_Para, all_sub_loss):
        super(Compute_Wasserstein_distance, self).__init__()
        self.Key_Para = Key_Para
        self.all_sub_loss = all_sub_loss

    def compute(self):
        Pre_W_dis = self.all_sub_loss[-1][0].cpu().detach().numpy()
        print('Pre Wasserstein distance: ', Pre_W_dis)


class Plot_Result(object):
    def __init__(self, Key_Para, gen_Nodes):
        super(Plot_Result, self).__init__()
        self.Key_Para = Key_Para
        self.gen_Nodes = gen_Nodes
        self.Dim_time = Key_Para['Dim_time']
        self.Dim_space = Key_Para['Dim_space']
        self.Time = Key_Para['Time']
        self.Space = Key_Para['Space']

    def plot_rho(self, model_rho, ite=''):
        if self.Dim_space == 3:
            type_surface = self.Key_Para['type_surface']
            if type_surface == 'Cylinder':
                r = self.Key_Para['r']
                T = int(self.Key_Para['Num_Nodes_t'])
                nodes, elements = self.gen_Nodes.forward()

                t = np.linspace(self.Time[0, 0], self.Time[0, 1], T)
                nodes = nodes.cpu().detach().numpy()
                nodes_space = nodes[0::int(self.Key_Para['Num_Nodes_t']), 1:]
                nodes = np.hstack((t.reshape(-1, 1).repeat(self.Key_Para['Num_Nodes_all_space'], axis=1).T.reshape(-1, 1), nodes_space.repeat(T, axis=0)))
                
                elements = elements.cpu().detach().numpy()

                fig = plt.figure(figsize=(int(3*T), 1))
                for i in range(1, T+1):
                    cur_nodes = nodes[(i-1)::T, :]

                    pre_cur_nodes = torch.tensor(cur_nodes, requires_grad=True, dtype=torch.float64).cuda()
                    pre_cur_rho = model_rho(pre_cur_nodes)
                    cur_rho = pre_cur_rho.reshape(-1, 1).cpu().detach().numpy()

                    surface_rho = (cur_rho[elements[:, 0]] + cur_rho[elements[:, 1]] + cur_rho[elements[:, 2]]) / 3
                    color_rho = (surface_rho - surface_rho.min()) / (surface_rho - surface_rho.min()).max()

                    ax = fig.add_subplot(1, T, i, projection='3d')
                    surf = ax.plot_trisurf(cur_nodes[:, 1], cur_nodes[:, 2], cur_nodes[:, 3], triangles=elements, vmin=0, vmax=1, antialiased=False)
                    surf.set_array(color_rho[:, 0])
                    ax.view_init(elev=25, azim=45)
                    # ax.view_init(elev=25, azim=135)
                    # ax.set_xlim3d(0, 1), ax.set_ylim3d(0, 1), ax.set_zlim3d(cur_nodes[:, 3].min(), cur_nodes[:, 3].max())
                    ax.set_xticks([]),  ax.set_yticks([]), ax.set_zticks([])
                    # ax.set_xlabel('x'), ax.set_ylabel('y'), ax.set_zlabel('z')
                    ax.set_box_aspect([1, 1, 2])
                    cb = fig.colorbar(surf, ax=[ax])
                    cb.set_ticks(np.linspace(surf.norm.vmin, surf.norm.vmax, num=5))
                    scale = np.linspace(surface_rho.min(), surface_rho.max(), num=5)
                    scale1 = []
                    for j in range(0, 5):
                        scale1.append('{:.3f}'.format(scale[j]))
                    cb.set_ticklabels(scale1)
                    plt.title('t=%1.3f' %(t[i-1]))

                plt.savefig(self.Key_Para['File_name'] + '/' + 'Fig-rho' + str(ite) + '.png', bbox_inches='tight', dpi=300)

                plt.close()

            elif type_surface == 'Sphere':
                r = self.Key_Para['r']
                T = 10 #int(self.Key_Para['Num_Nodes_t'])
                nodes, elements = self.gen_Nodes.forward(type_train='False')

                t = np.linspace(self.Time[0, 0], self.Time[0, 1], T)
                nodes = nodes.cpu().detach().numpy()
                nodes_space = nodes[0::10, 1:]
                nodes = np.hstack((t.reshape(-1, 1).repeat(self.Key_Para['Num_Nodes_all_space'], axis=1).T.reshape(-1, 1), nodes_space.repeat(T, axis=0)))
                
                elements = elements.cpu().detach().numpy()

                fig = plt.figure(figsize=(int(3*T), 1))
                for i in range(1, T+1):
                    cur_nodes = nodes[(i-1)::T, :]

                    pre_cur_nodes = torch.tensor(cur_nodes, requires_grad=True, dtype=torch.float64).cuda()
                    pre_cur_rho = model_rho(pre_cur_nodes)
                    cur_rho = pre_cur_rho.reshape(-1, 1).cpu().detach().numpy()

                    surface_rho = (cur_rho[elements[:, 0]] + cur_rho[elements[:, 1]] + cur_rho[elements[:, 2]]) / 3
                    color_rho = (surface_rho - surface_rho.min()) / (surface_rho - surface_rho.min()).max()

                    ax = fig.add_subplot(1, T, i, projection='3d')
                    surf = ax.plot_trisurf(cur_nodes[:, 1], cur_nodes[:, 2], cur_nodes[:, 3], triangles=elements, vmin=0, vmax=1, antialiased=False)
                    surf.set_array(color_rho[:, 0])
                    ax.view_init(elev=25, azim=45)
                    # ax.view_init(elev=90, azim=45)
                    ax.set_xlim3d(0, 1), ax.set_ylim3d(0, 1), ax.set_zlim3d(cur_nodes[:, 3].min(), cur_nodes[:, 3].max())
                    ax.set_xticks([]),  ax.set_yticks([]), ax.set_zticks([])
                    # ax.set_xlabel('x'), ax.set_ylabel('y'), ax.set_zlabel('z')
                    ax.set_box_aspect([1, 1, 1])
                    cb = fig.colorbar(surf, ax=[ax])
                    cb.set_ticks(np.linspace(surf.norm.vmin, surf.norm.vmax, num=5))
                    scale = np.linspace(surface_rho.min(), surface_rho.max(), num=5)
                    scale1 = []
                    for j in range(0, 5):
                        scale1.append('{:.3f}'.format(scale[j]))
                    cb.set_ticklabels(scale1)
                    plt.title('t=%1.3f' %(t[i-1]))

                plt.savefig(self.Key_Para['File_name'] + '/' + 'Fig-rho-' + str(ite) + '.png', bbox_inches='tight', dpi=300)

                plt.close()


                if 1:
                    T = 5
                    nodes, elements = self.gen_Nodes.forward()

                    t = np.linspace(self.Time[0, 0], self.Time[0, 1], T)
                    nodes = nodes.cpu().detach().numpy()
                    nodes_space = nodes[0::int(self.Key_Para['Num_Nodes_t']), 1:]
                    nodes = np.hstack((t.reshape(-1, 1).repeat(self.Key_Para['Num_Nodes_all_space'], axis=1).T.reshape(-1, 1), nodes_space.repeat(T, axis=0)))
                    
                    elements = elements.cpu().detach().numpy()

                    for i in range(1, T+1):
                        fig = plt.figure(figsize=(6, 6))
                        cur_nodes = nodes[(i-1)::T, :]

                        pre_cur_nodes = torch.tensor(cur_nodes, requires_grad=True, dtype=torch.float64).cuda()
                        pre_cur_rho = model_rho(pre_cur_nodes)
                        cur_rho = pre_cur_rho.reshape(-1, 1).cpu().detach().numpy()

                        surface_rho = (cur_rho[elements[:, 0]] + cur_rho[elements[:, 1]] + cur_rho[elements[:, 2]]) / 3
                        color_rho = (surface_rho - surface_rho.min()) / (surface_rho - surface_rho.min()).max()

                        ax = fig.add_subplot(1, 1, 1, projection='3d')
                        surf = ax.plot_trisurf(cur_nodes[:, 1], cur_nodes[:, 2], cur_nodes[:, 3], triangles=elements, vmin=0, vmax=1, antialiased=False)
                        surf.set_array(color_rho[:, 0])
                        # ax.view_init(elev=90 - (180/(T-1))*(i-1), azim=45)
                        ax.view_init(elev=25, azim=45)
                        ax.set_xlim3d(0, 1), ax.set_ylim3d(0, 1), ax.set_zlim3d(0, 1)
                        ax.set_xticks([0.0, 0.5, 1.0]),  ax.set_yticks([0.0, 0.5, 1.0]), ax.set_zticks([0.0, 0.5, 1.0])
                        ax.set_xlabel('x'), ax.set_ylabel('y'), ax.set_zlabel('z')
                        ax.set_box_aspect([1, 1, 1])
                        plt.title('t=%1.3f' %(t[i-1]))

                        ax.grid(None)
                        ax.xaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))
                        ax.yaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))
                        ax.zaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))
                        # ax.axis('off')

                        cb = fig.colorbar(surf, ax=[ax])
                        cb.set_ticks(np.linspace(surf.norm.vmin, surf.norm.vmax, num=5))
                        scale = np.linspace(surface_rho.min(), surface_rho.max(), num=5)
                        scale1 = []
                        for j in range(0, 5):
                            scale1.append('{:.3f}'.format(scale[j]))
                        cb.set_ticklabels(scale1)
                        plt.savefig(self.Key_Para['File_name'] + '/' + 'Fig-rho' + str(i) + '.png', dpi=300)
                        plt.close()

            elif type_surface == 'Ellipsoid':
                r = self.Key_Para['r']
                T = int(self.Key_Para['Num_Nodes_t'])
                nodes, elements = self.gen_Nodes.forward()

                t = np.linspace(self.Time[0, 0], self.Time[0, 1], T)
                nodes = nodes.cpu().detach().numpy()
                nodes_space = nodes[0::int(self.Key_Para['Num_Nodes_t']), 1:]
                nodes = np.hstack((t.reshape(-1, 1).repeat(self.Key_Para['Num_Nodes_all_space'], axis=1).T.reshape(-1, 1), nodes_space.repeat(T, axis=0)))
                
                elements = elements.cpu().detach().numpy()

                fig = plt.figure(figsize=(int(3*T), 1))
                for i in range(1, T+1):
                    cur_nodes = nodes[(i-1)::T, :]

                    pre_cur_nodes = torch.tensor(cur_nodes, requires_grad=True, dtype=torch.float64).cuda()
                    pre_cur_rho = model_rho(pre_cur_nodes)
                    cur_rho = pre_cur_rho.reshape(-1, 1).cpu().detach().numpy()

                    surface_rho = (cur_rho[elements[:, 0]] + cur_rho[elements[:, 1]] + cur_rho[elements[:, 2]]) / 3
                    color_rho = (surface_rho - surface_rho.min()) / (surface_rho - surface_rho.min()).max()

                    ax = fig.add_subplot(1, T, i, projection='3d')
                    surf = ax.plot_trisurf(cur_nodes[:, 1], cur_nodes[:, 2], cur_nodes[:, 3], triangles=elements, vmin=0, vmax=1, antialiased=False)
                    surf.set_array(color_rho[:, 0])
                    ax.view_init(elev=60, azim=0)
                    # ax.view_init(elev=25, azim=135)
                    ax.set_xlim3d(0, 1), ax.set_ylim3d(0, 1), ax.set_zlim3d(cur_nodes[:, 3].min(), cur_nodes[:, 3].max())
                    ax.set_xticks([]),  ax.set_yticks([]), ax.set_zticks([])
                    # ax.set_xlabel('x'), ax.set_ylabel('y'), ax.set_zlabel('z')
                    ax.set_box_aspect([6, 3, 1])
                    cb = fig.colorbar(surf, ax=[ax])
                    cb.set_ticks(np.linspace(surf.norm.vmin, surf.norm.vmax, num=5))
                    scale = np.linspace(surface_rho.min(), surface_rho.max(), num=5)
                    scale1 = []
                    for j in range(0, 5):
                        scale1.append('{:.3f}'.format(scale[j]))
                    cb.set_ticklabels(scale1)
                    plt.title('t=%1.3f' %(t[i-1]))

                plt.savefig(self.Key_Para['File_name'] + '/' + 'Fig-rho' + str(ite) + '.png', bbox_inches='tight', dpi=300)

                plt.close()

            elif type_surface == 'Torus':
                r = self.Key_Para['r']
                T = int(self.Key_Para['Num_Nodes_t'])
                nodes, elements = self.gen_Nodes.forward()

                t = np.linspace(self.Time[0, 0], self.Time[0, 1], T)
                nodes = nodes.cpu().detach().numpy()
                nodes_space = nodes[0::int(self.Key_Para['Num_Nodes_t']), 1:]
                nodes = np.hstack((t.reshape(-1, 1).repeat(self.Key_Para['Num_Nodes_all_space'], axis=1).T.reshape(-1, 1), nodes_space.repeat(T, axis=0)))
                
                elements = elements.cpu().detach().numpy()

                fig = plt.figure(figsize=(int(3*T), 1))
                for i in range(1, T+1):
                    cur_nodes = nodes[(i-1)::T, :]

                    pre_cur_nodes = torch.tensor(cur_nodes, requires_grad=True, dtype=torch.float64).cuda()
                    pre_cur_rho = model_rho(pre_cur_nodes)
                    cur_rho = pre_cur_rho.reshape(-1, 1).cpu().detach().numpy()

                    surface_rho = (cur_rho[elements[:, 0]] + cur_rho[elements[:, 1]] + cur_rho[elements[:, 2]]) / 3
                    color_rho = (surface_rho - surface_rho.min()) / (surface_rho - surface_rho.min()).max()

                    ax = fig.add_subplot(1, T, i, projection='3d')
                    surf = ax.plot_trisurf(cur_nodes[:, 1], cur_nodes[:, 2], cur_nodes[:, 3], triangles=elements, vmin=0, vmax=1, antialiased=False)
                    surf.set_array(color_rho[:, 0])
                    ax.view_init(elev=60, azim=45)
                    # ax.view_init(elev=25, azim=135)
                    # ax.set_xlim3d(0, 1), ax.set_ylim3d(0, 1), ax.set_zlim3d(cur_nodes[:, 3].min(), cur_nodes[:, 3].max())
                    ax.set_xticks([]),  ax.set_yticks([]), ax.set_zticks([])
                    # ax.set_xlabel('x'), ax.set_ylabel('y'), ax.set_zlabel('z')
                    ax.set_box_aspect([10, 10, 2])
                    cb = fig.colorbar(surf, ax=[ax])
                    cb.set_ticks(np.linspace(surf.norm.vmin, surf.norm.vmax, num=5))
                    scale = np.linspace(surface_rho.min(), surface_rho.max(), num=5)
                    scale1 = []
                    for j in range(0, 5):
                        scale1.append('{:.3f}'.format(scale[j]))
                    cb.set_ticklabels(scale1)
                    plt.title('t=%1.3f' %(t[i-1]))

                plt.savefig(self.Key_Para['File_name'] + '/' + 'Fig-rho-' + str(ite) + '.png', bbox_inches='tight', dpi=300)

                plt.close()



                if 1:
                    T = 5
                    nodes, elements = self.gen_Nodes.forward()

                    t = np.linspace(self.Time[0, 0], self.Time[0, 1], T)
                    nodes = nodes.cpu().detach().numpy()
                    nodes_space = nodes[0::int(self.Key_Para['Num_Nodes_t']), 1:]
                    nodes = np.hstack((t.reshape(-1, 1).repeat(self.Key_Para['Num_Nodes_all_space'], axis=1).T.reshape(-1, 1), nodes_space.repeat(T, axis=0)))
                    
                    elements = elements.cpu().detach().numpy()


                    
                    for i in range(1, T+1):
                        fig = plt.figure(figsize=(6, 6))
                        cur_nodes = nodes[(i-1)::T, :]

                        pre_cur_nodes = torch.tensor(cur_nodes, requires_grad=True, dtype=torch.float64).cuda()
                        pre_cur_rho = model_rho(pre_cur_nodes)
                        cur_rho = pre_cur_rho.reshape(-1, 1).cpu().detach().numpy()

                        surface_rho = (cur_rho[elements[:, 0]] + cur_rho[elements[:, 1]] + cur_rho[elements[:, 2]]) / 3
                        color_rho = (surface_rho - surface_rho.min()) / (surface_rho - surface_rho.min()).max()

                        ax = fig.add_subplot(1, 1, 1, projection='3d')
                        surf = ax.plot_trisurf(cur_nodes[:, 1], cur_nodes[:, 2], cur_nodes[:, 3], triangles=elements, vmin=0, vmax=1, antialiased=False)
                        surf.set_array(color_rho[:, 0])
                        # ax.view_init(elev=80 - 40*(i-1), azim=45)
                        ax.view_init(elev=60, azim=45)
                        # ax.set_xlim3d(0, 1), ax.set_ylim3d(0, 1), ax.set_zlim3d(0, 1)
                        # ax.set_xticks([0.0, 0.5, 1.0]),  ax.set_yticks([0.0, 0.5, 1.0]), 
                        ax.set_zticks([cur_nodes[:, 3].min(), cur_nodes[:, 3].max()])
                        ax.set_xlabel('x'), ax.set_ylabel('y'), ax.set_zlabel('z')
                        ax.set_box_aspect([10, 10, 2])

                        ax.grid(None)
                        ax.xaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))
                        ax.yaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))
                        ax.zaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))
                        # ax.axis('off')


                        cb = fig.colorbar(surf, ax=[ax])
                        cb.set_ticks(np.linspace(surf.norm.vmin, surf.norm.vmax, num=5))
                        scale = np.linspace(surface_rho.min(), surface_rho.max(), num=5)
                        scale1 = []
                        for j in range(0, 5):
                            scale1.append('{:.3f}'.format(scale[j]))
                        cb.set_ticklabels(scale1)
                        plt.savefig(self.Key_Para['File_name'] + '/' + 'Fig-rho' + str(i) + '.png', dpi=300)
                        plt.close()

            elif type_surface == 'Peanut':
                r = self.Key_Para['r']
                T = int(self.Key_Para['Num_Nodes_t'])
                nodes, elements = self.gen_Nodes.forward()

                t = np.linspace(self.Time[0, 0], self.Time[0, 1], T)
                nodes = nodes.cpu().detach().numpy()
                nodes_space = nodes[0::int(self.Key_Para['Num_Nodes_t']), 1:]
                nodes = np.hstack((t.reshape(-1, 1).repeat(self.Key_Para['Num_Nodes_all_space'], axis=1).T.reshape(-1, 1), nodes_space.repeat(T, axis=0)))
                
                elements = elements.cpu().detach().numpy()

                fig = plt.figure(figsize=(int(3*T), 1))
                for i in range(1, T+1):
                    cur_nodes = nodes[(i-1)::T, :]

                    pre_cur_nodes = torch.tensor(cur_nodes, requires_grad=True, dtype=torch.float64).cuda()
                    pre_cur_rho = model_rho(pre_cur_nodes)
                    cur_rho = pre_cur_rho.reshape(-1, 1).cpu().detach().numpy()

                    surface_rho = (cur_rho[elements[:, 0]] + cur_rho[elements[:, 1]] + cur_rho[elements[:, 2]]) / 3
                    color_rho = (surface_rho - surface_rho.min()) / (surface_rho - surface_rho.min()).max()

                    ax = fig.add_subplot(1, T, i, projection='3d')
                    surf = ax.plot_trisurf(cur_nodes[:, 1], cur_nodes[:, 2], cur_nodes[:, 3], triangles=elements, vmin=0, vmax=1, antialiased=False)
                    surf.set_array(color_rho[:, 0])
                    ax.view_init(elev=45, azim=45)
                    # ax.view_init(elev=25, azim=135)
                    # ax.set_xlim3d(0, 1), ax.set_ylim3d(0, 1), ax.set_zlim3d(cur_nodes[:, 3].min(), cur_nodes[:, 3].max())
                    ax.set_xticks([]),  ax.set_yticks([]), ax.set_zticks([])
                    # ax.set_xlabel('x'), ax.set_ylabel('y'), ax.set_zlabel('z')
                    ax.set_box_aspect([6, 4, 4])
                    cb = fig.colorbar(surf, ax=[ax])
                    cb.set_ticks(np.linspace(surf.norm.vmin, surf.norm.vmax, num=5))
                    scale = np.linspace(surface_rho.min(), surface_rho.max(), num=5)
                    scale1 = []
                    for j in range(0, 5):
                        scale1.append('{:.3f}'.format(scale[j]))
                    cb.set_ticklabels(scale1)
                    plt.title('t=%1.3f' %(t[i-1]))

                plt.savefig(self.Key_Para['File_name'] + '/' + 'Fig-rho' + str(ite) + '.png', bbox_inches='tight', dpi=300)

                plt.close()

            elif type_surface == 'Opener':
                T = int(self.Key_Para['Num_Nodes_t'])
                nodes, elements = self.gen_Nodes.forward()

                t = np.linspace(self.Time[0, 0], self.Time[0, 1], T)
                nodes = nodes.cpu().detach().numpy()
                nodes_space = nodes[0::int(self.Key_Para['Num_Nodes_t']), 1:]
                nodes = np.hstack((t.reshape(-1, 1).repeat(self.Key_Para['Num_Nodes_all_space'], axis=1).T.reshape(-1, 1), nodes_space.repeat(T, axis=0)))
                
                elements = elements.cpu().detach().numpy()

                fig = plt.figure(figsize=(int(3*T), 1))
                for i in range(1, T+1):
                    cur_nodes = nodes[(i-1)::T, :]

                    pre_cur_nodes = torch.tensor(cur_nodes, requires_grad=True, dtype=torch.float64).cuda()
                    pre_cur_rho = model_rho(pre_cur_nodes)
                    cur_rho = pre_cur_rho.reshape(-1, 1).cpu().detach().numpy()

                    surface_rho = (cur_rho[elements[:, 0]] + cur_rho[elements[:, 1]] + cur_rho[elements[:, 2]]) / 3
                    color_rho = (surface_rho - surface_rho.min()) / (surface_rho - surface_rho.min()).max()

                    ax = fig.add_subplot(1, T, i, projection='3d')
                    surf = ax.plot_trisurf(cur_nodes[:, 1], cur_nodes[:, 2], cur_nodes[:, 3], triangles=elements, vmin=0, vmax=1, antialiased=False)
                    surf.set_array(color_rho[:, 0])
                    ax.view_init(elev=45, azim=45)
                    # ax.view_init(elev=25, azim=135)
                    # ax.set_xlim3d(0, 1), ax.set_ylim3d(0, 1), ax.set_zlim3d(cur_nodes[:, 3].min(), cur_nodes[:, 3].max())
                    ax.set_xticks([]),  ax.set_yticks([]), ax.set_zticks([])
                    # ax.set_xlabel('x'), ax.set_ylabel('y'), ax.set_zlabel('z')
                    ax.set_box_aspect([10, 8, 2])
                    cb = fig.colorbar(surf, ax=[ax])
                    cb.set_ticks(np.linspace(surf.norm.vmin, surf.norm.vmax, num=5))
                    scale = np.linspace(surface_rho.min(), surface_rho.max(), num=5)
                    scale1 = []
                    for j in range(0, 5):
                        scale1.append('{:.3f}'.format(scale[j]))
                    cb.set_ticklabels(scale1)
                    plt.title('t=%1.3f' %(t[i-1]))

                plt.savefig(self.Key_Para['File_name'] + '/' + 'Fig-rho' + str(ite) + '.png', bbox_inches='tight', dpi=300)

                plt.close()

            elif type_surface == 'Cube':
                r = self.Key_Para['r']
                T = int(self.Key_Para['Num_Nodes_t'])
                nodes, elements = self.gen_Nodes.forward()

                t = np.linspace(self.Time[0, 0], self.Time[0, 1], T)
                nodes = nodes.cpu().detach().numpy()
                nodes_space = nodes[0::int(self.Key_Para['Num_Nodes_t']), 1:]
                nodes = np.hstack((t.reshape(-1, 1).repeat(self.Key_Para['Num_Nodes_all_space'], axis=1).T.reshape(-1, 1), nodes_space.repeat(T, axis=0)))
                
                elements = elements.cpu().detach().numpy()

                fig = plt.figure(figsize=(int(3*T), 1))
                for i in range(1, T+1):
                    cur_nodes = nodes[(i-1)::T, :]

                    pre_cur_nodes = torch.tensor(cur_nodes, requires_grad=True, dtype=torch.float64).cuda()
                    pre_cur_rho = model_rho(pre_cur_nodes)
                    cur_rho = pre_cur_rho.reshape(-1, 1).cpu().detach().numpy()

                    surface_rho = (cur_rho[elements[:, 0]] + cur_rho[elements[:, 1]] + cur_rho[elements[:, 2]]) / 3
                    color_rho = (surface_rho - surface_rho.min()) / (surface_rho - surface_rho.min()).max()

                    ax = fig.add_subplot(1, T, i, projection='3d')
                    surf = ax.plot_trisurf(cur_nodes[:, 1], cur_nodes[:, 2], cur_nodes[:, 3], triangles=elements, vmin=0, vmax=1, antialiased=False)
                    surf.set_array(color_rho[:, 0])
                    ax.view_init(elev=60, azim=45)
                    # ax.view_init(elev=25, azim=135)
                    ax.set_xlim3d(-0.5, 1.5), ax.set_ylim3d(-0.5, 1.5), ax.set_zlim3d(-0.5, 1.5)
                    ax.set_xticks([]),  ax.set_yticks([]), ax.set_zticks([])
                    # ax.set_xlabel('x'), ax.set_ylabel('y'), ax.set_zlabel('z')
                    ax.set_box_aspect([1, 1, 1])
                    cb = fig.colorbar(surf, ax=[ax])
                    cb.set_ticks(np.linspace(surf.norm.vmin, surf.norm.vmax, num=5))
                    scale = np.linspace(surface_rho.min(), surface_rho.max(), num=5)
                    scale1 = []
                    for j in range(0, 5):
                        scale1.append('{:.3f}'.format(scale[j]))
                    cb.set_ticklabels(scale1)
                    plt.title('t=%1.3f' %(t[i-1]))

                plt.savefig(self.Key_Para['File_name'] + '/' + 'Fig-rho' + str(ite) + '.png', bbox_inches='tight', dpi=300)

                plt.close()

            elif type_surface == 'Hand':
                T = int(self.Key_Para['Num_Nodes_t'])
                nodes, elements = self.gen_Nodes.forward()

                t = np.linspace(self.Time[0, 0], self.Time[0, 1], T)
                nodes = nodes.cpu().detach().numpy()
                nodes_space = nodes[0::int(self.Key_Para['Num_Nodes_t']), 1:]
                nodes = np.hstack((t.reshape(-1, 1).repeat(self.Key_Para['Num_Nodes_all_space'], axis=1).T.reshape(-1, 1), nodes_space.repeat(T, axis=0)))
                
                elements = elements.cpu().detach().numpy()

                fig = plt.figure(figsize=(int(3*T), 1))
                for i in range(1, T+1):
                    cur_nodes = nodes[(i-1)::T, :]

                    pre_cur_nodes = torch.tensor(cur_nodes, requires_grad=True, dtype=torch.float64).cuda()
                    pre_cur_rho = model_rho(pre_cur_nodes)
                    cur_rho = pre_cur_rho.reshape(-1, 1).cpu().detach().numpy()

                    surface_rho = (cur_rho[elements[:, 0]] + cur_rho[elements[:, 1]] + cur_rho[elements[:, 2]]) / 3
                    color_rho = (surface_rho - surface_rho.min()) / (surface_rho - surface_rho.min()).max()

                    ax = fig.add_subplot(1, T, i, projection='3d')
                    surf = ax.plot_trisurf(cur_nodes[:, 1], cur_nodes[:, 2], cur_nodes[:, 3], triangles=elements, vmin=0, vmax=1, antialiased=False)
                    surf.set_array(color_rho[:, 0])
                    # ax.view_init(elev=65, azim=165)
                    ax.view_init(elev=80, azim=60)
                    # ax.set_xlim3d(0, 1), ax.set_ylim3d(0, 1), #ax.set_zlim3d(cur_nodes[:, 3].min(), cur_nodes[:, 3].max())
                    ax.set_xticks([]),  ax.set_yticks([]), ax.set_zticks([])
                    # ax.set_xlabel('x'), ax.set_ylabel('y'), ax.set_zlabel('z')
                    ax.set_box_aspect([1, 1, 1])
                    cb = fig.colorbar(surf, ax=[ax])
                    cb.set_ticks(np.linspace(surf.norm.vmin, surf.norm.vmax, num=5))
                    scale = np.linspace(surface_rho.min(), surface_rho.max(), num=5)
                    scale1 = []
                    for j in range(0, 5):
                        scale1.append('{:.3f}'.format(scale[j]))
                    cb.set_ticklabels(scale1)
                    plt.title('t=%1.3f' %(t[i-1]))

                plt.savefig(self.Key_Para['File_name'] + '/' + 'Fig-rho' + str(ite) + '.png', bbox_inches='tight', dpi=300)

                plt.close()

            elif type_surface == 'Airplane':
                T = int(self.Key_Para['Num_Nodes_t'])
                nodes, elements = self.gen_Nodes.forward()

                t = np.linspace(self.Time[0, 0], self.Time[0, 1], T)
                nodes = nodes.cpu().detach().numpy()
                nodes_space = nodes[0::int(self.Key_Para['Num_Nodes_t']), 1:]
                nodes = np.hstack((t.reshape(-1, 1).repeat(self.Key_Para['Num_Nodes_all_space'], axis=1).T.reshape(-1, 1), nodes_space.repeat(T, axis=0)))
                
                elements = elements.cpu().detach().numpy()

                fig = plt.figure(figsize=(int(3*T), 1))
                for i in range(1, T+1):
                    cur_nodes = nodes[(i-1)::T, :]

                    pre_cur_nodes = torch.tensor(cur_nodes, requires_grad=True, dtype=torch.float64).cuda()
                    pre_cur_rho = model_rho(pre_cur_nodes)
                    cur_rho = pre_cur_rho.reshape(-1, 1).cpu().detach().numpy()

                    surface_rho = (cur_rho[elements[:, 0]] + cur_rho[elements[:, 1]] + cur_rho[elements[:, 2]]) / 3
                    color_rho = (surface_rho - surface_rho.min()) / (surface_rho - surface_rho.min()).max()

                    ax = fig.add_subplot(1, T, i, projection='3d')
                    surf = ax.plot_trisurf(cur_nodes[:, 1], cur_nodes[:, 2], cur_nodes[:, 3], triangles=elements, vmin=0, vmax=1, antialiased=False)
                    surf.set_array(color_rho[:, 0])
                    ax.view_init(elev=155, azim=-90)
                    # ax.view_init(elev=65, azim=165)
                    ax.set_xlim3d(-1, 1), ax.set_ylim3d(-0.5, 0.5), ax.set_zlim3d(-1, 1)
                    ax.set_xticks([]),  ax.set_yticks([]), ax.set_zticks([])
                    # ax.set_xlabel('x'), ax.set_ylabel('y'), ax.set_zlabel('z')
                    ax.set_box_aspect([2, 1, 2])
                    cb = fig.colorbar(surf, ax=[ax])
                    cb.set_ticks(np.linspace(surf.norm.vmin, surf.norm.vmax, num=5))
                    scale = np.linspace(surface_rho.min(), surface_rho.max(), num=5)
                    scale1 = []
                    for j in range(0, 5):
                        scale1.append('{:.3f}'.format(scale[j]))
                    cb.set_ticklabels(scale1)
                    plt.title('t=%1.3f' %(t[i-1]))

                plt.savefig(self.Key_Para['File_name'] + '/' + 'Fig-rho' + str(ite) + '.png', bbox_inches='tight', dpi=300)

                plt.close()

            elif type_surface == 'Mazes':
                T = int(self.Key_Para['Num_Nodes_t'])
                nodes, elements = self.gen_Nodes.forward()

                t = np.linspace(self.Time[0, 0], self.Time[0, 1], T)
                nodes = nodes.cpu().detach().numpy()
                nodes_space = nodes[0::int(self.Key_Para['Num_Nodes_t']), 1:]
                nodes = np.hstack((t.reshape(-1, 1).repeat(self.Key_Para['Num_Nodes_all_space'], axis=1).T.reshape(-1, 1), nodes_space.repeat(T, axis=0)))
                
                elements = elements.cpu().detach().numpy()

                fig = plt.figure(figsize=(int(3*T), 1))
                for i in range(1, T+1):
                    cur_nodes = nodes[(i-1)::T, :]

                    pre_cur_nodes = torch.tensor(cur_nodes, requires_grad=True, dtype=torch.float64).cuda()
                    pre_cur_rho = model_rho(pre_cur_nodes)
                    cur_rho = pre_cur_rho.reshape(-1, 1).cpu().detach().numpy()

                    surface_rho = (cur_rho[elements[:, 0]] + cur_rho[elements[:, 1]] + cur_rho[elements[:, 2]]) / 3
                    color_rho = (surface_rho - surface_rho.min()) / (surface_rho - surface_rho.min()).max()

                    ax = fig.add_subplot(1, T, i, projection='3d')
                    surf = ax.plot_trisurf(cur_nodes[:, 1], cur_nodes[:, 2], cur_nodes[:, 3], triangles=elements, vmin=0, vmax=1, antialiased=False)
                    surf.set_array(color_rho[:, 0])
                    ax.view_init(elev=90, azim=45)
                    ax.set_xlim3d(0, 1), ax.set_ylim3d(0, 1), ax.set_zlim3d(0, 0.1)
                    ax.set_xticks([]),  ax.set_yticks([]), ax.set_zticks([])
                    # ax.set_xlabel('x'), ax.set_ylabel('y'), ax.set_zlabel('z')

                    cb = fig.colorbar(surf, ax=[ax])
                    cb.set_ticks(np.linspace(surf.norm.vmin, surf.norm.vmax, num=5))
                    scale = np.linspace(surface_rho.min(), surface_rho.max(), num=5)
                    scale1 = []
                    for j in range(0, 5):
                        scale1.append('{:.3f}'.format(scale[j]))
                    cb.set_ticklabels(scale1)
                    plt.title('t=%1.3f' %(t[i-1]))

                plt.savefig(self.Key_Para['File_name'] + '/' + 'Fig-rho' + str(ite) + '.png', bbox_inches='tight', dpi=300)

                plt.close()

        elif self.Dim_space > 3:
            print('There need code')

    def plot_v(self, model_phi, ite=''):
        if self.Dim_space == 3:
            type_surface = self.Key_Para['type_surface']
            if type_surface == 'Cylinder':
                print('There need code!')
            
            elif type_surface == 'Sphere':
                w = 1

        elif self.Dim_space > 3:
            print('There need code')

    def plot_g(self, model_g, ite=''):
        if self.Dim_space == 3:
            type_surface = self.Key_Para['type_surface']
            if type_surface == 'Cylinder':
                print('There need code!')

            elif type_surface == 'Sphere':
                w = 1

        elif self.Dim_space > 3:
            print('There need code')

    def plot_loss(self, all_sub_loss):
        num = len(all_sub_loss)
        num_loss = len(all_sub_loss[0])
        all_loss = torch.zeros((num, num_loss + 1))
        for i in range(num):
            for j in range(1, num_loss + 1):
                all_loss[i, j] = all_sub_loss[i][j-1]
            all_loss[i, 0] = torch.sum(all_loss[i, 1:])
        all_loss = all_loss.cpu().detach().numpy()

        plt.figure(figsize=(4*all_loss.shape[1], 3))
        for i in range(all_loss.shape[1]):
            if i == 0:
                plt.subplot(1, all_loss.shape[1], i+1)
                plt.plot(all_loss[:, i], color='r')
                plt.title('all_loss')
            else:
                plt.subplot(1, all_loss.shape[1], i+1)
                plt.plot(all_loss[:, i])
                plt.title(self.Key_Para['loss_name'][i-1])
        plt.savefig(self.Key_Para['File_name'] + '/' + 'loss.png')
        plt.close()

        np.save(self.Key_Para['File_name'] + '/' + 'all_loss.npy', all_loss)

    def plot_c_eq(self, ite=''):
        if self.Dim_space == 3:
            type_surface = self.Key_Para['type_surface']
            if type_surface == 'Cylinder':
                r = self.Key_Para['r']
                T = int(self.Key_Para['Num_Nodes_t'])
                nodes, elements = self.gen_Nodes.forward()

                t = np.linspace(self.Time[0, 0], self.Time[0, 1], T)
                nodes = nodes.cpu().detach().numpy()
                nodes_space = nodes[0::int(self.Key_Para['Num_Nodes_t']), 1:]
                nodes = np.hstack((t.reshape(-1, 1).repeat(self.Key_Para['Num_Nodes_all_space'], axis=1).T.reshape(-1, 1), nodes_space.repeat(T, axis=0)))
                
                elements = elements.cpu().detach().numpy()

                fig = plt.figure(figsize=(int(3*T), 1))
                for i in range(1, T+1):
                    cur_nodes = nodes[(i-1)::T, :]
                    cur_plot = self.Key_Para['res_c_eq'][(i-1)::T, :]

                    pre_cur_nodes = torch.tensor(cur_nodes, requires_grad=True, dtype=torch.float64).cuda()
                    pre_cur_rho = cur_plot
                    cur_rho = np.abs(pre_cur_rho.reshape(-1, 1).cpu().detach().numpy())


                    surface_rho = (cur_rho[elements[:, 0]] + cur_rho[elements[:, 1]] + cur_rho[elements[:, 2]]) / 3
                    color_rho = (surface_rho - surface_rho.min()) / (surface_rho - surface_rho.min()).max()

                    ax = fig.add_subplot(1, T, i, projection='3d')
                    surf = ax.plot_trisurf(cur_nodes[:, 1], cur_nodes[:, 2], cur_nodes[:, 3], triangles=elements, vmin=0, vmax=1, antialiased=False)
                    surf.set_array(color_rho[:, 0])
                    ax.view_init(elev=25, azim=45)
                    # ax.view_init(elev=25, azim=135)
                    # ax.set_xlim3d(0, 1), ax.set_ylim3d(0, 1), ax.set_zlim3d(cur_nodes[:, 3].min(), cur_nodes[:, 3].max())
                    ax.set_xticks([]),  ax.set_yticks([]), ax.set_zticks([])
                    # ax.set_xlabel('x'), ax.set_ylabel('y'), ax.set_zlabel('z')
                    ax.set_box_aspect([1, 1, 2])
                    cb = fig.colorbar(surf, ax=[ax])
                    cb.set_ticks(np.linspace(surf.norm.vmin, surf.norm.vmax, num=5))
                    scale = np.linspace(surface_rho.min(), surface_rho.max(), num=5)
                    scale1 = []
                    for j in range(0, 5):
                        scale1.append('{:.3f}'.format(scale[j]))
                    cb.set_ticklabels(scale1)
                    plt.title('t=%1.3f' %(t[i-1]))

                plt.savefig(self.Key_Para['File_name'] + '/' + 'Fig-Res_C-eq' + str(ite) + '.png', bbox_inches='tight', dpi=300)

                plt.close()

            elif type_surface == 'Sphere':
                r = self.Key_Para['r']
                T = int(self.Key_Para['Num_Nodes_t'])
                nodes, elements = self.gen_Nodes.forward()

                t = np.linspace(self.Time[0, 0], self.Time[0, 1], T)
                nodes = nodes.cpu().detach().numpy()
                nodes_space = nodes[0::int(self.Key_Para['Num_Nodes_t']), 1:]
                nodes = np.hstack((t.reshape(-1, 1).repeat(self.Key_Para['Num_Nodes_all_space'], axis=1).T.reshape(-1, 1), nodes_space.repeat(T, axis=0)))
                
                elements = elements.cpu().detach().numpy()

                fig = plt.figure(figsize=(int(3*T), 1))
                for i in range(1, T+1):
                    cur_nodes = nodes[(i-1)::T, :]
                    cur_plot = self.Key_Para['res_c_eq'][(i-1)::T, :]

                    pre_cur_nodes = torch.tensor(cur_nodes, requires_grad=True, dtype=torch.float64).cuda()
                    pre_cur_rho = cur_plot
                    cur_rho = np.abs(pre_cur_rho.reshape(-1, 1).cpu().detach().numpy())

                    surface_rho = (cur_rho[elements[:, 0]] + cur_rho[elements[:, 1]] + cur_rho[elements[:, 2]]) / 3
                    color_rho = (surface_rho - surface_rho.min()) / (surface_rho - surface_rho.min()).max()

                    ax = fig.add_subplot(1, T, i, projection='3d')
                    surf = ax.plot_trisurf(cur_nodes[:, 1], cur_nodes[:, 2], cur_nodes[:, 3], triangles=elements, vmin=0, vmax=1, antialiased=False)
                    surf.set_array(color_rho[:, 0])
                    ax.view_init(elev=25, azim=45)
                    # ax.view_init(elev=25, azim=135)
                    ax.set_xlim3d(0, 1), ax.set_ylim3d(0, 1), ax.set_zlim3d(cur_nodes[:, 3].min(), cur_nodes[:, 3].max())
                    ax.set_xticks([]),  ax.set_yticks([]), ax.set_zticks([])
                    # ax.set_xlabel('x'), ax.set_ylabel('y'), ax.set_zlabel('z')
                    ax.set_box_aspect([1, 1, 1])
                    cb = fig.colorbar(surf, ax=[ax])
                    cb.set_ticks(np.linspace(surf.norm.vmin, surf.norm.vmax, num=5))
                    scale = np.linspace(surface_rho.min(), surface_rho.max(), num=5)
                    scale1 = []
                    for j in range(0, 5):
                        scale1.append('{:.3f}'.format(scale[j]))
                    cb.set_ticklabels(scale1)
                    plt.title('t=%1.3f' %(t[i-1]))

                plt.savefig(self.Key_Para['File_name'] + '/' + 'Fig-Res_C-eq' + str(ite) + '.png', bbox_inches='tight', dpi=300)

                plt.close()

            elif type_surface == 'Ellipsoid':
                r = self.Key_Para['r']
                T = int(self.Key_Para['Num_Nodes_t'])
                nodes, elements = self.gen_Nodes.forward()

                t = np.linspace(self.Time[0, 0], self.Time[0, 1], T)
                nodes = nodes.cpu().detach().numpy()
                nodes_space = nodes[0::int(self.Key_Para['Num_Nodes_t']), 1:]
                nodes = np.hstack((t.reshape(-1, 1).repeat(self.Key_Para['Num_Nodes_all_space'], axis=1).T.reshape(-1, 1), nodes_space.repeat(T, axis=0)))
                
                elements = elements.cpu().detach().numpy()

                fig = plt.figure(figsize=(int(3*T), 1))
                for i in range(1, T+1):
                    cur_nodes = nodes[(i-1)::T, :]
                    cur_plot = self.Key_Para['res_c_eq'][(i-1)::T, :]

                    pre_cur_nodes = torch.tensor(cur_nodes, requires_grad=True, dtype=torch.float64).cuda()
                    pre_cur_rho = cur_plot
                    cur_rho = np.abs(pre_cur_rho.reshape(-1, 1).cpu().detach().numpy())

                    surface_rho = (cur_rho[elements[:, 0]] + cur_rho[elements[:, 1]] + cur_rho[elements[:, 2]]) / 3
                    color_rho = (surface_rho - surface_rho.min()) / (surface_rho - surface_rho.min()).max()

                    ax = fig.add_subplot(1, T, i, projection='3d')
                    surf = ax.plot_trisurf(cur_nodes[:, 1], cur_nodes[:, 2], cur_nodes[:, 3], triangles=elements, vmin=0, vmax=1, antialiased=False)
                    surf.set_array(color_rho[:, 0])
                    ax.view_init(elev=60, azim=0)
                    # ax.view_init(elev=25, azim=135)
                    ax.set_xlim3d(0, 1), ax.set_ylim3d(0, 1), ax.set_zlim3d(cur_nodes[:, 3].min(), cur_nodes[:, 3].max())
                    ax.set_xticks([]),  ax.set_yticks([]), ax.set_zticks([])
                    # ax.set_xlabel('x'), ax.set_ylabel('y'), ax.set_zlabel('z')
                    ax.set_box_aspect([6, 3, 1])
                    cb = fig.colorbar(surf, ax=[ax])
                    cb.set_ticks(np.linspace(surf.norm.vmin, surf.norm.vmax, num=5))
                    scale = np.linspace(surface_rho.min(), surface_rho.max(), num=5)
                    scale1 = []
                    for j in range(0, 5):
                        scale1.append('{:.3f}'.format(scale[j]))
                    cb.set_ticklabels(scale1)
                    plt.title('t=%1.3f' %(t[i-1]))

                plt.savefig(self.Key_Para['File_name'] + '/' + 'Fig-Res_C-eq' + str(ite) + '.png', bbox_inches='tight', dpi=300)

                plt.close()

            elif type_surface == 'Torus':
                r = self.Key_Para['r']
                T = int(self.Key_Para['Num_Nodes_t'])
                nodes, elements = self.gen_Nodes.forward()

                t = np.linspace(self.Time[0, 0], self.Time[0, 1], T)
                nodes = nodes.cpu().detach().numpy()
                nodes_space = nodes[0::int(self.Key_Para['Num_Nodes_t']), 1:]
                nodes = np.hstack((t.reshape(-1, 1).repeat(self.Key_Para['Num_Nodes_all_space'], axis=1).T.reshape(-1, 1), nodes_space.repeat(T, axis=0)))
                
                elements = elements.cpu().detach().numpy()

                fig = plt.figure(figsize=(int(3*T), 1))
                for i in range(1, T+1):
                    cur_nodes = nodes[(i-1)::T, :]
                    cur_plot = self.Key_Para['res_c_eq'][(i-1)::T, :]

                    pre_cur_nodes = torch.tensor(cur_nodes, requires_grad=True, dtype=torch.float64).cuda()
                    pre_cur_rho = cur_plot
                    cur_rho = np.abs(pre_cur_rho.reshape(-1, 1).cpu().detach().numpy())

                    surface_rho = (cur_rho[elements[:, 0]] + cur_rho[elements[:, 1]] + cur_rho[elements[:, 2]]) / 3
                    color_rho = (surface_rho - surface_rho.min()) / (surface_rho - surface_rho.min()).max()

                    ax = fig.add_subplot(1, T, i, projection='3d')
                    surf = ax.plot_trisurf(cur_nodes[:, 1], cur_nodes[:, 2], cur_nodes[:, 3], triangles=elements, vmin=0, vmax=1, antialiased=False)
                    surf.set_array(color_rho[:, 0])
                    ax.view_init(elev=60, azim=45)
                    # ax.view_init(elev=25, azim=135)
                    # ax.set_xlim3d(0, 1), ax.set_ylim3d(0, 1), ax.set_zlim3d(cur_nodes[:, 3].min(), cur_nodes[:, 3].max())
                    ax.set_xticks([]),  ax.set_yticks([]), ax.set_zticks([])
                    # ax.set_xlabel('x'), ax.set_ylabel('y'), ax.set_zlabel('z')
                    ax.set_box_aspect([10, 10, 2])
                    cb = fig.colorbar(surf, ax=[ax])
                    cb.set_ticks(np.linspace(surf.norm.vmin, surf.norm.vmax, num=5))
                    scale = np.linspace(surface_rho.min(), surface_rho.max(), num=5)
                    scale1 = []
                    for j in range(0, 5):
                        scale1.append('{:.3f}'.format(scale[j]))
                    cb.set_ticklabels(scale1)
                    plt.title('t=%1.3f' %(t[i-1]))

                plt.savefig(self.Key_Para['File_name'] + '/' + 'Fig-Res_C-eq' + str(ite) + '.png', bbox_inches='tight', dpi=300)

                plt.close()

            elif type_surface == 'Peanut':
                r = self.Key_Para['r']
                T = int(self.Key_Para['Num_Nodes_t'])
                nodes, elements = self.gen_Nodes.forward()

                t = np.linspace(self.Time[0, 0], self.Time[0, 1], T)
                nodes = nodes.cpu().detach().numpy()
                nodes_space = nodes[0::int(self.Key_Para['Num_Nodes_t']), 1:]
                nodes = np.hstack((t.reshape(-1, 1).repeat(self.Key_Para['Num_Nodes_all_space'], axis=1).T.reshape(-1, 1), nodes_space.repeat(T, axis=0)))
                
                elements = elements.cpu().detach().numpy()

                fig = plt.figure(figsize=(int(3*T), 1))
                for i in range(1, T+1):
                    cur_nodes = nodes[(i-1)::T, :]
                    cur_plot = self.Key_Para['res_c_eq'][(i-1)::T, :]

                    pre_cur_nodes = torch.tensor(cur_nodes, requires_grad=True, dtype=torch.float64).cuda()
                    pre_cur_rho = cur_plot
                    cur_rho = np.abs(pre_cur_rho.reshape(-1, 1).cpu().detach().numpy())

                    surface_rho = (cur_rho[elements[:, 0]] + cur_rho[elements[:, 1]] + cur_rho[elements[:, 2]]) / 3
                    color_rho = (surface_rho - surface_rho.min()) / (surface_rho - surface_rho.min()).max()

                    ax = fig.add_subplot(1, T, i, projection='3d')
                    surf = ax.plot_trisurf(cur_nodes[:, 1], cur_nodes[:, 2], cur_nodes[:, 3], triangles=elements, vmin=0, vmax=1, antialiased=False)
                    surf.set_array(color_rho[:, 0])
                    ax.view_init(elev=45, azim=45)
                    # ax.view_init(elev=25, azim=135)
                    # ax.set_xlim3d(0, 1), ax.set_ylim3d(0, 1), ax.set_zlim3d(cur_nodes[:, 3].min(), cur_nodes[:, 3].max())
                    ax.set_xticks([]),  ax.set_yticks([]), ax.set_zticks([])
                    # ax.set_xlabel('x'), ax.set_ylabel('y'), ax.set_zlabel('z')
                    ax.set_box_aspect([6, 4, 4])
                    cb = fig.colorbar(surf, ax=[ax])
                    cb.set_ticks(np.linspace(surf.norm.vmin, surf.norm.vmax, num=5))
                    scale = np.linspace(surface_rho.min(), surface_rho.max(), num=5)
                    scale1 = []
                    for j in range(0, 5):
                        scale1.append('{:.3f}'.format(scale[j]))
                    cb.set_ticklabels(scale1)
                    plt.title('t=%1.3f' %(t[i-1]))

                plt.savefig(self.Key_Para['File_name'] + '/' + 'Fig-Res_C-eq' + str(ite) + '.png', bbox_inches='tight', dpi=300)

                plt.close()

            elif type_surface == 'Opener':
                T = int(self.Key_Para['Num_Nodes_t'])
                nodes, elements = self.gen_Nodes.forward()

                t = np.linspace(self.Time[0, 0], self.Time[0, 1], T)
                nodes = nodes.cpu().detach().numpy()
                nodes_space = nodes[0::int(self.Key_Para['Num_Nodes_t']), 1:]
                nodes = np.hstack((t.reshape(-1, 1).repeat(self.Key_Para['Num_Nodes_all_space'], axis=1).T.reshape(-1, 1), nodes_space.repeat(T, axis=0)))
                
                elements = elements.cpu().detach().numpy()

                fig = plt.figure(figsize=(int(3*T), 1))
                for i in range(1, T+1):
                    cur_nodes = nodes[(i-1)::T, :]
                    cur_plot = self.Key_Para['res_c_eq'][(i-1)::T, :]

                    pre_cur_nodes = torch.tensor(cur_nodes, requires_grad=True, dtype=torch.float64).cuda()
                    pre_cur_rho = cur_plot
                    cur_rho = np.abs(pre_cur_rho.reshape(-1, 1).cpu().detach().numpy())

                    ax = fig.add_subplot(1, T, i, projection='3d')
                    surf = ax.plot_trisurf(cur_nodes[:, 1], cur_nodes[:, 2], cur_nodes[:, 3], triangles=elements, vmin=0, vmax=1, antialiased=False)
                    surf.set_array(color_rho[:, 0])
                    ax.view_init(elev=45, azim=45)
                    # ax.view_init(elev=25, azim=135)
                    # ax.set_xlim3d(0, 1), ax.set_ylim3d(0, 1), ax.set_zlim3d(cur_nodes[:, 3].min(), cur_nodes[:, 3].max())
                    ax.set_xticks([]),  ax.set_yticks([]), ax.set_zticks([])
                    # ax.set_xlabel('x'), ax.set_ylabel('y'), ax.set_zlabel('z')
                    ax.set_box_aspect([10, 8, 2])
                    cb = fig.colorbar(surf, ax=[ax])
                    cb.set_ticks(np.linspace(surf.norm.vmin, surf.norm.vmax, num=5))
                    scale = np.linspace(surface_rho.min(), surface_rho.max(), num=5)
                    scale1 = []
                    for j in range(0, 5):
                        scale1.append('{:.3f}'.format(scale[j]))
                    cb.set_ticklabels(scale1)
                    plt.title('t=%1.3f' %(t[i-1]))

                plt.savefig(self.Key_Para['File_name'] + '/' + 'Fig-Res_C-eq' + str(ite) + '.png', bbox_inches='tight', dpi=300)

                plt.close()

            elif type_surface == 'Cube':
                r = self.Key_Para['r']
                T = int(self.Key_Para['Num_Nodes_t'])
                nodes, elements = self.gen_Nodes.forward()

                t = np.linspace(self.Time[0, 0], self.Time[0, 1], T)
                nodes = nodes.cpu().detach().numpy()
                nodes_space = nodes[0::int(self.Key_Para['Num_Nodes_t']), 1:]
                nodes = np.hstack((t.reshape(-1, 1).repeat(self.Key_Para['Num_Nodes_all_space'], axis=1).T.reshape(-1, 1), nodes_space.repeat(T, axis=0)))
                
                elements = elements.cpu().detach().numpy()

                fig = plt.figure(figsize=(int(3*T), 1))
                for i in range(1, T+1):
                    cur_nodes = nodes[(i-1)::T, :]
                    cur_plot = self.Key_Para['res_c_eq'][(i-1)::T, :]

                    pre_cur_nodes = torch.tensor(cur_nodes, requires_grad=True, dtype=torch.float64).cuda()
                    pre_cur_rho = cur_plot
                    cur_rho = np.abs(pre_cur_rho.reshape(-1, 1).cpu().detach().numpy())

                    surface_rho = (cur_rho[elements[:, 0]] + cur_rho[elements[:, 1]] + cur_rho[elements[:, 2]]) / 3
                    color_rho = (surface_rho - surface_rho.min()) / (surface_rho - surface_rho.min()).max()

                    ax = fig.add_subplot(1, T, i, projection='3d')
                    surf = ax.plot_trisurf(cur_nodes[:, 1], cur_nodes[:, 2], cur_nodes[:, 3], triangles=elements, vmin=0, vmax=1, antialiased=False)
                    surf.set_array(color_rho[:, 0])
                    ax.view_init(elev=60, azim=45)
                    # ax.view_init(elev=25, azim=135)
                    ax.set_xlim3d(-0.5, 1.5), ax.set_ylim3d(-0.5, 1.5), ax.set_zlim3d(-0.5, 1.5)
                    ax.set_xticks([]),  ax.set_yticks([]), ax.set_zticks([])
                    # ax.set_xlabel('x'), ax.set_ylabel('y'), ax.set_zlabel('z')
                    ax.set_box_aspect([1, 1, 1])
                    cb = fig.colorbar(surf, ax=[ax])
                    cb.set_ticks(np.linspace(surf.norm.vmin, surf.norm.vmax, num=5))
                    scale = np.linspace(surface_rho.min(), surface_rho.max(), num=5)
                    scale1 = []
                    for j in range(0, 5):
                        scale1.append('{:.3f}'.format(scale[j]))
                    cb.set_ticklabels(scale1)
                    plt.title('t=%1.3f' %(t[i-1]))

                plt.savefig(self.Key_Para['File_name'] + '/' + 'Fig-Res_C-eq' + str(ite) + '.png', bbox_inches='tight', dpi=300)

                plt.close()

            elif type_surface == 'Hand':
                T = int(self.Key_Para['Num_Nodes_t'])
                nodes, elements = self.gen_Nodes.forward()

                t = np.linspace(self.Time[0, 0], self.Time[0, 1], T)
                nodes = nodes.cpu().detach().numpy()
                nodes_space = nodes[0::int(self.Key_Para['Num_Nodes_t']), 1:]
                nodes = np.hstack((t.reshape(-1, 1).repeat(self.Key_Para['Num_Nodes_all_space'], axis=1).T.reshape(-1, 1), nodes_space.repeat(T, axis=0)))
                
                elements = elements.cpu().detach().numpy()

                fig = plt.figure(figsize=(int(3*T), 1))
                for i in range(1, T+1):
                    cur_nodes = nodes[(i-1)::T, :]
                    cur_plot = self.Key_Para['res_c_eq'][(i-1)::T, :]

                    pre_cur_nodes = torch.tensor(cur_nodes, requires_grad=True, dtype=torch.float64).cuda()
                    pre_cur_rho = cur_plot
                    cur_rho = np.abs(pre_cur_rho.reshape(-1, 1).cpu().detach().numpy())

                    surface_rho = (cur_rho[elements[:, 0]] + cur_rho[elements[:, 1]] + cur_rho[elements[:, 2]]) / 3
                    color_rho = (surface_rho - surface_rho.min()) / (surface_rho - surface_rho.min()).max()

                    ax = fig.add_subplot(1, T, i, projection='3d')
                    surf = ax.plot_trisurf(cur_nodes[:, 1], cur_nodes[:, 2], cur_nodes[:, 3], triangles=elements, vmin=0, vmax=1, antialiased=False)
                    surf.set_array(color_rho[:, 0])
                    # ax.view_init(elev=65, azim=165)
                    ax.view_init(elev=80, azim=60)
                    # ax.set_xlim3d(0, 1), ax.set_ylim3d(0, 1), #ax.set_zlim3d(cur_nodes[:, 3].min(), cur_nodes[:, 3].max())
                    ax.set_xticks([]),  ax.set_yticks([]), ax.set_zticks([])
                    # ax.set_xlabel('x'), ax.set_ylabel('y'), ax.set_zlabel('z')
                    ax.set_box_aspect([1, 1, 1])
                    cb = fig.colorbar(surf, ax=[ax])
                    cb.set_ticks(np.linspace(surf.norm.vmin, surf.norm.vmax, num=5))
                    scale = np.linspace(surface_rho.min(), surface_rho.max(), num=5)
                    scale1 = []
                    for j in range(0, 5):
                        scale1.append('{:.3f}'.format(scale[j]))
                    cb.set_ticklabels(scale1)
                    plt.title('t=%1.3f' %(t[i-1]))

                plt.savefig(self.Key_Para['File_name'] + '/' + 'Fig-Res_C-eq' + str(ite) + '.png', bbox_inches='tight', dpi=300)

                plt.close()

            elif type_surface == 'Airplane':
                T = int(self.Key_Para['Num_Nodes_t'])
                nodes, elements = self.gen_Nodes.forward()

                t = np.linspace(self.Time[0, 0], self.Time[0, 1], T)
                nodes = nodes.cpu().detach().numpy()
                nodes_space = nodes[0::int(self.Key_Para['Num_Nodes_t']), 1:]
                nodes = np.hstack((t.reshape(-1, 1).repeat(self.Key_Para['Num_Nodes_all_space'], axis=1).T.reshape(-1, 1), nodes_space.repeat(T, axis=0)))
                
                elements = elements.cpu().detach().numpy()

                fig = plt.figure(figsize=(int(3*T), 1))
                for i in range(1, T+1):
                    cur_nodes = nodes[(i-1)::T, :]
                    cur_plot = self.Key_Para['res_c_eq'][(i-1)::T, :]

                    pre_cur_nodes = torch.tensor(cur_nodes, requires_grad=True, dtype=torch.float64).cuda()
                    pre_cur_rho = cur_plot
                    cur_rho = np.abs(pre_cur_rho.reshape(-1, 1).cpu().detach().numpy())

                    surface_rho = (cur_rho[elements[:, 0]] + cur_rho[elements[:, 1]] + cur_rho[elements[:, 2]]) / 3
                    color_rho = (surface_rho - surface_rho.min()) / (surface_rho - surface_rho.min()).max()

                    ax = fig.add_subplot(1, T, i, projection='3d')
                    surf = ax.plot_trisurf(cur_nodes[:, 1], cur_nodes[:, 2], cur_nodes[:, 3], triangles=elements, vmin=0, vmax=1, antialiased=False)
                    surf.set_array(color_rho[:, 0])
                    ax.view_init(elev=155, azim=-90)
                    # ax.view_init(elev=65, azim=165)
                    ax.set_xlim3d(-1, 1), ax.set_ylim3d(-0.5, 0.5), ax.set_zlim3d(-1, 1)
                    ax.set_xticks([]),  ax.set_yticks([]), ax.set_zticks([])
                    # ax.set_xlabel('x'), ax.set_ylabel('y'), ax.set_zlabel('z')
                    ax.set_box_aspect([2, 1, 2])
                    cb = fig.colorbar(surf, ax=[ax])
                    cb.set_ticks(np.linspace(surf.norm.vmin, surf.norm.vmax, num=5))
                    scale = np.linspace(surface_rho.min(), surface_rho.max(), num=5)
                    scale1 = []
                    for j in range(0, 5):
                        scale1.append('{:.3f}'.format(scale[j]))
                    cb.set_ticklabels(scale1)
                    plt.title('t=%1.3f' %(t[i-1]))

                plt.savefig(self.Key_Para['File_name'] + '/' + 'Fig-Res_C-eq' + str(ite) + '.png', bbox_inches='tight', dpi=300)

                plt.close()

            elif type_surface == 'Mazes':
                T = int(self.Key_Para['Num_Nodes_t'])
                nodes, elements = self.gen_Nodes.forward()

                t = np.linspace(self.Time[0, 0], self.Time[0, 1], T)
                nodes = nodes.cpu().detach().numpy()
                nodes_space = nodes[0::int(self.Key_Para['Num_Nodes_t']), 1:]
                nodes = np.hstack((t.reshape(-1, 1).repeat(self.Key_Para['Num_Nodes_all_space'], axis=1).T.reshape(-1, 1), nodes_space.repeat(T, axis=0)))
                
                elements = elements.cpu().detach().numpy()

                fig = plt.figure(figsize=(int(3*T), 1))
                for i in range(1, T+1):
                    cur_nodes = nodes[(i-1)::T, :]
                    cur_plot = self.Key_Para['res_c_eq'][(i-1)::T, :]

                    pre_cur_nodes = torch.tensor(cur_nodes, requires_grad=True, dtype=torch.float64).cuda()
                    pre_cur_rho = cur_plot
                    cur_rho = np.abs(pre_cur_rho.reshape(-1, 1).cpu().detach().numpy())

                    surface_rho = (cur_rho[elements[:, 0]] + cur_rho[elements[:, 1]] + cur_rho[elements[:, 2]]) / 3
                    color_rho = (surface_rho - surface_rho.min()) / (surface_rho - surface_rho.min()).max()

                    ax = fig.add_subplot(1, T, i, projection='3d')
                    surf = ax.plot_trisurf(cur_nodes[:, 1], cur_nodes[:, 2], cur_nodes[:, 3], triangles=elements, vmin=0, vmax=1, antialiased=False)
                    surf.set_array(color_rho[:, 0])
                    ax.view_init(elev=90, azim=45)
                    ax.set_xlim3d(0, 1), ax.set_ylim3d(0, 1), ax.set_zlim3d(0, 0.1)
                    ax.set_xticks([]),  ax.set_yticks([]), ax.set_zticks([])
                    # ax.set_xlabel('x'), ax.set_ylabel('y'), ax.set_zlabel('z')

                    cb = fig.colorbar(surf, ax=[ax])
                    cb.set_ticks(np.linspace(surf.norm.vmin, surf.norm.vmax, num=5))
                    scale = np.linspace(surface_rho.min(), surface_rho.max(), num=5)
                    scale1 = []
                    for j in range(0, 5):
                        scale1.append('{:.3f}'.format(scale[j]))
                    cb.set_ticklabels(scale1)
                    plt.title('t=%1.3f' %(t[i-1]))

                plt.savefig(self.Key_Para['File_name'] + '/' + 'Fig-Res_C-eq' + str(ite) + '.png', bbox_inches='tight', dpi=300)

                plt.close()

        elif self.Dim_space > 3:
            print('There need code')

    def plot_hj_eq(self, ite=''):
        if self.Dim_space == 3:
            type_surface = self.Key_Para['type_surface']
            if type_surface == 'Cylinder':
                r = self.Key_Para['r']
                T = int(self.Key_Para['Num_Nodes_t'])
                nodes, elements = self.gen_Nodes.forward()

                t = np.linspace(self.Time[0, 0], self.Time[0, 1], T)
                nodes = nodes.cpu().detach().numpy()
                nodes_space = nodes[0::int(self.Key_Para['Num_Nodes_t']), 1:]
                nodes = np.hstack((t.reshape(-1, 1).repeat(self.Key_Para['Num_Nodes_all_space'], axis=1).T.reshape(-1, 1), nodes_space.repeat(T, axis=0)))
                
                elements = elements.cpu().detach().numpy()

                fig = plt.figure(figsize=(int(3*T), 1))
                for i in range(1, T+1):
                    cur_nodes = nodes[(i-1)::T, :]
                    cur_plot = self.Key_Para['res_c_eq'][(i-1)::T, :]

                    pre_cur_nodes = torch.tensor(cur_nodes, requires_grad=True, dtype=torch.float64).cuda()
                    pre_cur_rho = cur_plot
                    cur_rho = np.abs(pre_cur_rho.reshape(-1, 1).cpu().detach().numpy())


                    surface_rho = (cur_rho[elements[:, 0]] + cur_rho[elements[:, 1]] + cur_rho[elements[:, 2]]) / 3
                    color_rho = (surface_rho - surface_rho.min()) / (surface_rho - surface_rho.min()).max()

                    ax = fig.add_subplot(1, T, i, projection='3d')
                    surf = ax.plot_trisurf(cur_nodes[:, 1], cur_nodes[:, 2], cur_nodes[:, 3], triangles=elements, vmin=0, vmax=1, antialiased=False)
                    surf.set_array(color_rho[:, 0])
                    ax.view_init(elev=25, azim=45)
                    # ax.view_init(elev=25, azim=135)
                    # ax.set_xlim3d(0, 1), ax.set_ylim3d(0, 1), ax.set_zlim3d(cur_nodes[:, 3].min(), cur_nodes[:, 3].max())
                    ax.set_xticks([]),  ax.set_yticks([]), ax.set_zticks([])
                    # ax.set_xlabel('x'), ax.set_ylabel('y'), ax.set_zlabel('z')
                    ax.set_box_aspect([1, 1, 2])
                    cb = fig.colorbar(surf, ax=[ax])
                    cb.set_ticks(np.linspace(surf.norm.vmin, surf.norm.vmax, num=5))
                    scale = np.linspace(surface_rho.min(), surface_rho.max(), num=5)
                    scale1 = []
                    for j in range(0, 5):
                        scale1.append('{:.3f}'.format(scale[j]))
                    cb.set_ticklabels(scale1)
                    plt.title('t=%1.3f' %(t[i-1]))

                plt.savefig(self.Key_Para['File_name'] + '/' + 'Fig-Res_C-eq' + str(ite) + '.png', bbox_inches='tight', dpi=300)

                plt.close()

            elif type_surface == 'Sphere':
                r = self.Key_Para['r']
                T = int(self.Key_Para['Num_Nodes_t'])
                nodes, elements = self.gen_Nodes.forward()

                t = np.linspace(self.Time[0, 0], self.Time[0, 1], T)
                nodes = nodes.cpu().detach().numpy()
                nodes_space = nodes[0::int(self.Key_Para['Num_Nodes_t']), 1:]
                nodes = np.hstack((t.reshape(-1, 1).repeat(self.Key_Para['Num_Nodes_all_space'], axis=1).T.reshape(-1, 1), nodes_space.repeat(T, axis=0)))
                
                elements = elements.cpu().detach().numpy()

                fig = plt.figure(figsize=(int(3*T), 1))
                for i in range(1, T+1):
                    cur_nodes = nodes[(i-1)::T, :]
                    cur_plot = self.Key_Para['res_HJ_eq'][(i-1)::T, :]

                    pre_cur_nodes = torch.tensor(cur_nodes, requires_grad=True, dtype=torch.float64).cuda()
                    pre_cur_rho = cur_plot
                    cur_rho = np.abs(pre_cur_rho.reshape(-1, 1).cpu().detach().numpy())

                    surface_rho = (cur_rho[elements[:, 0]] + cur_rho[elements[:, 1]] + cur_rho[elements[:, 2]]) / 3
                    color_rho = (surface_rho - surface_rho.min()) / (surface_rho - surface_rho.min()).max()

                    ax = fig.add_subplot(1, T, i, projection='3d')
                    surf = ax.plot_trisurf(cur_nodes[:, 1], cur_nodes[:, 2], cur_nodes[:, 3], triangles=elements, vmin=0, vmax=1, antialiased=False)
                    surf.set_array(color_rho[:, 0])
                    ax.view_init(elev=25, azim=45)
                    # ax.view_init(elev=25, azim=135)
                    ax.set_xlim3d(0, 1), ax.set_ylim3d(0, 1), ax.set_zlim3d(cur_nodes[:, 3].min(), cur_nodes[:, 3].max())
                    ax.set_xticks([]),  ax.set_yticks([]), ax.set_zticks([])
                    # ax.set_xlabel('x'), ax.set_ylabel('y'), ax.set_zlabel('z')
                    ax.set_box_aspect([1, 1, 1])
                    cb = fig.colorbar(surf, ax=[ax])
                    cb.set_ticks(np.linspace(surf.norm.vmin, surf.norm.vmax, num=5))
                    scale = np.linspace(surface_rho.min(), surface_rho.max(), num=5)
                    scale1 = []
                    for j in range(0, 5):
                        scale1.append('{:.3f}'.format(scale[j]))
                    cb.set_ticklabels(scale1)
                    plt.title('t=%1.3f' %(t[i-1]))

                plt.savefig(self.Key_Para['File_name'] + '/' + 'Fig-Res_HJ-eq' + str(ite) + '.png', bbox_inches='tight', dpi=300)

                plt.close()

            elif type_surface == 'Ellipsoid':
                r = self.Key_Para['r']
                T = int(self.Key_Para['Num_Nodes_t'])
                nodes, elements = self.gen_Nodes.forward()

                t = np.linspace(self.Time[0, 0], self.Time[0, 1], T)
                nodes = nodes.cpu().detach().numpy()
                nodes_space = nodes[0::int(self.Key_Para['Num_Nodes_t']), 1:]
                nodes = np.hstack((t.reshape(-1, 1).repeat(self.Key_Para['Num_Nodes_all_space'], axis=1).T.reshape(-1, 1), nodes_space.repeat(T, axis=0)))
                
                elements = elements.cpu().detach().numpy()

                fig = plt.figure(figsize=(int(3*T), 1))
                for i in range(1, T+1):
                    cur_nodes = nodes[(i-1)::T, :]
                    cur_plot = self.Key_Para['res_c_eq'][(i-1)::T, :]

                    pre_cur_nodes = torch.tensor(cur_nodes, requires_grad=True, dtype=torch.float64).cuda()
                    pre_cur_rho = cur_plot
                    cur_rho = np.abs(pre_cur_rho.reshape(-1, 1).cpu().detach().numpy())

                    surface_rho = (cur_rho[elements[:, 0]] + cur_rho[elements[:, 1]] + cur_rho[elements[:, 2]]) / 3
                    color_rho = (surface_rho - surface_rho.min()) / (surface_rho - surface_rho.min()).max()

                    ax = fig.add_subplot(1, T, i, projection='3d')
                    surf = ax.plot_trisurf(cur_nodes[:, 1], cur_nodes[:, 2], cur_nodes[:, 3], triangles=elements, vmin=0, vmax=1, antialiased=False)
                    surf.set_array(color_rho[:, 0])
                    ax.view_init(elev=60, azim=0)
                    # ax.view_init(elev=25, azim=135)
                    ax.set_xlim3d(0, 1), ax.set_ylim3d(0, 1), ax.set_zlim3d(cur_nodes[:, 3].min(), cur_nodes[:, 3].max())
                    ax.set_xticks([]),  ax.set_yticks([]), ax.set_zticks([])
                    # ax.set_xlabel('x'), ax.set_ylabel('y'), ax.set_zlabel('z')
                    ax.set_box_aspect([6, 3, 1])
                    cb = fig.colorbar(surf, ax=[ax])
                    cb.set_ticks(np.linspace(surf.norm.vmin, surf.norm.vmax, num=5))
                    scale = np.linspace(surface_rho.min(), surface_rho.max(), num=5)
                    scale1 = []
                    for j in range(0, 5):
                        scale1.append('{:.3f}'.format(scale[j]))
                    cb.set_ticklabels(scale1)
                    plt.title('t=%1.3f' %(t[i-1]))

                plt.savefig(self.Key_Para['File_name'] + '/' + 'Fig-Res_C-eq' + str(ite) + '.png', bbox_inches='tight', dpi=300)

                plt.close()

            elif type_surface == 'Torus':
                r = self.Key_Para['r']
                T = int(self.Key_Para['Num_Nodes_t'])
                nodes, elements = self.gen_Nodes.forward()

                t = np.linspace(self.Time[0, 0], self.Time[0, 1], T)
                nodes = nodes.cpu().detach().numpy()
                nodes_space = nodes[0::int(self.Key_Para['Num_Nodes_t']), 1:]
                nodes = np.hstack((t.reshape(-1, 1).repeat(self.Key_Para['Num_Nodes_all_space'], axis=1).T.reshape(-1, 1), nodes_space.repeat(T, axis=0)))
                
                elements = elements.cpu().detach().numpy()

                fig = plt.figure(figsize=(int(3*T), 1))
                for i in range(1, T+1):
                    cur_nodes = nodes[(i-1)::T, :]
                    cur_plot = self.Key_Para['res_c_eq'][(i-1)::T, :]

                    pre_cur_nodes = torch.tensor(cur_nodes, requires_grad=True, dtype=torch.float64).cuda()
                    pre_cur_rho = cur_plot
                    cur_rho = np.abs(pre_cur_rho.reshape(-1, 1).cpu().detach().numpy())

                    surface_rho = (cur_rho[elements[:, 0]] + cur_rho[elements[:, 1]] + cur_rho[elements[:, 2]]) / 3
                    color_rho = (surface_rho - surface_rho.min()) / (surface_rho - surface_rho.min()).max()

                    ax = fig.add_subplot(1, T, i, projection='3d')
                    surf = ax.plot_trisurf(cur_nodes[:, 1], cur_nodes[:, 2], cur_nodes[:, 3], triangles=elements, vmin=0, vmax=1, antialiased=False)
                    surf.set_array(color_rho[:, 0])
                    ax.view_init(elev=60, azim=45)
                    # ax.view_init(elev=25, azim=135)
                    # ax.set_xlim3d(0, 1), ax.set_ylim3d(0, 1), ax.set_zlim3d(cur_nodes[:, 3].min(), cur_nodes[:, 3].max())
                    ax.set_xticks([]),  ax.set_yticks([]), ax.set_zticks([])
                    # ax.set_xlabel('x'), ax.set_ylabel('y'), ax.set_zlabel('z')
                    ax.set_box_aspect([10, 10, 2])
                    cb = fig.colorbar(surf, ax=[ax])
                    cb.set_ticks(np.linspace(surf.norm.vmin, surf.norm.vmax, num=5))
                    scale = np.linspace(surface_rho.min(), surface_rho.max(), num=5)
                    scale1 = []
                    for j in range(0, 5):
                        scale1.append('{:.3f}'.format(scale[j]))
                    cb.set_ticklabels(scale1)
                    plt.title('t=%1.3f' %(t[i-1]))

                plt.savefig(self.Key_Para['File_name'] + '/' + 'Fig-Res_C-eq' + str(ite) + '.png', bbox_inches='tight', dpi=300)

                plt.close()

            elif type_surface == 'Peanut':
                r = self.Key_Para['r']
                T = int(self.Key_Para['Num_Nodes_t'])
                nodes, elements = self.gen_Nodes.forward()

                t = np.linspace(self.Time[0, 0], self.Time[0, 1], T)
                nodes = nodes.cpu().detach().numpy()
                nodes_space = nodes[0::int(self.Key_Para['Num_Nodes_t']), 1:]
                nodes = np.hstack((t.reshape(-1, 1).repeat(self.Key_Para['Num_Nodes_all_space'], axis=1).T.reshape(-1, 1), nodes_space.repeat(T, axis=0)))
                
                elements = elements.cpu().detach().numpy()

                fig = plt.figure(figsize=(int(3*T), 1))
                for i in range(1, T+1):
                    cur_nodes = nodes[(i-1)::T, :]
                    cur_plot = self.Key_Para['res_c_eq'][(i-1)::T, :]

                    pre_cur_nodes = torch.tensor(cur_nodes, requires_grad=True, dtype=torch.float64).cuda()
                    pre_cur_rho = cur_plot
                    cur_rho = np.abs(pre_cur_rho.reshape(-1, 1).cpu().detach().numpy())

                    surface_rho = (cur_rho[elements[:, 0]] + cur_rho[elements[:, 1]] + cur_rho[elements[:, 2]]) / 3
                    color_rho = (surface_rho - surface_rho.min()) / (surface_rho - surface_rho.min()).max()

                    ax = fig.add_subplot(1, T, i, projection='3d')
                    surf = ax.plot_trisurf(cur_nodes[:, 1], cur_nodes[:, 2], cur_nodes[:, 3], triangles=elements, vmin=0, vmax=1, antialiased=False)
                    surf.set_array(color_rho[:, 0])
                    ax.view_init(elev=45, azim=45)
                    # ax.view_init(elev=25, azim=135)
                    # ax.set_xlim3d(0, 1), ax.set_ylim3d(0, 1), ax.set_zlim3d(cur_nodes[:, 3].min(), cur_nodes[:, 3].max())
                    ax.set_xticks([]),  ax.set_yticks([]), ax.set_zticks([])
                    # ax.set_xlabel('x'), ax.set_ylabel('y'), ax.set_zlabel('z')
                    ax.set_box_aspect([6, 4, 4])
                    cb = fig.colorbar(surf, ax=[ax])
                    cb.set_ticks(np.linspace(surf.norm.vmin, surf.norm.vmax, num=5))
                    scale = np.linspace(surface_rho.min(), surface_rho.max(), num=5)
                    scale1 = []
                    for j in range(0, 5):
                        scale1.append('{:.3f}'.format(scale[j]))
                    cb.set_ticklabels(scale1)
                    plt.title('t=%1.3f' %(t[i-1]))

                plt.savefig(self.Key_Para['File_name'] + '/' + 'Fig-Res_C-eq' + str(ite) + '.png', bbox_inches='tight', dpi=300)

                plt.close()

            elif type_surface == 'Opener':
                T = int(self.Key_Para['Num_Nodes_t'])
                nodes, elements = self.gen_Nodes.forward()

                t = np.linspace(self.Time[0, 0], self.Time[0, 1], T)
                nodes = nodes.cpu().detach().numpy()
                nodes_space = nodes[0::int(self.Key_Para['Num_Nodes_t']), 1:]
                nodes = np.hstack((t.reshape(-1, 1).repeat(self.Key_Para['Num_Nodes_all_space'], axis=1).T.reshape(-1, 1), nodes_space.repeat(T, axis=0)))
                
                elements = elements.cpu().detach().numpy()

                fig = plt.figure(figsize=(int(3*T), 1))
                for i in range(1, T+1):
                    cur_nodes = nodes[(i-1)::T, :]
                    cur_plot = self.Key_Para['res_c_eq'][(i-1)::T, :]

                    pre_cur_nodes = torch.tensor(cur_nodes, requires_grad=True, dtype=torch.float64).cuda()
                    pre_cur_rho = cur_plot
                    cur_rho = np.abs(pre_cur_rho.reshape(-1, 1).cpu().detach().numpy())

                    ax = fig.add_subplot(1, T, i, projection='3d')
                    surf = ax.plot_trisurf(cur_nodes[:, 1], cur_nodes[:, 2], cur_nodes[:, 3], triangles=elements, vmin=0, vmax=1, antialiased=False)
                    surf.set_array(color_rho[:, 0])
                    ax.view_init(elev=45, azim=45)
                    # ax.view_init(elev=25, azim=135)
                    # ax.set_xlim3d(0, 1), ax.set_ylim3d(0, 1), ax.set_zlim3d(cur_nodes[:, 3].min(), cur_nodes[:, 3].max())
                    ax.set_xticks([]),  ax.set_yticks([]), ax.set_zticks([])
                    # ax.set_xlabel('x'), ax.set_ylabel('y'), ax.set_zlabel('z')
                    ax.set_box_aspect([10, 8, 2])
                    cb = fig.colorbar(surf, ax=[ax])
                    cb.set_ticks(np.linspace(surf.norm.vmin, surf.norm.vmax, num=5))
                    scale = np.linspace(surface_rho.min(), surface_rho.max(), num=5)
                    scale1 = []
                    for j in range(0, 5):
                        scale1.append('{:.3f}'.format(scale[j]))
                    cb.set_ticklabels(scale1)
                    plt.title('t=%1.3f' %(t[i-1]))

                plt.savefig(self.Key_Para['File_name'] + '/' + 'Fig-Res_C-eq' + str(ite) + '.png', bbox_inches='tight', dpi=300)

                plt.close()

            elif type_surface == 'Cube':
                r = self.Key_Para['r']
                T = int(self.Key_Para['Num_Nodes_t'])
                nodes, elements = self.gen_Nodes.forward()

                t = np.linspace(self.Time[0, 0], self.Time[0, 1], T)
                nodes = nodes.cpu().detach().numpy()
                nodes_space = nodes[0::int(self.Key_Para['Num_Nodes_t']), 1:]
                nodes = np.hstack((t.reshape(-1, 1).repeat(self.Key_Para['Num_Nodes_all_space'], axis=1).T.reshape(-1, 1), nodes_space.repeat(T, axis=0)))
                
                elements = elements.cpu().detach().numpy()

                fig = plt.figure(figsize=(int(3*T), 1))
                for i in range(1, T+1):
                    cur_nodes = nodes[(i-1)::T, :]
                    cur_plot = self.Key_Para['res_c_eq'][(i-1)::T, :]

                    pre_cur_nodes = torch.tensor(cur_nodes, requires_grad=True, dtype=torch.float64).cuda()
                    pre_cur_rho = cur_plot
                    cur_rho = np.abs(pre_cur_rho.reshape(-1, 1).cpu().detach().numpy())

                    surface_rho = (cur_rho[elements[:, 0]] + cur_rho[elements[:, 1]] + cur_rho[elements[:, 2]]) / 3
                    color_rho = (surface_rho - surface_rho.min()) / (surface_rho - surface_rho.min()).max()

                    ax = fig.add_subplot(1, T, i, projection='3d')
                    surf = ax.plot_trisurf(cur_nodes[:, 1], cur_nodes[:, 2], cur_nodes[:, 3], triangles=elements, vmin=0, vmax=1, antialiased=False)
                    surf.set_array(color_rho[:, 0])
                    ax.view_init(elev=60, azim=45)
                    # ax.view_init(elev=25, azim=135)
                    ax.set_xlim3d(-0.5, 1.5), ax.set_ylim3d(-0.5, 1.5), ax.set_zlim3d(-0.5, 1.5)
                    ax.set_xticks([]),  ax.set_yticks([]), ax.set_zticks([])
                    # ax.set_xlabel('x'), ax.set_ylabel('y'), ax.set_zlabel('z')
                    ax.set_box_aspect([1, 1, 1])
                    cb = fig.colorbar(surf, ax=[ax])
                    cb.set_ticks(np.linspace(surf.norm.vmin, surf.norm.vmax, num=5))
                    scale = np.linspace(surface_rho.min(), surface_rho.max(), num=5)
                    scale1 = []
                    for j in range(0, 5):
                        scale1.append('{:.3f}'.format(scale[j]))
                    cb.set_ticklabels(scale1)
                    plt.title('t=%1.3f' %(t[i-1]))

                plt.savefig(self.Key_Para['File_name'] + '/' + 'Fig-Res_C-eq' + str(ite) + '.png', bbox_inches='tight', dpi=300)

                plt.close()

            elif type_surface == 'Hand':
                T = int(self.Key_Para['Num_Nodes_t'])
                nodes, elements = self.gen_Nodes.forward()

                t = np.linspace(self.Time[0, 0], self.Time[0, 1], T)
                nodes = nodes.cpu().detach().numpy()
                nodes_space = nodes[0::int(self.Key_Para['Num_Nodes_t']), 1:]
                nodes = np.hstack((t.reshape(-1, 1).repeat(self.Key_Para['Num_Nodes_all_space'], axis=1).T.reshape(-1, 1), nodes_space.repeat(T, axis=0)))
                
                elements = elements.cpu().detach().numpy()

                fig = plt.figure(figsize=(int(3*T), 1))
                for i in range(1, T+1):
                    cur_nodes = nodes[(i-1)::T, :]
                    cur_plot = self.Key_Para['res_c_eq'][(i-1)::T, :]

                    pre_cur_nodes = torch.tensor(cur_nodes, requires_grad=True, dtype=torch.float64).cuda()
                    pre_cur_rho = cur_plot
                    cur_rho = np.abs(pre_cur_rho.reshape(-1, 1).cpu().detach().numpy())

                    surface_rho = (cur_rho[elements[:, 0]] + cur_rho[elements[:, 1]] + cur_rho[elements[:, 2]]) / 3
                    color_rho = (surface_rho - surface_rho.min()) / (surface_rho - surface_rho.min()).max()

                    ax = fig.add_subplot(1, T, i, projection='3d')
                    surf = ax.plot_trisurf(cur_nodes[:, 1], cur_nodes[:, 2], cur_nodes[:, 3], triangles=elements, vmin=0, vmax=1, antialiased=False)
                    surf.set_array(color_rho[:, 0])
                    # ax.view_init(elev=65, azim=165)
                    ax.view_init(elev=80, azim=60)
                    # ax.set_xlim3d(0, 1), ax.set_ylim3d(0, 1), #ax.set_zlim3d(cur_nodes[:, 3].min(), cur_nodes[:, 3].max())
                    ax.set_xticks([]),  ax.set_yticks([]), ax.set_zticks([])
                    # ax.set_xlabel('x'), ax.set_ylabel('y'), ax.set_zlabel('z')
                    ax.set_box_aspect([1, 1, 1])
                    cb = fig.colorbar(surf, ax=[ax])
                    cb.set_ticks(np.linspace(surf.norm.vmin, surf.norm.vmax, num=5))
                    scale = np.linspace(surface_rho.min(), surface_rho.max(), num=5)
                    scale1 = []
                    for j in range(0, 5):
                        scale1.append('{:.3f}'.format(scale[j]))
                    cb.set_ticklabels(scale1)
                    plt.title('t=%1.3f' %(t[i-1]))

                plt.savefig(self.Key_Para['File_name'] + '/' + 'Fig-Res_C-eq' + str(ite) + '.png', bbox_inches='tight', dpi=300)

                plt.close()

            elif type_surface == 'Airplane':
                T = int(self.Key_Para['Num_Nodes_t'])
                nodes, elements = self.gen_Nodes.forward()

                t = np.linspace(self.Time[0, 0], self.Time[0, 1], T)
                nodes = nodes.cpu().detach().numpy()
                nodes_space = nodes[0::int(self.Key_Para['Num_Nodes_t']), 1:]
                nodes = np.hstack((t.reshape(-1, 1).repeat(self.Key_Para['Num_Nodes_all_space'], axis=1).T.reshape(-1, 1), nodes_space.repeat(T, axis=0)))
                
                elements = elements.cpu().detach().numpy()

                fig = plt.figure(figsize=(int(3*T), 1))
                for i in range(1, T+1):
                    cur_nodes = nodes[(i-1)::T, :]
                    cur_plot = self.Key_Para['res_c_eq'][(i-1)::T, :]

                    pre_cur_nodes = torch.tensor(cur_nodes, requires_grad=True, dtype=torch.float64).cuda()
                    pre_cur_rho = cur_plot
                    cur_rho = np.abs(pre_cur_rho.reshape(-1, 1).cpu().detach().numpy())

                    surface_rho = (cur_rho[elements[:, 0]] + cur_rho[elements[:, 1]] + cur_rho[elements[:, 2]]) / 3
                    color_rho = (surface_rho - surface_rho.min()) / (surface_rho - surface_rho.min()).max()

                    ax = fig.add_subplot(1, T, i, projection='3d')
                    surf = ax.plot_trisurf(cur_nodes[:, 1], cur_nodes[:, 2], cur_nodes[:, 3], triangles=elements, vmin=0, vmax=1, antialiased=False)
                    surf.set_array(color_rho[:, 0])
                    ax.view_init(elev=155, azim=-90)
                    # ax.view_init(elev=65, azim=165)
                    ax.set_xlim3d(-1, 1), ax.set_ylim3d(-0.5, 0.5), ax.set_zlim3d(-1, 1)
                    ax.set_xticks([]),  ax.set_yticks([]), ax.set_zticks([])
                    # ax.set_xlabel('x'), ax.set_ylabel('y'), ax.set_zlabel('z')
                    ax.set_box_aspect([2, 1, 2])
                    cb = fig.colorbar(surf, ax=[ax])
                    cb.set_ticks(np.linspace(surf.norm.vmin, surf.norm.vmax, num=5))
                    scale = np.linspace(surface_rho.min(), surface_rho.max(), num=5)
                    scale1 = []
                    for j in range(0, 5):
                        scale1.append('{:.3f}'.format(scale[j]))
                    cb.set_ticklabels(scale1)
                    plt.title('t=%1.3f' %(t[i-1]))

                plt.savefig(self.Key_Para['File_name'] + '/' + 'Fig-Res_C-eq' + str(ite) + '.png', bbox_inches='tight', dpi=300)

                plt.close()

            elif type_surface == 'Mazes':
                T = int(self.Key_Para['Num_Nodes_t'])
                nodes, elements = self.gen_Nodes.forward()

                t = np.linspace(self.Time[0, 0], self.Time[0, 1], T)
                nodes = nodes.cpu().detach().numpy()
                nodes_space = nodes[0::int(self.Key_Para['Num_Nodes_t']), 1:]
                nodes = np.hstack((t.reshape(-1, 1).repeat(self.Key_Para['Num_Nodes_all_space'], axis=1).T.reshape(-1, 1), nodes_space.repeat(T, axis=0)))
                
                elements = elements.cpu().detach().numpy()

                fig = plt.figure(figsize=(int(3*T), 1))
                for i in range(1, T+1):
                    cur_nodes = nodes[(i-1)::T, :]
                    cur_plot = self.Key_Para['res_c_eq'][(i-1)::T, :]

                    pre_cur_nodes = torch.tensor(cur_nodes, requires_grad=True, dtype=torch.float64).cuda()
                    pre_cur_rho = cur_plot
                    cur_rho = np.abs(pre_cur_rho.reshape(-1, 1).cpu().detach().numpy())

                    surface_rho = (cur_rho[elements[:, 0]] + cur_rho[elements[:, 1]] + cur_rho[elements[:, 2]]) / 3
                    color_rho = (surface_rho - surface_rho.min()) / (surface_rho - surface_rho.min()).max()

                    ax = fig.add_subplot(1, T, i, projection='3d')
                    surf = ax.plot_trisurf(cur_nodes[:, 1], cur_nodes[:, 2], cur_nodes[:, 3], triangles=elements, vmin=0, vmax=1, antialiased=False)
                    surf.set_array(color_rho[:, 0])
                    ax.view_init(elev=90, azim=45)
                    ax.set_xlim3d(0, 1), ax.set_ylim3d(0, 1), ax.set_zlim3d(0, 0.1)
                    ax.set_xticks([]),  ax.set_yticks([]), ax.set_zticks([])
                    # ax.set_xlabel('x'), ax.set_ylabel('y'), ax.set_zlabel('z')

                    cb = fig.colorbar(surf, ax=[ax])
                    cb.set_ticks(np.linspace(surf.norm.vmin, surf.norm.vmax, num=5))
                    scale = np.linspace(surface_rho.min(), surface_rho.max(), num=5)
                    scale1 = []
                    for j in range(0, 5):
                        scale1.append('{:.3f}'.format(scale[j]))
                    cb.set_ticklabels(scale1)
                    plt.title('t=%1.3f' %(t[i-1]))

                plt.savefig(self.Key_Para['File_name'] + '/' + 'Fig-Res_C-eq' + str(ite) + '.png', bbox_inches='tight', dpi=300)

                plt.close()

        elif self.Dim_space > 3:
            print('There need code')


def main(Key_Para):
    utilize = Utilize(Key_Para)
    utilize.make_file()
    utilize.setup_seed(1)
    utilize.mu_0_mu_1()
    utilize.sigma_0_sigma_1()
    utilize.Time_Space()
    utilize.print_key(Key_Para)


    gen_Nodes = Gnerate_node(Key_Para)
    model_rho, model_phi, model_g = Model_rho(Key_Para), Model_phi(Key_Para), Model_g(Key_Para)
    loss_net = Loss_Net(model_rho, model_phi, model_g, Key_Para).cuda()
    PR = Plot_Result(Key_Para, gen_Nodes)
    train_net = Train_net(Key_Para, gen_Nodes, model_rho, model_phi, model_g, loss_net, PR)
    all_sub_loss = train_net.train()

    compute_W2 = Compute_Wasserstein_distance(Key_Para, all_sub_loss)
    compute_W2.compute()


    PR.plot_rho(model_rho)
    PR.plot_v(model_phi)
    PR.plot_g(model_g)
    PR.plot_loss(all_sub_loss)
    PR.plot_c_eq()
    # PR.plot_hj_eq()



if __name__== "__main__" :
    time_begin = time.time()
    time_now = time.strftime("%Y%m%d-%H%M", time.localtime())
    name = os.path.basename(sys.argv[0])
    File_name = time_now + '-' + name[:-3]

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    torch.cuda.set_device(0)
    torch.set_default_dtype(torch.float64)


    test = 'Sphere'
    if test == 'Cylinder':
        Dim_time = 1
        Dim_space = 3
        Num_Nodes_t = 10
        Num_Nodes_all_space = (30)**2
        mu_0, mu_1 = [0.75, 0.5, 0.5], [0.25, 0.5, 0.5] 
        sigma_0, sigma_1 = 0.01, 0.01
        Time, Space = [0.0, 1.0], [0.0, 1.0]

        epochs_sample = 1
        epochs_train = 3000
        learning_rate = 1e-3
        beta = [0, 1000, 1000, 1000, 1]

        type_print = 'True'           # 'True'      'False' 
        type_surface = 'Cylinder'   
        type_node = 'Load'         # 'Load'  'Generate' 
        type_pre_train = 'True'   # 'True'  'False'
        type_pre_plot = 'True'   # 'True'  'False'
        type_lr = 'General'   # 'Gradual-Decline'   'General'
        type_loss = 'General'     # 'Res-Attention'   'General'
        type_UOT = 0

    elif test == 'Sphere':
        Dim_time = 1
        Dim_space = 3
        Num_Nodes_t = 10
        Num_Nodes_all_space = (30)**2
        # mu_0, mu_1 = [1.0, 0.5, 0.5], [0.5, 1.0, 0.5]
        # mu_0, mu_1 = [1.0, 0.5, 0.5], [0.5, 0.5, 1.0]  
        mu_0, mu_1 = [0.5, 0.5, 0.0], [0.5, 0.5, 1.0]  
        # mu_0, mu_1 = [0.5 - np.sqrt(3)/6, 0.5 - np.sqrt(3)/6, 0.5 - np.sqrt(3)/6], [0.5 + np.sqrt(3)/6, 0.5 + np.sqrt(3)/6, 0.5 + np.sqrt(3)/6]
        sigma_0, sigma_1 = 0.05, 0.05
        Time, Space = [0.0, 1.0], [0.0, 1.0]

        epochs_sample = 1
        epochs_train = 10001
        learning_rate = 1e-3
        beta = [0, 10, 10, 1000, 0, 0]
        break_delta = 1e-1
        noise = 0.01

        type_print = 'False'           # 'True'      'False' 
        type_surface = 'Sphere'  
        type_node = 'Load'         # 'Load'  'Generate' 
        type_pre_train = 'False'   # 'True'  'False'
        type_pre_plot = 'True'   # 'True'  'False'
        type_lr = 'General'   # 'Gradual-Decline'   'General'
        type_loss = 'General'     # 'Res-Attention'   'General'
        type_UOT = 0

    elif test == 'Ellipsoid':
        Dim_time = 1
        Dim_space = 3
        Num_Nodes_t = 10
        Num_Nodes_all_space = (30)**2
        mu_0, mu_1 = [0.5, 0.5, 0.5 + np.sqrt(1/24)], [0.5, 0.5, 0.5 - np.sqrt(1/24)]  
        sigma_0, sigma_1 = 0.01, 0.01
        Time, Space = [0.0, 1.0], [0.0, 1.0]

        epochs_sample = 1
        epochs_train = 3000
        learning_rate = 1e-3
        beta = [0, 1, 1, 1000, 0, 0]

        type_print = 'True'           # 'True'      'False' 
        type_surface = test  
        type_node = 'Load'         # 'Load'  'Generate' 
        type_pre_train = 'False'   # 'True'  'False'
        type_pre_plot = 'True'   # 'True'  'False'
        type_lr = 'General'   # 'Gradual-Decline'   'General'
        type_loss = 'General'     # 'Res-Attention'   'General'
        type_UOT = 0

    elif test == 'Torus':
        Dim_time = 1
        Dim_space = 3
        Num_Nodes_t = 10
        Num_Nodes_all_space = (30)**2
        # mu_0, mu_1 = [0.5 + 0.4*np.sqrt(1/2), 0.5 + 0.4*np.sqrt(1/2), 0.5], [0.5 - 0.4*np.sqrt(1/2), 0.5 - 0.4*np.sqrt(1/2), 0.5]
        mu_0, mu_1 = [0.5 + 0.5*np.sqrt(1/2), 0.5 + 0.5*np.sqrt(1/2), 0.5], [0.5 - 0.5*np.sqrt(1/2), 0.5 - 0.5*np.sqrt(1/2), 0.5]
        sigma_0, sigma_1 = 0.05, 0.05
        Time, Space = [0.0, 1.0], [0.0, 1.0]

        epochs_sample = 1
        epochs_train = 10001
        learning_rate = 1e-3
        beta = [0, 10, 10, 1000, 0, 0]
        break_delta = 1e-1
        noise = 0

        type_print = 'False'           # 'True'      'False' 
        type_surface = test  
        type_node = 'Load'         # 'Load'  'Generate' 
        type_pre_train = 'False'   # 'True'  'False'
        type_pre_plot = 'True'   # 'True'  'False'
        type_lr = 'General'   # 'Gradual-Decline'   'General'
        type_loss = 'General'     # 'Res-Attention'   'General'
        type_UOT = 0

    elif test == 'Peanut':
        Dim_time = 1
        Dim_space = 3
        Num_Nodes_t = 10
        Num_Nodes_all_space = (30)**2
        mu_0, mu_1 = [0.5 + 0.25*np.sqrt(1+np.sqrt(1.2)), 0.5, 0.5], [0.5 - 0.25*np.sqrt(1+np.sqrt(1.2)), 0.5, 0.5],
        sigma_0, sigma_1 = 0.01, 0.01
        Time, Space = [0.0, 1.0], [0.0, 1.0]

        epochs_sample = 1
        epochs_train = 3000
        learning_rate = 1e-3
        beta = [0, 1, 1, 1000, 0, 0]

        type_print = 'True'           # 'True'      'False' 
        type_surface = test  
        type_node = 'Load'         # 'Load'  'Generate' 
        type_pre_train = 'False'   # 'True'  'False'
        type_pre_plot = 'True'   # 'True'  'False'
        type_lr = 'General'   # 'Gradual-Decline'   'General'
        type_loss = 'General'     # 'Res-Attention'   'General'
        type_UOT = 0

    elif test == 'Opener':
        Dim_time = 1
        Dim_space = 3
        Num_Nodes_t = 10
        Num_Nodes_all_space = (30)**2
        # mu_0, mu_1 = [0.5 + np.sqrt((1 + np.sqrt(1 + 2*np.sqrt(1/15)))/10), 0.5, 0.5], \
        #              [0.5 - np.sqrt((1 + np.sqrt(1 + 2*np.sqrt(1/15)))/10), 0.5, 0.5]
        mu_0, mu_1 = [0.5 + np.sqrt((3 + np.sqrt(9 - 60*np.sqrt(1/60)))/30), 0.5, 0.5], \
                     [0.5 - np.sqrt((3 + np.sqrt(9 - 60*np.sqrt(1/60)))/30), 0.5, 0.5]
        sigma_0, sigma_1 = 0.01, 0.01
        Time, Space = [0.0, 1.0], [0.0, 1.0]


        epochs_sample = 1
        epochs_train = 3000
        learning_rate = 1e-3
        beta = [0, 1, 1, 1000, 0]

        type_print = 'True'           # 'True'      'False' 
        type_surface = 'Opener'  
        type_node = 'Load'         # 'Load'  'Generate' 
        type_pre_train = 'False'   # 'True'  'False'
        type_pre_plot = 'True'   # 'True'  'False'
        type_lr = 'General'   # 'Gradual-Decline'   'General'
        type_loss = 'General'     # 'Res-Attention'   'General'
        type_UOT = 0

    elif test == 'Cube':
        Dim_time = 1
        Dim_space = 3
        Num_Nodes_t = 10
        Num_Nodes_all_space = (30)**2
        # mu_0, mu_1 = [0.5, 1, 0.5], [0.5, 0.5, 1]
        mu_0, mu_1 = [0.5, 0.5, 0], [0.5, 0.5, 1]
        sigma_0, sigma_1 = 0.05, 0.05
        Time, Space = [0.0, 1.0], [0.0, 1.0]

        epochs_sample = 1
        epochs_train = 3000
        learning_rate = 1e-3
        beta = [0, 100, 100, 1000, 1]
        # beta = cur_list

        type_print = 'False'           # 'True'      'False' 
        type_surface = test  
        type_node = 'Load'         # 'Load'  'Generate' 
        type_pre_train = 'False'   # 'True'  'False'
        type_pre_plot = 'True'   # 'True'  'False'
        type_lr = 'General'   # 'Gradual-Decline'   'General'
        type_loss = 'General'     # 'Res-Attention'   'General'
        type_UOT = 0

    elif test == 'Hand':
        Dim_time = 1
        Dim_space = 3
        Num_Nodes_t = 10
        Num_Nodes_all_space = (30)**2
        mu_0, mu_1 = [0, 0.5, 0.5], [1, 0.5, 0.5]
        sigma_0, sigma_1 = 0.01, 0.01
        Time, Space = [0.0, 1.0], [0.0, 1.0]

        epochs_sample = 1
        epochs_train = 3000
        learning_rate = 5e-3
        beta = [1, 10, 1, 10, 1, 1]

        type_print = 'True'           # 'True'      'False' 
        type_surface = 'Hand'   
        type_node = 'Load'         # 'Load'  'Generate' 
        type_pre_train = 'False'   # 'True'  'False'
        type_pre_plot = 'True'   # 'True'  'False'
        type_lr = 'General'   # 'Gradual-Decline'   'General'
        type_loss = 'General'     # 'Res-Attention'   'General'
        type_UOT = 0

    elif test == 'Airplane':
        Dim_time = 1
        Dim_space = 3
        Num_Nodes_t = 10
        Num_Nodes_all_space = (30)**2
        mu_0, mu_1 = [0, 0.5, 0.5 + np.sqrt(1/20)/2], [1, 0.5, 0.5 + np.sqrt(1/20)/2]
        sigma_0, sigma_1 = 0.01, 0.01
        Time, Space = [0.0, 1.0], [0.0, 1.0]

        epochs_sample = 1
        epochs_train = 3000
        learning_rate = 1e-3
        beta = [1, 10, 1, 10, 1, 1]

        type_print = 'True'           # 'True'      'False' 
        type_surface = 'Airplane'   
        type_node = 'Load'         # 'Load'  'Generate' 
        type_pre_train = 'False'   # 'True'  'False'
        type_pre_plot = 'True'   # 'True'  'False'
        type_lr = 'General'   # 'Gradual-Decline'   'General'
        type_loss = 'General'     # 'Res-Attention'   'General'
        type_UOT = 0

    elif test == 'Mazes':
        Dim_time = 1
        Dim_space = 3
        Num_Nodes_t = 10
        Num_Nodes_all_space = (30)**2
        mu_0, mu_1 = [0.1, 0.825, 0], [0.8, 0.1, 0]
        sigma_0, sigma_1 = 0.0015, 0.0015
        Time, Space = [0.0, 1.0], [0.0, 1.0]

        epochs_sample = 1
        epochs_train = 3000
        learning_rate = 1e-3
        beta = [0, 1, 1, 1000, 1]

        type_print = 'True'           # 'True'      'False' 
        type_surface = 'Mazes'   
        type_node = 'Load'         # 'Load'  'Generate' 
        type_pre_train = 'False'   # 'True'  'False'
        type_pre_plot = 'True'   # 'True'  'False'
        type_lr = 'General'   # 'Gradual-Decline'   'General'
        type_loss = 'General'     # 'Res-Attention'   'General'
        type_UOT = 0



    File_name = File_name + '_' + test
    Key_Parameters = {
        'test': test,
        'File_name': File_name,
        'Dim_time': Dim_time, 
        'Dim_space': Dim_space,
        'Num_Nodes_t': Num_Nodes_t, 
        'Num_Nodes_all_space': Num_Nodes_all_space,
        'mu_0': mu_0, 
        'mu_1': mu_1, 
        'sigma_0': sigma_0, 
        'sigma_1': sigma_1, 
        'Time': Time, 
        'Space': Space,
        'epochs_sample': epochs_sample, 
        'epochs_train': epochs_train, 
        'learning_rate': learning_rate,
        'beta': beta, 
        'break_delta': break_delta,
        'noise': noise,
        'type_print': type_print,
        'type_node': type_node, 
        'type_lr': type_lr, 
        'type_loss': type_loss,
        'type_surface': type_surface,
        'type_pre_train': type_pre_train,
        'type_pre_plot': type_pre_plot,
        'type_UOT': type_UOT, 
            }

    main(Key_Parameters)
    print('Runing_time:', time.time() - time_begin, 's')
