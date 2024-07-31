from scipy import stats
import math, os, time, torch, copy, sys, random, inspect, matplotlib

import pprint as pp
import matplotlib.pyplot as plt
import numpy as np
import torch.nn as nn
from matplotlib import cm



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
        if len(self.Key_Para['mu_0']) == self.Dim_space and len(self.Key_Para['mu_1']) == self.Dim_space:
            mu_0, mu_1 = np.array([self.Key_Para['mu_0']]), np.array([self.Key_Para['mu_1']])
        elif len(self.Key_Para['mu_0']) != self.Dim_space and len(self.Key_Para['mu_1']) != self.Dim_space:
            mu_0, mu_1 = np.array([self.Key_Para['mu_0']]), np.array([self.Key_Para['mu_1']])
            mu_0, mu_1 = mu_0.repeat(self.Dim_space, axis=1), mu_1.repeat(self.Dim_space, axis=1)
        
        if self.Dim_space > 2:
            mu_0[0, 2:] = np.zeros((1, self.Dim_space-2))
            mu_1[0, 2:] = np.zeros((1, self.Dim_space-2))
        
        self.Key_Para['mu_0'], self.Key_Para['mu_1'] = mu_0, mu_1

    def sigma_0_sigma_1(self):
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


class Model_rho(nn.Module):
    def __init__(self, Key_Para): 
        super(Model_rho, self).__init__()
        self.Key_Para = Key_Para
        self.Dim_time = Key_Para['Dim_time']
        self.Dim_space = Key_Para['Dim_space']

        self.activation = nn.Tanh()
        self.num_layer = 5
        self.num_neuron = 100
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
        # x0 = Gaussian_distribution_torch(self.Key_Para['mu_0'], self.Key_Para['sigma_0'], nodes)
        x1 = self.net_rho(nodes)
        # x2 = 2*Gaussian_distribution_torch(self.Key_Para['mu_1'], self.Key_Para['sigma_1'], nodes)
        # out = (1 - cur_t) * x0 + (cur_t - 0) * (1 - cur_t) * x1  + (cur_t - 0) * x2
        out = fun_relu(x1)

        return out


class Model_phi(nn.Module):
    def __init__(self, Key_Para): 
        super(Model_phi, self).__init__()
        self.Dim_time = Key_Para['Dim_time']
        self.Dim_space = Key_Para['Dim_space']

        self.activation = nn.Tanh()
        self.num_layer = 5
        self.num_neuron = 100
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
        out = self.net_phi(nodes)
        return out


def gradients(outputs, inputs):
    return torch.autograd.grad(outputs, inputs, grad_outputs=torch.ones_like(outputs), create_graph=True)


def Gaussian_distribution_torch(mu, sigma, x):
    mu = torch.tensor(mu, requires_grad=False, dtype=torch.float64).cuda()
    sigma = torch.tensor(sigma, requires_grad=False, dtype=torch.float64).cuda()

    x = x[:, 1:]
    d = x.shape[1]

    coefficient = (1 / torch.sqrt(((2 * np.pi)**d) * torch.det(sigma)))
    if Dim_space == 1:
        out = coefficient * (torch.diag(torch.exp(-0.5 * (x - mu) * (torch.linalg.inv(sigma) * (x - mu).T))))
    else:
        out = coefficient * (torch.diag(torch.exp(-0.5 * torch.mm((x - mu), torch.mm(torch.linalg.inv(sigma), (x - mu).T)))))
    out = out.reshape((-1, 1))
    return out


class Loss_Net(nn.Module):
    def __init__(self, model_rho, model_phi, Key_Para):
        super(Loss_Net, self).__init__()
        self.Key_Para = Key_Para
        self.model_rho = model_rho
        self.model_phi = model_phi
        self.type_loss = Key_Para['type_loss']
        self.beta = Key_Para['beta']
        self.num_losses = len(Key_Para['beta'])
        self.params = torch.nn.Parameter(torch.ones(self.num_losses, requires_grad=True)) 

    def forward(self, nodes):
        def get_variable_name(var):
            def retrieve_name(var):
                for fi in inspect.stack()[2:]:
                    for item in fi.frame.f_locals.items():
                        if var is item[1]:
                            return item[0]
                return ""
            return retrieve_name(var)

        def W2(nodes):
            cur_rho = self.model_rho(nodes)
            cur_phi = self.model_phi(nodes)
            cur_v = gradients(cur_phi, nodes)[0][:, 1:]
            cur_g = cur_phi

            out = 0.5*torch.mean(torch.mul(cur_rho, torch.sum(torch.mul(cur_v, cur_v), dim=1).reshape(-1, 1))) + \
                torch.mean(torch.mul(cur_rho, torch.mul(cur_g, cur_g)))
            return out

        def Continuity_Equation(nodes):
            nodes = nodes[torch.where(nodes[:,0]!= self.Key_Para['Time'][0, 0])[0], :]
            nodes = nodes[torch.where(nodes[:,0]!= self.Key_Para['Time'][0, 1])[0], :]
            cur_rho = self.model_rho(nodes)
            cur_phi = self.model_phi(nodes)
            cur_v = gradients(cur_phi, nodes)[0][:, 1:]
            cur_g = cur_phi
            cur_grad_rho_t = gradients(cur_rho, nodes)[0][:, 0:1]

            rho_v = torch.mul(cur_rho.repeat(1, Dim_space), cur_v)  
            div_cur_rho_v = 0
            for i in range(Dim_space):
                grad_cur_rho_v = gradients(rho_v[:, i], nodes)[0]
                div_cur_rho_v = div_cur_rho_v + grad_cur_rho_v[:, i+1]

            cur_rho_g = torch.mul(cur_rho, cur_g)

            out = cur_grad_rho_t + 0.5*div_cur_rho_v.reshape(-1, 1) - cur_rho_g
            out = torch.mean(torch.mul(out, out))
            return out

        def Hamilton_Jacobi_Equation(nodes):
            cur_phi = self.model_phi(nodes)
            cur_g = cur_phi
            grad_cur_phi = gradients(cur_phi, nodes)[0]

            out = grad_cur_phi[:, 0].reshape(-1, 1) + 0.5*torch.sum(torch.mul(grad_cur_phi[:, 1:], grad_cur_phi[:, 1:]), dim=1).reshape(-1, 1) \
                + torch.mul(cur_phi, cur_g) - 0.5*torch.mul(cur_g, cur_g)
            out = torch.mean(torch.mul(out, out))
            return out

        def rho_boundary(nodes):
            def load_data():
                rho_initial = np.load('./Example_Shape_Data/data_zhang.npy').reshape((-1, 1))
                rho_end = np.load('./Example_Shape_Data/data_pan.npy').reshape((-1, 1)) 
                #np.load('./Example_Shape/data_triangle.npy').reshape((-1, 1))

                rho_initial = torch.tensor(rho_initial, requires_grad=False, dtype=torch.float64).cuda()
                rho_end = torch.tensor(rho_end, requires_grad=False, dtype=torch.float64).cuda()
                return rho_initial, rho_end

            def Gaussian_data():
                rho_initial = Gaussian_distribution_torch(self.Key_Para['mu_0'], self.Key_Para['sigma_0'], nodes[::self.Key_Para['Num_Nodes_t'],:])\
                #         + Gaussian_distribution_torch(self.Key_Para['mu_0'] + np.array([[0.4, 0.0]]), self.Key_Para['sigma_0'], nodes[::self.Key_Para['Num_Nodes_t'],:])\
                #         + Gaussian_distribution_torch(self.Key_Para['mu_0'] + np.array([[0.0, 0.4]]), self.Key_Para['sigma_0'], nodes[::self.Key_Para['Num_Nodes_t'],:])\
                #         + Gaussian_distribution_torch(self.Key_Para['mu_0'] + np.array([[0.4, 0.4]]), self.Key_Para['sigma_0'], nodes[::self.Key_Para['Num_Nodes_t'],:])
                rho_end = Gaussian_distribution_torch(self.Key_Para['mu_1'], self.Key_Para['sigma_1'], nodes[int(self.Key_Para['Num_Nodes_t']-1)::self.Key_Para['Num_Nodes_t'],:])\
                        + Gaussian_distribution_torch(self.Key_Para['mu_1'] + np.array([[0.4, 0.4]]), self.Key_Para['sigma_1'], nodes[::self.Key_Para['Num_Nodes_t'],:])\
                        + Gaussian_distribution_torch(self.Key_Para['mu_1'] + np.array([[0.0, 0.4]]), self.Key_Para['sigma_1'], nodes[::self.Key_Para['Num_Nodes_t'],:])\
                        + Gaussian_distribution_torch(self.Key_Para['mu_1'] + np.array([[0.4, 0.0]]), self.Key_Para['sigma_1'], nodes[::self.Key_Para['Num_Nodes_t'],:])
                return rho_initial, rho_end
            
            def set_data(nodes):
                def is_point_inside_tetrahedron(nodes):
                    nodes = nodes.cpu().detach().numpy()

                    # k = 1/6 * (3/4)
                    # A = np.array([0.5 - k*np.sqrt(6), 0.5 - k*np.sqrt(6), 0.5 - k*np.sqrt(6)])
                    # B = np.array([0.5 + k*np.sqrt(6), 0.5 - k*np.sqrt(6), 0.5 - k*np.sqrt(6)])
                    # C = np.array([0.5 - k*np.sqrt(6), 0.5 + k*np.sqrt(6), 0.5 - k*np.sqrt(6)])
                    # D = np.array([0.5 - k*np.sqrt(6), 0.5 - k*np.sqrt(6), 0.5 + k*np.sqrt(6)])
                    
                    
                    center = np.array([0.5, 0.5, 0.35])
                    A = np.array([center[0], center[1], 0.95])
                    a = (A[2] - center[2])*(2*np.sqrt(6)/3)
                    B = np.array([center[0] - a/2, center[1] - a*np.sqrt(3)/6, center[2] - a*np.sqrt(6)/12])
                    C = np.array([center[0] + a/2, center[1] - a*np.sqrt(3)/6, center[2]  - a*np.sqrt(6)/12])
                    D = np.array([center[0], center[1] + a*np.sqrt(3)/3, center[2]  - a*np.sqrt(6)/12])


                    # 计算四个子体积
                    V1 = np.abs(np.dot(np.cross(B-A, C-A), nodes-A)) / 6
                    V2 = np.abs(np.dot(np.cross(C-B, D-B), nodes-B)) / 6
                    V3 = np.abs(np.dot(np.cross(D-C, A-C), nodes-C)) / 6
                    V4 = np.abs(np.dot(np.cross(A-D, B-D), nodes-D)) / 6

                    tetrahedron_volume = V1 + V2 + V3 + V4

                    # 判断点是否在四面体内部
                    return np.isclose(tetrahedron_volume, np.abs(np.dot(np.cross(B-A, C-A), D-A)) / 6)

                def is_point_inside_cube(nodes):
                    nodes = nodes.cpu().detach().numpy()
                    return np.logical_and(np.logical_and(np.logical_and(nodes[0] >= 0.4, nodes[0] <= 0.6), np.logical_and(nodes[1] >= 0.4, nodes[1] <= 0.6)), np.logical_and(nodes[2] >= 0.4, nodes[2] <= 0.6))

                def is_point_inside_sphere(nodes):
                    nodes = nodes.cpu().detach().numpy()
                    return np.logical_or(np.power(nodes[0] - 0.5, 2) + np.power(nodes[1] - 0.5, 2) + np.power(nodes[2] - 0.5, 2) <= 0.25, 0)

                nodes = nodes[:, 1:]
                rho_initial = torch.zeros_like(nodes[:, 0:1])
                rho_end = torch.zeros_like(nodes[:, 0:1])
                for i in range(nodes.shape[0]):
                    if is_point_inside_sphere(nodes[i, :]):
                        rho_initial[i] = 1
                    # if is_point_inside_tetrahedron(nodes[i, :]):
                    #     rho_end[i] = 1
                    if is_point_inside_cube(nodes[i, :]):
                        rho_end[i] = 1


                # nodes = nodes.detach().cpu().numpy()
                # rho_initial = rho_initial.detach().cpu().numpy()
                # rho_end = rho_end.detach().cpu().numpy()
                # id_initial = np.where(rho_initial == 1)[0]
                # id_end = np.where(rho_end == 1)[0]
                # fig = plt.figure(figsize=(int(6*6), 10))
                # for i in range(1, 7):
                #     ax = fig.add_subplot(2, 6, i, projection='3d')
                #     im = ax.scatter(nodes[id_initial, 0], nodes[id_initial, 1], nodes[id_initial, 2], c=rho_initial[id_initial, :])
                #     ax.view_init(elev=25, azim=60*i)
                #     ax.set_xlim3d(0, 1), ax.set_ylim3d(0, 1), ax.set_zlim3d(0, 1)
                #     # ax.set_xticks([]),  ax.set_yticks([]), ax.set_zticks([])
                #     # ax.set_box_aspect([2, 1, 2])
                #     plt.colorbar(im)

                #     ax = fig.add_subplot(2, 6, i+6, projection='3d')
                #     im = ax.scatter(nodes[id_end, 0], nodes[id_end, 1], nodes[id_end, 2], c=rho_end[id_end ,:])
                #     ax.view_init(elev=25, azim=60*i)
                #     ax.set_xlim3d(0, 1), ax.set_ylim3d(0, 1), ax.set_zlim3d(0, 1)
                #     # ax.set_xticks([]),  ax.set_yticks([]), ax.set_zticks([])
                #     # ax.set_box_aspect([2, 1, 2])
                #     plt.colorbar(im)

                # plt.savefig('rho.png')
                # plt.close()

                return rho_initial, rho_end


            # rho_initial, rho_end = load_data()
            rho_initial, rho_end = Gaussian_data()
            # rho_initial, rho_end = set_data(nodes[::self.Key_Para['Num_Nodes_t'],:])

            cur_rho_initial = self.model_rho(nodes[::self.Key_Para['Num_Nodes_t'],:])
            cur_rho_end = self.model_rho(nodes[int(self.Key_Para['Num_Nodes_t']-1)::self.Key_Para['Num_Nodes_t'],:])

            initial = rho_initial - cur_rho_initial
            end = rho_end - cur_rho_end
            out = torch.mean(torch.mul(initial, initial)) + torch.mean(torch.mul(end, end))
            return out

        loss_W2 = 0 #W2(nodes)
        loss_C_eq = Continuity_Equation(nodes)
        loss_HJ_eq  = Hamilton_Jacobi_Equation(nodes)
        loss_rho_bc = rho_boundary(nodes)
        sub_loss = [loss_W2, loss_C_eq, loss_HJ_eq, loss_rho_bc]

        temp = []
        for i in range(len(sub_loss)):
            temp.append(get_variable_name(sub_loss[i]))
        self.Key_Para['loss_name'] = temp

        if self.type_loss == 'Auto-Weight':
            loss = 0
            # loss += sub_loss[0]
            for i in range(self.num_losses):
                loss += 0.5 / (self.params[i] ** 2) * sub_loss[i] + torch.log(1 + self.params[i] ** 2) 
        elif self.type_loss == 'General':
            loss = self.beta[0]*loss_W2 + self.beta[1]*loss_C_eq + self.beta[2]*loss_HJ_eq + self.beta[3]*loss_rho_bc
        return loss, sub_loss


class Gnerate_node(object):
    def __init__(self, Key_Para):
        super(Gnerate_node, self).__init__()
        self.Dim_time = Key_Para['Dim_time']
        self.Dim_space = Key_Para['Dim_space']
        self.Time = Key_Para['Time']
        self.Space = Key_Para['Space']
        self.Num_Nodes_t = Key_Para['Num_Nodes_t']
        self.Num_Nodes_all_space = Key_Para['Num_Nodes_all_space']
        self.type_node = Key_Para['type_node']

    def forward(self):
        if self.type_node == 'Regular':
            if self.Dim_space == 1:
                t = np.linspace(self.Time[0, 0], self.Time[0, 1], self.Num_Nodes_t)
                nodes_space = np.linspace(self.Space[0, 0], self.Space[0, 1], self.Num_Nodes_all_space).reshape(-1, 1)
                nodes = np.hstack((t.reshape(-1, 1).repeat(self.Num_Nodes_all_space, axis=1).T.reshape(-1, 1), nodes_space.repeat(self.Num_Nodes_t, axis=0)))

            elif self.Dim_space == 2:
                h = int(np.sqrt(self.Num_Nodes_all_space))

                t = np.linspace(self.Time[0, 0], self.Time[0, 1], self.Num_Nodes_t)
                x = np.linspace(self.Space[0, 0], self.Space[0, 1], h)
                y = np.linspace(self.Space[1, 0], self.Space[1, 1], h)
                mesh = np.meshgrid(y, x)
                nodes_space = np.array(list(zip(*(dim.flat for dim in mesh))))
                nodes = np.hstack((t.reshape(-1, 1).repeat(self.Num_Nodes_all_space, axis=1).T.reshape(-1, 1), nodes_space.repeat(self.Num_Nodes_t, axis=0)))

            elif self.Dim_space == 3:
                h = int(np.ceil(np.power(self.Num_Nodes_all_space, 1/3)))
                
                t = np.linspace(self.Time[0, 0], self.Time[0, 1], self.Num_Nodes_t)
                x = np.linspace(self.Space[0, 0], self.Space[0, 1], h)
                y = np.linspace(self.Space[1, 0], self.Space[1, 1], h)
                z = np.linspace(self.Space[2, 0], self.Space[2, 1], h)
                mesh = np.meshgrid(z, y, x)
                nodes_space = np.array(list(zip(*(dim.flat for dim in mesh))))
                nodes = np.hstack((t.reshape(-1, 1).repeat(self.Num_Nodes_all_space, axis=1).T.reshape(-1, 1), nodes_space.repeat(self.Num_Nodes_t, axis=0)))

            elif self.Dim_space > 3:
                print('There need code')
        
        elif self.type_node == 'Random':
            def Farthest_Point_Sample(data, npoints):
                N, D = data.shape 
                xyz = data 
                centroids = np.zeros(npoints) 
                dictance = np.ones(N)*1e10
                farthest = random.sample(range(0, N), 1)[0]
                for i in range(npoints):
                    centroids[i] = farthest
                    centroid = xyz[farthest, :]
                    dict = ((xyz - centroid)**2).sum(axis=1)
                    mask = dict < dictance
                    dictance[mask] = dict[mask]
                    farthest = np.argmax(dictance)
                data = data[centroids.astype(int)]
                return data


            Num_Nodes_all_space_internal = int(0.8*self.Num_Nodes_all_space)
            Num_Nodes_all_space_external = int(self.Num_Nodes_all_space - Num_Nodes_all_space_internal)

            t = np.linspace(self.Time[0, 0], self.Time[0, 1], self.Num_Nodes_t)
            id_value = np.random.randint(0, 2, Num_Nodes_all_space_external) * (self.Space[0, 1] - self.Space[0, 0]) + self.Space[0, 0]
            id_side = np.random.randint(0, self.Dim_space, Num_Nodes_all_space_external)

            nodes_space_internal = np.random.uniform(low=self.Space[0, 0], high=self.Space[0, 1], size=(50000, self.Dim_space))
            nodes_space_external = np.random.uniform(low=self.Space[0, 0], high=self.Space[0, 1], size=(Num_Nodes_all_space_external, self.Dim_space))
            
            nodes_space_external[list(range(Num_Nodes_all_space_external)), id_side] = np.array(id_value)
            nodes_space_internal = Farthest_Point_Sample(nodes_space_internal, Num_Nodes_all_space_internal)
            nodes_space = np.vstack((nodes_space_external, nodes_space_internal))
            nodes = np.hstack((t.reshape(-1, 1).repeat(self.Num_Nodes_all_space, axis=1).T.reshape(-1, 1), nodes_space.repeat(self.Num_Nodes_t, axis=0)))

        nodes = torch.tensor(nodes, requires_grad=True, dtype=torch.float64).cuda()
        return nodes


class Train_net(object):
    def __init__(self, Key_Para, gen_Nodes, model_rho, model_phi, loss_net, Plot_Result):
        super(Train_net, self).__init__()
        self.Key_Para = Key_Para
        self.gen_Nodes = gen_Nodes
        self.model_rho = model_rho
        self.model_phi = model_phi
        self.loss_net = loss_net
        self.Plot_Result = Plot_Result
        self.optimizer = torch.optim.Adam(self.loss_net.parameters(), lr=self.Key_Para['learning_rate'])

    def train(self):
        all_sub_loss = []
        for ep_s in range(self.Key_Para['epochs_sample']):
            print('Sampling point: ', ep_s)
            nodes = self.gen_Nodes.forward()
            for ep_t in range(self.Key_Para['epochs_train']):  

                self.optimizer.zero_grad() 
                loss, sub_loss = self.loss_net(nodes)
                loss.backward()
                self.optimizer.step()
                all_sub_loss.append(sub_loss)

                if self.Key_Para['type_pre_plot'] == 'True':
                    if ep_t % 500 == 0:
                        w = 1
                        self.Plot_Result.plot_rho(self.model_rho, ep_t)
                        # self.Plot_Result.plot_v(self.model_phi, ep_t)
                        # self.Plot_Result.plot_g(self.model_g, ep_t)

                if ep_t % 10 ==0:
                    if type_lr == 'General':
                        pass
                    elif type_lr == 'Gradual-Decline':
                        for p in self.optimizer.param_groups:
                            p['lr'] = p['lr']*0.98

                if ep_t % 100 ==0:
                    print('ep:', '%d' %ep_t,
                        'loss:',"%.6f " %loss,
                        '[loss_W2:',"%.6f" %sub_loss[0], 
                        'loss_C_eq:',"%.6f" %sub_loss[1],
                        'loss_HJ_eq:',"%.6f" %sub_loss[2],
                        'loss_rho_bc:',"%.6f]" %sub_loss[3]
                        )

        torch.save(self.model_rho, self.Key_Para['File_name'] + '/model_rho.pth')
        torch.save(self.model_phi, self.Key_Para['File_name'] + '/model_phi.pth')
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
    def __init__(self, Key_Para):
        super(Plot_Result, self).__init__()
        self.Key_Para = Key_Para
        self.Dim_time = Key_Para['Dim_time']
        self.Dim_space = Key_Para['Dim_space']
        self.Time = Key_Para['Time']
        self.Space = Key_Para['Space']

    def plot_rho(self, model_rho, ite=''):
        if self.Dim_space == 2:
            T = 10
            num = 40
            t = np.linspace(self.Time[0, 0], self.Time[0, 1], T)
            x = np.linspace(self.Space[0, 0], self.Space[0, 1], num)
            y = np.linspace(self.Space[1, 0], self.Space[1, 1], num)
            mesh = np.meshgrid(y, x)
            nodes_space = np.array(list(zip(*(dim.flat for dim in mesh))))
            nodes = np.hstack((t.reshape(-1, 1).repeat(num**2, axis=1).T.reshape(-1, 1), nodes_space.repeat(T, axis=0)))
            pre_nodes = torch.tensor(nodes, requires_grad=True, dtype=torch.float64).cuda()


            real_rho = torch.zeros((T*(num)**2, 1))
            gamma = t/(self.Time[0, 1] - self.Time[0, 0])
            for i in range(gamma.shape[0]):
                mu_n = (1 - gamma)[i] * self.Key_Para['mu_0'] + gamma[i] * self.Key_Para['mu_1']
                sigma_n = ((1 - gamma)[i]*np.sqrt(self.Key_Para['sigma_0']) + gamma[i]*np.sqrt(self.Key_Para['sigma_1']))**2
                real_rho[i::T, :] = Gaussian_distribution_torch(mu_n, sigma_n, pre_nodes[0::T, :])

            pre_rho = model_rho(pre_nodes)
            pre_rho = pre_rho.reshape(-1, 1).cpu().detach().numpy()
            real_rho = real_rho.cpu().detach().numpy()


            plt.figure(figsize=(int(2*T), 2))
            for i in range(1, T+1):
                cur_rho = pre_rho[(i-1)::T, :].reshape(num, num)
                cur_real_rho = real_rho[(i-1)::T, :].reshape(num, num)
                cur_res_rho = np.abs(cur_real_rho - cur_rho)

                plt.subplot(1, T, i)
                plt.imshow(cur_rho, extent=(np.min(nodes[:, 1]), np.max(nodes[:, 1]), np.min(nodes[:, 2]), np.max(nodes[:, 2])), origin='lower')
                plt.title('t=%1.3f' %(t[i-1]))
                plt.xticks([])
                plt.yticks([])
                plt.colorbar()

            plt.savefig(self.Key_Para['File_name'] + '/' + 'Fig-rho-' + str(ite) + '.png')
            plt.close()

            if 1:
                T = 5
                num = 40
                t = np.linspace(self.Time[0, 0], self.Time[0, 1], T)
                x = np.linspace(self.Space[0, 0], self.Space[0, 1], num)
                y = np.linspace(self.Space[1, 0], self.Space[1, 1], num)
                mesh = np.meshgrid(y, x)
                nodes_space = np.array(list(zip(*(dim.flat for dim in mesh))))
                nodes = np.hstack((t.reshape(-1, 1).repeat(num**2, axis=1).T.reshape(-1, 1), nodes_space.repeat(T, axis=0)))
                pre_nodes = torch.tensor(nodes, requires_grad=True, dtype=torch.float64).cuda()
                pre_rho = model_rho(pre_nodes)
                pre_rho = pre_rho.reshape(-1, 1).cpu().detach().numpy()
                np.save(self.Key_Para['File_name'] + '/' + 'pre_rho.npy', pre_rho)

                norm = matplotlib.colors.Normalize(vmin=pre_rho.min(), vmax=pre_rho.max())
                for i in range(5):
                    plt.figure(figsize=(6, 6))
                    cur_rho = pre_rho[i::T, :].reshape(num, num)

                    plt.imshow(cur_rho, extent=(np.min(nodes[:, 1]), np.max(nodes[:, 1]), np.min(nodes[:, 2]), np.max(nodes[:, 2])), origin='lower', norm=norm, cmap='hot')
                    # plt.xticks([])
                    # plt.yticks([])
                    plt.colorbar()
                    plt.savefig(self.Key_Para['File_name'] + '/' + 'Fig-rho' + str(i) + '.png', dpi=300)
                    plt.close()

    def plot_v(self, model_phi, ite=''):
        if self.Dim_space == 2:
            T = 10
            num = 25
            t = np.linspace(self.Time[0, 0], self.Time[0, 1], T)
            x = np.linspace(self.Space[0, 0], self.Space[0, 1], num)
            y = np.linspace(self.Space[1, 0], self.Space[1, 1], num)
            mesh = np.meshgrid(y, x)
            nodes_space = np.array(list(zip(*(dim.flat for dim in mesh))))
            nodes = np.hstack((t.reshape(-1, 1).repeat(num**2, axis=1).T.reshape(-1, 1), nodes_space.repeat(T, axis=0)))
            pre_nodes = torch.tensor(nodes, requires_grad=True, dtype=torch.float64).cuda()


            pre_phi = model_phi(pre_nodes)
            pre_v = gradients(pre_phi, pre_nodes)[0][:, 1:]
            pre_v = pre_v.cpu().detach().numpy()


            plt.figure(figsize=(int(2*T), 4))
            for i in range(1, T+1):
                cur_v_1 = pre_v[(i-1)::T, 0].reshape(num, num)
                cur_v_2 = pre_v[(i-1)::T, 1].reshape(num, num)

                plt.subplot(2, T, i)
                plt.imshow(cur_v_1, extent=(np.min(nodes[:, 1]), np.max(nodes[:, 1]), np.min(nodes[:, 2]), np.max(nodes[:, 2])), origin='lower')
                plt.title('t=%1.3f' %(t[i-1]))
                plt.colorbar()
                plt.xticks([])
                plt.yticks([])

                plt.subplot(2, T, T+i)
                plt.imshow(cur_v_2, extent=(np.min(nodes[:, 1]), np.max(nodes[:, 1]), np.min(nodes[:, 2]), np.max(nodes[:, 2])), origin='lower', cmap='jet')
                plt.colorbar()
                plt.xticks([])
                plt.yticks([])

            plt.savefig(self.Key_Para['File_name'] + '/' + 'Fig-v.png')
            plt.close()



        np.save(self.Key_Para['File_name'] + '/' + 'pre_v.npy', pre_v)

    def plot_g(self, model_phi, ite=''):
        if self.Dim_space == 2:
            T = 10
            num = 25
            t = np.linspace(self.Time[0, 0], self.Time[0, 1], T)
            x = np.linspace(self.Space[0, 0], self.Space[0, 1], num)
            y = np.linspace(self.Space[1, 0], self.Space[1, 1], num)
            mesh = np.meshgrid(y, x)
            nodes_space = np.array(list(zip(*(dim.flat for dim in mesh))))
            nodes = np.hstack((t.reshape(-1, 1).repeat(num**2, axis=1).T.reshape(-1, 1), nodes_space.repeat(T, axis=0)))

            pre_nodes = torch.tensor(nodes, requires_grad=True, dtype=torch.float64).cuda()


            pre_phi = model_phi(pre_nodes)
            pre_g = pre_phi
            pre_g = pre_g.cpu().detach().numpy()


            plt.figure(figsize=(int(2*T), 2))
            for i in range(1, T+1):
                cur_g = pre_g[(i-1)::T, :].reshape(num, num)

                plt.subplot(1, T, i)
                plt.imshow(cur_g, extent=(np.min(nodes[:, 1]), np.max(nodes[:, 1]), np.min(nodes[:, 2]), np.max(nodes[:, 2])), origin='lower')
                plt.title('t=%1.3f' %(t[i-1]))
                plt.colorbar()
                plt.xticks([])
                plt.yticks([])

            plt.savefig(self.Key_Para['File_name'] + '/' + 'Fig-g.png')
            plt.close()

        np.save(self.Key_Para['File_name'] + '/' + 'pre_g.npy', pre_g)


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





def main(Key_Para):
    utilize = Utilize(Key_Para)
    utilize.make_file()
    utilize.setup_seed(1)
    utilize.mu_0_mu_1()
    utilize.sigma_0_sigma_1()
    utilize.Time_Space()
    utilize.print_key(Key_Para)


    gen_Nodes = Gnerate_node(Key_Para)
    model_rho, model_phi = Model_rho(Key_Para), Model_phi(Key_Para)
    loss_net = Loss_Net(model_rho, model_phi, Key_Para).cuda()
    PR = Plot_Result(Key_Para)

    train_net = Train_net(Key_Para, gen_Nodes, model_rho, model_phi, loss_net, PR)
    all_sub_loss = train_net.train()

    compute_W2 = Compute_Wasserstein_distance(Key_Para, all_sub_loss)
    # compute_W2.compute()

    PR.plot_rho(model_rho)
    PR.plot_v(model_phi)
    PR.plot_g(model_phi)
    PR.plot_loss(all_sub_loss)



if __name__== "__main__" :
    for j in range(1):
        time_begin = time.time()
        time_now = time.strftime("%Y%m%d-%H%M", time.localtime())
        name = os.path.basename(sys.argv[0])
        File_name = time_now + '-' + name[:-3]

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        torch.cuda.set_device(0)
        torch.set_default_dtype(torch.float64)

        test = 'test1'
        if test == 'test1':
            Dim_time = 1
            Dim_space = 2
            Num_Nodes_t = 10
            Num_Nodes_all_space = (30)**2
            mu_0, mu_1 = [0.5, 0.5], [0.3, 0.3]
            sigma_0, sigma_1 = 0.005, 0.005
            Time, Space = [0.0, 1.0], [0.0, 1.0]

            epochs_sample = 1
            epochs_train = 2500
            learning_rate = 1e-3
            beta = [0, 1000, 1000, 1000]

            type_print = 'False'           # 'True'      'False' 
            type_node = 'Regular'         # 'Regular'   'Random'
            type_lr = 'General'   # 'Gradual-Decline'   'General'
            type_loss = 'General'     # 'Auto-Weight'   'General'
            type_pre_plot = 'True'   # 'True'  'False'



        Key_Parameters = {
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
            'type_print': type_print,
            'type_node': type_node, 
            'type_lr': type_lr, 
            'type_loss': type_loss,
            'type_pre_plot': type_pre_plot,
                }

        main(Key_Parameters)
        print('Runing_time:', time.time() - time_begin, 's')
