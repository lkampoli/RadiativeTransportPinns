from ImportFile import *
from pyevtk.hl import gridToVTK

pi = math.pi
extrema_values = None
parameters_values = torch.tensor([[0, 2 * pi],
                                  [0, pi],
                                  [0, 1]])
space_dimensions = 3
time_dimensions = 0
output_dimension = 2
assign_g = True
average = False
type_of_points = "sobol"
r_min = 0.0
input_dimensions = 3
kernel_type = "isotropic"
n_quad = 10

if torch.cuda.is_available():
    dev = torch.device('cuda')
else:
    dev = torch.device("cpu")


def get_points(samples, dim, type_point_param, random_seed):
    if type_point_param == "uniform":
        torch.random.manual_seed(random_seed)
        points = torch.rand([samples, dim]).type(torch.FloatTensor)
    elif type_point_param == "sobol":
        skip = random_seed
        data = np.full((samples, dim), np.nan)
        for j in range(samples):
            seed = j + skip
            data[j, :], next_seed = sobol_seq.i4_sobol(dim, seed)
        points = torch.from_numpy(data).type(torch.FloatTensor)
    return points


def get_G(intensity, phys_coord, n_quad):
    quad_points, w = np.polynomial.legendre.leggauss(n_quad)

    mat_quad = np.transpose([np.repeat(quad_points, len(quad_points)), np.tile(quad_points, len(quad_points))])
    w = w.reshape(-1, 1)
    mat_weight = np.matmul(w, w.T).reshape(n_quad * n_quad, 1)
    mat_weight = torch.from_numpy(mat_weight).type(torch.FloatTensor).to(dev)

    mat_quad[:, 0] = pi * (mat_quad[:, 0] + 1)
    mat_quad[:, 1] = pi / 2 * (mat_quad[:, 1] + 1)
    mat_quad = torch.from_numpy(mat_quad).type(torch.FloatTensor).to(dev)

    s = get_s(mat_quad)

    s_new = torch.cat(phys_coord.shape[0] * [s]).to(dev)
    phys_coord_new = tile(phys_coord, dim=0, n_tile=s.shape[0]).to(dev)

    inputs = torch.cat([phys_coord_new, s_new], 1).to(dev)

    u = intensity(inputs)[:, 0].reshape(-1, )
    u = u.reshape(phys_coord.shape[0], n_quad * n_quad)

    sin_theta_v = torch.sin(mat_quad[:, 1]).reshape(-1, 1)

    G = pi ** 2 / 2 * torch.matmul(u, sin_theta_v * mat_weight).reshape(-1, )

    return G


def get_average_inf_q(intensity, phys_coord, n_quad):
    quad_points, w = np.polynomial.legendre.leggauss(n_quad)

    mat_quad = np.transpose([np.repeat(quad_points, len(quad_points)), np.tile(quad_points, len(quad_points))])
    w = w.reshape(-1, 1)
    mat_weight = np.matmul(w, w.T).reshape(n_quad * n_quad, 1)
    mat_weight = torch.from_numpy(mat_weight).type(torch.FloatTensor).to(dev)

    mat_quad[:, 0] = pi * (mat_quad[:, 0] + 1)
    mat_quad[:, 1] = pi / 2 * (mat_quad[:, 1] + 1)
    mat_quad = torch.from_numpy(mat_quad).type(torch.FloatTensor).to(dev)

    s = get_s(mat_quad)

    s_new = torch.cat(phys_coord.shape[0] * [s]).to(dev)
    phys_coord_new = tile(phys_coord, dim=0, n_tile=s.shape[0]).to(dev)

    inputs = torch.cat([phys_coord_new, s_new], 1).to(dev)

    u = intensity(inputs)[:, 1].reshape(-1, )
    u = u.reshape(phys_coord.shape[0], n_quad * n_quad)

    sin_theta_v = torch.sin(mat_quad[:, 1]).reshape(-1, 1)

    average = pi ** 2 / 2 * torch.matmul(u, sin_theta_v * mat_weight).reshape(-1, )

    return average / (4 * pi)


def tile(a, dim, n_tile):
    init_dim = a.size(dim)
    repeat_idx = [1] * a.dim()
    repeat_idx[dim] = n_tile
    a = a.repeat(*(repeat_idx))
    order_index = torch.LongTensor(np.concatenate([init_dim * np.arange(n_tile) + i for i in range(init_dim)]))
    return torch.index_select(a, dim, order_index.to(dev))


def f(x, y, z, s1, s2, s3):
    c = -4 / pi * (3 + (s1 + s2 + s3) ** 2)
    source = c * (s1 * (2 * x - 1) * (y ** 2 - y) * (z ** 2 - z) +
                  s2 * (x ** 2 - x) * (2 * y - 1) * (z ** 2 - z) +
                  s3 * (x ** 2 - x) * (y ** 2 - y) * (2 * z - 1))

    source = source + (K(x, y, z) + S(x, y, z)) * exact(x, y, z, s1, s2, s3)
    source = source - S(x, y, z) * G(x, y, z) / (4 * pi)

    return source


def exact(x, y, z, s1, s2, s3):
    return -4 / pi * (3 + (s1 + s2 + s3) ** 2) * (x ** 2 - x) * (y ** 2 - y) * (z ** 2 - z)


def G(x, y, z):
    return -64 * (x ** 2 - x) * (y ** 2 - y) * (z ** 2 - z)


def K(x, y, z):
    k = x ** 2 * y ** 2 * z ** 2
    return k


def S(x, y, z):
    sigma = 0.5 * torch.ones(x.shape)
    return sigma.to(dev)


def compute_scatter(x_train, model):
    phys_coord = x_train[:, :3]
    s_train = x_train[:, 3:]
    scatter_values = get_G(model, phys_coord, n_quad) / (4 * pi)
    return scatter_values


def get_s(params):
    s = torch.tensor(()).new_full(size=(params.shape[0], 3), fill_value=0.0)
    phi = params[:, 0]
    theta = params[:, 1]
    s[:, 0] = torch.cos(phi) * torch.sin(theta)
    s[:, 1] = torch.sin(phi) * torch.sin(theta)
    s[:, 2] = torch.cos(theta)
    return s


def generator_param_samples(points):
    extrema_0 = parameters_values[:2, 0]
    extrema_f = parameters_values[:2, 1]
    points = points * (extrema_f - extrema_0) + extrema_0
    s = torch.tensor(()).new_full(size=(points.shape[0], 3), fill_value=0.0)
    phi = points[:, 0]
    theta = points[:, 1]
    s[:, 0] = torch.cos(phi) * torch.sin(theta)
    s[:, 1] = torch.sin(phi) * torch.sin(theta)
    s[:, 2] = torch.cos(theta)
    return s


def generator_domain_samples(points, boundary):
    if boundary:
        n_single_dim = int(points.shape[0] / input_dimensions)
        print(n_single_dim)
        for i in range(input_dimensions):
            n = int(n_single_dim / 2)
            points[2 * i * n:n * (2 * i + 1), i] = torch.tensor(()).new_full(size=(n,), fill_value=0.0)
            points[n * (2 * i + 1):2 * n * (i + 1), i] = torch.tensor(()).new_full(size=(n,), fill_value=1.0)

    return points


def compute_res(network, x_f_train, space_dimensions, solid_object, computing_error):
    x_f_train.requires_grad = True
    x = x_f_train[:, 0]
    y = x_f_train[:, 1]
    z = x_f_train[:, 2]
    s = x_f_train[:, 3:]

    phys_coord = x_f_train[:, :3]

    u = network(x_f_train)[:, 0].reshape(-1, )
    if average:
        absorb = get_average_inf_q(network, phys_coord, n_quad)
    else:
        absorb = network(x_f_train)[:, 1].reshape(-1, )

    grad_u = torch.autograd.grad(u, x_f_train, grad_outputs=torch.ones(x_f_train.shape[0], ).to(dev), create_graph=True)[0]

    grad_u_x = grad_u[:, 0]
    grad_u_y = grad_u[:, 1]
    grad_u_z = grad_u[:, 2]
    s1 = s[:, 0]
    s2 = s[:, 1]
    s3 = s[:, 2]

    scatter_kernel = compute_scatter(x_f_train, network)

    res = s1 * grad_u_x + s2 * grad_u_y + s3 * grad_u_z + (absorb + S(x, y, z)) * u - f(x, y, z, s1, s2, s3) - S(x, y, z) * scatter_kernel

    return res


def add_internal_points(n_internal):
    n_int = int(n_internal * 3 / 4)
    print("Adding Internal Points")
    points = get_points(n_internal, 5, "uniform", 16)
    dom_int = points[:n_int, :3]
    angles_int = points[:n_int, 3:]
    dom_b = points[n_int:, :3]
    angles_b = points[n_int:, 3:]
    dom = torch.cat([generator_domain_samples(dom_int, boundary=False), generator_domain_samples(dom_b, boundary=True)])
    s = torch.cat([generator_param_samples(angles_int), generator_param_samples(angles_b)])

    x = dom[:, 0].reshape(-1, 1)
    y = dom[:, 1].reshape(-1, 1)
    z = dom[:, 2].reshape(-1, 1)

    u = exact(x.reshape(-1, ), y.reshape(-1, ), z.reshape(-1, ), s[:, 0], s[:, 1], s[:, 2]).reshape(-1, 1)
    g = G(x.reshape(-1, ), y.reshape(-1, ), z.reshape(-1, )).reshape(-1, 1)

    if assign_g:
        return torch.cat([x, y, z, s], 1), g
    else:
        return torch.cat([x, y, z, s], 1), u


def add_boundary(n_boundary):
    print("Adding Boundary")
    points = get_points(n_boundary, 5, type_of_points, 16)
    dom = points[:, :3]
    angles = points[:, 3:]
    dom = generator_domain_samples(dom, boundary=True)
    s = generator_param_samples(angles)

    x = dom[:, 0].reshape(-1, 1)
    y = dom[:, 1].reshape(-1, 1)
    z = dom[:, 2].reshape(-1, 1)
    ub = torch.tensor(()).new_full(size=(n_boundary, 1), fill_value=0.0)

    return torch.cat([x, y, z, s], 1), ub


def add_collocations(n_collocation):
    n_coll_int = int(3 / 4 * n_collocation)
    print("Adding Collocation")
    points = get_points(n_collocation, 5, "sobol", 16)
    dom_int = points[:n_coll_int, :3]
    angles_int = points[:n_coll_int, 3:]
    dom_b = points[n_coll_int:, :3]
    angles_b = points[n_coll_int:, 3:]
    dom = torch.cat([generator_domain_samples(dom_int, boundary=False), generator_domain_samples(dom_b, boundary=True)])
    s = torch.cat([generator_param_samples(angles_int), generator_param_samples(angles_b)])

    x = dom[:, 0].reshape(-1, 1)
    y = dom[:, 1].reshape(-1, 1)
    z = dom[:, 2].reshape(-1, 1)

    u = torch.tensor(()).new_full(size=(n_collocation, 1), fill_value=np.nan)

    return torch.cat([x, y, z, s], 1), u


def apply_BC(x_boundary, u_boundary, model):
    x = x_boundary[:, 0]
    y = x_boundary[:, 1]
    z = x_boundary[:, 2]

    s = x_boundary[:, 3:]

    xyz = torch.cat([x.reshape(-1, 1), y.reshape(-1, 1), z.reshape(-1, 1)], 1)
    if torch.cuda.is_available():
        xyz = xyz.cuda()
    xyz_mod = torch.where(xyz == 0, torch.tensor(-1.).to(dev), xyz)
    if torch.cuda.is_available():
        xyz_mod = xyz_mod.cuda()

    n = torch.where(((xyz_mod != 1) & (xyz_mod != -1)), torch.tensor(0.).to(dev), xyz_mod)
    if torch.cuda.is_available():
        n = n.cuda()

    n1 = n[:, 0]
    n2 = n[:, 1]
    n3 = n[:, 2]
    s1 = s[:, 0]
    s2 = s[:, 1]
    s3 = s[:, 2]

    scalar = (n1 * s1 + n2 * s2 + n3 * s3) < 0

    x_boundary_inf = x_boundary[scalar, :]
    x_boundary_out = x_boundary[~scalar, :]
    u_boundary_inf = u_boundary[scalar, :]

    u_pred = model(x_boundary_inf)[:, 0]

    return u_pred.reshape(-1, ), u_boundary_inf.reshape(-1, )


def u0(inputs):
    return


def convert(vector):
    vector = np.array(vector)
    max_val = np.max(np.array(extrema_values), axis=1)
    min_val = np.min(np.array(extrema_values), axis=1)
    vector = vector * (max_val - min_val) + min_val
    return torch.from_numpy(vector).type(torch.FloatTensor)


def compute_generalization_error(model, extrema, images_path=None):
    return 0, 0


def plotting(model, images_path, extrema, solid):
    from scipy import integrate
    n_sam = 100000

    points = get_points(n_sam, 5, "uniform", 12)

    phys_coord = generator_domain_samples(points[:, :3], boundary=False).to(dev)
    s = generator_param_samples(points[:, 3:]).to(dev)

    x = phys_coord[:, 0]
    y = phys_coord[:, 1]
    z = phys_coord[:, 2]

    s1 = s[:, 0]
    s2 = s[:, 1]
    s3 = s[:, 2]

    inputs = torch.cat([phys_coord, s], 1).to(dev)

    g_ex = G(x, y, z).detach().cpu().numpy()
    u_ex = exact(x, y, z, s1, s2, s3).detach().cpu().numpy()

    g = get_G(model, phys_coord, n_quad).detach().cpu().numpy()
    u = model(inputs)[:, 0].detach().cpu().numpy()

    k_ex = K(x, y, z).detach().cpu().numpy()
    k = get_average_inf_q(model, phys_coord, n_quad).detach().cpu().numpy()

    g_rel_err = (np.mean((g_ex - g) ** 2) / np.mean(g_ex ** 2)) ** 0.5
    k_rel_err = (np.mean((k_ex - k) ** 2) / np.mean(k_ex ** 2)) ** 0.5
    u_rel_err = (np.mean((u_ex - u) ** 2) / np.mean(u_ex ** 2)) ** 0.5

    with open(images_path + '/errors.txt', 'w') as file:
        file.write("g_rel_err,"
                   "k_rel_err,"
                   "u_rel_err\n")
        file.write(str(float(g_rel_err)) + "," +
                   str(float(k_rel_err)) + "," +
                   str(float(u_rel_err)))

    x = torch.linspace(0, 1, 60).reshape(-1, 1)
    xyz = torch.cat([x, x, x], 1).to(dev)

    g_plot = get_G(model, xyz, n_quad).detach().cpu().numpy()
    k_plot = get_average_inf_q(model, xyz, n_quad).detach().cpu().numpy()

    g_plot_ex = G(xyz[:, 0], xyz[:, 1], xyz[:, 2]).detach().cpu().numpy()
    k_plot_ex = K(xyz[:, 0], xyz[:, 1], xyz[:, 2]).detach().cpu().numpy()

    r = torch.sqrt(3 * x ** 2).detach().cpu().numpy()

    fig = plt.figure()
    plt.grid(True, which="both", ls=":")
    plt.plot(r, g_plot_ex, linewidth=2, color="grey", label=r'Exact')
    plt.scatter(r, g_plot, label=r'Predicted', marker="o", s=14)
    plt.xlabel(r'$r$')
    plt.ylabel(r'G')
    plt.legend()
    plt.savefig(images_path + "/Samples_G.png", dpi=500)

    fig = plt.figure()
    plt.grid(True, which="both", ls=":")
    plt.plot(r, k_plot_ex, linewidth=2, color="grey", label=r'Exact')
    plt.scatter(r, k_plot, label=r'Predicted', marker="o", s=14)
    plt.xlabel(r'$r$')
    plt.ylabel(r'k')
    plt.legend()
    plt.savefig(images_path + "/Samples_K.png", dpi=500)
