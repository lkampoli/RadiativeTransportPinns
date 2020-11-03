from ImportFile import *
from pyevtk.hl import gridToVTK

pi = math.pi

space_dimensions = 3
time_dimensions = 0
output_dimension = 1
extrema_values = None
parameters_values = torch.tensor([[0, 2 * pi],  # phi
                                  [0, pi],
                                  [0, 1]])
type_of_points = "sobol"
r_min = 0.0
input_dimensions = 3
kernel_type = "isotropic"
n_quad = 10

if torch.cuda.is_available():
    dev = torch.device('cuda')
else:
    dev = torch.device("cpu")


def tile(a, dim, n_tile):
    init_dim = a.size(dim)
    repeat_idx = [1] * a.dim()
    repeat_idx[dim] = n_tile
    a = a.repeat(*(repeat_idx))
    order_index = torch.LongTensor(np.concatenate([init_dim * np.arange(n_tile) + i for i in range(init_dim)]))
    return torch.index_select(a, dim, order_index.to(dev))


def I0(x, y, z):
    x0 = 0.5
    y0 = 0.5
    z0 = 0.5

    r0 = torch.sqrt((x - x0) ** 2 + (y - y0) ** 2 + (z - z0) ** 2)
    source = 0.5 - r0
    source_mod = torch.where(r0 > 0.5, torch.tensor(0.).to(dev), source)

    return source_mod


def K(x, y, z):
    k = I0(x, y, z)
    return k.to(dev)


def S(x, y, z):
    sigma = x / x
    return sigma.to(dev)


def kernel(s, s_prime, kernel_type):
    if kernel_type == "isotropic":
        return 1 / (4 * pi)
    elif kernel_type == "HG":
        gamma = 0.5
        k = (1 - gamma ** 2) / (1 + gamma ** 2 - 2 * gamma * torch.sum(s * s_prime, 1))
        return k / (4 * pi)


def compute_scatter(x_train, model):
    quad_points, w = np.polynomial.legendre.leggauss(n_quad)

    phys_coord = x_train[:, :3]

    mat_quad = np.transpose([np.repeat(quad_points, len(quad_points)), np.tile(quad_points, len(quad_points))])
    w = w.reshape(-1, 1)
    mat_weight = np.matmul(w, w.T).reshape(n_quad * n_quad, 1)
    mat_weight = torch.from_numpy(mat_weight).type(torch.FloatTensor).to(dev)

    mat_quad[:, 0] = pi * (mat_quad[:, 0] + 1)
    mat_quad[:, 1] = pi / 2 * (mat_quad[:, 1] + 1)
    mat_quad = torch.from_numpy(mat_quad).type(torch.FloatTensor).to(dev)

    s = get_s(mat_quad)

    s_new = torch.cat(x_train.shape[0] * [s]).to(dev)
    phys_coord_new = tile(phys_coord, dim=0, n_tile=s.shape[0]).to(dev)

    inputs = torch.cat([phys_coord_new, s_new], 1).to(dev)

    u = model(inputs).reshape(-1, )
    u = u.reshape(x_train.shape[0], n_quad * n_quad)

    sin_theta_v = torch.sin(mat_quad[:, 1]).reshape(-1, 1)

    scatter_values = pi ** 2 / 2 * (1 / (4 * pi)) * torch.matmul(u, sin_theta_v * mat_weight).reshape(-1, )

    return scatter_values


def get_s(params):
    s = torch.tensor(()).new_full(size=(params.shape[0], 3), fill_value=0.0)
    phi = params[:, 0]
    theta = params[:, 1]
    s[:, 0] = torch.cos(phi) * torch.sin(theta)
    s[:, 1] = torch.sin(phi) * torch.sin(theta)
    s[:, 2] = torch.cos(theta)
    return s


def get_points(samples, dim, type_point_param, random_seed):
    if type_point_param == "uniform":
        torch.random.manual_seed(random_seed)
        points = torch.rand([samples, dim]).type(torch.FloatTensor)
    elif type_point_param == "sobol":
        # if n_time_step is None:
        skip = random_seed
        data = np.full((samples, dim), np.nan)
        for j in range(samples):
            seed = j + skip
            data[j, :], next_seed = sobol_seq.i4_sobol(dim, seed)
        points = torch.from_numpy(data).type(torch.FloatTensor)
    return points


def generator_param_samples(points):
    extrema_0 = parameters_values[:2, 0]
    extrema_f = parameters_values[:2, 1]
    points = points * (extrema_f - extrema_0) + extrema_0
    s_nu = torch.tensor(()).new_full(size=(points.shape[0], 3), fill_value=0.0)
    phi = points[:, 0]
    theta = points[:, 1]
    s_nu[:, 0] = torch.cos(phi) * torch.sin(theta)
    s_nu[:, 1] = torch.sin(phi) * torch.sin(theta)
    s_nu[:, 2] = torch.cos(theta)
    return s_nu


def generator_domain_samples(points, boundary):
    if boundary:
        n_single_dim = int(points.shape[0] / input_dimensions)
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

    u = network(x_f_train).reshape(-1, )
    grad_u = torch.autograd.grad(u, x_f_train, grad_outputs=torch.ones(x_f_train.shape[0], ).to(dev), create_graph=True)[0]

    grad_u_x = grad_u[:, 0]
    grad_u_y = grad_u[:, 1]
    grad_u_z = grad_u[:, 2]
    s1 = s[:, 0]
    s2 = s[:, 1]
    s3 = s[:, 2]

    scatter_kernel = compute_scatter(x_f_train, network)

    res = s1 * grad_u_x + s2 * grad_u_y + s3 * grad_u_z + (K(x, y, z) + S(x, y, z)) * u - S(x, y, z) * scatter_kernel - K(x, y, z) * I0(x, y, z)

    return res


def add_internal_points(n_internal):
    points = get_points(n_internal, 5, type_of_points, 16)
    dom_int = points[:n_internal, :3]
    angles_int = points[:n_internal, 3:]
    dom = generator_domain_samples(dom_int, boundary=False)
    s = generator_param_samples(angles_int)

    x = dom[:, 0].reshape(-1, 1)
    y = dom[:, 1].reshape(-1, 1)
    z = dom[:, 2].reshape(-1, 1)

    u = torch.full((n_internal, 1), 0)
    return torch.cat([x, y, z, s], 1), u


def exact(x, y, s1, s2):
    return 4 * s1 ** 2 * s2 ** 2 * torch.sin(2 * pi * (x ** 2 + y ** 2))


def add_boundary(n_boundary):
    print("Adding Boundary")
    points = get_points(n_boundary, 5, type_of_points, 16)
    dom = points[:, :3]
    angles = points[:, 3:]
    dom = generator_domain_samples(dom.clone(), boundary=True)
    s = generator_param_samples(angles)

    x = dom[:, 0].reshape(-1, 1)
    y = dom[:, 1].reshape(-1, 1)
    z = dom[:, 2].reshape(-1, 1)
    ub = torch.tensor(()).new_full(size=(n_boundary, 1), fill_value=0.0)

    return torch.cat([x, y, z, s], 1), ub


def add_collocations(n_collocation):
    print("Adding Collocation")
    n_coll_int = int(n_collocation)
    points = get_points(n_collocation, 5, type_of_points, 16)
    dom_int = points[:n_coll_int, :3]
    angles_int = points[:n_coll_int, 3:]
    dom = generator_domain_samples(dom_int, boundary=False)
    s = generator_param_samples(angles_int)

    dom[-100:, :3] = 0.5 * torch.ones_like(dom[-100:, :3])

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

    xyz = torch.cat([x.reshape(-1, 1), y.reshape(-1, 1), z.reshape(-1, 1)], 1).to(dev)
    xyz_mod = torch.where(xyz == 0, torch.tensor(-1.).to(dev), xyz).to(dev)
    n = torch.where(((xyz_mod != 1) & (xyz_mod != -1)), torch.tensor(0.).to(dev), xyz_mod).to(dev)

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

    u_pred = model(x_boundary_inf)

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

    n = 50
    phys_coord = torch.full((n * n * n, 3), 0)
    x = torch.linspace(0, 1, n)
    y = torch.linspace(0, 1, n)
    z = torch.linspace(0, 1, n)

    index = 0

    for k, z_k in enumerate(z):
        for j, y_j in enumerate(y):
            for i, x_i in enumerate(x):
                phys_coord[index, 0] = x_i
                phys_coord[index, 1] = y_j
                phys_coord[index, 2] = z_k
                index = index + 1

    phys_coord = phys_coord.to(dev)

    G = 4 * pi * compute_scatter(phys_coord, model)

    X = np.zeros((n, n, n))

    Y = np.zeros((n, n, n))

    Z = np.zeros((n, n, n))

    G_ = np.zeros((n, n, n))
    divq = np.zeros((n, n, n))

    index = 0

    for k, z_k in enumerate(z):
        for j, y_j in enumerate(y):
            for i, x_i in enumerate(x):
                X[i, j, k] = x[i]
                Y[i, j, k] = y[j]
                Z[i, j, k] = z[k]

                G_[i, j, k] = G[index].cpu().detach().numpy()
                index = index + 1

    gridToVTK(images_path + "/structured", X, Y, Z, pointData={"G": G_})
