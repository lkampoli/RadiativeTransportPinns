from ImportFile import *
from pyevtk.hl import gridToVTK

pi = math.pi
extrema_values = None
time_dimensions = 0
space_dimensions = 3
parameters_values = torch.tensor([[0, 2 * pi],  # phi
                                  [0, pi],
                                  [-6, 6],
                                  [0, 1]])

type_of_points = "sobol"
r_min = 0.0
input_dimensions = 3
kernel_type = "isotropic"
n_quad = 10
a = 8

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


def get_G(intensity, phys_coord, nu, n_quad):
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
    nu_new = tile(nu, dim=0, n_tile=s.shape[0]).to(dev).reshape(-1, 1)

    inputs = torch.cat([phys_coord_new, s_new, nu_new], 1).to(dev)

    u = intensity(inputs)[:, 0].reshape(-1, )
    u = u.reshape(phys_coord.shape[0], n_quad * n_quad)

    sin_theta_v = torch.sin(mat_quad[:, 1]).reshape(-1, 1)

    G = pi ** 2 / 2 * torch.matmul(u, sin_theta_v * mat_weight).reshape(-1, )

    return G


def get_F(intensity, phys_coord, nu, n_quad):
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
    nu_new = tile(nu, dim=0, n_tile=s.shape[0]).to(dev).reshape(-1, 1)

    inputs = torch.cat([phys_coord_new, s_new, nu_new], 1).to(dev)

    u = intensity(inputs)[:, 0].reshape(-1, )
    u = u.reshape(phys_coord.shape[0], n_quad * n_quad)

    theta_v = mat_quad[:, 1].reshape(-1, 1)
    phi_v = mat_quad[:, 0].reshape(-1, 1)

    fx = pi ** 2 / 2 * torch.matmul(u, torch.cos(phi_v) * torch.sin(theta_v) * torch.sin(theta_v) * mat_weight).reshape(-1, )
    fy = pi ** 2 / 2 * torch.matmul(u, torch.sin(phi_v) * torch.sin(theta_v) * torch.sin(theta_v) * mat_weight).reshape(-1, )
    fz = pi ** 2 / 2 * torch.matmul(u, torch.cos(theta_v) * torch.sin(theta_v) * mat_weight).reshape(-1, )

    return fx, fy, fz


def tile(a, dim, n_tile):
    init_dim = a.size(dim)
    repeat_idx = [1] * a.dim()
    repeat_idx[dim] = n_tile
    a = a.repeat(*(repeat_idx))
    order_index = torch.LongTensor(np.concatenate([init_dim * np.arange(n_tile) + i for i in range(init_dim)]))
    return torch.index_select(a, dim, order_index.to(dev))


def gamma(nu):
    return torch.exp(-nu ** 2 / 2) / pi ** 0.5


def f(x, y, z, nu):
    r = ((x - 0.5) ** 2 + (y - 0.5) ** 2 + (z - 0.5) ** 2) ** 0.5
    r = r.reshape(-1, )
    nu = nu.reshape(-1, )
    source = pi ** 0.5 * gamma(nu) * (1 - r / 0.5)
    source_mod = torch.where(r >= 0.5, torch.tensor(0.).to(dev), source)

    return source_mod


def exact_flux(x, y, z, nu):
    r = ((x - 0.5) ** 2 + (y - 0.5) ** 2 + (z - 0.5) ** 2) ** 0.5
    r = r.reshape(-1, )
    nu = nu.reshape(-1, ).reshape(-1, )
    exact_f = torch.zeros_like(r).to(dev)
    for i in range(r.shape[0]):
        r_i = r[i].to(dev)
        if r_i < 0.5:
            exact_f[i] = r_i / 3 - r_i ** 2 / 2
        else:
            exact_f[i] = 1 / (96 * r_i ** 2)
    exact_flux = 4 * pi * pi ** 0.5 * gamma(nu).reshape(-1, ) * exact_f.reshape(-1, )
    return exact_flux.to(dev)


def G(x, y, z, nu):
    return -64 * gamma(nu) * (x ** 2 - x) * (y ** 2 - y) * (z ** 2 - z)


def K(x, y, z):
    k = torch.zeros_like(x)
    return k.to(dev)


def S(x, y, z):
    sigma = torch.ones_like(x)
    return sigma.to(dev)


def exact_kernel(x, y, z, nu):
    if kernel_type == "isotropic":
        return G(x, y, z, nu)
    elif kernel_type == "redistribution":
        return -64 * pi ** 0.5 * math.erf(4) * (x ** 2 - x) * (y ** 2 - y) * (z ** 2 - z)


def compute_scatter(x_train, model):
    phys_coord = x_train[:, :3]
    nu_train = x_train[:, -1]
    if kernel_type == "isotropic":
        scatter_values = get_G(model, phys_coord, nu_train, n_quad)
    elif kernel_type == "redistribution":
        quad_points, w = np.polynomial.legendre.leggauss(n_quad)
        phys_coord = x_train[:, :3]
        w = w.reshape(-1, 1)

        freq_min = parameters_values[2, 0]
        freq_max = parameters_values[2, 1]

        w = torch.from_numpy(w).type(torch.FloatTensor).to(dev)
        quad_points = torch.from_numpy(quad_points).type(torch.FloatTensor).to(dev)

        nu = (freq_max - freq_min) / 2 * quad_points + (freq_max + freq_min) / 2
        phys_coord_new = tile(phys_coord, dim=0, n_tile=w.shape[0]).to(dev)
        nu_new = torch.cat(phys_coord.shape[0] * [nu]).to(dev)

        integral = get_G(model, phys_coord_new, nu_new, n_quad) * gamma(nu_new)
        integral = integral.reshape(phys_coord.shape[0], w.shape[0])

        scatter_values = (freq_max - freq_min) / 2 * torch.matmul(integral, w).reshape(-1, )

    else:
        raise ValueError()

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
    extrema_0 = parameters_values[:3, 0]
    extrema_f = parameters_values[:3, 1]
    points = points * (extrema_f - extrema_0) + extrema_0
    s_nu = torch.tensor(()).new_full(size=(points.shape[0], 4), fill_value=0.0)
    phi = points[:, 0]
    theta = points[:, 1]
    nu = points[:, 2]
    s_nu[:, 0] = torch.cos(phi) * torch.sin(theta)
    s_nu[:, 1] = torch.sin(phi) * torch.sin(theta)
    s_nu[:, 2] = torch.cos(theta)
    s_nu[:, 3] = nu
    return s_nu


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
    s = x_f_train[:, 3:-1]
    nu = x_f_train[:, -1]
    # s = get_s(angles)

    u = network(x_f_train)[:, 0].reshape(-1, )

    grad_u = torch.autograd.grad(u, x_f_train, grad_outputs=torch.ones(x_f_train.shape[0], ).to(dev), create_graph=True)[0]

    grad_u_x = grad_u[:, 0]
    grad_u_y = grad_u[:, 1]
    grad_u_z = grad_u[:, 2]
    s1 = s[:, 0]
    s2 = s[:, 1]
    s3 = s[:, 2]

    scatter_kernel = compute_scatter(x_f_train, network)

    res = s1 * grad_u_x + s2 * grad_u_y + s3 * grad_u_z + (K(x, y, z) + S(x, y, z)) * u - f(x, y, z, nu) - S(x, y, z) * scatter_kernel / (4 * pi)

    return res


def add_internal_points(n_internal):
    n_int = int(n_internal * 3 / 4)
    print("Adding Collocation")
    points = get_points(n_internal, 6, "uniform", 16)
    dom_int = points[:n_int, :3]
    angles_int = points[:n_int, 3:]
    dom_b = points[n_int:, :3]
    angles_b = points[n_int:, 3:]
    dom = torch.cat([generator_domain_samples(dom_int, boundary=False), generator_domain_samples(dom_b, boundary=True)])
    s_nu = torch.cat([generator_param_samples(angles_int), generator_param_samples(angles_b)])

    x = dom[:, 0].reshape(-1, 1)
    y = dom[:, 1].reshape(-1, 1)
    z = dom[:, 2].reshape(-1, 1)

    u = torch.full((n_internal, 1), 0)
    return torch.cat([x, y, z, s_nu], 1), u


def add_boundary(n_boundary):
    freq_min = parameters_values[2, 0]
    freq_max = parameters_values[2, 1]
    print("Adding Boundary")
    n_boundary_phys = int(n_boundary * 4 / 4)
    n_boundary_freq = n_boundary - n_boundary_phys
    points = get_points(n_boundary, 6, "sobol", 16)
    dom = points[:, :3]
    angles = points[:, 3:]
    dom = torch.cat([generator_domain_samples(dom[:n_boundary_phys, :].clone(), boundary=True), generator_domain_samples(dom[n_boundary_phys:, :].clone(), boundary=True)])
    s = generator_param_samples(angles)
    s[n_boundary_phys:(n_boundary_phys + int(n_boundary_freq / 2)), -1] = freq_min * torch.ones((int(n_boundary_freq / 2),))
    s[(n_boundary_phys + int(n_boundary_freq / 2)):, -1] = freq_max * torch.ones((int(n_boundary_freq / 2),))

    x = dom[:, 0].reshape(-1, 1)
    y = dom[:, 1].reshape(-1, 1)
    z = dom[:, 2].reshape(-1, 1)
    ub = torch.tensor(()).new_full(size=(n_boundary, 1), fill_value=0.0)

    return torch.cat([x, y, z, s], 1), ub


def add_collocations(n_collocation):
    n_coll_int = int(n_collocation)
    print("Adding Collocation")
    points = get_points(n_collocation, 6, "sobol", 16)
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
    n_boundary = x_boundary.shape[0]
    n_boundary_phys = int(n_boundary * 4 / 4)

    x_boundary_phys = x_boundary[:n_boundary_phys, :]
    u_boundary_phys = u_boundary[:n_boundary_phys, :]

    x_boundary_freq = x_boundary[n_boundary_phys:, :]
    u_boundary_freq = u_boundary[n_boundary_phys:, :]

    x = x_boundary_phys[:, 0]
    y = x_boundary_phys[:, 1]
    z = x_boundary_phys[:, 2]

    s = x_boundary_phys[:, 3:-1]

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

    x_boundary_inf = x_boundary_phys[scalar, :]
    u_boundary_inf = u_boundary_phys[scalar, :]

    u_pred_phys = model(x_boundary_inf)
    u_pred_freq = model(x_boundary_freq)

    u_pred = torch.cat([u_pred_phys, u_pred_freq])
    u_bound = torch.cat([u_boundary_inf, u_boundary_freq])

    return u_pred.reshape(-1, ), u_bound.reshape(-1, )


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
    freq_min = parameters_values[2, 0]
    freq_max = parameters_values[2, 1]

    phys_coord_nu = torch.rand((20000, 4))
    phys_coord_nu[:, -1] = phys_coord_nu[:, -1] * (freq_max - freq_min) + freq_min
    phys_coord_nu = phys_coord_nu.to(dev)
    xyz = phys_coord_nu[:, :3]
    nu = phys_coord_nu[:, -1]
    fx, fy, fz = get_F(model, xyz, nu, n_quad)
    fr = torch.sqrt(fx ** 2 + fy ** 2 + fz ** 2).detach().cpu().numpy()
    fr_exact = exact_flux(xyz[:, 0], xyz[:, 1], xyz[:, 2], nu).detach().cpu().numpy()

    err = np.sqrt(np.mean((fr - fr_exact) ** 2) / (np.mean(fr_exact ** 2)))
    print(err)

    nu = torch.linspace(freq_min, freq_max, 150).reshape(-1, 1).to(dev)
    x = torch.linspace(0.5, 1, 3).reshape(-1, 1)
    y = torch.linspace(0.5, 1, 3).reshape(-1, 1)
    z = torch.linspace(0.5, 1, 3).reshape(-1, 1)

    scale_vec = np.linspace(0.65, 1.55, len(x))

    fig = plt.figure()
    plt.grid(True, which="both", ls=":")

    for x_k, y_k, z_k, scale in zip(x, y, z, scale_vec):
        x_k = float(x_k.numpy())
        y_k = float(y_k.numpy())
        z_k = float(z_k.numpy())
        x_v = torch.full((nu.shape[0], 1), x_k).to(dev)
        y_v = torch.full((nu.shape[0], 1), y_k).to(dev)
        z_v = torch.full((nu.shape[0], 1), z_k).to(dev)
        xyz = torch.cat([x_v, y_v, z_v], 1)

        fx, fy, fz = get_F(model, xyz, nu, n_quad)
        fr = torch.sqrt(fx ** 2 + fy ** 2 + fz ** 2).detach().cpu().numpy()
        fr_exact = exact_flux(x_v, y_v, z_v, nu).detach().cpu().numpy()

        plt.scatter(nu.cpu().numpy(), fr, label=r'Learned Solution, $x=$' + str(round(x_k, 1)), marker="o", s=14, color=lighten_color('C0', scale), zorder=10)
        plt.plot(nu.cpu().numpy(), fr_exact, label=r'Exact Solution, $x=$' + str(round(x_k, 1)), color=lighten_color("grey", scale), zorder=10)
        plt.xlabel(r'$\nu$')
        plt.ylabel(r'$F_\nu$')
        plt.legend()
    plt.savefig(images_path + "/Samples_q.png", dpi=500)

    with open(images_path + '/errors.txt', 'w') as file:
        file.write("err_1,"
                   "err_2\n")
        file.write(str(float(err)) + "," +
                   str(float(0)))
