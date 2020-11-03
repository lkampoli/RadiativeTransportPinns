from ImportFile import *
from scipy import special

pi = math.pi
extrema_values = None
time_dimensions = 1
space_dimensions = 3
parameters_values = torch.tensor([[0, 2 * pi],  # phi
                                  [0, pi],
                                  [0, 1],
                                  [0, 1]])

type_of_points = "sobol"
r_min = 0.0
input_dimensions = 3
output_dimension = 1
kernel_type = "isotropic"
n_quad = 10
Ri = 2
Re = 4
Tm = 120 * 11604.52500617
Ts = 150 * 11604.52500617
tf = 1

if torch.cuda.is_available():
    dev = torch.device('cuda')
else:
    dev = torch.device("cpu")


def b(T, nu):
    return 4 * np.pi * B(T, nu)


def B(T, nu):
    v = (nu.clone().cpu().detach().numpy()).astype(np.float)
    v = v.reshape(-1, )
    # print(v)
    kb = 1.38064852 * 10 ** (-23)
    h = 6.62607004 * 10 ** (-34)
    c = 299792458

    # print(2 * h * v ** 3)
    # print(c ** 2 * (np.exp(h * v / (T * kb)) - 1))

    if T < 1:
        b = torch.full((nu.shape[0], 1), 0)
    else:
        source = 2 * h * v ** 3 / (c ** 2 * (np.exp(h * v / (T * kb)) - 1))
        b = torch.from_numpy(source).type(torch.FloatTensor).reshape(-1, 1)

    return b


def Source(inputs, T):
    nu = trasform_freq(inputs[:, -1])
    v = (nu.clone().cpu().detach().numpy()).astype(np.float)
    v = v.reshape(-1, )
    kb = 1.38064852 * 10 ** (-23)
    h = 6.62607004 * 10 ** (-34)
    c = 299792458

    source = 2 * h * v ** 3 / (c ** 2 * (np.exp(h * v / (T * kb)) - 1))
    return torch.from_numpy(source).type(torch.FloatTensor).to(dev).reshape(-1, 1)


def trasform_freq(freq):
    max_exp = 18
    min_exp = 15
    return torch.pow(10, freq * (max_exp - min_exp) + min_exp)


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


def get_average_B(phys_coord, nu, n_quad, T):
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

    u = Source(inputs, T)[:, 0].reshape(-1, )
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


def exact(t, x, y, z, nu):
    x0 = 1
    y0 = 1
    z0 = 1
    nu = trasform_freq(nu).to(dev)
    r = torch.sqrt((x - x0) ** 2 + (y - y0) ** 2 + (z - z0) ** 2).to(dev)
    flux = exact_flux(t, x, y, z, nu)
    G = b(Tm, nu).to(dev) + (Ri / r) * (b(Ts, nu).to(dev) - b(Tm, nu).to(dev)) * flux.to(dev)
    return G


def exact_flux(t, x, y, z, nu):
    x0 = 1
    y0 = 1
    z0 = 1
    r = torch.sqrt((x - x0) ** 2 + (y - y0) ** 2 + (z - z0) ** 2)
    flux = 0.5 * torch.exp(-np.sqrt(3.) * K(x, y, z, nu).cpu() * (r - Ri).cpu()) * (
                special.erfc(0.5 * torch.sqrt(3 * K(x, y, z, nu).cpu() / t.cpu()) * (r - Ri).cpu() - torch.sqrt(K(x, y, z, nu).cpu() * t.cpu())) +
                special.erfc(0.5 * torch.sqrt(3 * K(x, y, z, nu).cpu() / t.cpu()) * (r - Ri).cpu() + torch.sqrt(K(x, y, z, nu).cpu() * t.cpu())))
    return flux.to(dev)


def K(x, y, z, nu):
    k = 10 * nu / nu
    return k


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
        for i in range(input_dimensions):
            n = int(n_single_dim / 2)
            points[2 * i * n:n * (2 * i + 1), i] = torch.tensor(()).new_full(size=(n,), fill_value=0.0)
            points[n * (2 * i + 1):2 * n * (i + 1), i] = torch.tensor(()).new_full(size=(n,), fill_value=1.0)

    return points


def compute_res(network, x_f_train, space_dimensions, solid_object, computing_error):
    x_f_train.requires_grad = True
    t = x_f_train[:, 0]
    x = x_f_train[:, 1]
    y = x_f_train[:, 2]
    z = x_f_train[:, 3]
    s = x_f_train[:, 4:-1]
    nu = trasform_freq(x_f_train[:, -1])

    u = network(x_f_train)[:, 0].reshape(-1, )

    grad_u = torch.autograd.grad(u, x_f_train, grad_outputs=torch.ones(x_f_train.shape[0], ).to(dev), create_graph=True)[0]

    grad_u_t = grad_u[:, 0]
    grad_u_x = grad_u[:, 1]
    grad_u_y = grad_u[:, 2]
    grad_u_z = grad_u[:, 3]

    grad_u_xx = torch.autograd.grad(grad_u_x, x_f_train, grad_outputs=torch.ones(x_f_train.shape[0], ).to(dev), create_graph=True)[0][:, 1]
    grad_u_yy = torch.autograd.grad(grad_u_y, x_f_train, grad_outputs=torch.ones(x_f_train.shape[0], ).to(dev), create_graph=True)[0][:, 2]
    grad_u_zz = torch.autograd.grad(grad_u_z, x_f_train, grad_outputs=torch.ones(x_f_train.shape[0], ).to(dev), create_graph=True)[0][:, 3]

    lap_u = grad_u_xx + grad_u_yy + grad_u_zz
    s1 = s[:, 0]
    s2 = s[:, 1]
    s3 = s[:, 2]

    f = B(Tm, nu).reshape(-1, ).to(dev)
    res = grad_u_t + s1 * grad_u_x + s2 * grad_u_y + s3 * grad_u_z + K(x, y, z, nu) * (u - f)
    # res = grad_u_t - lap_u/(3*K(x, y, z, nu))  + K(x, y, z, nu) * (u - f)
    # res = s1 * grad_u_x + s2 * grad_u_y + s3 * grad_u_z + K(x, y, z, nu) * (u - f)

    return res


def add_internal_points(n_internal):
    n_coll_int = int(n_internal)
    print("Adding Internal Points")
    points = get_points(n_internal, 7, "sobol", 16)
    t = points[:n_coll_int, 0].reshape(-1, 1)
    dom_int = points[:n_coll_int, 1:4]
    angles_int = points[:n_coll_int, 4:]
    dom = generator_domain_samples(dom_int, boundary=False)
    s = generator_param_samples(angles_int)

    x = dom[:, 0].reshape(-1, 1)
    y = dom[:, 1].reshape(-1, 1)
    z = dom[:, 2].reshape(-1, 1)

    u = torch.tensor(()).new_full(size=(n_internal, 1), fill_value=0.0)

    return torch.cat([t, x, y, z, s], 1), u


def add_boundary(n_boundary):
    freq_min = parameters_values[2, 0]
    freq_max = parameters_values[2, 1]
    print("Adding Boundary")
    n_boundary_phys = int(n_boundary)
    n_boundary_freq = n_boundary - n_boundary_phys
    points = get_points(n_boundary, 7, "sobol", 16)
    t = tf * points[:, 0].reshape(-1, 1)
    ti = t[:int(n_boundary / 2), :]
    te = t[int(n_boundary / 2):, :]
    dom = points[:, 1:4]
    angles = points[:, 4:]

    re = torch.full(size=(int(n_boundary / 2),), fill_value=Re)
    phi_e = 2 * pi * dom[:int(n_boundary / 2), 1]
    theta_e = pi * dom[:int(n_boundary / 2), 2]

    ri = torch.full(size=(int(n_boundary / 2),), fill_value=Ri)
    phi_i = 2 * pi * dom[int(n_boundary / 2):, 1]
    theta_i = pi * dom[int(n_boundary / 2):, 2]

    xe = (re * torch.cos(phi_e) * torch.sin(theta_e) + 1).reshape(-1, 1)
    ye = (re * torch.sin(phi_e) * torch.sin(theta_e) + 1).reshape(-1, 1)
    ze = (re * torch.cos(theta_e) + 1).reshape(-1, 1)

    xi = (ri * torch.cos(phi_i) * torch.sin(theta_i) + 1).reshape(-1, 1)
    yi = (ri * torch.sin(phi_i) * torch.sin(theta_i) + 1).reshape(-1, 1)
    zi = (ri * torch.cos(theta_i) + 1).reshape(-1, 1)

    s = generator_param_samples(angles)
    se = s[:int(n_boundary / 2), :]
    si = s[int(n_boundary / 2):, :]
    s[n_boundary_phys:(n_boundary_phys + int(n_boundary_freq / 2)), -1] = freq_min * torch.ones((int(n_boundary_freq / 2),))
    s[(n_boundary_phys + int(n_boundary_freq / 2)):, -1] = freq_max * torch.ones((int(n_boundary_freq / 2),))

    freq = s[:, -1].reshape(-1, 1)
    transf_freq = trasform_freq(freq)
    transf_freq_e = transf_freq[:int(n_boundary / 2), :]
    transf_freq_i = transf_freq[int(n_boundary / 2):, :]

    ube = B(Tm, transf_freq_e)
    ubi = B(Ts, transf_freq_i)

    inputs_i = torch.cat([ti, xi, yi, zi, si], 1)
    inputs_e = torch.cat([te, xe, ye, ze, se], 1)

    inputs = torch.cat([inputs_e, inputs_i], 0)
    ub = torch.cat([ube, ubi], 0)

    return inputs, ub


def add_initial_points(n_initial):
    print("Adding Initial Points")
    points = get_points(n_initial, 7, "sobol", 16)
    t = 0 * points[:n_initial, 0].reshape(-1, 1)
    dom_int = points[:n_initial, 1:4]

    r = dom_int[:, 0] * (Re - Ri) + Ri
    phi = 2 * pi * dom_int[:, 1]
    theta = pi * dom_int[:, 2]
    x = (r * torch.cos(phi) * torch.sin(theta) + 1).reshape(-1, 1)
    y = (r * torch.sin(phi) * torch.sin(theta) + 1).reshape(-1, 1)
    z = (r * torch.cos(theta) + 1).reshape(-1, 1)
    angles_int = points[:n_initial, 4:]
    dom = 2 * generator_domain_samples(dom_int, boundary=False)
    s = generator_param_samples(angles_int)

    freq = s[:, -1].reshape(-1, 1)
    transf_freq = trasform_freq(freq)

    u = B(Tm, transf_freq)

    return torch.cat([t, x, y, z, s], 1), u


def add_collocations(n_collocation):
    n_coll_int = int(n_collocation)
    print("Adding Collocation")
    points = get_points(n_collocation, 7, "sobol", 16)
    t = tf * points[:n_coll_int, 0].reshape(-1, 1)
    dom_int = points[:n_coll_int, 1:4]
    r = dom_int[:, 0] * (Re - Ri) + Ri
    phi = 2 * pi * dom_int[:, 1]
    theta = pi * dom_int[:, 2]
    x = (r * torch.cos(phi) * torch.sin(theta) + 1).reshape(-1, 1)
    y = (r * torch.sin(phi) * torch.sin(theta) + 1).reshape(-1, 1)
    z = (r * torch.cos(theta) + 1).reshape(-1, 1)
    angles_int = points[:n_coll_int, 4:]
    s = generator_param_samples(angles_int)

    u = torch.tensor(()).new_full(size=(n_collocation, 1), fill_value=np.nan)

    return torch.cat([t, x, y, z, s], 1), u


def apply_BC(x_boundary, u_boundary, model):

    '''
    freq = trasform_freq(x_boundary[:, -1])
    n_boundary = int(x_boundary.shape[0] / 2)
    x_be = x_boundary[:n_boundary, 1:4]
    x_bi = x_boundary[n_boundary:, 1:4]

    s_be = x_boundary[:n_boundary, 4:-1]
    s_bi = x_boundary[n_boundary:, 4:-1]

    scalar_e = (x_be[:, 0] * s_be[:, 0] + x_be[:, 1] * s_be[:, 1] + x_be[:, 2] * s_be[:, 2]) < 0
    scalar_i = (x_bi[:, 0] * s_bi[:, 0] + x_bi[:, 1] * s_bi[:, 1] + x_bi[:, 2] * s_bi[:, 2]) > 0

    x_boundary_e_tot = x_boundary[:n_boundary, :]
    x_boundary_i_tot = x_boundary[n_boundary:, :]
    u_boundary_e_tot = u_boundary[:n_boundary, :]
    u_boundary_i_tot = u_boundary[n_boundary:, :]

    x_boundary_e = x_boundary_e_tot[scalar_e, :]
    x_boundary_i = x_boundary_i_tot[scalar_i, :]

    u_boundary_e = u_boundary_e_tot[scalar_e, :]
    u_boundary_i = u_boundary_i_tot[scalar_i, :]

    u_pred_e = model(x_boundary_e)
    u_pred_i = model(x_boundary_i)'''

    u_pred = model(x_boundary)


    return u_pred.reshape(-1, ), u_boundary.reshape(-1, )


def apply_IC(x_initial, u_initial, model):
    freq = trasform_freq(x_initial[:, -1])
    u_pred = model(x_initial)
    return u_pred.reshape(-1, ), u_initial.reshape(-1, )


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

    freq_min = parameters_values[2, 0]
    freq_max = parameters_values[2, 1]

    nu = torch.linspace(freq_min, freq_max, 150).reshape(-1, 1).to(dev)
    # nu = trasform_freq(nu)
    t = torch.linspace(tf, tf, 4).reshape(-1, 1)
    x = torch.linspace(1 + Ri / 3 ** 0.5, 1 + 2.1 / 3 ** 0.5, 4).reshape(-1, 1)
    y = torch.linspace(1 + Ri / 3 ** 0.5, 1 + 2.1 / 3 ** 0.5, 4).reshape(-1, 1)
    z = torch.linspace(1 + Ri / 3 ** 0.5, 1 + 2.1 / 3 ** 0.5, 4).reshape(-1, 1)
    scale_vec = np.linspace(0.65, 1.55, len(x))

    fig = plt.figure()
    plt.grid(True, which="both", ls=":")

    for t_k, x_k, y_k, z_k, scale in zip(t, x, y, z, scale_vec):
        t_k = float(t_k.numpy())
        x_k = float(x_k.numpy())
        y_k = float(y_k.numpy())
        z_k = float(z_k.numpy())
        t_v = torch.full((nu.shape[0], 1), t_k).to(dev)
        x_v = torch.full((nu.shape[0], 1), x_k).to(dev)
        y_v = torch.full((nu.shape[0], 1), y_k).to(dev)
        z_v = torch.full((nu.shape[0], 1), z_k).to(dev)
        txyz = torch.cat([t_v, x_v, y_v, z_v], 1)

        fx, fy, fz = get_F(model, txyz, nu, n_quad)
        g = get_G(model, txyz, nu, n_quad).detach().cpu().numpy()
        g_ex = exact(t_v, x_v, y_v, z_v, nu).detach().cpu().numpy()
        Bm = get_average_B(txyz, nu, n_quad, Tm).detach().cpu().numpy()
        Bs = get_average_B(txyz, nu, n_quad, Ts).detach().cpu().numpy()
        fr = torch.sqrt(fx ** 2 + fy ** 2 + fz ** 2).detach().cpu().numpy()

        x0 = 1
        y0 = 1
        z0 = 1
        r_s = round(np.sqrt((x_k - x0) ** 2 + (y_k - y0) ** 2 + (z_k - z0) ** 2), 2)

        plt.scatter(torch.log10(trasform_freq(nu)).cpu().numpy(), g, label=r'Predicted, $r=$' + str(r_s), marker="o", s=14, color=lighten_color('C0', scale), zorder=10)
        plt.plot(torch.log10(trasform_freq(nu)).cpu().numpy(), g_ex, label=r'Exact, $r=$' + str(r_s), color=lighten_color('grey', scale), zorder=10)

        plt.xlabel(r'$\nu$')
        plt.ylabel(r'$F_\nu$')
        plt.legend()
    plt.savefig(images_path + "/Samples_q.png", dpi=500)

    n = 150
    t = torch.linspace(tf, tf, n).reshape(-1, 1)
    x = torch.linspace(1 + Ri / 3 ** 0.5, 1 + Re / 3 ** 0.5, n).reshape(-1, 1)
    y = torch.linspace(1 + Ri / 3 ** 0.5, 1 + Re / 3 ** 0.5, n).reshape(-1, 1)
    z = torch.linspace(1 + Ri / 3 ** 0.5, 1 + Re / 3 ** 0.5, n).reshape(-1, 1)

    nu = torch.linspace(freq_min, freq_max, n).reshape(-1, 1).to(dev)

    radius = torch.sqrt((x - x0) ** 2 + (y - y0) ** 2 + (z - z0) ** 2)

    g_countour = np.zeros((n, n))
    g_countour_ex = np.zeros((n, n))

    k = 0
    for t_k, x_k, y_k, z_k in zip(t, x, y, z):
        t_k = float(t_k.numpy())
        x_k = float(x_k.numpy())
        y_k = float(y_k.numpy())
        z_k = float(z_k.numpy())
        t_v = torch.full((nu.shape[0], 1), t_k).to(dev)
        x_v = torch.full((nu.shape[0], 1), x_k).to(dev)
        y_v = torch.full((nu.shape[0], 1), y_k).to(dev)
        z_v = torch.full((nu.shape[0], 1), z_k).to(dev)
        txyz = torch.cat([t_v, x_v, y_v, z_v], 1)

        r = round(np.sqrt((x_k - x0) ** 2 + (y_k - y0) ** 2 + (z_k - z0) ** 2), 2)

        g = get_G(model, txyz, nu, n_quad).detach().cpu().numpy()
        g_ex = exact(t_v, x_v, y_v, z_v, nu).detach().cpu().numpy()

        g_countour[:, k] = g.reshape(-1, )
        g_countour_ex[:, k] = g_ex.reshape(-1, )

        k = k + 1

    plt.figure()
    plt.title(r'$G^\ast(r, \nu)$')
    ax = plt.gca()
    ttl = ax.title
    ttl.set_position([.5, 1.035])
    plt.contourf(radius.detach().numpy().reshape(-1, ), torch.log10(trasform_freq(nu)).cpu().numpy().reshape(-1, ), g_countour, 20, cmap='Spectral')
    plt.colorbar()
    plt.xlabel(r'$r$')
    plt.ylabel(r'$\nu$')
    plt.savefig(images_path + "/contour_G.png", dpi=400)

    plt.figure()
    plt.title(r'$G(r, \nu)$')
    ax = plt.gca()
    ttl = ax.title
    ttl.set_position([.5, 1.035])
    plt.contourf(radius.detach().numpy().reshape(-1, ), torch.log10(trasform_freq(nu)).cpu().numpy().reshape(-1, ), g_countour_ex, 20, cmap='Spectral')
    plt.colorbar()
    plt.xlabel(r'$r$')
    plt.ylabel(r'$\nu$')
    plt.savefig(images_path + "/contour_G_ex.png", dpi=400)
