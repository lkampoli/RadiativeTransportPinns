from ImportFile import *
import time
from scipy.special import legendre

pi = math.pi

extrema_values = None
space_dimensions = 1
time_dimensions = 0
domain_values = torch.tensor([[0.00, 1.0]])
parameters_values = torch.tensor([[-1.0, 1.0]])  # mu=cos(theta)

type_of_points = "sobol"
type_of_points_dom = "sobol"
r_min = 0.0
input_dimensions = 1
output_dimension = 1

ub_0 = 1.
ub_1 = 0.
strip = 0.05

n_quad = 10

if torch.cuda.is_available():
    dev = torch.device('cuda')
else:
    dev = torch.device("cpu")


def I0(x, mu):
    x = x.reshape(-1, )
    mu = mu.reshape(-1, )
    return (-mu * pi / 2 * torch.sin(pi / 2 * x) + torch.cos(pi / 2 * x)) * torch.sin(pi * mu) ** 2 - x / 2 * torch.cos(pi / 2 * x)


def sigma(x):
    return x


def kernel(mu, mu_prime):
    d = [1.0, 1.98398, 1.50823, 0.70075, 0.23489, 0.05133, 0.00760, 0.00048]
    k = torch.tensor(()).new_full(size=(mu.shape[0], mu_prime.shape[0]), fill_value=0.0)

    for p in range(len(d)):
        pn_mu = torch.from_numpy(legendre(p)(mu.detach().cpu().numpy()).reshape(-1, 1)).type(torch.FloatTensor)
        # plt.scatter(mu.detach().numpy(), pn_mu.detach().numpy())
        pn_mu_prime = torch.from_numpy(legendre(p)(mu_prime.detach().cpu().numpy()).reshape(-1, 1).T).type(torch.FloatTensor)
        kn = torch.matmul(pn_mu, pn_mu_prime)
        k = k + d[p] * kn

    return k.to(dev)


def compute_scattering(x, mu, model):
    mu_prime, w = np.polynomial.legendre.leggauss(n_quad)
    w = torch.from_numpy(w).type(torch.FloatTensor)
    mu_prime = torch.from_numpy(mu_prime).type(torch.FloatTensor)

    x_l = list(x.detach().cpu().numpy())
    mu_prime_l = list(mu_prime.detach().cpu().numpy())

    inputs = torch.from_numpy(np.transpose([np.repeat(x_l, len(mu_prime_l)), np.tile(mu_prime_l, len(x_l))])).type(torch.FloatTensor).to(dev)

    u = model(inputs)
    u = u.reshape(x.shape[0], mu_prime.shape[0])

    kern = kernel(mu, mu_prime)

    scatter_values = torch.zeros_like(x)

    for i in range(len(w)):
        scatter_values = scatter_values + w[i] * kern[:, i] * u[:, i]

    return scatter_values.to(dev)


def generator_samples(type_point_param, samples, dim, random_seed):
    extrema = torch.cat([domain_values, parameters_values], 0)
    extrema_0 = extrema[:, 0]
    extrema_f = extrema[:, 1]
    if type_point_param == "uniform":
        if random_seed is not None:
            torch.random.manual_seed(random_seed)
        params = torch.rand([samples, dim]).type(torch.FloatTensor) * (extrema_f - extrema_0) + extrema_0

        return params
    elif type_point_param == "sobol":
        # if n_time_step is None:
        skip = random_seed
        data = np.full((samples, dim), np.nan)
        for j in range(samples):
            seed = j + skip
            data[j, :], next_seed = sobol_seq.i4_sobol(dim, seed)
        params = torch.from_numpy(data).type(torch.FloatTensor) * (extrema_f - extrema_0) + extrema_0
        return params
    elif type_point_param == "grid":
        # if n_time_step is None:
        if dim == 2:
            n_mu = 128
            n_x = int(samples / n_mu)
            x = np.linspace(0, 1, n_x + 2)
            mu = np.linspace(0, 1, n_mu)
            x = x[1:-1]
            inputs = torch.from_numpy(np.transpose([np.repeat(x, len(mu)), np.tile(mu, len(x))])).type(torch.FloatTensor)
            inputs = inputs * (extrema_f - extrema_0) + extrema_0
        elif dim == 1:
            x = torch.linspace(0, 1, samples).reshape(-1, 1)
            mu = torch.linspace(0, 1, samples).reshape(-1, 1)
            inputs = torch.cat([x, mu], 1)
            inputs = inputs * (extrema_f - extrema_0) + extrema_0
        else:
            raise ValueError()

        return inputs.to(dev)


def compute_res(network, x_f_train, space_dimensions, solid_object, computing_error):
    x_f_train.requires_grad = True
    x = x_f_train[:, 0]
    mu = x_f_train[:, 1]

    x_f_train = x_f_train[~((x < 0.01) & (mu < 0.01) & (mu > -0.01))]
    x = x_f_train[:, 0]
    mu = x_f_train[:, 1]

    u = network(x_f_train).reshape(-1, )
    grad_u = torch.autograd.grad(u, x_f_train, grad_outputs=torch.ones(x_f_train.shape[0], ).to(dev), create_graph=True)[0]

    grad_u_x = grad_u[:, 0]

    scatter_values = compute_scattering(x, mu, network)
    residual = (mu * grad_u_x + u) - sigma(x) / 2 * scatter_values  # - I0(x, mu)

    res = residual

    return res


def add_internal_points(n_internal):
    x_internal = torch.tensor(()).new_full(size=(n_internal, parameters_values.shape[0] + domain_values.shape[0]), fill_value=0.0)
    y_internal = torch.tensor(()).new_full(size=(n_internal, 1), fill_value=0.0)

    return x_internal, y_internal


def add_boundary(n_boundary):
    mu0 = generator_samples(type_of_points, int(n_boundary / 2), parameters_values.shape[0], 1024)[:, 1].reshape(-1, 1)
    mu1 = generator_samples(type_of_points, int(n_boundary / 2), parameters_values.shape[0], 1024)[:, 1].reshape(-1, 1)
    x0 = torch.tensor(()).new_full(size=(int(n_boundary / 2), 1), fill_value=float(domain_values[0, 0]))
    x1 = torch.tensor(()).new_full(size=(int(n_boundary / 2), 1), fill_value=float(domain_values[0, 1]))
    x = torch.cat([x0, x1], 0)
    mu = torch.cat([mu0, mu1], 0)
    ub = torch.tensor(()).new_full(size=(int(n_boundary), 1), fill_value=ub_1)
    return torch.cat([x, mu], 1), ub


def add_collocations(n_collocation):
    inputs = generator_samples(type_of_points_dom, int(n_collocation), parameters_values.shape[0] + domain_values.shape[0], 1024)

    u = torch.tensor(()).new_full(size=(n_collocation, 1), fill_value=np.nan)
    return inputs, u


def apply_BC(x_boundary, u_boundary, model):
    x = x_boundary[:, 0]
    mu = x_boundary[:, 1]

    x0 = x[x == domain_values[0, 0]]
    x1 = x[x == domain_values[0, 1]]

    n0_len = x0.shape[0]
    n1_len = x1.shape[0]

    n0 = torch.tensor(()).new_full(size=(n0_len,), fill_value=-1.0)
    n1 = torch.tensor(()).new_full(size=(n1_len,), fill_value=1.0)
    n = torch.cat([n0, n1], 0).to(dev)

    scalar = n * mu < 0

    x_boundary_inf = x_boundary[scalar, :]
    u_boundary_inf = u_boundary[scalar, :]

    where_x_equal_0 = x_boundary_inf[:, 0] == domain_values[0, 0]

    u_boundary_inf = u_boundary_inf.reshape(-1, )
    u_boundary_inf_mod = torch.where(where_x_equal_0, torch.tensor(ub_0).to(dev), u_boundary_inf)

    u_pred = model(x_boundary_inf)

    return u_pred.reshape(-1, ), u_boundary_inf_mod.reshape(-1, )


def convert(vector):
    vector = np.array(vector)
    max_val = np.max(np.array(extrema_values), axis=1)
    min_val = np.min(np.array(extrema_values), axis=1)
    vector = vector * (max_val - min_val) + min_val
    return torch.from_numpy(vector).type(torch.FloatTensor)


def compute_generalization_error(model, extrema, images_path=None):
    return 0, 0


def plotting(model, images_path, extrema, solid):
    model.cpu()
    model = model.eval()
    n = 500

    x = np.linspace(domain_values[0, 0], domain_values[0, 1], n)
    mu = np.linspace(parameters_values[0, 0], parameters_values[0, 1], n)

    inputs = torch.from_numpy(np.transpose([np.repeat(x, len(mu)), np.tile(mu, len(x))])).type(torch.FloatTensor)
    sol = model(inputs)
    sol = sol.reshape(x.shape[0], mu.shape[0])

    x_l = inputs[:, 0]
    mu_l = inputs[:, 1]

    exact = torch.sin(pi * mu_l) ** 2 * torch.cos(pi / 2 * x_l)
    exact = exact.reshape(x.shape[0], mu.shape[0])

    print(torch.mean(abs(sol - exact) / torch.mean(abs(exact))))

    levels = [0.00, 0.006, 0.013, 0.021, 0.029, 0.04, 0.047, 0.06, 0.071, 0.099, 0.143, 0.214, 0.286, 0.357, 0.429, 0.5, 0.571, 0.643, 0.714, 0.786, 0.857, 0.929, 1]
    norml = matplotlib.colors.BoundaryNorm(levels, 256)
    plt.figure()
    plt.contourf(x.reshape(-1, ), mu.reshape(-1, ), sol.detach().numpy().T, cmap='jet', levels=levels, norm=norml)
    plt.axes().set_aspect('equal')
    plt.colorbar()
    plt.xlabel(r'$x$')
    plt.ylabel(r'$\mu$')
    plt.title(r'$u(x,\mu)$')
    plt.savefig(images_path + "/net_sol.png", dpi=400)

    plt.figure()
    plt.contourf(x.reshape(-1, ), mu.reshape(-1, ), exact.detach().numpy().T, cmap='jet')
    plt.axes().set_aspect('equal')
    plt.colorbar()
    plt.xlabel(r'$x$')
    plt.ylabel(r'$\mu$')
    plt.title(r'$u(x,\mu)$')
    plt.savefig(images_path + "/exact.png", dpi=400)

    x = np.linspace(0, 0, 1)
    mu = np.linspace(-1, 1, n)
    theta_ex = [0 + pi, pi / 12 + pi, pi / 6 + pi, pi / 4 + pi, pi / 3 + pi, 5 * pi / 12 + pi, 0, pi / 12, pi / 6, pi / 4, pi / 3, 5 * pi / 12, pi / 2]
    mu_ex = np.cos(theta_ex)
    sol_ex = np.array([0.0079, 0.0089, 0.0123, 0.0189, 0.0297, 0.0385, 1, 1, 1, 1, 1, 1, 1])
    inputs = torch.from_numpy(np.transpose([np.repeat(x, len(mu)), np.tile(mu, len(x))])).type(torch.FloatTensor)
    sol = model(inputs).detach().numpy()

    inputs_err = torch.from_numpy(np.concatenate([np.linspace(0, 0, len(mu_ex)).reshape(-1, 1), mu_ex.reshape(-1, 1)], 1)).type(torch.FloatTensor)
    sol_err = model(inputs_err).detach().numpy()

    err_1 = np.sqrt(np.mean((sol_ex.reshape(-1, ) - sol_err.reshape(-1, )) ** 2) / np.mean(sol_ex.reshape(-1, ) ** 2))

    plt.figure()
    plt.grid(True, which="both", ls=":")
    plt.plot(mu, sol, color="grey", lw=2, label=r'Learned Solution')
    plt.scatter(mu_ex, sol_ex, label=r'Exact Solution')
    plt.xlabel(r'$\mu$')
    plt.ylabel(r'$u^-(x=0)$')
    plt.legend()
    plt.savefig(images_path + "/u0.png", dpi=400)

    x = np.linspace(1, 1, 1)
    mu = np.linspace(-1, 1, n)
    theta_ex = [0 + pi, pi / 12 + pi, pi / 6 + pi, pi / 4 + pi, pi / 3 + pi, 5 * pi / 12 + pi, pi / 2 + pi, 0, pi / 12, pi / 6, pi / 4, pi / 3, 5 * pi / 12]
    mu_ex = np.cos(theta_ex)
    sol_ex = np.array([0, 0, 0, 0, 0, 0, 0, 0.5363, 0.5234, 0.4830, 0.4104, 0.3020, 0.1848])
    inputs = torch.from_numpy(np.transpose([np.repeat(x, len(mu)), np.tile(mu, len(x))])).type(torch.FloatTensor)
    sol = model(inputs).detach().numpy()

    inputs_err = torch.from_numpy(np.concatenate([np.linspace(1, 1, len(mu_ex)).reshape(-1, 1), mu_ex.reshape(-1, 1)], 1)).type(torch.FloatTensor)
    sol_err = model(inputs_err).detach().numpy()

    err_2 = np.sqrt(np.mean((sol_ex.reshape(-1, ) - sol_err.reshape(-1, )) ** 2) / np.mean(sol_ex.reshape(-1, ) ** 2))

    plt.figure()
    plt.grid(True, which="both", ls=":")
    plt.plot(mu, sol, color="grey", lw=2, label=r'Learned Solution')
    plt.scatter(mu_ex, sol_ex, label=r'Exact Solution')
    plt.xlabel(r'$\mu$')
    plt.ylabel(r'$u^+(x=1)$')
    plt.legend()
    plt.savefig(images_path + "/u1.png", dpi=400)

    with open(images_path + '/errors.txt', 'w') as file:
        file.write("err_1,"
                   "err_2\n")
        file.write(str(float(err_1)) + "," +
                   str(float(err_2)))
