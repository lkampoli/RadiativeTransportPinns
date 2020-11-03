from ImportFile import *

pi = math.pi


class SolidObject:

    def __init__(self, n_object_space, n_object_time, space_dimensions, time_dimensions, extrema_values):
        self.n_object_space = n_object_space
        self.n_object_time = n_object_time
        self.space_dimensions = space_dimensions
        self.time_dimensions = time_dimensions
        self.extrema_values = extrema_values

    def construct_object(self):
        x_ob = self.draw_shape()
        y_ob = torch.tensor(()).new_full(size=(x_ob.shape[0], 3), fill_value=0.0).type(torch.FloatTensor)
        BC = ["func", "func", "der"]
        return x_ob, y_ob, BC


class Cylinder(SolidObject):

    def __init__(self, n_object_space, n_object_time, space_dimensions, time_dimensions, extrema_values, r, xc, yc):
        super().__init__(n_object_space, n_object_time, space_dimensions, time_dimensions, extrema_values)
        self.radius = r
        self.xc = xc
        self.yc = yc

    def draw_shape(self):
        val_0 = self.extrema_values[:, 0]
        val_f = self.extrema_values[:, 1]
        x_ob = torch.tensor(()).new_full(size=[self.n_object_space * self.n_object_time, self.space_dimensions + self.time_dimensions], fill_value=0.0).type(torch.FloatTensor)
        x_ob = x_ob * (val_f - val_0) + val_0
        points = torch.from_numpy(np.arange(0, self.n_object_space * self.n_object_time)).type(torch.FloatTensor)

        if self.time_dimensions != 0:
            x_ob[:, 1] = self.radius * torch.cos(2 * pi * points / self.n_object_space * self.n_object_time) + self.xc
            x_ob[:, 2] = self.radius * torch.sin(2 * pi * points / self.n_object_space * self.n_object_time) + self.yc
        else:
            x_ob[:, 0] = self.radius * torch.cos(2 * pi * points / self.n_object_space * self.n_object_time) + self.xc
            x_ob[:, 1] = self.radius * torch.sin(2 * pi * points / self.n_object_space * self.n_object_time) + self.yc
        # x_ob = x_ob[torch.randperm(x_ob.shape[0])]
        x_ob = x_ob.type(torch.FloatTensor)
        return x_ob

    def im_in(self, x):
        if x.shape[1] > 2:
            raise ValueError("Implemented only for steady 2D problems")
        im_in = (x[:, 0] - self.xc) ** 2 + (x[:, 1] - self.yc) ** 2 < self.radius ** 2
        return im_in

    def im_out(self, x):
        if x.shape[1] > 2:
            raise ValueError("Implemented only for steady 2D problems")
        im_out = (x[:, 0] - self.xc) ** 2 + (x[:, 1] - self.yc) ** 2 > self.radius ** 2 * 1.01
        return im_out

    def plot_object(self):
        circle = plt.Circle((self.xc, self.yc), self.radius, color='grey')
        plt.axes().add_artist(circle)


class Square(SolidObject):

    def __init__(self, n_object_space, n_object_time, space_dimensions, time_dimensions, extrema_values, Lx, Ly, xc, yc):
        super().__init__(n_object_space, n_object_time, space_dimensions, time_dimensions, extrema_values)
        self.Lx = Lx
        self.Ly = Ly
        self.xc = xc
        self.yc = yc

    def draw_shape(self):

        sample_sides = int(self.n_object_space / 4)
        x_side = torch.linspace(0, self.Lx, sample_sides).reshape(-1, 1)
        y_side = torch.linspace(0, self.Ly, sample_sides).reshape(-1, 1)
        zero_vec = torch.tensor(()).new_full(size=[sample_sides, 1], fill_value=0.0).type(torch.FloatTensor)
        Lx_vec = torch.tensor(()).new_full(size=[sample_sides, 1], fill_value=self.Lx).type(torch.FloatTensor)
        Ly_vec = torch.tensor(()).new_full(size=[sample_sides, 1], fill_value=self.Ly).type(torch.FloatTensor)
        side_b = torch.cat([x_side, zero_vec], 1)
        side_u = torch.cat([x_side, Ly_vec], 1)
        side_l = torch.cat([zero_vec, y_side], 1)
        side_r = torch.cat([Lx_vec, y_side], 1)
        x_ob = torch.cat([side_b, side_u, side_l, side_r], 0)

        x_ob[:, 0] = x_ob[:, 0] + (self.xc - self.Lx / 2)
        x_ob[:, 1] = x_ob[:, 1] + (self.yc - self.Ly / 2)

        # plt.scatter(x_ob[:, 0].detach().numpy(), x_ob[:, 1].detach().numpy())

        # plt.show()
        # quit()
        # x_ob = x_ob[torch.randperm(x_ob.shape[0])]
        x_ob = x_ob.type(torch.FloatTensor)
        return x_ob

    def im_in(self, x):
        if x.shape[1] > 2:
            raise ValueError("Implemented only for steady 2D problems")
        im_in = (-self.Lx / 2 < (x[:, 0] - self.xc)) & ((x[:, 0] - self.xc) < self.Lx / 2) & \
                (-self.Ly / 2 < (x[:, 1] - self.yc)) & ((x[:, 1] - self.yc) < self.Ly / 2)
        return im_in

    def im_out(self, x):
        if x.shape[1] > 2:
            raise ValueError("Implemented only for steady 2D problems")
        im_out = (-self.Lx / 2* 1.01 > (x[:, 0] - self.xc)) | ((x[:, 0] - self.xc) > self.Lx / 2* 1.01) | \
                 (-self.Ly / 2* 1.01 > (x[:, 1] - self.yc)) | ((x[:, 1] - self.yc) > self.Ly / 2* 1.01)
        return im_out

    def im_on(self, x):
        if x.shape[1] > 2:
            raise ValueError("Implemented only for steady 2D problems")
        im_in = (-self.Lx / 2 == (x[:, 0] - self.xc)) & ((x[:, 0] - self.xc) == self.Lx / 2) & \
                (-self.Ly / 2 == (x[:, 1] - self.yc)) & ((x[:, 1] - self.yc) == self.Ly / 2)
        return im_in

    def plot_object(self):
        print("no ob")
