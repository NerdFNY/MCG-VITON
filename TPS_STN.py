import numpy as np

def TPS_STN(U, nx, ny, cp, out_size):
    """Thin Plate Spline Spatial Transformer Layer
    TPS control points are arranged in a regular grid.

    U : float Tensor
        shape [num_batch, height, width, num_channels].
    nx : int
        The number of control points on x-axis
    ny : int
        The number of control points on y-axis
    cp : float Tensor
        control points. shape [num_batch, nx*ny, 2].
    out_size: tuple of two ints
        The size of the output of the network (height, width)

    ----------
    Reference :
      https://github.com/daviddao/spatial-transformer-tensorflow/blob/master/spatial_transformer.py
    """

    def _repeat(x, n_repeats):
        rep = np.transpose(
            np.expand_dims(np.ones(shape=np.stack([n_repeats, ])), 1), [1, 0])
        rep =np.array(rep, dtype=int)
        x = np.matmul(np.reshape(x, (-1, 1)), rep)
        return np.reshape(x, [-1])

    def _interpolate(im, x, y, out_size):
        # constants
        num_batch = im.shape[0]
        height = im.shape[1]
        width = im.shape[2]
        channels = im.shape[3]

        x = np.array(x, dtype=float)
        y = np.array(y, dtype=float)
        height_f =float(height)
        width_f = float(width)
        out_height = out_size[0]
        out_width = out_size[1]
        zero = np.zeros([], dtype='int32')
        max_y = int(im.shape[1] - 1)
        max_x = int(im.shape[2] - 1)

        # scale indices from [-1, 1] to [0, width/height]
        x = (x + 1.0)*(width_f) / 2.0
        y = (y + 1.0)*(height_f) / 2.0

        # do sampling
        x0 = np.array(np.floor(x), dtype=int)
        x1 = x0 + 1
        y0 = np.array(np.floor(y), dtype=int)
        y1 = y0 + 1

        x0 = np.clip(x0, zero, max_x)
        x1 = np.clip(x1, zero, max_x)
        y0 = np.clip(y0, zero, max_y)
        y1 = np.clip(y1, zero, max_y)
        dim2 = width
        dim1 = width*height
        base = _repeat(np.arange(num_batch)*dim1, out_height*out_width)
        base_y0 = base + y0*dim2
        base_y1 = base + y1*dim2
        idx_a = base_y0 + x0
        idx_b = base_y1 + x0
        idx_c = base_y0 + x1
        idx_d = base_y1 + x1

        # use indices to lookup pixels in the flat image and restore
        # channels dim
        im_flat = np.reshape(im, np.stack([-1, channels]))
        im_flat = np.array(im_flat, dtype=float)

        Ia=[]
        Ib=[]
        Ic=[]
        Id=[]
        for i in range(len(idx_a)):
            Ia.append(im_flat[idx_a[i]])
        for i in range(len(idx_b)):
            Ib.append(im_flat[idx_b[i]])
        for i in range(len(idx_c)):
            Ic.append(im_flat[idx_c[i]])
        for i in range(len(idx_d)):
            Id.append(im_flat[idx_d[i]])
        # and finally calculate interpolated values
        x0_f = np.array(x0, dtype=float)
        x1_f = np.array(x1, dtype=float)
        y0_f = np.array(y0, dtype=float)
        y1_f = np.array(y1, dtype=float)
        wa = np.expand_dims(((x1_f-x) * (y1_f-y)), 1)
        wb = np.expand_dims(((x1_f-x) * (y-y0_f)), 1)
        wc = np.expand_dims(((x-x0_f) * (y1_f-y)), 1)
        wd = np.expand_dims(((x-x0_f) * (y-y0_f)), 1)
        output = wa*Ia+wb*Ib+wc*Ic+wd*Id
        return output

    def _meshgrid(height, width, fp):
        x_t = np.matmul(
            np.ones(shape=np.stack([height, 1])),
            np.transpose(np.expand_dims(np.linspace(-1.0, 1.0, width), 1), [1, 0]))
        y_t = np.matmul(
            np.expand_dims(np.linspace(-1.0, 1.0, height), 1),
            np.ones(shape=np.stack([1, width])))

        x_t_flat = np.reshape(x_t, (1, -1))
        y_t_flat = np.reshape(y_t, (1, -1))

        x_t_flat_b = np.expand_dims(x_t_flat, 0) # [1, 1, h*w]
        y_t_flat_b = np.expand_dims(y_t_flat, 0) # [1, 1, h*w]

        num_batch = fp.shape[0]
        px = np.expand_dims(fp[:,:,0], 2) # [n, nx*ny, 1]
        py = np.expand_dims(fp[:,:,1], 2) # [n, nx*ny, 1]
        d = np.sqrt(np.power(x_t_flat_b - px, 2.) + np.power(y_t_flat_b - py, 2.))
        r = np.power(d, 2) * np.log(d + 1e-6) # [n, nx*ny, h*w]
        x_t_flat_g = np.tile(x_t_flat_b, np.stack([num_batch, 1, 1])) # [n, 1, h*w]
        y_t_flat_g = np.tile(y_t_flat_b, np.stack([num_batch, 1, 1])) # [n, 1, h*w]
        ones = np.ones_like(x_t_flat_g) # [n, 1, h*w]

        grid = np.concatenate([ones, x_t_flat_g, y_t_flat_g, r], 1) # [n, nx*ny+3, h*w]
        return grid

    def _transform(T, fp, input_dim, out_size):
        num_batch = input_dim.shape[0]
        height = input_dim.shape[1]
        width = input_dim.shape[2]
        num_channels = input_dim.shape[3]

        # grid of (x_t, y_t, 1), eq (1) in ref [1]
        height_f = float(height)
        width_f = float(width)
        out_height = out_size[0]
        out_width = out_size[1]
        grid = _meshgrid(out_height, out_width, fp) # [2, h*w]

        # transform A x (1, x_t, y_t, r1, r2, ..., rn) -> (x_s, y_s)
        T_g = np.matmul(T, grid) # MARK
        x_s = T_g[:,0,:],
        y_s = T_g[:,1,:]
        x_s_flat = np.reshape(x_s, [-1])
        y_s_flat = np.reshape(y_s, [-1])

        input_transformed = _interpolate(
            input_dim, x_s_flat, y_s_flat, out_size)

        output = np.reshape(
            input_transformed, 
            np.stack([num_batch, out_height, out_width, num_channels]))
        return output

    def _solve_system(cp, nx, ny):
        gx = 2. / nx # grid x size
        gy = 2. / ny # grid y size
        cx = -1. + gx/2. # x coordinate
        cy = -1. + gy/2. # y coordinate

        p_ = np.empty([nx*ny, 3], dtype='float32')
        i = 0
        for _ in range(ny):
          for _ in range(nx):
            p_[i, :] = 1, cx, cy
            i += 1
            cx += gx
          cx = -1. + gx/2
          cy += gy

        p_1 = p_.reshape([nx*ny,1,3])
        p_2 = p_.reshape([1, nx*ny, 3])
        d = np.sqrt(np.sum((p_1-p_2)**2, 2)) # [nx*ny, nx*ny]
        r = d*d*np.log(d*d+1e-5)
        W = np.zeros([nx*ny+3, nx*ny+3], dtype='float32')
        W[:nx*ny, 3:] = r
        W[:nx*ny, :3] = p_
        W[nx*ny:, 3:] = p_.T

        num_batch = cp.shape[0]
        fp = np.array(p_[:,1:], dtype='float32') # [nx*ny, 2]
        fp = np.expand_dims(fp, 0) # [1, nx*ny, 2]
        fp = np.tile(fp, np.stack([num_batch, 1, 1])) # [n, nx*ny, 2]

        W_inv = np.linalg.inv(W)
        W_inv_t =np.array(W_inv, dtype='float32') # [nx*ny+3, nx*ny+3]
        W_inv_t = np.expand_dims(W_inv_t, 0)          # [1, nx*ny+3, nx*ny+3]
        W_inv_t = np.tile(W_inv_t, np.stack([num_batch, 1, 1]))

        cp_pad = np.pad(cp, [[0, 0], [0, 3], [0, 0]], "constant")
        T = np.matmul(W_inv_t, cp_pad)
        T = np.transpose(T, [0,2,1])

        return T, fp
        
    T, fp = _solve_system(cp, nx, ny)
    output = _transform(T, fp, U, out_size)
    return output
