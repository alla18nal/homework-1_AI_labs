import numpy as np

class SimulatedBench:
    def __init__(self, dmd_side=32, cam_side=64, noise=0.05, seed=0):
        rng = np.random.default_rng(seed)
        self.dmd_side, self.cam_side, self.noise = dmd_side, cam_side, noise
        n_in  = dmd_side * dmd_side      # number of DMD mirrors (inputs)
        n_out = cam_side * cam_side      # number of camera pixels (outputs)
        # The "frosted glass": a fixed random complex matrix.
        # Each entry = how much light, and with what phase, one mirror sends to one camera pixel.
        self.T = (rng.normal(size=(n_out, n_in)) + 1j * rng.normal(size=(n_out, n_in))) / np.sqrt(2 * n_in)
        self.rng = rng

    def shoot(self, dmd_pattern):
        """dmd_pattern: 2D array of 0/1, shape (dmd_side, dmd_side). Returns a noisy camera image."""
        E_in = dmd_pattern.reshape(-1).astype(float)        # mirrors ON = 1, OFF = 0
        E_out = self.T @ E_in                               # add up every mirror's light at each pixel
        I = np.abs(E_out) ** 2                              # the camera sees brightness: |T E_in|^2
        I = I * (1 + self.noise * self.rng.normal(size=I.shape))   # noise: a second shot isn't a copy
        return np.clip(I, 0, None).reshape(self.cam_side, self.cam_side)

bench = SimulatedBench()
pattern = np.random.default_rng(1).integers(0, 2, size=(32, 32))
img = bench.shoot(pattern)

def focus(bench, macro=4, focus_size=2, bg_size=4, n_regions=4, iterations=500, n_flip=1, seed=0):
    rng = np.random.default_rng(seed)
    n = bench.dmd_side // macro  #number of blocks
    blocks = rng.integers(0, 2, size=(n, n))

    def enhancement(img):
        center = bench.cam_side // 2
        h = focus_size // 2
        b = bg_size // 2
        focus_region = img[center - h: center + h, center - h : center + h] #focuses on center
        I_focus = focus_region.mean()
        bg_region_bottom = img[center+h+1-b : center+h+1+b, center - h : center + h]
        bg_region_top = img[center-h-1-b : center-h-1+b, center - h : center + h]
        bg_region_left = img[center - h : center + h, center-h-1-b : center-h-1+b]
        bg_region_right = img[center+h+1-b : center+h+1+b, center - h : center + h]
        regions = [bg_region_bottom, bg_region_left, bg_region_right, bg_region_top]
        bg_mean = 0
        for reg in range(max(n_regions, 4)):
            bg_mean += regions[reg].mean()


        return I_focus / bg_mean #I am not sure about this bc this way bg contains focus region too




    def measure(blocks):
        dmd = np.kron(blocks, np.ones((macro, macro)))
        return enhancement(bench.shoot(dmd))

    best = measure(blocks)
    history = [best]
    
    for it in range(iterations):
        id_1 = np.random.choice(n * n, size = n_flip)
        id_2 = np.random.choice(n * n, size = n_flip)
        new_blocks = blocks
        for block in n_flip:
            new_blocks[id_1][id_2] = 1 if  blocks[id_1][id_2] == 0 else 0

        if measure(blocks) > history[-1]:
            history.append(best)
            blocks = new_blocks

    return best, history, blocks