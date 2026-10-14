import numpy as np

class SimulatedBench:
    def __init__(self, dmd_side=50, cam_side=64, noise=0.05, seed=0):
        rng = np.random.default_rng(seed)
        self.dmd_side, self.cam_side, self.noise = dmd_side, cam_side, noise
        n_in  = dmd_side * dmd_side      
        n_out = cam_side * cam_side      
        
        
        self.T = (rng.normal(size=(n_out, n_in)) + 1j * rng.normal(size=(n_out, n_in))) / np.sqrt(2 * n_in)
        self.rng = rng

    def shoot(self, dmd_pattern):
        """dmd_pattern: 2D array of 0/1, shape (dmd_side, dmd_side). Returns a noisy camera image."""
        E_in = dmd_pattern.reshape(-1).astype(float)        
        E_out = self.T @ E_in                              
        I = np.abs(E_out) ** 2                              
        I = I * (1 + self.noise * self.rng.normal(size=I.shape))   
        return np.clip(I, 0, None).reshape(self.cam_side, self.cam_side)

bench = SimulatedBench()
pattern = np.random.default_rng(1).integers(0, 2, size=(50, 50))
img = bench.shoot(pattern)

def focus(bench, macro=4, focus_size=2, bg_size=2, n_regions=4, iterations=500, n_flip=1, seed=0):
    rng = np.random.default_rng(seed)
    n = bench.dmd_side // macro  #number of blocks
    blocks = rng.integers(0, 2, size=(n, n))
    if (2 * bg_size + focus_size) > bench.cam_side:
        raise ValueError("Wrong input for focus/background size")
    def enhancement(img):
        center = bench.cam_side // 2
        h = focus_size // 2
        b = bg_size // 2
        focus_region = img[center - h: center + h, center - h : center + h] #focuses on center
        I_focus = focus_region.mean()
        bg_region_bottom = img[center+h : center+h+2*b, center - b : center + b]
        bg_region_top = img[center-h-2*b : center-h, center - b : center + b]
        bg_region_left = img[center - b : center + b, center-h-2*b : center-h]
        bg_region_right = img[center - b : center + b, center+h : center+h+2*b]
        regions = [bg_region_bottom, bg_region_left, bg_region_right, bg_region_top]
        bg_mean = 0
        for reg in range(min(n_regions, 4)):
            bg_mean += regions[reg].mean()
        bg_mean = bg_mean / (min(n_regions, 4))


        return I_focus / bg_mean 




    def measure(blocks):
        dmd = np.kron(blocks, np.ones((macro, macro)))
        return enhancement(bench.shoot(dmd))

    best = measure(blocks)
    history = [best]

    best_blocks = blocks
    
    for it in range(iterations):
        id_1 = rng.choice(n * n, size = n_flip, replace=False)
        ids = [True if i in id_1 else False for i in range(n * n)]
        blocks = blocks.reshape(-1)
        new_blocks = blocks.copy()
        new_blocks_1 = np.bitwise_invert(new_blocks, out=new_blocks, where=ids)
        new_blocks_1[ids] += 2
        new_blocks = new_blocks_1.reshape(n, n)
        res_1 = measure(new_blocks)
        res_2 = measure(new_blocks)
        res = (res_1 + res_2) / 2 #to eradicate the possibility of random noise being better
        if res > best:
            best = res
            best_blocks = new_blocks
        blocks = best_blocks
        history.append(res)
        

    return best, history, best_blocks

bench = SimulatedBench()
best, history, best_blocks = focus(bench, focus_size=15, bg_size=16, macro=2, iterations=5000, n_flip=10)
print("final enhancement:", best)

import matplotlib.pyplot as plt
img = bench.shoot(np.kron(best_blocks, np.ones((2, 2))))
fig, ax = plt.subplots(1, 2, figsize=(10, 4))
ax[0].plot(history); ax[0].set_xlabel("iteration"); ax[0].set_ylabel("enhancement")
ax[1].imshow(img); ax[1].set_title("best camera image")
plt.show()