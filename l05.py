import os
import time
from multiprocessing import Pool, shared_memory

import numpy as np
import gymnasium as gym

D_TYPE = np.float32


def worker(args):
    """Fill rows [start, start + n) of the shared x / y arrays."""
    shm_x_name, shm_y_name, total, start, n, seed = args
    shm_x = shared_memory.SharedMemory(name=shm_x_name)
    shm_y = shared_memory.SharedMemory(name=shm_y_name)
    x = np.ndarray((total, 5), dtype=D_TYPE, buffer=shm_x.buf)
    y = np.ndarray((total, 1), dtype=D_TYPE, buffer=shm_y.buf)

    rng = np.random.default_rng(seed)  # independent stream per worker
    env = gym.make("CartPole-v1")
    obs, info = env.reset(seed=seed)

    for i in range(start, start + n):
        action = int(rng.integers(0, 2))
        x[i, :4] = obs          # state BEFORE the action
        x[i, 4] = action
        obs, reward, terminated, truncated, info = env.step(action)
        y[i, 0] = reward
        if terminated or truncated:
            obs, info = env.reset()

    env.close()
    # drop this process's views before closing the mapping
    del x, y
    shm_x.close()
    shm_y.close()


def generate_data(n_steps=1000, n_workers=None, seed=0):
    n_workers = n_workers or os.cpu_count()
    shm_x = shared_memory.SharedMemory(create=True, size=n_steps * 5 * np.dtype(D_TYPE).itemsize)
    shm_y = shared_memory.SharedMemory(create=True, size=n_steps * 1 * np.dtype(D_TYPE).itemsize)

    try:
        # split rows into n_workers contiguous chunks
        bounds = np.linspace(0, n_steps, n_workers + 1, dtype=np.int64)
        jobs = [
            (shm_x.name, shm_y.name, n_steps, int(bounds[k]), int(bounds[k + 1] - bounds[k]), seed + k)
            for k in range(n_workers)
        ]
        with Pool(n_workers) as pool:
            pool.map(worker, jobs)

        # copy out of shared memory so it can be released
        x = np.ndarray((n_steps, 5), dtype=D_TYPE, buffer=shm_x.buf).copy()
        y = np.ndarray((n_steps, 1), dtype=D_TYPE, buffer=shm_y.buf).copy()
    finally:
        shm_x.close()
        shm_x.unlink()
        shm_y.close()
        shm_y.unlink()

    return x, y


def main():
    t0 = time.time()
    x, y = generate_data(2_000_000)  # change to 100_000_000 for the full run
    print("Generated data in %.1fs with %d workers" % (time.time() - t0, os.cpu_count()))
    print("X shape:", x.shape)
    print("Y shape:", y.shape)


if __name__ == "__main__":
    main()