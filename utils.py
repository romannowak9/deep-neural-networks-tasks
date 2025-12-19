import os, requests, zipfile
import numpy as np
import matplotlib.pyplot as plt

def read_dataset(filename):
    with open(filename, 'rb') as f:
        raw_data = np.fromfile(f, dtype=np.uint8)

    raw_data = np.uint32(raw_data)
    all_y = raw_data[1::5]
    all_x = raw_data[0::5]
    all_p = (raw_data[2::5] & 128) >> 7
    all_p = all_p
    all_ts = ((raw_data[2::5] & 127) << 16) | (raw_data[3::5] << 8) | raw_data[4::5]

    time_increment = 2 ** 13
    overflow_indices = np.where(all_y == 240)[0]
    for overflow_index in overflow_indices:
        all_ts[overflow_index:] += time_increment

    td_indices = np.where(all_y != 240)[0]
    x, y, ts, p = all_x[td_indices], all_y[td_indices], all_ts[td_indices], all_p[td_indices]

    mask = (x < 32) & (y < 32)
    x, y, ts, p  = x[mask], y[mask], ts[mask], p[mask]

    return ts, x, y, p

def get_nmnist():
    base_url = "https://data.mendeley.com/public-files/datasets/468j46mzdv/files/"
    files = [
        ("39c25547-014b-4137-a934-9d29fa53c7a0/file_downloaded", "train.zip", "Train"),
        ("05a4d654-7e03-4c15-bdfa-9bb2bcbea494/file_downloaded", "test.zip",  "Test")
    ]

    for path, zip_name, folder in files:
        if os.path.exists(zip_name) or os.path.exists(folder):
            print(f"Skipping {zip_name} (already exists)")
            continue
        url = base_url + path
        print(f"Downloading and extracting {zip_name} ...")
        with requests.get(url, stream=True) as r:
            r.raise_for_status()
            with open(zip_name, "wb") as f:
                f.write(r.content)
        with zipfile.ZipFile(zip_name, "r") as z:
            z.extractall(".")
    print("Done")

def plot_spike_activity(spikes, mems, max_neurons=64, max_time=None, layer_names=None):
    """
    Visualize spike raster (top) and membrane heatmap (bottom) for each layer.
    
    Args:
        spikes: list of tensors [ (T, B, N), ... ]  – spike history per layer
        mems:   list of tensors [ (T, B, N), ... ]  – membrane history per layer
        max_neurons: int, number of neurons to show per layer
        max_time: int or None, number of time steps to show (defaults to all)
        layer_names: list of strings, optional layer names
    """

    n_layers = len(spikes)
    layer_names = layer_names or [f"Layer {i+1}" for i in range(n_layers)]
    max_time = max_time or spikes[0].shape[0]

    fig, axes = plt.subplots(2, n_layers, figsize=(6 * n_layers, 8))
    if n_layers == 1:
        axes = axes.reshape(2, 1)

    for li in range(n_layers):
        spk = spikes[li][:max_time, 0, :max_neurons].detach().cpu().numpy()  # (T, N)
        mem = mems[li][:max_time, 0, :max_neurons].detach().cpu().numpy()    # (T, N)

        T, N = spk.shape

        # ---- Spike raster ----
        ax_raster = axes[0, li]
        ax_raster.set_title(f"{layer_names[li]} — Spike Raster")
        ax_raster.set_xlabel("Time step")
        ax_raster.set_ylabel("Neuron index")

        # get (t, n) coordinates for spikes
        t_idx, n_idx = np.where(spk > 0)
        ax_raster.scatter(t_idx, n_idx, c='red', s=4, marker='|', linewidths=1)
        ax_raster.set_xlim(0, T)
        ax_raster.set_ylim(0, N)
        ax_raster.invert_yaxis()  # neuron 0 on top

        # ---- Membrane heatmap ----
        ax_mem = axes[1, li]
        im = ax_mem.imshow(
            mem.T, aspect='auto', origin='lower',
            extent=[0, T, 0, N], cmap='viridis'
        )
        ax_mem.set_title(f"{layer_names[li]} — Membrane Potential")
        ax_mem.set_xlabel("Time step")
        ax_mem.set_ylabel("Neuron index")
        fig.colorbar(im, ax=ax_mem, fraction=0.046, pad=0.04)

    plt.tight_layout()
    return fig