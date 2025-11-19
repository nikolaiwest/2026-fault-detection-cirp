# %%
import pickle
from pathlib import Path

import pyscrew


def load_data(force_reload=False):
    """Load and cache preprocessed screw driving data."""
    # Define cache path
    cache_file = Path("data/processed/pyscrew_s04.pkl")
    cache_file.parent.mkdir(parents=True, exist_ok=True)

    # Check if cache exists and should be used
    if cache_file.exists() and not force_reload:
        print(f"Loading cached data from {cache_file}")
        with open(cache_file, "rb") as f:
            data = pickle.load(f)
        print(f"Loaded {len(data['torque_values'])} samples from cache")
        return data

    # Load fresh data from pyscrew
    print("Loading data from pyscrew (this may take a few minutes)...")
    data = pyscrew.get_data(
        scenario="s04",
        screw_positions="left",
        cache_dir="data/pyscrew/",
        force_download=force_reload,
        handle_duplicates="first",
        handle_missings="mean",
        target_length=2000,
    )

    # Save to cache
    print(f"Saving to cache: {cache_file}")
    with open(cache_file, "wb") as f:
        pickle.dump(data, f)

    print(f"Loaded and cached {len(data['torque_values'])} samples")
    return data


# %%

data = load_data()

print(data.keys())


# %%
