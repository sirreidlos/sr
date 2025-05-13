import os
import pandas as pd

# dataset_path = "./out-min-n-20-max-t-30-clean/"
dataset_path = "./out-min-n-30-max-t-30-augmented/"
df = pd.read_csv(os.path.join(dataset_path, "accent_data.csv"))
# max_duration = df["duration"].max()
accent_mapping = {k:v for (v, k) in enumerate(df["accent"].unique())}

class SharedConfig:
    max_duration = 30 # Maximum audio duration in seconds
    sample_rate = 16000  # Target sample rate
    max_length = 16000 * 30 # Maximum sequence length (sample_rate * max_duration)

shared_cfg = SharedConfig()

