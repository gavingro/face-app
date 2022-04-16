import os

import pandas as pd

# Setup Ages CSV
image_df = pd.DataFrame(
    [filename for filename in os.listdir("data/UTKFace/crop_part1")]
).rename(columns={0: "id"})
image_df = image_df.assign(age=image_df.id.str.split("_", expand=True)[0])
image_df.set_index("id").to_csv("data/UTKFace/ages-cropped.csv")
