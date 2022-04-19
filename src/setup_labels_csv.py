import os

import pandas as pd


def setup_agelabel_df_from_filepath(img_folder: str, output_csv_path: str) -> None:
    """
    Generates a csv of age-labels for images from the UTKFace
    dataset.

    Grabs all of the images from the UTKFace dataset held within
    the folder provided, extracts each images "age" from its
    file name, then saves the information to a csv
    at the filepath provided.

    Parameters
    ----------
    img_folder : str
        Path to the folder holding UKTFace images.
    output_csv_path : str
        Location to save .csv to.
    """
    image_df = pd.DataFrame([filename for filename in os.listdir(img_folder)]).rename(
        columns={0: "id"}
    )
    image_df = image_df.assign(age=image_df.id.str.split("_", expand=True)[0])
    image_df.set_index("id").to_csv(output_csv_path)
    return


if __name__ == "__main__":
    setup_agelabel_df_from_filepath(
        "data/UTKFace/crop_part1", "data/UTKFace/ages-cropped.csv"
    )
    setup_agelabel_df_from_filepath("assets/app_img_subset", "assets/app_ages.csv")
