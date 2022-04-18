# For the sake of a cleaner repository,
# All code for the minimal dash app will be
# contained within this file instead
# of being modularized out.

import dash
from dash import dcc
from dash import html
from dash.dependencies import Input, Output
import dash_bootstrap_components as dbc
import pandas as pd

from src.current_model import model
from src.helper_func import preprocess_single_img


# Define APP and SERVER
app = dash.Dash(
    __name__,
    suppress_callback_exceptions=True,
    external_stylesheets=[dbc.themes.ZEPHYR],
    title="How Old Are They?",
    meta_tags=[{"name": "viewport", "content": "width=device-width, initial-scale=1"}],
)
server = app.server

# Age CSV Dataframe
ages_df = pd.read_csv("data/UTKFace/ages-cropped.csv")
model = model

# APP LAYOUT
app.layout = html.Div(
    [
        dbc.Container(
            children=[
                dbc.Col(
                    class_name="col-xxl-auto px-4 py-5",
                    children=[
                        dbc.Row(
                            class_name="flex-lg-row-reverse g-5 py-5",
                            align="center",
                            children=[
                                # Image
                                dbc.Col(
                                    class_name="text-center",
                                    width=10,
                                    sm=8,
                                    lg=6,
                                    children=[
                                        html.Img(
                                            id="img-display",
                                            className="d-block mx-lg-auto img-fluid",
                                            width=500,
                                            height=500,
                                            alt="image to be predicted",
                                            src=app.get_asset_url(
                                                "crop_part1/90_1_0_20170110182426286.jpg.chip.jpg"
                                            ),
                                        ),
                                        html.P(
                                            "Predicted age: 25 years old",
                                            id="prediction",
                                            className="font-weight-light font-italic px-2 text-muted",
                                        ),
                                    ],
                                ),
                                # Text and Upload
                                dbc.Col(
                                    lg=6,
                                    children=[
                                        html.H1(
                                            "How Old Are They?",
                                            className="fw-bold lh-1 mb-3",
                                        ),
                                        html.P(
                                            children=[
                                                "This is a Convolutional Neural Network that predicts how old someone is based on an image of their face.",
                                                html.Br(),
                                                html.Br(),
                                                "You can explore how the model performs on the types of ",
                                                html.A(
                                                    "images it was trained on",
                                                    id="dataset",
                                                    style={
                                                        "textDecoration": "underline",
                                                        "cursor": "pointer",
                                                    },
                                                ),
                                                ", or test out the model yourself by uploading an image of a face with the buttons below.",
                                                html.Br(),
                                                "For best results, try to match the cropping and 244x244 resolution of the images from the dataset.",
                                                dbc.Tooltip(
                                                    html.P(
                                                        [
                                                            "Our model was trained on the UTKFace Dataset.",
                                                            html.Br(),
                                                            "This dataset is collection of portraits described as 'a large-scale face dataset with [a] long age span (range from 0 to 116 years old)'.",
                                                            html.Br(),
                                                            html.Br(),
                                                            "As you can see from our images, this model was trained on pre-cropped and pre-centered images of faces. ",
                                                            "As a result, any inference on more noisy or uncropped pictures 'in the wild' is limited.",
                                                        ]
                                                    ),
                                                    target="dataset",
                                                ),
                                            ],
                                            className="lead",
                                        ),
                                        html.Div(
                                            className="d-grid gap-2 d-md-flex justify-content-md-start",
                                            children=[
                                                dcc.Upload(
                                                    children=[
                                                        dbc.Button(
                                                            "Upload Portrait",
                                                            class_name="btn btn-warning btn-lg px-4 me-md-2",
                                                        )
                                                    ],
                                                    id="image-upload",
                                                    max_size=9e7,
                                                    accept="image/*",
                                                ),
                                                dbc.Button(
                                                    "Explore Dataset",
                                                    class_name="btn btn-secondary btn-lg px-4",
                                                    id="explore-button",
                                                ),
                                            ],
                                        ),
                                    ],
                                ),
                            ],
                        )
                    ],
                )
            ],
        ),
        html.Footer(
            className="footer",
            children=[
                dbc.Container(
                    children=dbc.Row(
                        className="text-center text-light bg-warning fixed-bottom pt-2",
                        children=[
                            html.P(
                                [
                                    "Created By Gavin Grochowski  |  ",
                                    html.A(
                                        "Source Code",
                                        href="https://github.com/gavingro/portrait-age-regression",
                                        className="text-white",
                                        target="_blank",
                                    ),
                                    " | ",
                                    html.A(
                                        "Data Source",
                                        href="https://susanqq.github.io/UTKFace/",
                                        className="text-white",
                                        target="_blank",
                                    ),
                                ]
                            ),
                        ],
                    ),
                )
            ],
        ),
    ]
)

# DYNAMIC CALLBACKS

# Update image and prediction based
# on "random image" button or
# user upload
@app.callback(
    Output("img-display", "src"),
    Output("prediction", "children"),
    Input("explore-button", "n_clicks"),
    Input("image-upload", "contents"),
    Input("image-upload", "filename"),
)
def update_img_display_from_button(button_clicks, upload_contents, upload_filename):
    # Uploaded Image
    if dash.callback_context.triggered[0]["prop_id"] == "image-upload.contents":
        # Easy SRC for Display
        image_src = upload_contents

        # Get Age Prediction.
        model.eval()
        # predicted_age = model(preprocess_single_img(upload_filename))
        predicted_age = 25
        prediction = f"Predicted Age: {predicted_age:.2f} years old."
    # Button Click
    else:
        # Get Random Image / Age from labels dataframe
        picture = ages_df.sample(1, replace=True)
        age = picture["age"].values[0]
        image_src = app.get_asset_url("crop_part1/" + picture["id"].values[0])

        # Get Age Prediction
        model.eval()
        # predicted_age = model(preprocess_single_img(image_src))
        predicted_age = 25
        prediction = (
            f"Predicted Age: {predicted_age} Years old.\t True Age: {age} years old."
        )
    return image_src, prediction


if __name__ == "__main__":
    app.run_server(debug=True)
