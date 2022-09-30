import requests
import uuid
import os
import shutil 
import glob 

import torch
import scann
import clip
import numpy as np
import pandas as pd
from PIL import Image


def download_image(url, outdir="matches"):
    try:
        res = requests.get(url)
        imfile = str(uuid.uuid1())
        with open(f"{outdir}/{imfile}.jpg", 'wb') as handler:
            handler.write(res.content)
    except Exception as e:
        print(url, "\n", e)


def build_searcher(embeds):
    n_samples = len(embeds)

    searcher = (
        scann.scann_ops_pybind.builder(
            embeds,  # the dataset that ScaNN will search over; a 2d array of 32-bit floats with one data point per row
            num_neighbors=500,  # the default # neighbors the index will return per query
            distance_measure="dot_product",  # one of "squared_l2" or "dot_product"
        )
        .tree(
            # The higher num_leaves, the higher-quality the partitioning will be (makes partitioning take longer)
            # num_leaves_to_search / num_leaves determines the proportion of the dataset that is pruned
            # Raising this proportion increases accuracy but leads to more points being scored and therefore less speed.
            num_leaves=250,
            num_leaves_to_search=250,
            training_sample_size=250000,  
        )
        .score_brute_force(quantize=False)
        .reorder(
            reordering_num_neighbors=n_samples
        )  # "reordering_num_neighbors should be greater than k" per docs
        .build()
    )

    return searcher


def index():
    image_embeds = np.load("laion_data/img_emb_0.npy")
    image_searcher = build_searcher(image_embeds)
    image_searcher.serialize("searcher")


def get_neighbors_and_sims(path_to_searcher, embedding, final_num_neighbors=100):
    searcher = scann.scann_ops_pybind.load_searcher(path_to_searcher)
    neighbors, similarity = searcher.search_batched(
        embedding, final_num_neighbors=final_num_neighbors
    )
    return neighbors, similarity


def get_urls(indices):
    df = pd.read_parquet("laion_data/metadata_0.parquet")

    urls = []
    for index in indices:
        urls.extend([df.iloc[i].url for i in index])

    return set(urls)

def get_embeddings(images):
    device = "cpu"
    model, preprocess = clip.load("ViT-B/32", device=device)

    im_features = []
    with torch.no_grad():
        for im in images:
            image = preprocess(Image.open(im)).unsqueeze(0).to(device)
            img_features = model.encode_image(image).numpy().squeeze()
            img_features = img_features / np.linalg.norm(img_features, keepdims=True)
            im_features.append(img_features)
    return im_features 

if __name__ == "__main__":

    if os.path.exists("matches"):
        shutil.rmtree("matches")
    os.mkdir("matches")

    if not os.path.exists("searcher/scann_config.pb"):
        index()

    images = glob.glob("assets/*")
    im_features = get_embeddings(images)

    im_neighbors, im_similarity = get_neighbors_and_sims(
        path_to_searcher="searchers/image_searcher", embedding=im_features
    )

    urls = get_urls(im_neighbors)
    print(f"Attempting retrieval of {len(urls)} images")
    _ = [download_image(url) for url in urls]
    n_retrieved = os.listdir("matches")
    print(f"{len(n_retrieved)}/{len(urls)} retrieved")
