"""
This is meant to be a comparison of my method and this 
repo: https://github.com/rom1504/clip-retrieval, which
has more bells and whistles but seems to lack what 
I'm looking for which is the ability to return any 
results past a certain similarity score. Also more often 
than not image search returns very few results. See below.
"""

from clip_retrieval.clip_client import ClipClient

client = ClipClient(url="https://knn5.laion.ai/knn-service", indice_name="laion5B", num_images = 1000)
results = client.query(image="assets/spaghetti_1.jpg")
print(*results, sep="\n")