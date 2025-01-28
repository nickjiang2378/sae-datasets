import streamlit as st
import pandas as pd
import numpy as np
import pickle
import scipy
import asyncio
from goodfire import Client, Variant

# Initialize the client and variant
client = Client(api_key="sk-goodfire-MaIHyE13tEGWYNTgmSx04KG1VE873Hugse7TQVMtTTuINCj-iDWb-Q")
variant = Variant("meta-llama/Llama-3.3-70B-Instruct")

def best_nonzero_feature(activations):
    nonzero_indices = np.nonzero(activations)[0]
    lowest_nonzero_index = nonzero_indices[0] if nonzero_indices.size > 0 else None
    if lowest_nonzero_index is None:
        return float("inf"), -float("inf")
    return lowest_nonzero_index, -activations[lowest_nonzero_index]

async def run_analysis_async(query):
    # Load checkpoint data
    with open("ds_analysis_ckpt.pkl", "rb") as f:
        ckpt = pickle.load(f)

    rand_activations_sparse_loaded = scipy.sparse.load_npz("rand_activations_sparse.npz")

    # Verify the loaded matrix
    rand_activations = rand_activations_sparse_loaded.toarray()
    # print(rand_activations_dense)

    # rand_activations = ckpt["rand_activations"]
    rand_labels = ckpt["rand_labels"]
    rand_ds = ckpt["rand_ds"]

    # Search for top features based on the query
    top_features = client.features.search(query, model=variant, top_k=100)
    top_feature_indices = [feature.index_in_sae for feature in top_features]

    # Process activations
    top_feat_activations = []
    for activations in rand_activations:
        top_feat_activations.append(activations[top_feature_indices])

    # Create a DataFrame to store the data
    data = {
        "Data": [],
        "Category": [],
        "Feature": []
    }

    ds_indices = list(range(len(rand_ds)))
    ds_indices.sort(key=lambda i: best_nonzero_feature(top_feat_activations[i]))

    cnt = 0
    for i in ds_indices:
        ds_feature = top_features[int(best_nonzero_feature(top_feat_activations[i])[0])]
        data["Data"].append(rand_ds[i])
        data["Category"].append(rand_labels[i])
        data["Feature"].append(ds_feature.label)
        cnt += 1
        if cnt > 50:
            break

    # Create a DataFrame
    df = pd.DataFrame(data)
    return df

def run_analysis(query):
    # Create a new event loop and run the async function
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        df = loop.run_until_complete(run_analysis_async(query))
    finally:
        loop.close()
    return df

# Streamlit app
st.title("Dataset Analysis using SAEs")

query = st.text_input("Enter your query:", "presence of misspelled words")

if st.button("Run Analysis"):
    df = run_analysis(query)
    st.dataframe(df)