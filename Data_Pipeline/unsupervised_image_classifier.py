import argparse
import os
import json
import hdbscan
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np  
import spacy
import pickle
from sentence_transformers import SentenceTransformer
from collections import defaultdict, Counter
import umap


nlp = spacy.load("en_core_web_sm")

def only_nouns(text):
    doc = nlp(text)
    return " ".join([t.lemma_ for t in doc if t.pos_ == "NOUN"])

def one_word_label(captions):
    vectorizer = TfidfVectorizer(stop_words='english')
    X = vectorizer.fit_transform(captions)
    scores = np.asarray(X.sum(axis=0)).ravel()
    vocab = np.array(vectorizer.get_feature_names_out())
    return vocab[scores.argmax()]

def label_clusters(captions, labels):
    """Assigns one-word label with TF-IDF"""
    clusters = defaultdict(list)
    for caption, lab in zip(captions, labels):
        if lab != -1:  # ignore noise if using HDBSCAN
            clusters[lab].append(caption)

    cluster_names = {}
    for lab, caps in clusters.items():
        cluster_names[lab] = one_word_label(caps)
    return cluster_names

def main():
    parser = argparse.ArgumentParser(description="Unsupervised Image Classifier")
    parser.add_argument('--input_dir', type=str, required=True, help='Directory containing input images')
    parser.add_argument('--output_dir', type=str, required=True, help='Directory to classification metadata will be saved as a json')
    
    args = parser.parse_args()
    
    input_dir = args.input_dir
    output_dir = args.output_dir
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    ###sanity check messages
    print(f"Loading images from {input_dir}...")
    print(f"Saving classified images to {output_dir}...")

    ## open input directory and store file path and captions in a list
    file_paths = []
    captions = []
    for filename in os.listdir(input_dir):
        if filename.lower().endswith(('.p')):
            image_path = os.path.join(input_dir, filename)
            file_paths.append(image_path)
            with open(image_path, 'rb') as f:
                data = pickle.load(f)
                captions.append(data['text'])

### CLUSTERING AND LABELING
    # prepare noun-only text for each caption
    noun_texts = [only_nouns(c) for c in captions]

    # compute embeddings: prefer semantic SentenceTransformer embeddings, fallback to TF-IDF on nouns
    try:
        st_model = SentenceTransformer("all-MiniLM-L6-v2")
        embeddings = st_model.encode(captions, show_progress_bar=False, convert_to_numpy=True)
        #umap dimensionality reduction
        umap_reducer = umap.UMAP(n_neighbors=15, n_components=10, metric='cosine', random_state=67)
        embeddings = umap_reducer.fit_transform(embeddings)
    except Exception:
        vectorizer = TfidfVectorizer(stop_words='english')
        embeddings = vectorizer.fit_transform(noun_texts).toarray()

        #umap dimensionality reduction
        umap_reducer = umap.UMAP(n_neighbors=15, n_components=10, metric='cosine', random_state=67)
        embeddings = umap_reducer.fit_transform(embeddings)

    # cluster with HDBSCAN
    clusterer = hdbscan.HDBSCAN(min_cluster_size=2, metric='euclidean')
    labels = clusterer.fit_predict(embeddings)


    # group captions by cluster (ignore noise -1)
    clusters = defaultdict(list)
    for file_path, label, cap in zip(file_paths, labels, captions):
        if label != -1:
            clusters[label].append(cap)

    # assign each cluster a label = most common lemmatized noun in the cluster
    cluster_names = {}
    for lab, caps in clusters.items():
        noun_counter = Counter()
        for cap in caps:
            nouns = [t for t in only_nouns(cap).split() if t]
            noun_counter.update(nouns)
        cluster_names[lab] = noun_counter.most_common(1)[0][0] if noun_counter else "unlabeled"

    # save classification metadata
    classification_metadata = {}
    for file_path, label, caption in zip(file_paths, labels, captions):
        lab_int = int(label)
        if lab_int == -1:
            cluster_name = "noise"
        else:
            # cluster_names stores plain strings, so return it directly (don't index [0])
            cluster_name = cluster_names.get(lab_int, "unlabeled")

        classification_metadata[file_path] = {
            'cluster_label': lab_int,
            'cluster_name': cluster_name,
            'caption': caption
        }
    with open(os.path.join(output_dir, 'classification_metadata.json'), 'w') as f:
        json.dump(classification_metadata, f, indent=4)
    print("Classification metadata saved.")


if __name__ == "__main__":
    main()