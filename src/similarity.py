import numpy as np
import os
from numpy.linalg import norm

TEMPLATE_DIR = "data/templates"

def load_templates():
    templates = {}
    for f in os.listdir(TEMPLATE_DIR):
        if f.endswith(".npy"):
            templates[f.replace(".npy","")] = np.load(os.path.join(TEMPLATE_DIR, f))
    return templates

def cosine_similarity(a, b):
    a = a.flatten()
    b = b.flatten()
    return np.dot(a, b) / (norm(a) * norm(b))

def predict(input_pose, templates):
    scores = {}

    for word, temp in templates.items():
        score = cosine_similarity(input_pose, temp)
        scores[word] = score

    # Urutkan dari skor tertinggi
    sorted_scores = sorted(scores.items(), key=lambda x: x[1], reverse=True)

    best_word, best_score = sorted_scores[0]

    # Jika hanya 1 template (aman)
    if len(sorted_scores) > 1:
        second_score = sorted_scores[1][1]
    else:
        second_score = 0.0

    margin = best_score - second_score

    return best_word, best_score, margin

