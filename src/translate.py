import json
import os

KAMUS_PATH = "kamus/kamus_jawa.json"

# Load kamus sekali saja
with open(KAMUS_PATH, encoding="utf-8") as f:
    KAMUS = json.load(f)


def translate(word, mode="ngoko"):
    """
    mode: ngoko | krama | krama_inggil
    """
    word = word.upper()

    if word not in KAMUS:
        return None

    entry = KAMUS[word]

    if mode == "ngoko":
        jawa = entry.get("jawa_ngoko", "")
    elif mode == "krama":
        jawa = entry.get("jawa_krama", "")
    else:
        jawa = entry.get("jawa_krama_inggil", "")

    return {
        "indonesia": entry.get("indonesia", ""),
        "jawa": jawa
    }


# # TEST MANUAL
# if __name__ == "__main__":
#     for w in ["MAKAN", "MINUM", "KAMU"]:
#         print(w, "→", translate(w, mode="ngoko"))
