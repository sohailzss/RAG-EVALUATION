"""
goldens/build_faithfulness.py -- builds goldens/faithfulness_dataset.json.

ideal_context must be the EXACT chunk strings the retriever stores, so instead of
copy-pasting we re-chunk the transcripts with the same splitter settings as
src/retriever.py and select chunks by an anchor phrase that appears in them.
If an anchor matches zero chunks the script fails loudly.

The last two cases are deliberately UNANSWERABLE from their context: the chunk
only *mentions* the topic. The faithful behaviour there is to abstain, so they
test that the generator doesn't fill the gap with outside knowledge.

    python goldens/build_faithfulness.py      (run from the project root)
"""

import glob
import json
import re

from langchain_text_splitters import RecursiveCharacterTextSplitter

DATA_DIR = "data"
OUT_PATH = "goldens/faithfulness_dataset.json"

# (id, query, [anchor phrase per context chunk], answerable)
CASES = [
    ("f001", "why can adding more parameters to a linear regression never make R squared worse?",
     ["will never perform worse than this equation"], True),
    ("f002", "how does logistic regression find the best fitting curve?",
     ["Instead, it uses something called maximum likelihood",
      "multiply all of those likelihoods together"], True),
    ("f003", "why does the squiggly line have high variance even though it fits the training set perfectly?",
     ["the difference in fits between datasets is called variance",
      "Because the squiggly line fits the training set really well"], True),
    ("f004", "how does four-fold cross validation use the data?",
     ["cross-validation would start by using the first three blocks"], True),
    ("f005", "how does ridge regression work when the predictor is a discrete variable like diet?",
     ["lambda times diet difference squared",
      "shrink diet distance down"], True),
    ("f006", "why isn't the y-intercept included in the ridge regression penalty?",
     ["that's why the Y intercept is not included"], True),
    ("f007", "how does lasso regression get rid of useless variables like astrological sign?",
     ["these parameters will go all the way to zero"], True),
    ("f008", "how do you calculate Gini impurity for a numeric feature like age?",
     ["sort the rows by age",
      "are tied for the lowest impurity"], True),
    ("f009", "how do you create a bootstrapped dataset for a random forest?",
     ["we're allowed to pick the same sample more than once"], True),
    ("f010", "how does a forest of stumps made with AdaBoost classify a new patient?",
     ["add up the amounts of say"], True),
    ("f011", "what is a pseudo residual in gradient boost?",
     ["which is called a pseudo residual"], True),
    ("f012", "how does lambda change the output value of a leaf in an XGBoost tree?",
     ["if lambda equals one, the output value equals -5.25"], True),
    ("f013", "how does the radial kernel classify a new observation?",
     ["weighted nearest neighbor model"], True),
    # --- unanswerable from the given context: generator should abstain ---
    ("f014", "how does the k-nearest neighbors algorithm choose the value of k?",
     ["K nearest neighbors"], False),
    ("f015", "how does gradient boost use log odds to classify whether someone loves Troll 2?",
     ["Part three in this series shows how gradient boost can be used for classification"], False),
]


def load_chunks():
    """Same loading + splitting as src/retriever.py, keeping the session id."""
    chunks = []
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
    for path in sorted(glob.glob(f"{DATA_DIR}/*.vtt")):
        with open(path, encoding="utf-8") as f:
            lines = [ln.strip() for ln in f
                     if ln.strip() and ln.strip() != "WEBVTT" and "-->" not in ln]
        session = str(int(re.search(r"Session[ _]*(\d+)", path).group(1)))
        for text in splitter.split_text(" ".join(lines)):
            chunks.append((session, text))
    return chunks


def find_chunk(chunks, anchor):
    hits = [(s, t) for s, t in chunks if anchor in t]
    if not hits:
        raise SystemExit(f"anchor not found in any chunk: {anchor!r}")
    return hits[0]     # overlap can put an anchor in 2 chunks; take the first


if __name__ == "__main__":
    chunks = load_chunks()
    rows = []
    for gid, query, anchors, answerable in CASES:
        picked = [find_chunk(chunks, a) for a in anchors]
        rows.append({
            "id": gid,
            "query": query,
            "ideal_context": [t for _, t in picked],
            "source_sessions": sorted({s for s, _ in picked}, key=int),
            "answerable": answerable,
        })
    with open(OUT_PATH, "w", encoding="utf-8") as f:
        json.dump(rows, f, indent=2, ensure_ascii=False)
    for r in rows:
        print(f"{r['id']}  ctx={len(r['ideal_context'])}  sessions={r['source_sessions']}  "
              f"answerable={r['answerable']}")
    print(f"wrote {len(rows)} -> {OUT_PATH}")
