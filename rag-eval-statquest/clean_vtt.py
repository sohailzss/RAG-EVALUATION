"""
clean_vtt.py -- turn raw YouTube .vtt captions into clean, timestamped .vtt files.

Raw files (data/raw/*.vtt) come in two flavours:
  - MANUAL captions: one phrase per cue, HTML entities like &gt;&gt;
  - AUTO captions:   per-word <00:00:01.760><c> tags, and every line repeated
                     2-3 times as the caption "rolls" up the screen

Output (data/StatQuest Session NN - <Title>.vtt): one cue per NEW line of speech,
with its original start/end timestamp, no inline tags, no duplicates, no [Music].
The format stays valid WEBVTT, so the retriever's loader works unchanged.

    python clean_vtt.py
"""

import glob
import html
import os
import re

RAW_DIR = "data/raw"
OUT_DIR = "data"

CUE_TIME = re.compile(r"^(\d{2}:\d{2}:\d{2}\.\d{3}) --> (\d{2}:\d{2}:\d{2}\.\d{3})")
INLINE_TAG = re.compile(r"<[^>]+>")                 # <00:00:01.760>, <c>, </c>
NOISE_ONLY = re.compile(r"^(\[[^\]]*\]|[♪\s]+)$")   # [Music], [Applause], ♪ ♪


def to_seconds(ts):
    h, m, s = ts.split(":")
    return int(h) * 3600 + int(m) * 60 + float(s)


def parse_cues(path):
    """Yield (start, end, [text lines]) for every cue in a raw VTT file."""
    cues, cur = [], None
    with open(path, encoding="utf-8") as f:
        for raw in f:
            line = raw.rstrip("\n")
            m = CUE_TIME.match(line)
            if m:
                cur = [m.group(1), m.group(2), []]
                cues.append(cur)
            elif cur is not None and line.strip():
                cur[2].append(line)
            elif line == "":          # only a truly empty line ends a cue;
                cur = None            # auto cues contain " " placeholder lines
    return cues


def clean_file(path):
    out_cues, last_line = [], None
    for start, end, lines in parse_cues(path):
        # auto captions insert ~10ms "transition" cues that only repeat text
        if to_seconds(end) - to_seconds(start) < 0.05:
            continue
        new_text = []
        for line in lines:
            text = html.unescape(INLINE_TAG.sub("", line)).strip()
            text = re.sub(r"\s+", " ", text)
            if not text or NOISE_ONLY.match(text):
                continue
            if text == last_line:            # rolling-caption duplicate
                continue
            new_text.append(text)
            last_line = text
        if new_text:
            out_cues.append((start, end, " ".join(new_text)))
    return out_cues


def clean_name(path):
    """'StatQuest Session 02 - StatQuest： Logistic Regression.en.vtt'
       -> 'StatQuest Session 02 - Logistic Regression.vtt'"""
    name = os.path.basename(path)
    name = re.sub(r"\.en(-\w+)?\.vtt$", "", name)
    session, title = re.match(r"(StatQuest Session \d+) - (.*)", name).groups()
    title = title.replace("：", ":").replace("StatQuest:", "")
    title = re.sub(r",?\s*Clearly Explained", "", title)
    title = re.sub(r"[!:\\/?*\"<>|]", "", title)
    title = re.sub(r"\s+", " ", title).strip(" -")
    return f"{session} - {title}.vtt"


if __name__ == "__main__":
    for path in sorted(glob.glob(f"{RAW_DIR}/*.vtt")):
        cues = clean_file(path)
        out_path = os.path.join(OUT_DIR, clean_name(path))
        with open(out_path, "w", encoding="utf-8") as f:
            f.write("WEBVTT\n\n")
            for start, end, text in cues:
                f.write(f"{start} --> {end}\n{text}\n\n")
        words = sum(len(t.split()) for _, _, t in cues)
        print(f"{os.path.basename(out_path):70s} {len(cues):4d} cues {words:6d} words")
