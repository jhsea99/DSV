"""Prompts and model-free text helpers of DSV."""

from __future__ import annotations

import re

CAPTION_PROMPT = "Please describe this image in detail."
EXTRACT_PROMPT = ("Here is a sentence from an image description: \"{s}\"\nList the physical objects (nouns) the sentence "
                  "mentions, as a comma-separated list copied from the sentence. Reply with the list only.")
POPE_SUFFIX = "\nAnswer the question using a single word or phrase."

WORD_RE = re.compile(r"[A-Za-z]+")
SENT_END = (".", "!", "?")
DETERMINERS = {"a", "an", "the", "some", "several", "many", "few", "various", "multiple", "other", "another", "its",
               "their", "his", "her", "two", "three", "four", "five", "six", "numerous"}
# Word starts that are never looked ahead for the noun multiplication (speed only; the noun decision itself is nltk POS).
SKIP_LOOKAHEAD = set("""
the a an and or but of in on at to with is are was were be been being it its this that these those there their his her
he she they them as by for from which while who whose also can may might could would should will appears appear seems
seem some several various other another many few more most each both into onto near next behind over under above below
between among around such like than very well just only not no so too then here where when what how all any every own
additionally furthermore overall likely possibly perhaps
""".split())


def forms(w: str) -> set[str]:
    """Singular and plural surface forms of a word (no lemmatizer: the model's own word and its obvious variants)."""
    w = w.lower()
    out = {w}
    if w.endswith("ies"):
        out.add(w[:-3] + "y")
    if w.endswith("es"):
        out.add(w[:-2])
    if w.endswith("s") and not w.endswith("ss"):
        out.add(w[:-1])
    for b in list(out):
        out |= {b + "s", b + "es"}
        if b.endswith("y"):
            out.add(b[:-1] + "ies")
    return out


def head(phrase: str) -> str:
    ws = WORD_RE.findall(phrase.lower())
    return ws[-1] if ws else ""


def parse_phrases(text: str, max_items: int = 15) -> list[str]:
    """Phrases of the model's comma-separated object list, leading determiners removed."""
    items = []
    for w in re.split(r",|\band\b|;|\n", text):
        ws = [x for x in WORD_RE.findall(w.lower())]
        while ws and ws[0] in DETERMINERS:
            ws = ws[1:]
        p = " ".join(ws)
        if p and p not in items:
            items.append(p)
    return items[:max_items]


def question(phrase: str) -> str:
    """The POPE-style question DSV asks about a phrase."""
    h = head(phrase)
    if h.endswith("s") and not h.endswith(("ss", "us", "is")):
        return f"Are there {phrase} in the image?"
    return f"Is there {'an' if phrase[0] in 'aeiou' else 'a'} {phrase} in the image?"


def surface(phrase: str, words: set[str]) -> set[str]:
    """Sentence words that are a form of the phrase's head word or end with it ("cellphone" for "phone")."""
    fs = forms(head(phrase))
    return {w for w in words if w in fs or (len(head(phrase)) >= 4 and any(w.endswith(fm) for fm in fs))}


def sample_words(samples: list[str]) -> list[set[str]]:
    """Lower-case word sets of the sampled captions (what the sample consensus counts)."""
    return [set(w.lower() for w in WORD_RE.findall(x)) for x in samples]
