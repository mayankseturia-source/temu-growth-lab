"""Lightweight review text analytics: cleaning, spam filter, lexicon sentiment, topic tagging.
Built to cope with noisy text (typos, caps, emojis, mixed languages, sarcasm)."""
from __future__ import annotations

import re

import numpy as np
import pandas as pd

POS = {"good": 1, "great": 1, "quick": 1, "fast": 1, "faster": 1, "exactly": 0.6, "sturdy": 1, "works": 0.7,
       "cheap": 0.3, "worth": 1, "easy": 1, "fun": 1, "refunded": 0.8, "love": 1.2, "excellent": 1.2,
       "surprised": 0.4, "bueno": 1, "ممتاز": 1, "pas mal": 0.6, "in stock": 0.8}
NEG = {"slow": -1, "delayed": -1.2, "waiting": -0.8, "stuck": -1, "broke": -1.3, "nothing like": -1.2,
       "wrong": -1, "stopped": -1, "not apply": -1, "not free": -1.2, "expired": -1, "went up": -0.8,
       "out of stock": -1.3, "cancelled": -1.2, "unavailable": -1, "back-ordered": -1, "annoying": -1.2,
       "spam": -1.2, "uninstalling": -1.5, "too many": -1, "never": -1, "pending": -0.8, "never came": -1.2,
       "schlecht": -1.2, "não chegou": -1, "بطيء": -1, "too slow": -1.2, "cheap material": -1.2,
       "stop sending": -1.2, "never replied": -1.2}
EMOJI = {"😍": 1, "👍": 0.8, "🔥": 0.8, "✨": 0.6, "😊": 0.8, "😡": -1.2, "👎": -1, "🙄": -0.9, "😤": -1, "💀": -0.6}
TOPIC_WORDS = {
    "delivery": ["arriv", "deliver", "tracking", "days", "shipping", "chegou", "شحن"],
    "quality": ["quality", "material", "broke", "picture", "photos", "size", "working", "sturdy"],
    "price_coupon": ["coupon", "price", "free gift", "bogo", "deal", "wheel", "cheap", "tariff", "referral"],
    "stock": ["stock", "unavailable", "cancelled", "back-ordered", "shipped"],
    "app_notifications": ["notification", "pop-up", "push", "app", "games", "uninstall"],
    "service": ["support", "refund", "return"],
}
SPAM_PAT = re.compile(r"(bit\.ly|click here|follow me|visit now|code \d{3,}|^(.)\2{3,}$|^\W*$)", re.I)


def clean(text: str) -> str:
    t = str(text).lower()
    t = re.sub(r"\s+", " ", t)
    return t.strip()


def is_spam(text: str) -> bool:
    return bool(SPAM_PAT.search(str(text).strip()))


def sentiment(text: str) -> float:
    """Score in [-1, 1]. Lexicon + emoji, with negation flip and sarcasm ('great, ... 🙄') detection."""
    raw = str(text)
    t = clean(raw)
    score = 0.0
    for w, v in {**POS, **NEG}.items():
        if w in t:
            neg_before = re.search(r"(not|never|no|n't)\s+(\w+\s+)?" + re.escape(w), t)
            score += -v if (neg_before and v > 0) else v
    for e, v in EMOJI.items():
        if e in raw:
            score += v
    if t.startswith("great,") and any(k in t for k in ("late", "delayed", "waiting", "days", "🙄")):
        score -= 2.0  # sarcasm
    if raw.isupper() and score < 0:
        score *= 1.3  # shouting amplifies negative tone
    if "!!" in raw and score < 0:
        score -= 0.3
    return float(np.tanh(score / 1.5))


def topic(text: str) -> str:
    t = clean(text)
    best, hits = "other", 0
    for k, words in TOPIC_WORDS.items():
        h = sum(w in t for w in words)
        if h > hits:
            best, hits = k, h
    return best


def analyse(df: pd.DataFrame) -> pd.DataFrame:
    d = df.copy()
    d["spam_flag"] = d["text"].map(is_spam)
    d["sentiment"] = d["text"].map(sentiment)
    d["label"] = np.where(d["spam_flag"], "Spam", np.where(d["sentiment"] > 0.15, "Positive",
                          np.where(d["sentiment"] < -0.15, "Negative", "Neutral")))
    d["topic_detected"] = np.where(d["spam_flag"], "spam", d["text"].map(topic))
    d["rating_text_mismatch"] = ((d["stars"] >= 4) & (d["label"] == "Negative")) | ((d["stars"] <= 2) & (d["label"] == "Positive"))
    return d
