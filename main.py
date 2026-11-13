# ==============================================================================
# requirements.txt
# ------------------------------------------------------------------------------
# torch>=2.1.0
# transformers>=4.36.0
# scikit-learn>=1.3.0
# scipy>=1.11.0
# gradio>=4.15.0
# matplotlib>=3.8.0
# numpy>=1.24.0
# pandas>=2.1.0
#
# Run:
#   pip install torch transformers scikit-learn scipy gradio matplotlib numpy pandas
#   python app.py
# ==============================================================================

import os
import re
import gc
import json
import math
import random
import warnings
from collections import Counter
from typing import List, Dict, Tuple, Optional, Any

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModel
from sklearn.preprocessing import RobustScaler
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
)
import gradio as gr

warnings.filterwarnings("ignore")

# ==============================================================================
# 1. CONFIGURATION & HARDWARE SETUP (Apple Silicon M1 8GB Optimized)
# ==============================================================================
MODEL_NAME = "cointegrated/rubert-tiny2"
WEIGHTS_PATH = "stylometry_model_v3.pt"
TELEGRAM_JSON_PATH = "result.json"
CUSTOM_CSV_PATH = "data.csv"

MAX_SEQ_LENGTH = 256
ENCODE_BATCH_SIZE = 8
TRAIN_BATCH_SIZE = 16
RANDOM_SEED = 42


def set_seed(seed: int = RANDOM_SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def get_device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    elif torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def clear_memory() -> None:
    gc.collect()
    if torch.backends.mps.is_available():
        try:
            torch.mps.empty_cache()
        except Exception:
            pass
    elif torch.cuda.is_available():
        torch.cuda.empty_cache()


set_seed()
DEVICE = get_device()

# ==============================================================================
# 2. UNIVERSAL LEXICAL & SYNTACTIC DICTIONARIES (ENGLISH-FIRST + MULTILINGUAL)
# ==============================================================================
ARTICLES_AND_DETERMINERS = {
    "a", "an", "the", "this", "that", "these", "those", "some", "any", "each",
    "every", "either", "neither", "another", "such", "what", "which", "whose",
    "этот", "эта", "это", "эти", "тот", "та", "то", "те", "весь", "вся", "всё",
    "все", "каждый", "любой", "некоторый", "такой", "какой",
}

PREPOSITIONS = {
    "in", "on", "at", "to", "for", "with", "from", "by", "about", "as", "into",
    "like", "through", "after", "over", "between", "out", "against", "during",
    "without", "before", "under", "around", "among", "towards", "toward", "upon",
    "within", "along", "following", "across", "behind", "beyond", "plus", "except",
    "up", "down", "off", "above", "near", "inside", "outside", "onto", "until",
    "в", "во", "на", "с", "со", "к", "ко", "у", "за", "по", "от", "о", "об",
    "из", "из-за", "до", "для", "без", "под", "при", "после", "над", "через",
    "про", "между", "перед", "около", "вокруг", "среди", "ради",
}

FIRST_PERSON_PRONOUNS = {
    "i", "me", "my", "mine", "myself", "we", "us", "our", "ours", "ourselves",
    "i'm", "i've", "i'd", "i'll", "we're", "we've", "we'd", "we'll",
    "я", "мне", "меня", "мной", "мною", "мой", "моя", "моё", "мое", "мои",
    "мы", "нам", "нас", "нами", "наш", "наша", "наше", "наши",
}

SECOND_PERSON_PRONOUNS = {
    "you", "your", "yours", "yourself", "yourselves", "you're", "you've",
    "you'd", "you'll",
    "ты", "тебе", "тебя", "тобой", "твой", "твоя", "твои", "вы", "вам", "вас",
    "вами", "ваш", "ваша", "ваши",
}

THIRD_PERSON_PRONOUNS = {
    "he", "him", "his", "himself", "she", "her", "hers", "herself", "it", "its",
    "itself", "they", "them", "their", "theirs", "themselves", "it's", "he's",
    "she's", "they're", "they've", "they'd", "they'll",
    "он", "она", "оно", "они", "его", "ее", "её", "их", "ему", "ей", "им",
    "ним", "ней", "них", "него", "нее", "неё", "ими",
}

MODAL_VERBS = {
    "can", "cannot", "can't", "could", "couldn't", "may", "might", "must",
    "mustn't", "shall", "shan't", "should", "shouldn't", "will", "won't",
    "would", "wouldn't", "ought",
    "может", "могут", "могу", "можешь", "могли", "могла", "мог", "нужно",
    "надо", "должен", "должна", "должны", "следует", "приходится", "пришлось",
}

AUXILIARY_AND_COPULA_VERBS = {
    "be", "am", "is", "are", "was", "were", "been", "being", "have", "has",
    "had", "having", "do", "does", "did", "isn't", "aren't", "wasn't", "weren't",
    "haven't", "hasn't", "hadn't", "doesn't", "didn't", "don't", "that's",
    "there's", "here's", "what's", "who's",
    "быть", "был", "была", "было", "были", "будет", "будут", "есть", "является",
}

IRREGULAR_PAST_VERBS = {
    "was", "were", "had", "did", "said", "went", "got", "made", "knew", "thought",
    "took", "saw", "came", "wanted", "looked", "used", "found", "gave", "told",
    "worked", "called", "tried", "asked", "needed", "felt", "became", "left",
    "put", "meant", "kept", "let", "began", "seemed", "helped", "talked", "turned",
    "started", "might", "showed", "heard", "played", "ran", "moved", "liked",
    "lived", "believed", "held", "brought", "happened", "wrote", "provided",
    "sat", "stood", "lost", "paid", "met", "included", "continued", "set",
    "learned", "learnt", "changed", "led", "understood", "watched", "followed",
    "stopped", "created", "spoke", "read", "allowed", "added", "spent", "grew",
    "opened", "walked", "won", "offered", "remembered", "loved", "considered",
    "appeared", "bought", "waited", "served", "died", "sent", "expected", "built",
    "stayed", "fell", "cut", "reached", "killed", "remained", "suggested",
    "raised", "passed", "sold", "required", "reported", "decided", "pulled",
    "shot", "trapped", "crumpled", "blocked", "drove", "broke", "caught", "chose",
    "drew", "drank", "ate", "flew", "forgot", "forgave", "froze", "hid", "hit",
    "hurt", "lay", "rode", "rang", "rose", "sang", "sank", "shook", "shone",
    "shut", "slept", "slid", "smelt", "spat", "split", "spread", "ank", "stole",
    "stuck", "stung", "struck", "swore", "swept", "swam", "swung", "taught",
    "tore", "threw", "woke", "wore", "wound", "won", "wasn't", "weren't",
    "hadn't", "didn't", "couldn't", "wouldn't",
}

ADVERBS_AND_INTENSIFIERS = {
    "very", "really", "quite", "just", "much", "even", "still", "also", "too",
    "only", "well", "how", "then", "now", "always", "never", "often", "sometimes",
    "usually", "already", "almost", "enough", "perhaps", "probably", "actually",
    "literally", "simply", "especially", "particularly", "decidedly", "eventually",
    "finally", "honestly", "basically", "clearly", "obviously", "apparently",
    "completely", "totally", "absolutely", "hardly", "barely", "merely", "rather",
    "pretty", "super", "extremely", "highly", "mostly", "nearly", "forever",
    "очень", "просто", "вообще", "даже", "уже", "еще", "ещё", "только", "почти",
    "совсем", "абсолютно", "реально", "короче", "типа", "прям", "буквально",
    "именно", "действительно", "наконец", "особенно", "отчасти", "вполне",
}

CONJUNCTIONS_AND_CAUSALS = {
    "and", "but", "or", "so", "if", "because", "since", "although", "though",
    "while", "whereas", "unless", "until", "whether", "before", "after", "when",
    "whenever", "where", "wherever", "than", "that", "as", "however", "therefore",
    "thus", "hence", "moreover", "furthermore", "nevertheless", "nonetheless",
    "consequently", "otherwise", "instead", "meanwhile", "yet", "nor",
    "и", "а", "но", "да", "или", "либо", "что", "чтобы", "чтоб", "как", "когда",
    "если", "потому", "поэтому", "так", "хотя", "хоть", "пока", "зато", "однако",
    "причем", "причём", "поскольку", "ибо", "ведь", "же", "ли", "бы",
}

NEGATION_WORDS = {
    "not", "no", "never", "nothing", "nobody", "noone", "nowhere", "neither",
    "nor", "none", "without", "cannot", "can't", "won't", "don't", "doesn't",
    "didn't", "isn't", "aren't", "wasn't", "weren't", "haven't", "hasn't",
    "hadn't", "couldn't", "wouldn't", "shouldn't", "mustn't",
    "не", "нет", "ни", "никогда", "никуда", "никто", "ничего", "нигде", "никак",
    "никакой", "нельзя", "без",
}

FUNCTION_WORDS = (
    ARTICLES_AND_DETERMINERS
    | PREPOSITIONS
    | FIRST_PERSON_PRONOUNS
    | SECOND_PERSON_PRONOUNS
    | THIRD_PERSON_PRONOUNS
    | MODAL_VERBS
    | AUXILIARY_AND_COPULA_VERBS
    | ADVERBS_AND_INTENSIFIERS
    | CONJUNCTIONS_AND_CAUSALS
    | NEGATION_WORDS
)


# ==============================================================================
# 3. SYNTACTIC SKELETONIZER (TOPIC MASKING FOR TRANSFORMER)
# ==============================================================================
APOSTROPHE_NORM_RE = re.compile(r"[’‘`´]")
WORD_WITH_CONTRACTION_RE = re.compile(
    r"[A-Za-zА-Яа-яЁё0-9]+(?:'[A-Za-zА-Яа-яЁё]+)?(?:-[A-Za-zА-Яа-яЁё0-9]+)*",
    re.UNICODE,
)
TOKEN_OR_PUNCT_RE = re.compile(
    r"[A-Za-zА-Яа-яЁё0-9]+(?:'[A-Za-zА-Яа-яЁё]+)?(?:-[A-Za-zА-Яа-яЁё0-9]+)*|[^\sA-Za-zА-Яа-яЁё0-9]+",
    re.UNICODE,
)


def normalize_apostrophes(text: str) -> str:
    return APOSTROPHE_NORM_RE.sub("'", text)


def mask_word_to_morphology(word: str, is_mid_sentence: bool = False) -> str:
    w_low = word.lower()
    if w_low in FUNCTION_WORDS:
        return w_low

    if "-" in w_low and len(w_low) >= 4:
        return "<COMPOUND_HYPH>"

    if "'" in w_low:
        if w_low.endswith("n't"):
            return "<NEG_CONTR>"
        if w_low.endswith("'s"):
            return "<POSS_OR_IS>"
        return "<CONTR>"

    if w_low in IRREGULAR_PAST_VERBS or (len(w_low) >= 4 and w_low.endswith("ed")):
        return "<V_PAST>"
    if len(w_low) >= 5 and w_low.endswith("ing"):
        return "<V_ING>"
    if len(w_low) >= 5 and w_low.endswith("ly"):
        return "<ADV_LY>"
    if len(w_low) >= 6 and w_low.endswith(("tion", "sion", "ment", "ness", "ity", "ance", "ence")):
        return "<NOUN_ABSTR>"
    if len(w_low) >= 5 and w_low.endswith(("able", "ible", "ous", "ive", "ful", "less", "al", "ic")):
        return "<ADJ>"

    # Russian morphological fallbacks
    if len(w_low) >= 4 and w_low.endswith(("ла", "лась", "лся", "ли", "лись")):
        return "<V_PAST>"
    if len(w_low) >= 4 and w_low.endswith(("ть", "ться", "ти")):
        return "<V_INF>"

    if is_mid_sentence and word[0].isupper():
        return "<PROPER_CAP>"

    if len(w_low) <= 4:
        return "<CONTENT_S>"
    elif len(w_low) <= 7:
        return "<CONTENT_M>"
    return "<CONTENT_L>"


def build_syntactic_skeleton(text: str) -> str:
    """
    Preserves all function words, pronouns, auxiliaries, modals, negations,
    contractions, and punctuation while replacing topical nouns/verbs/adjectives
    with grammatical shape tokens.
    """
    if not text or not text.strip():
        return "."
    norm_text = normalize_apostrophes(text)
    tokens = TOKEN_OR_PUNCT_RE.findall(norm_text)
    out = []
    at_sentence_start = True
    for tok in tokens:
        if tok[0].isalpha():
            out.append(mask_word_to_morphology(tok, is_mid_sentence=not at_sentence_start))
            at_sentence_start = False
        elif tok[0].isdigit():
            out.append("<NUM>")
            at_sentence_start = False
        else:
            out.append(tok)
            if any(p in tok for p in (".", "!", "?", "\n")):
                at_sentence_start = True
    return " ".join(out)


# ==============================================================================
# 4. UNIVERSAL 36-FEATURE STYLOMETRIC EXTRACTOR (ZERO DEAD-FEATURE BIAS)
# ==============================================================================
class StylometricFeatureExtractor:
    """
    Extracts 36 universal, dense psycholinguistic, syntactic, and lexical features.
    Tracks which features are dense vs. sparse so that '0 vs 0' on rare punctuation
    never artificially inflates similarity scores.
    """

    FEATURE_NAMES = [
        # 1. Lexical Richness & Word Length Distribution (0..9)
        "avg_word_len",
        "std_word_len",
        "word_len_iqr",
        "herdan_c_ttr",              # Length-invariant log(V)/log(N)
        "mattr_window",              # Moving-Average TTR (25-word window)
        "yules_k_inv",               # Yule's K lexical richness index
        "hapax_ratio",               # Single-occurrence words ratio
        "short_word_ratio",          # Words <= 3 chars
        "long_word_ratio",           # Words >= 7 chars
        "content_repetition_rate",   # Repeated content words rate
        # 2. Syntactic Rhythm & Clause/Sentence Cadence (10..15)
        "avg_clause_len",            # Words per punctuation-bounded clause
        "clause_len_cv",             # Clause length coefficient of variation
        "clause_boundary_per_100",   # Total punctuation boundaries per 100 chars
        "avg_sentence_len",          # Words per sentence
        "sentence_len_cv",           # Sentence length coefficient of variation
        "para_density_per_100w",     # Paragraphs per 100 words
        # 3. Function Words & Grammatical Glue (16..22)
        "function_word_ratio",       # Overall function word ratio
        "article_det_ratio",         # Determiners & articles (the, a, an, this, that)
        "preposition_ratio",         # Prepositions (in, on, to, with, from, by, at)
        "conjunction_causal_ratio",  # Coordinating/subordinating/causal links
        "adverb_intensifier_ratio",  # Adverbs (-ly) & intensifiers (really, quite, very, just)
        "negation_ratio",            # Negations (not, n't, no, never, nothing, without)
        "contraction_rate_per_100w", # Contractions & apostrophes per 100 words
        # 4. Pronoun Perspective & Verb Tense / Modality (23..28)
        "first_person_ratio",        # 1st-person pronouns (I, me, my, we, us)
        "second_person_ratio",       # 2nd-person pronouns (you, your)
        "third_person_ratio",        # 3rd-person pronouns (it, its, he, she, they, them)
        "modal_verb_ratio",          # Modal verbs (would, could, should, can, will, might)
        "aux_copula_ratio",          # Primary auxiliaries/copulas (is, was, were, have, had, did)
        "past_tense_ratio",          # Past-tense verbs (-ed + irregular past verbs)
        "participle_gerund_ratio",   # Present participles / gerunds (-ing forms)
        # 5. Punctuation, Orthography & Register Habits (30..35)
        "comma_per_100w",            # Commas per 100 words
        "period_per_100w",           # Periods per 100 words
        "hyphen_dash_per_100w",      # Hyphens & dashes per 100 words (e.g., toss-up, left-handed)
        "parens_quotes_colon_per_100w",  # Parentheses, quotes, colons, semicolons per 100 words
        "expressive_punct_per_100w", # Exclamations, questions, ellipses, emoticons per 100 words
        "mid_sentence_cap_and_num_rate", # Mid-sentence capitalized words + numeric tokens rate
    ]

    # Features that can legitimately be 0 in normal paragraphs: if BOTH texts are 0,
    # we do NOT reward them with a 100% match weight.
    SPARSE_ZERO_INDICES = {21, 22, 23, 24, 26, 29, 32, 33, 34, 35}

    # Features susceptible to temporary emotional spikes (e.g. '!!!' replacing ',')
    VOLATILE_EMOTION_INDICES = {13, 14, 30, 31, 34}

    SENT_SPLIT_RE = re.compile(r"(?:[.!?…]+)+\s*|\n+", re.UNICODE)
    CLAUSE_SPLIT_RE = re.compile(r"[,;:—–]+|(?:[.!?…]+)+|\n+", re.UNICODE)
    ELLIPSIS_CLUSTER_RE = re.compile(r"\.{2,}|…+")
    EXCL_CLUSTER_RE = re.compile(r"!+")
    QUEST_CLUSTER_RE = re.compile(r"[?]+|[?!]{2,}")
    EMDASH_RE = re.compile(r"—|--|–")
    QUOTES_RE = re.compile(r"[\"“”«»]")
    EMOTICON_RE = re.compile(r"(?::\)|:\(|;\)|\(:\s*|\(-:|\bxd\b)", re.IGNORECASE)

    @staticmethod
    def _compute_mattr(words: List[str], window: int = 25) -> float:
        n = len(words)
        if n == 0:
            return 0.0
        if n <= window:
            v = len(set(words))
            return float(v / (n ** 0.88)) if n > 1 else 1.0
        ttrs = [
            len(set(words[i : i + window])) / float(window)
            for i in range(n - window + 1)
        ]
        return float(np.mean(ttrs))

    @staticmethod
    def _compute_yules_k_inv(words: List[str]) -> float:
        n = len(words)
        if n < 5:
            return 0.5
        counts = Counter(words)
        freq_spectrum = Counter(counts.values())
        s2 = sum((m ** 2) * v_m for m, v_m in freq_spectrum.items())
        k = 10000.0 * (s2 - n) / float(n ** 2)
        return float(1.0 / (1.0 + max(0.0, k) / 140.0))

    def extract_vector(self, text: str) -> np.ndarray:
        if not text or not isinstance(text, str):
            return np.zeros(len(self.FEATURE_NAMES), dtype=np.float32)

        lines = [line.strip() for line in text.strip().splitlines()]
        paragraphs = [p for p in lines if len(p) > 0]
        clean_text = "\n".join(paragraphs)
        char_count = len(clean_text)
        if char_count == 0:
            return np.zeros(len(self.FEATURE_NAMES), dtype=np.float32)

        norm_text = normalize_apostrophes(clean_text)
        raw_tokens = WORD_WITH_CONTRACTION_RE.findall(norm_text)
        if not raw_tokens:
            return np.zeros(len(self.FEATURE_NAMES), dtype=np.float32)

        lower_words = [w.lower() for w in raw_tokens]
        word_count = len(lower_words)
        scale_100w = 100.0 / float(word_count)
        scale_100c = 100.0 / float(char_count)

        # ------------------------------------------------------------------
        # 1. Lexical Richness & Word Length Distribution
        # ------------------------------------------------------------------
        # Strip apostrophes/hyphens when measuring pure character length of words
        clean_word_lengths = np.array(
            [len(w.replace("'", "").replace("-", "")) for w in lower_words],
            dtype=np.float32,
        )
        avg_word_len = float(np.mean(clean_word_lengths))
        std_word_len = float(np.std(clean_word_lengths))
        q75, q25 = np.percentile(clean_word_lengths, [75, 25])
        word_len_iqr = float(q75 - q25)

        word_counts = Counter(lower_words)
        unique_count = len(word_counts)
        if word_count > 1 and unique_count > 1:
            herdan_c = float(math.log(unique_count) / math.log(word_count))
        else:
            herdan_c = 1.0

        mattr_val = self._compute_mattr(lower_words, window=25)
        yules_k_inv = self._compute_yules_k_inv(lower_words)

        hapax_count = sum(1 for _, c in word_counts.items() if c == 1)
        hapax_ratio = float(hapax_count) / float(word_count ** 0.90)

        short_word_ratio = float(np.sum(clean_word_lengths <= 3)) / float(word_count)
        long_word_ratio = float(np.sum(clean_word_lengths >= 7)) / float(word_count)

        content_words = [w for w in lower_words if w not in FUNCTION_WORDS and len(w) >= 4]
        if content_words:
            cw_counts = Counter(content_words)
            repeated_cw = sum(c for w, c in cw_counts.items() if c >= 2)
            content_repetition_rate = float(repeated_cw) / float(len(content_words))
        else:
            content_repetition_rate = 0.0

        # ------------------------------------------------------------------
        # 2. Syntactic Rhythm & Clause/Sentence Cadence
        # ------------------------------------------------------------------
        clauses = [c.strip() for c in self.CLAUSE_SPLIT_RE.split(norm_text) if c.strip()]
        clause_lengths = [
            len(WORD_WITH_CONTRACTION_RE.findall(c)) for c in clauses
        ]
        clause_lengths = [cl for cl in clause_lengths if cl > 0]
        if clause_lengths:
            avg_clause_len = float(np.mean(clause_lengths))
            clause_len_cv = (
                float(np.std(clause_lengths) / (avg_clause_len + 1e-5))
                if len(clause_lengths) > 1
                else 0.0
            )
        else:
            avg_clause_len = float(word_count)
            clause_len_cv = 0.0

        raw_sentences = [s.strip() for s in self.SENT_SPLIT_RE.split(norm_text) if s.strip()]
        sent_lengths = [
            len(WORD_WITH_CONTRACTION_RE.findall(s)) for s in raw_sentences
        ]
        sent_lengths = [sl for sl in sent_lengths if sl > 0]
        if sent_lengths:
            avg_sentence_len = float(np.mean(sent_lengths))
            sentence_len_cv = (
                float(np.std(sent_lengths) / (avg_sentence_len + 1e-5))
                if len(sent_lengths) > 1
                else 0.0
            )
        else:
            avg_sentence_len = float(word_count)
            sentence_len_cv = 0.0

        para_density_per_100w = float(len(paragraphs)) * scale_100w

        # ------------------------------------------------------------------
        # 3. Function Words, Grammatical Glue, Negations & Contractions
        # ------------------------------------------------------------------
        func_hits = sum(1 for w in lower_words if w in FUNCTION_WORDS)
        function_word_ratio = float(func_hits) / float(word_count)

        art_det_hits = sum(1 for w in lower_words if w in ARTICLES_AND_DETERMINERS)
        article_det_ratio = float(art_det_hits) / float(word_count)

        prep_hits = sum(1 for w in lower_words if w in PREPOSITIONS)
        preposition_ratio = float(prep_hits) / float(word_count)

        conj_hits = sum(1 for w in lower_words if w in CONJUNCTIONS_AND_CAUSALS)
        conjunction_causal_ratio = float(conj_hits) / float(word_count)

        adv_hits = sum(
            1 for w in lower_words
            if w in ADVERBS_AND_INTENSIFIERS
            or (len(w) >= 5 and w.endswith("ly") and w not in {"family", "early", "only", "belly", "bully", "holly"})
        )
        adverb_intensifier_ratio = float(adv_hits) / float(word_count)

        neg_hits = sum(
            1 for w in lower_words
            if w in NEGATION_WORDS or w.endswith("n't")
        )
        negation_ratio = float(neg_hits) / float(word_count)

        contraction_hits = sum(1 for w in lower_words if "'" in w)
        contraction_rate_per_100w = float(contraction_hits) * scale_100w

        # ------------------------------------------------------------------
        # 4. Pronoun Perspective & Verb Tense / Modality
        # ------------------------------------------------------------------
        fp_hits = sum(1 for w in lower_words if w in FIRST_PERSON_PRONOUNS)
        first_person_ratio = float(fp_hits) / float(word_count)

        sp_hits = sum(1 for w in lower_words if w in SECOND_PERSON_PRONOUNS)
        second_person_ratio = float(sp_hits) / float(word_count)

        tp_hits = sum(1 for w in lower_words if w in THIRD_PERSON_PRONOUNS)
        third_person_ratio = float(tp_hits) / float(word_count)

        modal_hits = sum(1 for w in lower_words if w in MODAL_VERBS)
        modal_verb_ratio = float(modal_hits) / float(word_count)

        aux_hits = sum(1 for w in lower_words if w in AUXILIARY_AND_COPULA_VERBS)
        aux_copula_ratio = float(aux_hits) / float(word_count)

        past_hits = sum(
            1 for w in lower_words
            if w in IRREGULAR_PAST_VERBS
            or (len(w) >= 4 and w.endswith("ed") and w not in {"red", "bed", "fed", "led", "shed", "sled", "hundred"})
            or (len(w) >= 4 and w.endswith(("ла", "лась", "лся", "лись")))
        )
        past_tense_ratio = float(past_hits) / float(word_count)

        ing_hits = sum(
            1 for w in lower_words
            if (len(w) >= 5 and w.endswith("ing") and w not in {"thing", "bring", "spring", "string", "swing", "king", "ring", "wing"})
            or (len(w) >= 5 and w.endswith(("ущий", "ющий", "ащий", "ящий", "вший")))
        )
        participle_gerund_ratio = float(ing_hits) / float(word_count)

        # ------------------------------------------------------------------
        # 5. Punctuation, Orthography & Register Habits
        # ------------------------------------------------------------------
        ellipsis_clusters = len(self.ELLIPSIS_CLUSTER_RE.findall(clean_text))
        text_no_ell = self.ELLIPSIS_CLUSTER_RE.sub(" ", clean_text)

        quest_clusters = len(self.QUEST_CLUSTER_RE.findall(text_no_ell))
        text_no_q = self.QUEST_CLUSTER_RE.sub(" ", text_no_ell)

        excl_clusters = len(self.EXCL_CLUSTER_RE.findall(text_no_q))
        emoticon_hits = len(self.EMOTICON_RE.findall(clean_text))

        period_count = text_no_ell.count(".")
        comma_count = clean_text.count(",")

        total_boundaries = (
            period_count + comma_count + excl_clusters + quest_clusters + ellipsis_clusters
        )
        clause_boundary_per_100 = float(total_boundaries) * scale_100c

        comma_per_100w = float(comma_count) * scale_100w
        period_per_100w = float(period_count) * scale_100w

        emdash_count = len(self.EMDASH_RE.findall(clean_text))
        text_no_em = self.EMDASH_RE.sub(" ", clean_text)
        hyphen_count = text_no_em.count("-")
        hyphen_dash_per_100w = float(emdash_count + hyphen_count) * scale_100w

        parens_quotes_colon = (
            clean_text.count("(")
            + clean_text.count(")")
            + clean_text.count(":")
            + clean_text.count(";")
            + len(self.QUOTES_RE.findall(clean_text))
        )
        parens_quotes_colon_per_100w = float(parens_quotes_colon) * scale_100w

        expressive_punct_per_100w = (
            float(excl_clusters + quest_clusters + ellipsis_clusters + emoticon_hits) * scale_100w
        )

        # Mid-sentence capitalized words (excluding single-letter "I") + numeric tokens
        all_seq_tokens = TOKEN_OR_PUNCT_RE.findall(norm_text)
        mid_sent_caps = 0
        num_tokens = 0
        at_sent_start = True
        for tok in all_seq_tokens:
            if tok[0].isalpha():
                if not at_sent_start and tok[0].isupper() and tok != "I":
                    mid_sent_caps += 1
                at_sent_start = False
            elif tok[0].isdigit():
                num_tokens += 1
                at_sent_start = False
            else:
                if any(p in tok for p in (".", "!", "?", "\n", ":")):
                    at_sent_start = True

        mid_sentence_cap_and_num_rate = float(mid_sent_caps + num_tokens) / float(word_count)

        feats = np.array(
            [
                avg_word_len,
                std_word_len,
                word_len_iqr,
                herdan_c,
                mattr_val,
                yules_k_inv,
                hapax_ratio,
                short_word_ratio,
                long_word_ratio,
                content_repetition_rate,
                avg_clause_len,
                clause_len_cv,
                clause_boundary_per_100,
                avg_sentence_len,
                sentence_len_cv,
                para_density_per_100w,
                function_word_ratio,
                article_det_ratio,
                preposition_ratio,
                conjunction_causal_ratio,
                adverb_intensifier_ratio,
                negation_ratio,
                contraction_rate_per_100w,
                first_person_ratio,
                second_person_ratio,
                third_person_ratio,
                modal_verb_ratio,
                aux_copula_ratio,
                past_tense_ratio,
                participle_gerund_ratio,
                comma_per_100w,
                period_per_100w,
                hyphen_dash_per_100w,
                parens_quotes_colon_per_100w,
                expressive_punct_per_100w,
                mid_sentence_cap_and_num_rate,
            ],
            dtype=np.float32,
        )
        return np.nan_to_num(feats, nan=0.0, posinf=0.0, neginf=0.0)

    def extract_batch(self, texts: List[str]) -> np.ndarray:
        return np.vstack([self.extract_vector(t) for t in texts])

    def get_radar_metrics(self, text: str) -> Dict[str, float]:
        """
        6 balanced, non-sparse axes for the Radar Chart so differences between
        narrative, expository, formal, and conversational authors are visually clear.
        """
        v = self.extract_vector(text)
        idx = {name: i for i, name in enumerate(self.FEATURE_NAMES)}

        clause_cadence = float(np.clip(v[idx["avg_clause_len"]] / 14.0, 0.0, 1.0))
        lexical_diversity = float(
            np.clip(0.5 * v[idx["herdan_c_ttr"]] + 0.5 * v[idx["mattr_window"]], 0.0, 1.0)
        )
        subjective_past = float(
            np.clip(
                (v[idx["first_person_ratio"]] * 5.5)
                + (v[idx["past_tense_ratio"]] * 3.8)
                + (v[idx["contraction_rate_per_100w"]] * 0.08),
                0.0,
                1.0,
            )
        )
        modal_analytical = float(
            np.clip(
                (v[idx["modal_verb_ratio"]] * 8.0)
                + (v[idx["article_det_ratio"]] * 2.5)
                + (v[idx["preposition_ratio"]] * 2.0),
                0.0,
                1.0,
            )
        )
        punct_complexity = float(
            np.clip(
                (v[idx["comma_per_100w"]] * 0.06)
                + (v[idx["hyphen_dash_per_100w"]] * 0.12)
                + (v[idx["parens_quotes_colon_per_100w"]] * 0.10),
                0.0,
                1.0,
            )
        )
        word_complexity = float(
            np.clip((v[idx["avg_word_len"]] - 3.2) / 2.5, 0.0, 1.0)
        )

        return {
            "Clause Length": clause_cadence,
            "Lexical Richness": lexical_diversity,
            "1st-Person & Past Narrative": subjective_past,
            "Modal & Analytical Syntax": modal_analytical,
            "Compound & Punctuation": punct_complexity,
            "Word Complexity": word_complexity,
        }


# ==============================================================================
# 5. STRUCTURAL FUNCTION-WORD & MORPHOLOGICAL N-GRAM KERNEL
# ==============================================================================
class StructuralNgramKernel:
    """
    Computes TF-IDF Cosine Similarity over function-word bigrams, morphological
    suffixes, contractions, and punctuation transitions (completely topic-free).
    """

    def __init__(self):
        self.vectorizer = TfidfVectorizer(
            analyzer="word",
            ngram_range=(1, 3),
            min_df=1,
            max_features=1500,
            sublinear_tf=True,
            token_pattern=r"[^\s]+",
        )
        self.fitted = False

    @staticmethod
    def _to_structural_tokens(text: str) -> str:
        norm_text = normalize_apostrophes(text)
        tokens = TOKEN_OR_PUNCT_RE.findall(norm_text)
        stream = []
        at_start = True
        for t in tokens:
            t_low = t.lower()
            if t[0].isalpha():
                if t_low in FUNCTION_WORDS:
                    stream.append(t_low)
                else:
                    stream.append(mask_word_to_morphology(t, is_mid_sentence=not at_start))
                at_start = False
            elif t[0].isdigit():
                stream.append("<NUM>")
                at_start = False
            else:
                stream.append(t[0])
                if any(p in t for p in (".", "!", "?", "\n")):
                    at_start = True
        return " ".join(stream)

    def fit(self, texts: List[str]) -> None:
        streams = [self._to_structural_tokens(t) for t in texts]
        self.vectorizer.fit(streams)
        self.fitted = True

    def compute_pair_similarity(self, text_a: str, text_b: str) -> float:
        s_a = self._to_structural_tokens(text_a)
        s_b = self._to_structural_tokens(text_b)
        if not self.fitted:
            vec = TfidfVectorizer(
                analyzer="word", ngram_range=(1, 3), sublinear_tf=True, token_pattern=r"[^\s]+"
            )
            mat = vec.fit_transform([s_a, s_b]).toarray()
        else:
            mat = self.vectorizer.transform([s_a, s_b]).toarray()
        norm_a = np.linalg.norm(mat[0]) + 1e-9
        norm_b = np.linalg.norm(mat[1]) + 1e-9
        return float(np.dot(mat[0], mat[1]) / (norm_a * norm_b))


# ==============================================================================
# 6. DATA INGESTION & MULTI-AUTHOR CALIBRATION CORPUS
# ==============================================================================
def extract_telegram_text(text_field: Any) -> str:
    if isinstance(text_field, str):
        return text_field
    if isinstance(text_field, list):
        parts = []
        for item in text_field:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict) and "text" in item:
                parts.append(str(item["text"]))
        return "".join(parts)
    return ""


def generate_synthetic_dataset() -> Tuple[pd.DataFrame, str]:
    """
    Generates a diverse 8-author calibration corpus (160 samples across 8 topics)
    spanning distinct syntactic, narrative, punctuation, and lexical personas.
    """
    topics = [
        ("arctic wildlife", "Marine Predator", "pack ice", "thermal insulation", "hunting territory"),
        ("urban transit", "Subway Line", "signal failure", "platform crowd", "evening commute"),
        ("distributed systems", "Cache Node", "network latency", "memory buffer", "failover cluster"),
        ("culinary fermentation", "Sourdough Starter", "ambient temperature", "crumb texture", "cast-iron oven"),
        ("cinema history", "Soundstage Lighting", "camera movement", "practical effects", "theatrical cut"),
        ("aviation safety", "Flight Deck", "wind shear", "runway approach", "hydraulic pressure"),
        ("archaeology", "Bronze Artifact", "sediment layer", "trade route", "radiocarbon dating"),
        ("personal finance", "Index Fund", "interest rate", "liquidity reserve", "market volatility"),
    ]

    personas: Dict[str, List[str]] = {
        "Author_01 (Expository Lore & Compound-Heavy)": [],
        "Author_02 (1st-Person Eyewitness & Contractions)": [],
        "Author_03 (Formal Academic & Semicolon-Heavy)": [],
        "Author_04 (High-Arousal Expressive & Informal)": [],
        "Author_05 (Terse Lowercase Minimalist)": [],
        "Author_06 (2nd-Person Rhetorical Debater)": [],
        "Author_07 (Quantitative Telemetry Analyst)": [],
        "Author_08 (Conversational Gerund Storyteller)": [],
    }

    for i in range(20):
        t = topics[i % len(topics)]
        w1, w2, w3, w4, w5 = t

        # Author 1: Impersonal, modal 'would/can', hyphenated words, mid-sentence caps, parentheticals, colons
        p1 = (
            f"In practice, a well-adapted {w2} can hold its own during {w1}, and will often dominate "
            f"the {w3} if it encounters it in deep-water conditions. On open ground it would be a toss-up, "
            f"with the edge decidedly towards the {w2} (a seasoned {w2} is much, much faster than average). "
            f"Fun fact: a long-standing benchmark for a field-tested operator would be to navigate {w4} "
            f"without triggering {w5}. If you were unlucky enough to misjudge the left-hand margin, you, "
            f"as veterans like to say \"would have learned a hard lesson\". (:"
        )
        personas["Author_01 (Expository Lore & Compound-Heavy)"].append(p1)

        # Author 2: 1st-person 'I', past tense (-ed/irregulars), contractions (it's, they're, couldn't), numbers
        p2 = (
            f"I saw that {w2} breaking down near the facility where I worked during the {w1} incident "
            f"after it ruined it's {w3} (a complete disaster). Everybody thinks they're an expert on {w4}. "
            f"Eventually they isolated the whole unit a couple blocks from where I worked and they ran "
            f"diagnostics hundreds and hundreds of times until it finally dropped to it's baseline, exhausted. "
            f"I couldn't see the final stage, it was sealed off, but I saw it on the report. They just kept "
            f"testing for something like 30- 40 minutes until {w5} was confirmed."
        )
        personas["Author_02 (1st-Person Eyewitness & Contractions)"].append(p2)

        # Author 3: Formal academic, long words, semicolons, em-dashes, zero contractions
        p3 = (
            f"Consequently, when evaluating the overarching framework of {w1}, one must inevitably account "
            f"for {w3}; furthermore, empirical observations regarding {w4} — particularly under constrained "
            f"environmental parameters — demonstrate that {w5} remains fundamentally intertwined with the "
            f"operational stability of each {w2}."
        )
        personas["Author_03 (Formal Academic & Semicolon-Heavy)"].append(p3)

        # Author 4: Emotional / expressive (alternating between exclamation-heavy and calm posts by same author)
        if i % 2 == 0:
            p4 = (
                f"It happened again that I am stuck dealing with {w1} by myself. I feel overwhelmed and "
                f"it is so hard for me to fix {w3}. Partly because of my previous experience ({w4} was a mess).\n\n"
                f"With {w5} I also have huge problems.\n\n"
                f"Now they refuse to help me with {w2}! They tell me to handle it myself! And I cannot fix it "
                f"alone!!! Nobody listens to me anywhere!!!\n\n"
                f"Am I really supposed to tolerate this {w4} situation until everything collapses ?!?!?"
            )
        else:
            p4 = (
                f"My supervisor locked the settings on {w2}, so that I could not adjust {w3} at night, "
                f"because according to him, I am disrupting {w1}.\n\n"
                f"Every evening, I have to ask him for access, so that he calibrates {w4} for half an hour, "
                f"at least before closing. If I do not ask, he will never configure {w5} himself.\n\n"
                f"Is this a normal situation?"
            )
        personas["Author_04 (High-Arousal Expressive & Informal)"].append(p4)

        # Author 5: Terse lowercase minimalist
        p5 = (
            f"checked {w1} yesterday. {w3} looks unstable. replaced the {w2} module. {w4} still slow. "
            f"skip {w5} for now. nothing else to report."
        )
        personas["Author_05 (Terse Lowercase Minimalist)"].append(p5)

        # Author 6: 2nd-person rhetorical debater
        p6 = (
            f"Have you ever actually considered why {w1} always fails whenever you ignore {w3}? If you look "
            f"closely at {w4}, you will immediately realize that your assumptions about {w2} are completely "
            f"backwards — because without {w5}, your entire strategy falls apart."
        )
        personas["Author_06 (2nd-Person Rhetorical Debater)"].append(p6)

        # Author 7: Quantitative telemetry analyst
        p7 = (
            f"AUDIT LOG [ID-{100 + i}]: (1) Inspected {w1} subsystem at 08:30; (2) {w2} metric {w3} deviated "
            f"by 14.2% vs baseline; (3) Both {w4} and {w5} require 2x recalibration within 24h (limit: 512MB)."
        )
        personas["Author_07 (Quantitative Telemetry Analyst)"].append(p7)

        # Author 8: Conversational storyteller with gerunds (-ing)
        p8 = (
            f"Walking into the {w1} session, we ended up spending hours tweaking the {w3} while trying "
            f"to keep the {w2} from overheating, and honestly, watching {w4} come together after dealing "
            f"with {w5} all week was genuinely satisfying."
        )
        personas["Author_08 (Conversational Gerund Storyteller)"].append(p8)

    rows = []
    for author, texts in personas.items():
        for txt in texts:
            rows.append({"author": author, "text": txt})

    df = pd.DataFrame(rows)
    return df, f"Built-in Calibration Corpus ({df['author'].nunique()} authors, {len(df)} texts)"


def load_corpus() -> Tuple[pd.DataFrame, str]:
    if os.path.exists(TELEGRAM_JSON_PATH):
        try:
            with open(TELEGRAM_JSON_PATH, "r", encoding="utf-8") as f:
                data = json.load(f)
            messages = data.get("messages", [])
            rows = []
            for msg in messages:
                if msg.get("type") != "message":
                    continue
                author = msg.get("from") or msg.get("from_id")
                if not author:
                    continue
                text = extract_telegram_text(msg.get("text", "")).strip()
                if len(text) >= 40:
                    rows.append({"author": str(author), "text": text})

            df = pd.DataFrame(rows)
            if not df.empty:
                counts = df["author"].value_counts()
                valid_authors = counts[counts >= 15].index
                df = df[df["author"].isin(valid_authors)].reset_index(drop=True)
                if df["author"].nunique() >= 2:
                    df = (
                        df.groupby("author", group_keys=False)
                        .apply(lambda x: x.sample(min(len(x), 150), random_state=RANDOM_SEED))
                        .reset_index(drop=True)
                    )
                    summary = (
                        f"Telegram JSON (`{TELEGRAM_JSON_PATH}`): "
                        f"{len(df)} messages across {df['author'].nunique()} authors"
                    )
                    return df, summary
        except Exception as e:
            print(f"[Warning] Failed to parse {TELEGRAM_JSON_PATH}: {e}")

    if os.path.exists(CUSTOM_CSV_PATH):
        try:
            df = pd.read_csv(CUSTOM_CSV_PATH)
            if {"author", "text"}.issubset(set(df.columns)):
                df = df.dropna(subset=["author", "text"]).copy()
                df["author"] = df["author"].astype(str)
                df["text"] = df["text"].astype(str).str.strip()
                df = df[df["text"].str.len() >= 20].reset_index(drop=True)
                if df["author"].nunique() >= 2 and len(df) >= 10:
                    summary = (
                        f"Custom CSV (`{CUSTOM_CSV_PATH}`): "
                        f"{len(df)} texts across {df['author'].nunique()} authors"
                    )
                    return df, summary
        except Exception as e:
            print(f"[Warning] Failed to parse {CUSTOM_CSV_PATH}: {e}")

    return generate_synthetic_dataset()


def build_balanced_pairs(
    df: pd.DataFrame, max_pairs_per_author: int = 80
) -> List[Tuple[int, int, int]]:
    rng = random.Random(RANDOM_SEED)
    author_to_indices: Dict[str, List[int]] = {}
    for idx, row in df.iterrows():
        author_to_indices.setdefault(row["author"], []).append(int(idx))

    authors = list(author_to_indices.keys())
    positive_pairs: List[Tuple[int, int, int]] = []
    negative_pairs: List[Tuple[int, int, int]] = []

    for author, indices in author_to_indices.items():
        if len(indices) < 2:
            continue
        possible_pos = []
        for i in range(len(indices)):
            for j in range(i + 1, len(indices)):
                possible_pos.append((indices[i], indices[j], 1))
        rng.shuffle(possible_pos)
        selected_pos = possible_pos[:max_pairs_per_author]
        positive_pairs.extend(selected_pos)

        other_authors = [a for a in authors if a != author]
        if not other_authors:
            continue
        for _ in range(len(selected_pos)):
            idx_a = rng.choice(indices)
            other_a = rng.choice(other_authors)
            idx_b = rng.choice(author_to_indices[other_a])
            negative_pairs.append((idx_a, idx_b, 0))

    all_pairs = positive_pairs + negative_pairs
    rng.shuffle(all_pairs)
    return all_pairs


# ==============================================================================
# 7. HYBRID SIAMESE FUSION ARCHITECTURE
# ==============================================================================
class SiameseFusionHead(nn.Module):
    def __init__(self, neural_dim: int, style_dim: int, hidden_dim: int = 256, out_dim: int = 128):
        super().__init__()
        self.style_encoder = nn.Sequential(
            nn.Linear(style_dim, 96),
            nn.LayerNorm(96),
            nn.GELU(),
            nn.Linear(96, 96),
            nn.LayerNorm(96),
        )
        self.neural_proj = nn.Sequential(
            nn.Linear(neural_dim, 96),
            nn.LayerNorm(96),
            nn.GELU(),
        )
        self.fusion_mlp = nn.Sequential(
            nn.Linear(192, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(0.12),
            nn.Linear(hidden_dim, out_dim),
            nn.LayerNorm(out_dim),
        )

    def forward_once(self, neural_emb: torch.Tensor, style_vec: torch.Tensor) -> torch.Tensor:
        s_emb = self.style_encoder(style_vec) * 1.65
        n_emb = self.neural_proj(neural_emb) * 0.75
        combined = torch.cat([n_emb, s_emb], dim=-1)
        out = self.fusion_mlp(combined)
        return F.normalize(out, p=2, dim=-1)

    def forward(
        self,
        neural_a: torch.Tensor,
        style_a: torch.Tensor,
        neural_b: torch.Tensor,
        style_b: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        emb_a = self.forward_once(neural_a, style_a)
        emb_b = self.forward_once(neural_b, style_b)
        cos_sim = F.cosine_similarity(emb_a, emb_b, dim=-1)
        return emb_a, emb_b, cos_sim


class PairFeatureDataset(Dataset):
    def __init__(
        self,
        pairs: List[Tuple[int, int, int]],
        neural_matrix: np.ndarray,
        style_matrix: np.ndarray,
    ):
        self.pairs = pairs
        self.neural = torch.tensor(neural_matrix, dtype=torch.float32)
        self.style = torch.tensor(style_matrix, dtype=torch.float32)

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, idx: int):
        i, j, label = self.pairs[idx]
        target = 1.0 if label == 1 else -1.0
        return (
            self.neural[i],
            self.style[i],
            self.neural[j],
            self.style[j],
            torch.tensor(target, dtype=torch.float32),
            torch.tensor(label, dtype=torch.long),
        )


# ==============================================================================
# 8. ENGINE: BURROWS' COSINE DELTA + 4-KERNEL FORENSIC ENSEMBLE
# ==============================================================================
class StylometryEngine:
    def __init__(self):
        self.device = DEVICE
        self.extractor = StylometricFeatureExtractor()
        self.scaler = RobustScaler()
        self.scaler_fitted = False
        self.ngram_kernel = StructuralNgramKernel()

        print(f"[Init] Loading transformer '{MODEL_NAME}' on device '{self.device}'...")
        self.tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
        self.backbone = AutoModel.from_pretrained(MODEL_NAME)
        self.backbone.to(self.device)
        self.backbone.eval()
        for param in self.backbone.parameters():
            param.requires_grad = False

        self.neural_dim = int(self.backbone.config.hidden_size)
        self.style_dim = len(StylometricFeatureExtractor.FEATURE_NAMES)

        self.fusion_head = SiameseFusionHead(
            neural_dim=self.neural_dim,
            style_dim=self.style_dim,
            hidden_dim=256,
            out_dim=128,
        ).to(self.device)

        self.sim_threshold: float = 0.48
        self.sim_temperature: float = 9.5

        self.df, self.corpus_summary = load_corpus()
        self.author_centroids: Dict[str, np.ndarray] = {}
        self.author_style_profiles: Dict[str, np.ndarray] = {}
        self.author_texts_concat: Dict[str, str] = {}
        self.author_sample_counts: Dict[str, int] = {}

        self.corpus_neural: Optional[np.ndarray] = None
        self.corpus_style_raw: Optional[np.ndarray] = None

        if not self._try_load_weights():
            self._quick_warmup_calibration()
        else:
            self._refresh_author_database()

    @torch.no_grad()
    def _encode_raw_strings(self, texts: List[str]) -> np.ndarray:
        self.backbone.eval()
        all_embeddings = []
        for start in range(0, len(texts), ENCODE_BATCH_SIZE):
            batch_texts = [t if t.strip() else "." for t in texts[start : start + ENCODE_BATCH_SIZE]]
            encoded = self.tokenizer(
                batch_texts,
                padding=True,
                truncation=True,
                max_length=MAX_SEQ_LENGTH,
                return_tensors="pt",
            ).to(self.device)

            outputs = self.backbone(**encoded)
            token_embeddings = outputs.last_hidden_state
            attention_mask = encoded["attention_mask"].unsqueeze(-1).expand(token_embeddings.size()).float()
            sum_embeddings = torch.sum(token_embeddings * attention_mask, dim=1)
            sum_mask = torch.clamp(attention_mask.sum(dim=1), min=1e-9)
            mean_pooled = sum_embeddings / sum_mask
            mean_pooled = F.normalize(mean_pooled, p=2, dim=-1)
            all_embeddings.append(mean_pooled.cpu().numpy())

        clear_memory()
        return np.vstack(all_embeddings).astype(np.float32)

    @torch.no_grad()
    def encode_texts_deconfounded(self, texts: List[str]) -> np.ndarray:
        skeletons = [build_syntactic_skeleton(t) for t in texts]
        emb_skel = self._encode_raw_strings(skeletons)
        emb_raw = self._encode_raw_strings(texts)
        blended = 0.75 * emb_skel + 0.25 * emb_raw
        norms = np.linalg.norm(blended, axis=1, keepdims=True) + 1e-9
        return (blended / norms).astype(np.float32)

    def _ensure_corpus_encoded(self) -> None:
        if self.corpus_neural is None or len(self.corpus_neural) != len(self.df):
            texts = self.df["text"].tolist()
            self.corpus_neural = self.encode_texts_deconfounded(texts)
            self.corpus_style_raw = self.extractor.extract_batch(texts)
            self.ngram_kernel.fit(texts)

        if not self.scaler_fitted:
            self.scaler.fit(self.corpus_style_raw)
            self.scaler_fitted = True

    def _scale_style(self, style_raw: np.ndarray) -> np.ndarray:
        scaled = self.scaler.transform(style_raw).astype(np.float32)
        return np.clip(scaled, -4.0, 4.0)

    @torch.no_grad()
    def get_hybrid_embeddings(self, texts: List[str]) -> np.ndarray:
        self._ensure_corpus_encoded()
        self.fusion_head.eval()

        neural_np = self.encode_texts_deconfounded(texts)
        style_raw = self.extractor.extract_batch(texts)
        style_scaled = self._scale_style(style_raw)

        n_tensor = torch.tensor(neural_np, dtype=torch.float32, device=self.device)
        s_tensor = torch.tensor(style_scaled, dtype=torch.float32, device=self.device)
        fused = self.fusion_head.forward_once(n_tensor, s_tensor)
        return fused.cpu().numpy()

    def compute_stylometric_similarity(
        self, raw_a: np.ndarray, raw_b: np.ndarray
    ) -> Tuple[float, float, float, List[Optional[float]]]:
        """
        Combines:
        1. Active-Only Calibrated Feature Kernel (excludes '0 vs 0' sparse matches!)
        2. Burrows' Cosine Delta (directional z-score alignment across active features)
        3. Arousal Shift Compensator (down-weights exclamation/comma swaps under high emotional delta)
        """
        self._ensure_corpus_encoded()
        iqr = np.maximum(self.scaler.scale_, 1e-4)
        z_a = np.clip((raw_a - self.scaler.center_) / iqr, -4.0, 4.0)
        z_b = np.clip((raw_b - self.scaler.center_) / iqr, -4.0, 4.0)

        norm_diffs = np.abs(raw_a - raw_b) / iqr
        # Steep calibrated decay: 1.0 SD difference -> 31.7% match (never inflates random pairs)
        per_feature_sim = np.exp(-1.15 * np.power(norm_diffs, 1.35))

        expressive_idx = StylometricFeatureExtractor.FEATURE_NAMES.index("expressive_punct_per_100w")
        arousal_shift = float(abs(raw_a[expressive_idx] - raw_b[expressive_idx]) / max(iqr[expressive_idx], 1.0))

        weights = np.ones(self.style_dim, dtype=np.float32)
        feature_match_list: List[Optional[float]] = []

        for idx in range(self.style_dim):
            # Eliminate the '0 vs 0 = 100%' sparsity trap on rare markers
            if (
                idx in StylometricFeatureExtractor.SPARSE_ZERO_INDICES
                and abs(raw_a[idx]) < 1e-6
                and abs(raw_b[idx]) < 1e-6
            ):
                weights[idx] = 0.0
                feature_match_list.append(None)
            else:
                feature_match_list.append(float(per_feature_sim[idx] * 100.0))

        # If one text has a strong emotional burst (!!! / ?!?) vs calm punctuation,
        # down-weight volatile sentence/comma/exclamation counts and boost clause rhythm
        if arousal_shift >= 1.25:
            for v_idx in StylometricFeatureExtractor.VOLATILE_EMOTION_INDICES:
                weights[v_idx] *= 0.15
            clause_idx = StylometricFeatureExtractor.FEATURE_NAMES.index("avg_clause_len")
            bound_idx = StylometricFeatureExtractor.FEATURE_NAMES.index("clause_boundary_per_100")
            weights[clause_idx] = 2.5
            weights[bound_idx] = 2.2

        active_mask = weights > 0
        if np.sum(weights) > 0:
            active_kernel_sim = float(np.sum(per_feature_sim * weights) / np.sum(weights))
        else:
            active_kernel_sim = 0.0

        # Burrows' Cosine Delta on active z-scored features
        z_a_act = z_a * np.sqrt(weights)
        z_b_act = z_b * np.sqrt(weights)
        denom = (np.linalg.norm(z_a_act) * np.linalg.norm(z_b_act)) + 1e-9
        cosine_delta = float(np.dot(z_a_act, z_b_act) / denom)

        return active_kernel_sim, cosine_delta, arousal_shift, feature_match_list

    def compute_composite_forensic_score(
        self, text_a: str, text_b: str
    ) -> Dict[str, Any]:
        self._ensure_corpus_encoded()
        raw_a = self.extractor.extract_vector(text_a)
        raw_b = self.extractor.extract_vector(text_b)

        s_kernel, cos_delta, arousal_shift, feat_matches = self.compute_stylometric_similarity(
            raw_a, raw_b
        )
        # Map Cosine Delta [-1, 1] -> [0, 1] where 0.0 (orthogonal/random authors) -> 0.30
        s_cos_delta = float(np.clip((cos_delta + 0.35) / 1.15, 0.0, 1.0))

        embs = self.get_hybrid_embeddings([text_a, text_b])
        cos_siamese = float(np.dot(embs[0], embs[1]))
        s_siamese = float(np.clip((cos_siamese + 0.25) / 1.15, 0.0, 1.0))

        s_ngram = self.ngram_kernel.compute_pair_similarity(text_a, text_b)

        composite_sim = (
            0.35 * s_kernel
            + 0.30 * s_cos_delta
            + 0.20 * s_siamese
            + 0.15 * s_ngram
        )

        logit = (composite_sim - self.sim_threshold) * self.sim_temperature
        prob = 100.0 / (1.0 + math.exp(-max(-20.0, min(20.0, logit))))

        return {
            "probability": float(prob),
            "composite_sim": float(composite_sim),
            "s_kernel": float(s_kernel),
            "cos_delta": float(cos_delta),
            "s_siamese": float(cos_siamese),
            "s_ngram": float(s_ngram),
            "arousal_shift": float(arousal_shift),
            "raw_a": raw_a,
            "raw_b": raw_b,
            "feat_matches": feat_matches,
        }

    def _quick_warmup_calibration(self) -> None:
        print("[Init] Running startup calibration of Siamese Fusion Head...")
        self._ensure_corpus_encoded()
        pairs = build_balanced_pairs(self.df, max_pairs_per_author=40)
        style_scaled = self._scale_style(self.corpus_style_raw)
        ds = PairFeatureDataset(pairs, self.corpus_neural, style_scaled)
        loader = DataLoader(ds, batch_size=TRAIN_BATCH_SIZE, shuffle=True)

        criterion = nn.CosineEmbeddingLoss(margin=0.20)
        optimizer = torch.optim.AdamW(self.fusion_head.parameters(), lr=1.8e-3, weight_decay=1e-4)

        self.fusion_head.train()
        for _ in range(3):
            for n_a, s_a, n_b, s_b, target, _ in loader:
                n_a, s_a = n_a.to(self.device), s_a.to(self.device)
                n_b, s_b = n_b.to(self.device), s_b.to(self.device)
                target = target.to(self.device)
                optimizer.zero_grad()
                emb_a, emb_b, _ = self.fusion_head(n_a, s_a, n_b, s_b)
                loss = criterion(emb_a, emb_b, target)
                loss.backward()
                optimizer.step()

        self.fusion_head.eval()
        self.sim_threshold = 0.48
        self.sim_temperature = 9.5
        self._refresh_author_database()
        clear_memory()

    def _refresh_author_database(self) -> None:
        self._ensure_corpus_encoded()
        self.fusion_head.eval()
        style_scaled = self._scale_style(self.corpus_style_raw)

        with torch.no_grad():
            n_tensor = torch.tensor(self.corpus_neural, dtype=torch.float32, device=self.device)
            s_tensor = torch.tensor(style_scaled, dtype=torch.float32, device=self.device)
            all_fused = self.fusion_head.forward_once(n_tensor, s_tensor).cpu().numpy()

        self.author_centroids.clear()
        self.author_style_profiles.clear()
        self.author_texts_concat.clear()
        self.author_sample_counts.clear()

        for author, group in self.df.groupby("author"):
            idx = group.index.to_numpy()
            author_embs = all_fused[idx]
            centroid = np.mean(author_embs, axis=0)
            norm = np.linalg.norm(centroid) + 1e-9
            self.author_centroids[str(author)] = (centroid / norm).astype(np.float32)
            self.author_style_profiles[str(author)] = np.median(self.corpus_style_raw[idx], axis=0)
            self.author_texts_concat[str(author)] = "\n".join(group["text"].tolist()[:15])
            self.author_sample_counts[str(author)] = len(idx)

    def save_weights(self) -> None:
        payload = {
            "fusion_state_dict": self.fusion_head.state_dict(),
            "scaler_center": self.scaler.center_.tolist() if self.scaler_fitted else None,
            "scaler_scale": self.scaler.scale_.tolist() if self.scaler_fitted else None,
            "sim_threshold": self.sim_threshold,
            "sim_temperature": self.sim_temperature,
            "model_name": MODEL_NAME,
            "style_dim": self.style_dim,
        }
        torch.save(payload, WEIGHTS_PATH)

    def _try_load_weights(self) -> bool:
        if not os.path.exists(WEIGHTS_PATH):
            return False
        try:
            checkpoint = torch.load(WEIGHTS_PATH, map_location=self.device, weights_only=False)
            if (
                checkpoint.get("model_name") != MODEL_NAME
                or checkpoint.get("style_dim") != self.style_dim
            ):
                return False
            self.fusion_head.load_state_dict(checkpoint["fusion_state_dict"])
            if checkpoint.get("scaler_center") is not None and checkpoint.get("scaler_scale") is not None:
                self.scaler.center_ = np.array(checkpoint["scaler_center"], dtype=np.float64)
                self.scaler.scale_ = np.array(checkpoint["scaler_scale"], dtype=np.float64)
                self.scaler.n_features_in_ = len(self.scaler.center_)
                self.scaler_fitted = True
            self.sim_threshold = float(checkpoint.get("sim_threshold", 0.48))
            self.sim_temperature = float(checkpoint.get("sim_temperature", 9.5))
            self.fusion_head.eval()
            print(f"[Init] Loaded pre-trained weights from '{WEIGHTS_PATH}'.")
            return True
        except Exception as e:
            print(f"[Warning] Could not load '{WEIGHTS_PATH}': {e}")
            return False

    def train_and_evaluate(
        self, epochs: int = 5, lr: float = 1.5e-3, progress=gr.Progress()
    ) -> Tuple[str, plt.Figure]:
        set_seed()
        progress(0.05, desc="Extracting 36 universal stylometric features & syntactic skeletons...")
        self._ensure_corpus_encoded()

        pairs = build_balanced_pairs(self.df, max_pairs_per_author=80)
        labels_only = [p[2] for p in pairs]
        train_pairs, val_pairs = train_test_split(
            pairs, test_size=0.25, random_state=RANDOM_SEED, stratify=labels_only
        )

        train_indices = sorted({p[0] for p in train_pairs} | {p[1] for p in train_pairs})
        self.scaler.fit(self.corpus_style_raw[train_indices])
        self.scaler_fitted = True

        style_scaled = self._scale_style(self.corpus_style_raw)

        train_ds = PairFeatureDataset(train_pairs, self.corpus_neural, style_scaled)
        train_loader = DataLoader(train_ds, batch_size=TRAIN_BATCH_SIZE, shuffle=True)

        self.fusion_head = SiameseFusionHead(
            neural_dim=self.neural_dim,
            style_dim=self.style_dim,
            hidden_dim=256,
            out_dim=128,
        ).to(self.device)

        criterion = nn.CosineEmbeddingLoss(margin=0.20)
        optimizer = torch.optim.AdamW(self.fusion_head.parameters(), lr=lr, weight_decay=1e-4)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

        log_lines = [
            "### Hybrid Forensic Ensemble Summary",
            f"- **Compute Device:** `{self.device}` (Batch Size: `{TRAIN_BATCH_SIZE}`)",
            f"- **Neural Backbone:** `{MODEL_NAME}` with Syntactic Skeleton Topic-Masking",
            f"- **Stylometric Space:** `{self.style_dim}` dense universal features + Burrows' Cosine Delta",
            f"- **Active Dataset:** {self.corpus_summary}",
            f"- **Generated Pairs:** `{len(train_pairs)}` train | `{len(val_pairs)}` validation\n",
            "### Epoch Progression",
        ]

        for epoch in range(1, int(epochs) + 1):
            self.fusion_head.train()
            running_loss = 0.0
            for n_a, s_a, n_b, s_b, target, _ in train_loader:
                n_a, s_a = n_a.to(self.device), s_a.to(self.device)
                n_b, s_b = n_b.to(self.device), s_b.to(self.device)
                target = target.to(self.device)

                optimizer.zero_grad()
                emb_a, emb_b, _ = self.fusion_head(n_a, s_a, n_b, s_b)
                loss = criterion(emb_a, emb_b, target)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.fusion_head.parameters(), 1.0)
                optimizer.step()
                running_loss += loss.item() * len(target)

            scheduler.step()
            avg_loss = running_loss / len(train_ds)
            progress(0.15 + 0.65 * (epoch / epochs), desc=f"Epoch {epoch}/{epochs} — Loss: {avg_loss:.4f}")
            log_lines.append(f"- **Epoch {epoch}/{epochs}:** Train CosineEmbeddingLoss = `{avg_loss:.4f}`")

        self.fusion_head.eval()
        texts_list = self.df["text"].tolist()
        val_scores, val_labels = [], []

        for i, j, lbl in val_pairs:
            res = self.compute_composite_forensic_score(texts_list[i], texts_list[j])
            val_scores.append(res["composite_sim"])
            val_labels.append(lbl)

        val_scores_np = np.array(val_scores, dtype=np.float32)
        val_labels_np = np.array(val_labels, dtype=np.int32)

        best_th, best_f1 = 0.48, -1.0
        for th in np.linspace(0.25, 0.75, 101):
            preds = (val_scores_np >= th).astype(np.int32)
            score = f1_score(val_labels_np, preds, zero_division=0)
            if score > best_f1:
                best_f1 = score
                best_th = float(th)

        self.sim_threshold = best_th
        pos_sims = val_scores_np[val_labels_np == 1]
        neg_sims = val_scores_np[val_labels_np == 0]
        gap = max(0.12, float(np.mean(pos_sims) - np.mean(neg_sims)))
        self.sim_temperature = float(np.clip(3.4 / gap, 8.0, 14.0))

        final_preds = (val_scores_np >= self.sim_threshold).astype(np.int32)
        acc = accuracy_score(val_labels_np, final_preds)
        f1 = f1_score(val_labels_np, final_preds, zero_division=0)
        roc_auc = roc_auc_score(val_labels_np, val_scores_np)
        cm = confusion_matrix(val_labels_np, final_preds)

        self.save_weights()
        self._refresh_author_database()
        clear_memory()

        log_lines.extend(
            [
                "\n### Final Validation Metrics",
                f"- **Validation Accuracy:** `{acc * 100:.2f}%`",
                f"- **Validation ROC-AUC:** `{roc_auc:.4f}`",
                f"- **Validation F1-Score:** `{f1:.4f}` (Calibrated Threshold: `{self.sim_threshold:.3f}`)",
                f"- **Weights Saved:** `{WEIGHTS_PATH}`",
            ]
        )

        fig = self._plot_evaluation_diagnostics(cm, pos_sims, neg_sims)
        progress(1.0, desc="Training & Calibration Complete!")
        return "\n".join(log_lines), fig

    def _plot_evaluation_diagnostics(
        self, cm: np.ndarray, pos_sims: np.ndarray, neg_sims: np.ndarray
    ) -> plt.Figure:
        plt.style.use("dark_background")
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.8), dpi=110)
        fig.patch.set_facecolor("#0f172a")

        ax1.set_facecolor("#1e293b")
        ax1.imshow(cm, cmap="Blues", alpha=0.85)
        ax1.set_title("Validation Confusion Matrix", fontsize=12, fontweight="bold", pad=12, color="#f8fafc")
        ax1.set_xticks([0, 1])
        ax1.set_yticks([0, 1])
        ax1.set_xticklabels(["Diff Author (0)", "Same Author (1)"], fontsize=10)
        ax1.set_yticklabels(["Diff Author (0)", "Same Author (1)"], fontsize=10)
        ax1.set_xlabel("Predicted Verdict", fontsize=10, color="#cbd5e1")
        ax1.set_ylabel("Ground Truth", fontsize=10, color="#cbd5e1")

        for i in range(2):
            for j in range(2):
                ax1.text(
                    j,
                    i,
                    f"{cm[i, j]}",
                    ha="center",
                    va="center",
                    color="#ffffff",
                    fontsize=14,
                    fontweight="bold",
                )

        ax2.set_facecolor("#1e293b")
        bins = np.linspace(0.05, 0.95, 28)
        ax2.hist(
            neg_sims,
            bins=bins,
            alpha=0.65,
            label="Different Authors (Label=0)",
            color="#f43f5e",
            edgecolor="#881337",
        )
        ax2.hist(
            pos_sims,
            bins=bins,
            alpha=0.65,
            label="Same Author (Label=1)",
            color="#10b981",
            edgecolor="#064e3b",
        )
        ax2.axvline(
            self.sim_threshold,
            color="#fbbf24",
            linestyle="--",
            linewidth=2,
            label=f"Decision Threshold ({self.sim_threshold:.2f})",
        )
        ax2.set_title("Ensemble Similarity Score Distribution", fontsize=12, fontweight="bold", pad=12, color="#f8fafc")
        ax2.set_xlabel("Composite Forensic Similarity", fontsize=10, color="#cbd5e1")
        ax2.set_ylabel("Pair Count", fontsize=10, color="#cbd5e1")
        ax2.legend(loc="upper left", frameon=True, facecolor="#0f172a", edgecolor="#334155")
        ax2.grid(True, linestyle=":", alpha=0.25)

        plt.tight_layout()
        return fig

    def compare_two_texts(self, text_a: str, text_b: str) -> Tuple[str, plt.Figure, pd.DataFrame]:
        text_a = (text_a or "").strip()
        text_b = (text_b or "").strip()

        if not text_a or not text_b:
            empty_fig = self._plot_radar_comparison({}, {})
            return (
                "⚠️ **Please enter non-empty text into both Text A and Text B.**",
                empty_fig,
                pd.DataFrame(),
            )

        res = self.compute_composite_forensic_score(text_a, text_b)
        prob = res["probability"]
        comp_sim = res["composite_sim"]

        if prob >= 65.0:
            verdict_badge = "🟢 HIGH LIKELIHOOD: SAME AUTHOR"
        elif prob >= 42.0:
            verdict_badge = "🟡 INCONCLUSIVE / BORDERLINE STYLE MATCH"
        else:
            verdict_badge = "🔴 DIFFERENT AUTHORS DETECTED"

        arousal_note = ""
        if res["arousal_shift"] >= 1.25:
            arousal_note = (
                f"\n- ⚡ **Emotional Shift Compensator Active:** High expressive punctuation delta detected "
                f"(`Δ = {res['arousal_shift']:.2f} SD`). Volatile exclamation/comma swaps are down-weighted in favor "
                f"of invariant clause cadence and syntactic markers."
            )

        report_md = (
            f"## {verdict_badge}\n"
            f"- **Same Author Probability:** `{prob:.1f}%`\n"
            f"- **Composite Forensic Score:** `{comp_sim:.4f}` *(Calibrated Threshold: `{self.sim_threshold:.3f}`)*\n"
            f"- **Sub-Kernel Telemetry:**\n"
            f"  1. **Active Stylometric Kernel (Non-Zero Weighted):** `{res['s_kernel'] * 100:.1f}%`\n"
            f"  2. **Burrows' Cosine Delta (Directional Z-Alignment):** `{res['cos_delta']:+.4f}`\n"
            f"  3. **Structural Function-Word / Suffix N-Gram Similarity:** `{res['s_ngram'] * 100:.1f}%`\n"
            f"  4. **Topic-Deconfounded Siamese MLP Cosine:** `{res['s_siamese']:+.4f}`"
            f"{arousal_note}"
        )

        radar_a = self.extractor.get_radar_metrics(text_a)
        radar_b = self.extractor.get_radar_metrics(text_b)
        radar_fig = self._plot_radar_comparison(radar_a, radar_b)

        raw_a = res["raw_a"]
        raw_b = res["raw_b"]
        feat_matches = res["feat_matches"]

        breakdown_rows = []
        for idx, (name, val_a, val_b, match_val) in enumerate(
            zip(StylometricFeatureExtractor.FEATURE_NAMES, raw_a, raw_b, feat_matches)
        ):
            if match_val is None:
                status_str = "Inactive (0/0)"
                match_str = "Excluded (0/0)"
            else:
                status_str = "Active"
                match_str = f"{match_val:.1f}%"

            breakdown_rows.append(
                {
                    "Stylometric Feature": name,
                    "Status": status_str,
                    "Text A": round(float(val_a), 3),
                    "Text B": round(float(val_b), 3),
                    "Abs Delta": round(abs(float(val_a - val_b)), 3),
                    "Calibrated Match": match_str,
                }
            )
        df_breakdown = pd.DataFrame(breakdown_rows)

        return report_md, radar_fig, df_breakdown

    def _plot_radar_comparison(
        self, metrics_a: Dict[str, float], metrics_b: Dict[str, float]
    ) -> plt.Figure:
        plt.style.use("dark_background")
        fig, ax = plt.subplots(figsize=(6.6, 5.8), subplot_kw=dict(polar=True), dpi=110)
        fig.patch.set_facecolor("#0f172a")
        ax.set_facecolor("#1e293b")

        if not metrics_a or not metrics_b:
            ax.text(0, 0, "Awaiting Text Input...", ha="center", va="center", color="#94a3b8", fontsize=12)
            return fig

        categories = list(metrics_a.keys())
        N = len(categories)
        angles = [n / float(N) * 2 * math.pi for n in range(N)]
        angles += angles[:1]

        values_a = [metrics_a[c] for c in categories] + [metrics_a[categories[0]]]
        values_b = [metrics_b[c] for c in categories] + [metrics_b[categories[0]]]

        ax.set_theta_offset(math.pi / 2)
        ax.set_theta_direction(-1)

        plt.xticks(angles[:-1], categories, color="#e2e8f0", size=9.5, fontweight="bold")
        ax.set_rlabel_position(0)
        plt.yticks([0.25, 0.50, 0.75, 1.0], ["0.25", "0.50", "0.75", "1.00"], color="#64748b", size=8)
        plt.ylim(0, 1.05)

        ax.plot(angles, values_a, linewidth=2.3, linestyle="solid", label="Text A", color="#38bdf8")
        ax.fill(angles, values_a, color="#38bdf8", alpha=0.25)

        ax.plot(angles, values_b, linewidth=2.3, linestyle="solid", label="Text B", color="#f43f5e")
        ax.fill(angles, values_b, color="#f43f5e", alpha=0.25)

        ax.legend(loc="upper right", bbox_to_anchor=(1.22, 1.12), facecolor="#0f172a", edgecolor="#334155")
        ax.grid(color="#334155", linestyle="--", linewidth=0.8)

        fig.subplots_adjust(top=0.84, bottom=0.08, left=0.10, right=0.86)
        return fig

    def identify_author(self, anonymous_text: str, top_k: int = 5) -> Tuple[Dict[str, float], pd.DataFrame, str]:
        anonymous_text = (anonymous_text or "").strip()
        if not anonymous_text:
            return {}, pd.DataFrame(), "⚠️ **Please paste an anonymous text snippet to analyze.**"

        if not self.author_centroids:
            self._refresh_author_database()

        query_emb = self.get_hybrid_embeddings([anonymous_text])[0]
        query_raw = self.extractor.extract_vector(anonymous_text)

        raw_scores: List[Tuple[str, float, float, float, float]] = []
        for author, centroid in self.author_centroids.items():
            cos_siamese = float(np.dot(query_emb, centroid))
            s_siamese = float(np.clip((cos_siamese + 0.25) / 1.15, 0.0, 1.0))

            author_prof = self.author_style_profiles[author]
            s_kernel, cos_delta, _, _ = self.compute_stylometric_similarity(query_raw, author_prof)
            s_cos_delta = float(np.clip((cos_delta + 0.35) / 1.15, 0.0, 1.0))

            s_ngram = self.ngram_kernel.compute_pair_similarity(
                anonymous_text, self.author_texts_concat[author]
            )

            comp_sim = (
                0.35 * s_kernel
                + 0.30 * s_cos_delta
                + 0.20 * s_siamese
                + 0.15 * s_ngram
            )
            logit = (comp_sim - self.sim_threshold) * self.sim_temperature
            cal_prob = 100.0 / (1.0 + math.exp(-max(-20.0, min(20.0, logit))))
            raw_scores.append((author, comp_sim, cos_delta, s_kernel, cal_prob))

        raw_scores.sort(key=lambda x: x[1], reverse=True)
        top_results = raw_scores[: max(1, int(top_k))]

        sims_arr = np.array([r[1] for r in top_results], dtype=np.float64)
        exp_scores = np.exp((sims_arr - np.max(sims_arr)) * 12.0)
        softmax_probs = exp_scores / np.sum(exp_scores)

        label_dict: Dict[str, float] = {}
        table_rows = []
        for idx, ((author, comp_sim, cos_delta, s_kernel, cal_prob), rel_conf) in enumerate(
            zip(top_results, softmax_probs), start=1
        ):
            blended_conf = float(0.50 * rel_conf + 0.50 * (cal_prob / 100.0))
            label_dict[author] = round(blended_conf, 4)
            table_rows.append(
                {
                    "Rank": f"#{idx}",
                    "Candidate Author": author,
                    "Composite Score": round(comp_sim, 4),
                    "Cosine Delta": round(cos_delta, 4),
                    "Active Kernel (%)": f"{s_kernel * 100:.1f}%",
                    "Match Likelihood (%)": f"{cal_prob:.1f}%",
                    "Corpus Samples": self.author_sample_counts.get(author, 0),
                }
            )

        best_author, best_sim, best_delta, _, best_prob = top_results[0]
        summary_md = (
            f"### Top Attribution Match: **{best_author}**\n"
            f"- **Composite Score:** `{best_sim:.4f}` | **Cosine Delta:** `{best_delta:+.4f}` | **Match Likelihood:** `{best_prob:.1f}%`\n"
            f"- **Database Coverage:** Compared against `{len(self.author_centroids)}` author style profiles."
        )
        return label_dict, pd.DataFrame(table_rows), summary_md


# ==============================================================================
# 9. GRADIO DARK-THEMED WEB INTERFACE (CLEAN ENGLISH UI, NO PRESET TEXTS)
# ==============================================================================
ENGINE = StylometryEngine()


def build_ui() -> gr.Blocks:
    dark_theme = gr.themes.Base(
        primary_hue="sky",
        secondary_hue="rose",
        neutral_hue="slate",
    ).set(
        body_background_fill="#0b0f19",
        body_text_color="#f1f5f9",
        block_background_fill="#111827",
        block_border_color="#1f2937",
        input_background_fill="#1e293b",
    )

    with gr.Blocks(theme=dark_theme, title="Digital Forensics / Style Profiler") as demo:
        gr.Markdown(
            """
            # 🕵️‍♂️ Digital Forensics & Stylometry Profiler
            **Hybrid Siamese Authorship Verification** — Combines Burrows' Cosine Delta across 36 universal psycholinguistic & syntactic features, active-only sparsity filtering, structural function-word N-grams, and topic-masked neural representations.
            """
        )

        with gr.Tabs():
            # ------------------------------------------------------------------
            # TAB 1: TRAIN & EVALUATE
            # ------------------------------------------------------------------
            with gr.TabItem("1. Train & Evaluate"):
                with gr.Row():
                    with gr.Column(scale=1):
                        gr.Markdown("### ⚙️ Training Configuration")
                        gr.Markdown(f"**Active Corpus:** {ENGINE.corpus_summary}")
                        epochs_slider = gr.Slider(
                            minimum=2,
                            maximum=8,
                            value=5,
                            step=1,
                            label="Training Epochs",
                        )
                        lr_number = gr.Number(
                            value=0.0015,
                            label="Learning Rate (AdamW)",
                            precision=5,
                        )
                        train_btn = gr.Button("🚀 Run Siamese Training & Calibration", variant="primary")
                        metrics_md = gr.Markdown(
                            "The model is pre-calibrated at startup. Click **Run Siamese Training & Calibration** "
                            "to re-train on the active dataset and render validation diagnostics."
                        )

                    with gr.Column(scale=2):
                        eval_plot = gr.Plot(label="Confusion Matrix & Similarity Score Distribution")

                train_btn.click(
                    fn=ENGINE.train_and_evaluate,
                    inputs=[epochs_slider, lr_number],
                    outputs=[metrics_md, eval_plot],
                )

            # ------------------------------------------------------------------
            # TAB 2: 1-VS-1 FORENSIC COMPARISON
            # ------------------------------------------------------------------
            with gr.TabItem("2. 1-vs-1 Forensic Comparison"):
                with gr.Row():
                    text_a_input = gr.Textbox(
                        label="Text Sample A",
                        lines=8,
                        value="",
                        placeholder="Paste first text sample here...",
                    )
                    text_b_input = gr.Textbox(
                        label="Text Sample B",
                        lines=8,
                        value="",
                        placeholder="Paste second text sample here...",
                    )

                compare_btn = gr.Button("🔍 Run Forensic Style Comparison", variant="primary")

                verdict_output = gr.Markdown()

                with gr.Row():
                    with gr.Column(scale=1):
                        radar_output = gr.Plot(label="6-Axis Stylometric Fingerprint Radar")
                    with gr.Column(scale=1):
                        feature_table = gr.Dataframe(
                            label="36-Feature Quantitative Stylometric Breakdown",
                            interactive=False,
                        )

                compare_btn.click(
                    fn=ENGINE.compare_two_texts,
                    inputs=[text_a_input, text_b_input],
                    outputs=[verdict_output, radar_output, feature_table],
                )

            # ------------------------------------------------------------------
            # TAB 3: AUTHOR IDENTIFICATION (DATABASE SEARCH)
            # ------------------------------------------------------------------
            with gr.TabItem("3. Author Identification (Database Search)"):
                with gr.Row():
                    with gr.Column(scale=1):
                        anon_input = gr.Textbox(
                            label="Anonymous Text Snippet",
                            lines=6,
                            value="",
                            placeholder="Paste an anonymous text snippet to rank candidate authors...",
                        )
                        top_k_slider = gr.Slider(
                            minimum=2,
                            maximum=10,
                            value=5,
                            step=1,
                            label="Top-N Authors to Rank",
                        )
                        search_btn = gr.Button("🎯 Identify Most Likely Author", variant="primary")
                        id_summary_md = gr.Markdown()

                    with gr.Column(scale=1):
                        author_bars = gr.Label(
                            num_top_classes=5,
                            label="Candidate Author Confidence Ranking",
                        )

                ranking_table = gr.Dataframe(
                    label="Detailed Candidate Telemetry",
                    interactive=False,
                )

                search_btn.click(
                    fn=ENGINE.identify_author,
                    inputs=[anon_input, top_k_slider],
                    outputs=[author_bars, ranking_table, id_summary_md],
                )

    return demo


if __name__ == "__main__":
    print(f"[Launch] Starting Digital Forensics Style Profiler on device: {DEVICE}")
    app = build_ui()
    app.launch(server_name="127.0.0.1", server_port=7860, inbrowser=False)
