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
WEIGHTS_PATH = "stylometry_model.pt"
LEGACY_WEIGHT_FILES = ("stylometry_model_v2.pt", "stylometry_model_v3.pt")
TELEGRAM_JSON_PATH = "result.json"
CUSTOM_CSV_PATH = "data.csv"

MAX_SEQ_LENGTH = 256
ENCODE_BATCH_SIZE = 8
TRAIN_BATCH_SIZE = 16
RANDOM_SEED = 42


def cleanup_legacy_weights() -> None:
    """Removes old versioned .pt files so only a single stylometry_model.pt exists for Git."""
    for old_file in LEGACY_WEIGHT_FILES:
        if os.path.exists(old_file):
            try:
                os.remove(old_file)
                print(f"[Cleanup] Removed legacy weights file: {old_file}")
            except OSError:
                pass


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


cleanup_legacy_weights()
set_seed()
DEVICE = get_device()

# ==============================================================================
# 2. MULTILINGUAL LEXICAL & GRAMMATICAL DICTIONARIES (EN + RU)
# ==============================================================================
ARTICLES_AND_DETERMINERS = {
    # English determiners & articles
    "a", "an", "the", "this", "that", "these", "those", "some", "any", "each",
    "every", "either", "neither", "another", "such", "what", "which", "whose",
    # Russian determiners & demonstrative pronouns
    "этот", "эта", "это", "эти", "этого", "этой", "этих", "этим", "этими", "этом",
    "тот", "та", "то", "те", "того", "той", "тех", "тем", "теми", "том",
    "весь", "вся", "всё", "все", "всего", "всей", "всех", "всем", "всеми",
    "каждый", "каждая", "каждое", "каждые", "любой", "любая", "любое", "любые",
    "такой", "такая", "такое", "такие", "такого", "таких", "сам", "сама", "само",
    "сами", "самый", "самая", "самое", "самые", "самых", "свой", "своя", "свое",
    "своё", "свои", "своих", "своими", "своего", "своей", "некоторый", "никакой",
}

PREPOSITIONS = {
    "in", "on", "at", "to", "for", "with", "from", "by", "about", "as", "into",
    "like", "through", "after", "over", "between", "out", "against", "during",
    "without", "before", "under", "around", "among", "towards", "toward", "upon",
    "within", "along", "following", "across", "behind", "beyond", "plus", "except",
    "up", "down", "off", "above", "near", "inside", "outside", "onto", "until",
    "в", "во", "на", "с", "со", "к", "ко", "у", "за", "по", "от", "о", "об",
    "обо", "из", "из-за", "из-под", "до", "для", "без", "под", "при", "после",
    "над", "через", "про", "между", "перед", "около", "вокруг", "среди", "ради",
    "сквозь", "вместо", "вроде", "насчет", "насчёт", "вдоль", "помимо", "кроме",
}

FIRST_PERSON_PRONOUNS = {
    "i", "me", "my", "mine", "myself", "we", "us", "our", "ours", "ourselves",
    "i'm", "i've", "i'd", "i'll", "we're", "we've", "we'd", "we'll",
    "я", "мне", "меня", "мной", "мною", "мой", "моя", "моё", "мое", "мои",
    "моего", "моей", "моих", "моим", "моими", "по-моему",
    "мы", "нам", "нас", "нами", "наш", "наша", "наше", "наши", "нашего", "нашей",
    "наших", "нашим", "нашими",
}

SECOND_PERSON_PRONOUNS = {
    "you", "your", "yours", "yourself", "yourselves", "you're", "you've",
    "you'd", "you'll",
    "ты", "тебе", "тебя", "тобой", "тобою", "твой", "твоя", "твоё", "твое",
    "твои", "твоего", "твоей", "твоих", "вы", "вам", "вас", "вами", "ваш",
    "ваша", "ваше", "ваши", "вашего", "вашей", "ваших",
}

THIRD_PERSON_PRONOUNS = {
    "he", "him", "his", "himself", "she", "her", "hers", "herself", "it", "its",
    "itself", "they", "them", "their", "theirs", "themselves", "it's", "he's",
    "she's", "they're", "they've", "they'd", "they'll",
    "он", "она", "оно", "они", "его", "ее", "её", "их", "ему", "ей", "им",
    "ним", "ней", "них", "него", "нее", "неё", "ими", "себя", "себе", "собой",
}

MODAL_AND_EPISTEMIC_WORDS = {
    "can", "cannot", "can't", "could", "couldn't", "may", "might", "must",
    "mustn't", "shall", "shan't", "should", "shouldn't", "will", "won't",
    "would", "wouldn't", "ought", "need", "needs", "needed",
    "может", "могут", "могу", "можешь", "могли", "могла", "мог", "могло",
    "нужно", "надо", "необходимо", "должен", "должна", "должно", "должны",
    "следует", "приходится", "пришлось", "можно", "нельзя", "стоит", "стоило",
    "возможно", "вероятно", "очевидно", "видимо", "кажется", "похоже", "наверное",
    "скорее", "вряд",
}

AUXILIARY_AND_COPULA_VERBS = {
    "be", "am", "is", "are", "was", "were", "been", "being", "have", "has",
    "had", "having", "do", "does", "did", "isn't", "aren't", "wasn't", "weren't",
    "haven't", "hasn't", "hadn't", "doesn't", "didn't", "don't", "that's",
    "there's", "here's", "what's", "who's",
    "быть", "был", "была", "было", "были", "будет", "будут", "буду", "будешь",
    "будем", "будь", "есть", "суть", "является", "являются", "становится",
    "становятся", "оказался", "оказалась", "оказалось", "оказались",
}

IRREGULAR_PAST_VERBS_EN = {
    "was", "were", "had", "did", "said", "went", "got", "made", "knew", "thought",
    "took", "saw", "came", "wanted", "looked", "used", "found", "gave", "told",
    "worked", "called", "tried", "asked", "needed", "felt", "became", "left",
    "put", "meant", "kept", "let", "began", "seemed", "helped", "talked", "turned",
    "started", "showed", "heard", "played", "ran", "moved", "liked", "lived",
    "believed", "held", "brought", "happened", "wrote", "provided", "sat", "stood",
    "lost", "paid", "met", "included", "continued", "set", "learned", "learnt",
    "changed", "led", "understood", "watched", "followed", "stopped", "created",
    "spoke", "read", "allowed", "added", "spent", "grew", "opened", "walked",
    "won", "offered", "remembered", "loved", "considered", "appeared", "bought",
    "waited", "served", "died", "sent", "expected", "built", "stayed", "fell",
    "cut", "reached", "killed", "remained", "suggested", "raised", "passed",
    "sold", "required", "reported", "decided", "pulled", "shot", "trapped",
    "crumpled", "blocked", "drove", "broke", "caught", "chose", "drew", "drank",
    "ate", "flew", "forgot", "forgave", "froze", "hid", "hit", "hurt", "lay",
    "rode", "rang", "rose", "sang", "sank", "shook", "shone", "shut", "slept",
    "slid", "smelt", "spat", "split", "spread", "stole", "stuck", "stung",
    "struck", "swore", "swept", "swam", "swung", "taught", "tore", "threw",
    "woke", "wore", "wound", "wasn't", "weren't", "hadn't", "didn't", "couldn't",
    "wouldn't",
}

RU_NON_PAST_EXCEPTIONS = {
    "стол", "стул", "пол", "зал", "бал", "вал", "мал", "мил", "сил", "дел",
    "тела", "дела", "крыла", "школа", "правила", "сначала", "мало", "много",
    "около", "число", "село", "стекло", "тепло", "светло", "тяжело", "легко",
    "если", "или", "нежели", "едва", "пока", "весьма", "туда", "сюда", "когда",
    "тогда", "всегда", "иногда", "никогда", "никуда", "откуда", "мысли", "земли",
    "цели", "роли", "боли", "соли", "пыли", "дали", "стали", "детали", "модели",
    "недели", "постели", "мебели", "гибели", "отрасли", "капли", "воли", "доли",
    "политики", "республики", "культуры", "литературы", "природы", "свободы",
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
    "совсем", "абсолютно", "реально", "короче", "типа", "прям", "прямо", "буквально",
    "именно", "действительно", "наконец", "особенно", "отчасти", "вполне", "всячески",
    "насильно", "хреново", "странно", "серьезно", "серьёзно", "теперь", "сейчас",
    "тут", "там", "здесь", "потом", "сразу", "вдруг", "опять", "снова", "вместе",
}

CONJUNCTIONS_AND_CAUSALS = {
    "and", "but", "or", "so", "if", "because", "since", "although", "though",
    "while", "whereas", "unless", "until", "whether", "before", "after", "when",
    "whenever", "where", "wherever", "than", "that", "as", "however", "therefore",
    "thus", "hence", "moreover", "furthermore", "nevertheless", "nonetheless",
    "consequently", "otherwise", "instead", "meanwhile", "yet", "nor", "plus",
    "и", "а", "но", "да", "или", "либо", "что", "чтобы", "чтоб", "как", "когда",
    "если", "потому", "поэтому", "так", "хотя", "хоть", "пока", "зато", "однако",
    "причем", "причём", "притом", "поскольку", "ибо", "ведь", "же", "ли", "бы",
    "б", "куда", "откуда", "где", "который", "которая", "которое", "которые",
    "которого", "которой", "которых", "которым", "которыми", "кто", "что", "чей",
    "чья", "чье", "чьё", "чьи", "чьим", "чьих", "тоже", "также", "затем", "иначе",
}

NEGATION_WORDS = {
    "not", "no", "never", "nothing", "nobody", "noone", "nowhere", "neither",
    "nor", "none", "without", "cannot", "can't", "won't", "don't", "doesn't",
    "didn't", "isn't", "aren't", "wasn't", "weren't", "haven't", "hasn't",
    "hadn't", "couldn't", "wouldn't", "shouldn't", "mustn't",
    "не", "нет", "ни", "никогда", "никуда", "никто", "ничего", "нигде", "никак",
    "никакой", "никакая", "никакие", "нельзя", "некуда", "нечего", "непонятно",
    "без",
}

FUNCTION_WORDS = (
    ARTICLES_AND_DETERMINERS
    | PREPOSITIONS
    | FIRST_PERSON_PRONOUNS
    | SECOND_PERSON_PRONOUNS
    | THIRD_PERSON_PRONOUNS
    | MODAL_AND_EPISTEMIC_WORDS
    | AUXILIARY_AND_COPULA_VERBS
    | ADVERBS_AND_INTENSIFIERS
    | CONJUNCTIONS_AND_CAUSALS
    | NEGATION_WORDS
)


# ==============================================================================
# 3. LANGUAGE DETECTION & EMPIRICAL POPULATION PRIORS (EN vs RU)
# ==============================================================================
CYRILLIC_RE = re.compile(r"[А-Яа-яЁё]")
LATIN_RE = re.compile(r"[A-Za-z]")


def detect_text_language(text: str) -> str:
    cyr = len(CYRILLIC_RE.findall(text))
    lat = len(LATIN_RE.findall(text))
    return "ru" if cyr > lat else "en"


def is_russian_past_verb(w_low: str) -> bool:
    if len(w_low) < 3 or w_low in RU_NON_PAST_EXCEPTIONS or w_low in FUNCTION_WORDS:
        return False
    if w_low in {"был", "была", "было", "были", "стал", "стала", "стало", "стали", "шел", "шёл", "шла", "шли"}:
        return True
    if len(w_low) >= 4 and w_low.endswith(("лся", "лась", "лось", "лись")):
        return True
    if len(w_low) >= 4 and w_low.endswith((
        "ал", "ала", "ало", "али", "ял", "яла", "яло", "яли",
        "ил", "ила", "ило", "или", "ел", "ела", "ело", "ели",
        "ул", "ула", "уло", "ули", "ыла", "ыли", "ола", "оли"
    )):
        return True
    return False


def is_russian_participle_or_gerund(w_low: str) -> bool:
    if len(w_low) < 5 or w_low in FUNCTION_WORDS:
        return False
    return w_low.endswith((
        "ущий", "ющий", "ащий", "ящий", "ущая", "ющая", "ащая", "ящая",
        "ущее", "ющее", "ащее", "ящее", "ущие", "ющие", "ащие", "ящие",
        "вший", "вшая", "вшее", "вшие", "ший", "шая", "шее", "шие",
        "вшись", "ши", "учи", "ючи",
        "емый", "емая", "емое", "емые", "имый", "имая", "имое", "имые",
        "нный", "нная", "нное", "нные", "нного", "нной", "нным", "нными", "нных",
        "тый", "тая", "тое", "тые", "того", "тым", "тыми",
        "вшем", "вшей", "вшего", "вшему", "вшими", "вших",
    ))


POPULATION_PRIORS: Dict[str, Dict[str, np.ndarray]] = {
    "en": {
        "mu": np.array([
            4.25,  # 0: avg_word_len
            2.15,  # 1: std_word_len
            2.60,  # 2: word_len_iqr
            0.915, # 3: herdan_c_ttr
            0.890, # 4: mattr_window
            0.570, # 5: yules_k_inv
            0.820, # 6: hapax_ratio
            0.435, # 7: short_word_ratio
            0.155, # 8: long_word_ratio
            0.210, # 9: content_repetition_rate
            10.50, # 10: avg_clause_len
            0.520, # 11: clause_len_cv
            1.750, # 12: clause_boundary_per_100
            19.50, # 13: avg_sentence_len
            0.550, # 14: sentence_len_cv
            1.800, # 15: para_density_per_100w
            0.520, # 16: function_word_ratio
            0.090, # 17: article_det_ratio
            0.120, # 18: preposition_ratio
            0.080, # 19: conjunction_causal_ratio
            0.050, # 20: adverb_intensifier_ratio
            0.025, # 21: negation_ratio
            2.600, # 22: contraction_clitic_per_100w
            0.038, # 23: first_person_ratio
            0.015, # 24: second_person_ratio
            0.068, # 25: third_person_ratio
            0.024, # 26: modal_epistemic_ratio
            0.070, # 27: aux_copula_ratio
            0.065, # 28: past_tense_ratio
            0.030, # 29: participle_gerund_ratio
            4.300, # 30: comma_per_100w
            4.800, # 31: period_per_100w
            0.900, # 32: hyphen_dash_per_100w
            1.200, # 33: parens_quotes_colon_slash_per_100w
            0.800, # 34: expressive_punct_per_100w
            0.030, # 35: mid_sentence_cap_and_num_rate
        ], dtype=np.float32),
        "sigma": np.array([
            0.32,  # 0: avg_word_len
            0.24,  # 1: std_word_len
            0.75,  # 2: word_len_iqr
            0.028, # 3: herdan_c_ttr
            0.038, # 4: mattr_window
            0.085, # 5: yules_k_inv
            0.140, # 6: hapax_ratio
            0.055, # 7: short_word_ratio
            0.052, # 8: long_word_ratio
            0.085, # 9: content_repetition_rate
            3.20,  # 10: avg_clause_len
            0.180, # 11: clause_len_cv
            0.550, # 12: clause_boundary_per_100
            6.50,  # 13: avg_sentence_len
            0.280, # 14: sentence_len_cv
            1.100, # 15: para_density_per_100w
            0.055, # 16: function_word_ratio
            0.030, # 17: article_det_ratio
            0.032, # 18: preposition_ratio
            0.026, # 19: conjunction_causal_ratio
            0.022, # 20: adverb_intensifier_ratio
            0.016, # 21: negation_ratio
            1.800, # 22: contraction_clitic_per_100w
            0.028, # 23: first_person_ratio
            0.016, # 24: second_person_ratio
            0.032, # 25: third_person_ratio
            0.016, # 26: modal_epistemic_ratio
            0.026, # 27: aux_copula_ratio
            0.035, # 28: past_tense_ratio
            0.018, # 29: participle_gerund_ratio
            2.100, # 30: comma_per_100w
            1.900, # 31: period_per_100w
            1.100, # 32: hyphen_dash_per_100w
            1.300, # 33: parens_quotes_colon_slash_per_100w
            1.400, # 34: expressive_punct_per_100w
            0.024, # 35: mid_sentence_cap_and_num_rate
        ], dtype=np.float32),
    },
    "ru": {
        "mu": np.array([
            5.25,  # 0: avg_word_len
            2.85,  # 1: std_word_len
            3.50,  # 2: word_len_iqr
            0.945, # 3: herdan_c_ttr
            0.915, # 4: mattr_window
            0.630, # 5: yules_k_inv
            1.050, # 6: hapax_ratio
            0.335, # 7: short_word_ratio
            0.320, # 8: long_word_ratio
            0.130, # 9: content_repetition_rate
            5.80,  # 10: avg_clause_len
            0.480, # 11: clause_len_cv
            2.500, # 12: clause_boundary_per_100
            14.20, # 13: avg_sentence_len
            0.550, # 14: sentence_len_cv
            2.600, # 15: para_density_per_100w
            0.410, # 16: function_word_ratio
            0.035, # 17: article_det_ratio
            0.115, # 18: preposition_ratio
            0.105, # 19: conjunction_causal_ratio
            0.055, # 20: adverb_intensifier_ratio
            0.038, # 21: negation_ratio
            0.800, # 22: contraction_clitic_per_100w
            0.042, # 23: first_person_ratio
            0.012, # 24: second_person_ratio
            0.045, # 25: third_person_ratio
            0.026, # 26: modal_epistemic_ratio
            0.025, # 27: aux_copula_ratio
            0.068, # 28: past_tense_ratio
            0.032, # 29: participle_gerund_ratio
            9.500, # 30: comma_per_100w
            6.500, # 31: period_per_100w
            2.400, # 32: hyphen_dash_per_100w
            1.600, # 33: parens_quotes_colon_slash_per_100w
            1.200, # 34: expressive_punct_per_100w
            0.018, # 35: mid_sentence_cap_and_num_rate
        ], dtype=np.float32),
        "sigma": np.array([
            0.38,  # 0: avg_word_len
            0.28,  # 1: std_word_len
            0.80,  # 2: word_len_iqr
            0.025, # 3: herdan_c_ttr
            0.035, # 4: mattr_window
            0.085, # 5: yules_k_inv
            0.160, # 6: hapax_ratio
            0.055, # 7: short_word_ratio
            0.062, # 8: long_word_ratio
            0.075, # 9: content_repetition_rate
            1.75,  # 10: avg_clause_len
            0.170, # 11: clause_len_cv
            0.650, # 12: clause_boundary_per_100
            4.80,  # 13: avg_sentence_len
            0.260, # 14: sentence_len_cv
            1.300, # 15: para_density_per_100w
            0.058, # 16: function_word_ratio
            0.022, # 17: article_det_ratio
            0.036, # 18: preposition_ratio
            0.032, # 19: conjunction_causal_ratio
            0.025, # 20: adverb_intensifier_ratio
            0.022, # 21: negation_ratio
            0.900, # 22: contraction_clitic_per_100w
            0.035, # 23: first_person_ratio
            0.015, # 24: second_person_ratio
            0.028, # 25: third_person_ratio
            0.018, # 26: modal_epistemic_ratio
            0.018, # 27: aux_copula_ratio
            0.036, # 28: past_tense_ratio
            0.022, # 29: participle_gerund_ratio
            2.800, # 30: comma_per_100w
            2.400, # 31: period_per_100w
            1.400, # 32: hyphen_dash_per_100w
            1.400, # 33: parens_quotes_colon_slash_per_100w
            1.800, # 34: expressive_punct_per_100w
            0.018, # 35: mid_sentence_cap_and_num_rate
        ], dtype=np.float32),
    },
}


# ==============================================================================
# 4. SYNTACTIC SKELETONIZER (TOPIC MASKING FOR TRANSFORMER)
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
RU_CLITIC_SUFFIXES = ("-то", "-либо", "-нибудь", "-ка", "-таки", "-де", "-с")


def normalize_apostrophes(text: str) -> str:
    return APOSTROPHE_NORM_RE.sub("'", text)


def mask_word_to_morphology(word: str, is_mid_sentence: bool = False) -> str:
    w_low = word.lower()
    if w_low in FUNCTION_WORDS:
        return w_low

    if w_low.endswith(RU_CLITIC_SUFFIXES):
        return "<RU_CLITIC>"

    if "-" in w_low and len(w_low) >= 4:
        return "<COMPOUND_HYPH>"

    if "'" in w_low:
        if w_low.endswith("n't"):
            return "<NEG_CONTR>"
        if w_low.endswith("'s"):
            return "<POSS_OR_IS>"
        return "<CONTR>"

    if (
        w_low in IRREGULAR_PAST_VERBS_EN
        or (len(w_low) >= 4 and w_low.endswith("ed"))
        or is_russian_past_verb(w_low)
    ):
        return "<V_PAST>"
    if (len(w_low) >= 5 and w_low.endswith("ing")) or is_russian_participle_or_gerund(w_low):
        return "<V_PART_GER>"
    if len(w_low) >= 5 and w_low.endswith("ly"):
        return "<ADV_LY>"
    if len(w_low) >= 4 and w_low.endswith(("ть", "ться", "ти", "чь")):
        return "<V_INF>"
    if len(w_low) >= 6 and w_low.endswith((
        "tion", "sion", "ment", "ness", "ity", "ance", "ence",
        "ость", "ение", "ание", "ство", "ация", "яция"
    )):
        return "<NOUN_ABSTR>"
    if len(w_low) >= 5 and w_low.endswith((
        "able", "ible", "ous", "ive", "ful", "less", "al", "ic",
        "ый", "ий", "ой", "ая", "яя", "ое", "ее", "ые", "ие", "ого", "его", "ому", "ему"
    )):
        return "<ADJ>"

    if is_mid_sentence and word[0].isupper():
        return "<PROPER_CAP>"

    if len(w_low) <= 4:
        return "<CONTENT_S>"
    elif len(w_low) <= 7:
        return "<CONTENT_M>"
    return "<CONTENT_L>"


def build_syntactic_skeleton(text: str) -> str:
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
# 5. UNIVERSAL 36-FEATURE STYLOMETRIC EXTRACTOR (EN + RU BILINGUAL)
# ==============================================================================
class StylometricFeatureExtractor:
    FEATURE_NAMES = [
        # 1. Lexical Richness & Word Length Distribution (0..9)
        "avg_word_len",
        "std_word_len",
        "word_len_iqr",
        "herdan_c_ttr",
        "mattr_window",
        "yules_k_inv",
        "hapax_ratio",
        "short_word_ratio",
        "long_word_ratio",
        "content_repetition_rate",
        # 2. Syntactic Rhythm & Clause/Sentence Cadence (10..15)
        "avg_clause_len",
        "clause_len_cv",
        "clause_boundary_per_100",
        "avg_sentence_len",
        "sentence_len_cv",
        "para_density_per_100w",
        # 3. Function Words & Grammatical Glue (16..22)
        "function_word_ratio",
        "article_det_ratio",
        "preposition_ratio",
        "conjunction_causal_ratio",
        "adverb_intensifier_ratio",
        "negation_ratio",
        "contraction_clitic_per_100w",
        # 4. Pronoun Perspective & Verb Tense / Modality (23..29)
        "first_person_ratio",
        "second_person_ratio",
        "third_person_ratio",
        "modal_epistemic_ratio",
        "aux_copula_ratio",
        "past_tense_ratio",
        "participle_gerund_ratio",
        # 5. Punctuation, Orthography & Register Habits (30..35)
        "comma_per_100w",
        "period_per_100w",
        "hyphen_dash_per_100w",
        "parens_quotes_colon_slash_per_100w",
        "expressive_punct_per_100w",
        "mid_sentence_cap_and_num_rate",
    ]

    SPARSE_ZERO_INDICES = {21, 22, 23, 24, 26, 29, 32, 33, 34, 35}
    VOLATILE_EMOTION_INDICES = {13, 14, 30, 31, 34}

    SENT_SPLIT_RE = re.compile(r"(?:[.!?…]+)+\s*|\n+", re.UNICODE)
    CLAUSE_SPLIT_RE = re.compile(r"[,;:—–]+|(?:[.!?…]+)+|\n+", re.UNICODE)
    ELLIPSIS_CLUSTER_RE = re.compile(r"\.{2,}|…+")
    EXCL_CLUSTER_RE = re.compile(r"!+")
    QUEST_CLUSTER_RE = re.compile(r"[?]+|[?!]{2,}")
    EMDASH_RE = re.compile(r"—|--|–|\s-\s")
    QUOTES_RE = re.compile(r"[\"“”«»]")
    EMOTICON_RE = re.compile(r"(?::\)|:\(|;\)|\(:\s*|\(-:|\bxd\b|[💀😂😭🔥👍])", re.IGNORECASE)

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
        clean_word_lengths = np.array(
            [max(1, len(w.replace("'", "").replace("-", ""))) for w in lower_words],
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
        clause_lengths = [len(WORD_WITH_CONTRACTION_RE.findall(c)) for c in clauses]
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
        sent_lengths = [len(WORD_WITH_CONTRACTION_RE.findall(s)) for s in raw_sentences]
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
        # 3. Function Words, Grammatical Glue, Negations & Contractions/Clitics
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

        neg_hits = sum(1 for w in lower_words if w in NEGATION_WORDS or w.endswith("n't"))
        negation_ratio = float(neg_hits) / float(word_count)

        contraction_clitic_hits = sum(
            1 for w in lower_words
            if "'" in w or w.endswith(RU_CLITIC_SUFFIXES)
        )
        contraction_clitic_per_100w = float(contraction_clitic_hits) * scale_100w

        # ------------------------------------------------------------------
        # 4. Pronoun Perspective & Verb Tense / Modality (EN + RU)
        # ------------------------------------------------------------------
        fp_hits = sum(1 for w in lower_words if w in FIRST_PERSON_PRONOUNS)
        first_person_ratio = float(fp_hits) / float(word_count)

        sp_hits = sum(1 for w in lower_words if w in SECOND_PERSON_PRONOUNS)
        second_person_ratio = float(sp_hits) / float(word_count)

        tp_hits = sum(1 for w in lower_words if w in THIRD_PERSON_PRONOUNS)
        third_person_ratio = float(tp_hits) / float(word_count)

        modal_hits = sum(1 for w in lower_words if w in MODAL_AND_EPISTEMIC_WORDS)
        modal_epistemic_ratio = float(modal_hits) / float(word_count)

        aux_hits = sum(1 for w in lower_words if w in AUXILIARY_AND_COPULA_VERBS)
        aux_copula_ratio = float(aux_hits) / float(word_count)

        past_hits = sum(
            1 for w in lower_words
            if w in IRREGULAR_PAST_VERBS_EN
            or (len(w) >= 4 and w.endswith("ed") and w not in {"red", "bed", "fed", "led", "shed", "sled", "hundred"})
            or is_russian_past_verb(w)
        )
        past_tense_ratio = float(past_hits) / float(word_count)

        ing_part_hits = sum(
            1 for w in lower_words
            if (len(w) >= 5 and w.endswith("ing") and w not in {"thing", "bring", "spring", "string", "swing", "king", "ring", "wing"})
            or is_russian_participle_or_gerund(w)
        )
        participle_gerund_ratio = float(ing_part_hits) / float(word_count)

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

        parens_quotes_colon_slash = (
            clean_text.count("(")
            + clean_text.count(")")
            + clean_text.count(":")
            + clean_text.count(";")
            + clean_text.count("/")
            + len(self.QUOTES_RE.findall(clean_text))
        )
        parens_quotes_colon_slash_per_100w = float(parens_quotes_colon_slash) * scale_100w

        expressive_punct_per_100w = (
            float(excl_clusters + quest_clusters + ellipsis_clusters + emoticon_hits) * scale_100w
        )

        all_seq_tokens = TOKEN_OR_PUNCT_RE.findall(norm_text)
        mid_sent_caps = 0
        num_tokens = 0
        at_sent_start = True
        for tok in all_seq_tokens:
            if tok[0].isalpha():
                if not at_sent_start and tok[0].isupper() and tok not in {"I", "Я"}:
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
                contraction_clitic_per_100w,
                first_person_ratio,
                second_person_ratio,
                third_person_ratio,
                modal_epistemic_ratio,
                aux_copula_ratio,
                past_tense_ratio,
                participle_gerund_ratio,
                comma_per_100w,
                period_per_100w,
                hyphen_dash_per_100w,
                parens_quotes_colon_slash_per_100w,
                expressive_punct_per_100w,
                mid_sentence_cap_and_num_rate,
            ],
            dtype=np.float32,
        )
        return np.nan_to_num(feats, nan=0.0, posinf=0.0, neginf=0.0)

    def extract_batch(self, texts: List[str]) -> np.ndarray:
        """Extracts raw 36-feature vectors for a batch of texts."""
        return np.vstack([self.extract_vector(t) for t in texts])

    def standardize_vector(self, raw_vec: np.ndarray, lang: str) -> np.ndarray:
        priors = POPULATION_PRIORS.get(lang, POPULATION_PRIORS["en"])
        z = (raw_vec - priors["mu"]) / np.maximum(priors["sigma"], 1e-5)
        return np.clip(z, -4.5, 4.5).astype(np.float32)

    def extract_batch_standardized(self, texts: List[str]) -> np.ndarray:
        out = []
        for t in texts:
            lang = detect_text_language(t)
            raw = self.extract_vector(t)
            out.append(self.standardize_vector(raw, lang))
        return np.vstack(out)

    def get_radar_metrics(self, text: str) -> Dict[str, float]:
        """
        Maps language-conditional Z-scores via sigmoid(0.75 * z) into [0, 1] where 0.50
        is the exact population average for that language (EN or RU).
        """
        lang = detect_text_language(text)
        raw = self.extract_vector(text)
        z = self.standardize_vector(raw, lang)
        idx = {name: i for i, name in enumerate(self.FEATURE_NAMES)}

        def z_to_unit(z_val: float) -> float:
            return float(1.0 / (1.0 + math.exp(-0.75 * max(-5.0, min(5.0, z_val)))))

        clause_cadence = z_to_unit(0.6 * z[idx["avg_clause_len"]] + 0.4 * z[idx["avg_sentence_len"]])
        lexical_richness = z_to_unit(0.5 * z[idx["mattr_window"]] + 0.5 * z[idx["herdan_c_ttr"]])
        narrative_past = z_to_unit(
            0.45 * z[idx["past_tense_ratio"]]
            + 0.35 * z[idx["first_person_ratio"]]
            + 0.20 * z[idx["third_person_ratio"]]
        )
        modal_analytical = z_to_unit(
            0.45 * z[idx["modal_epistemic_ratio"]]
            + 0.30 * z[idx["conjunction_causal_ratio"]]
            + 0.25 * z[idx["preposition_ratio"]]
        )
        punct_complexity = z_to_unit(
            0.40 * z[idx["parens_quotes_colon_slash_per_100w"]]
            + 0.35 * z[idx["hyphen_dash_per_100w"]]
            + 0.25 * z[idx["contraction_clitic_per_100w"]]
        )
        word_complexity = z_to_unit(0.6 * z[idx["avg_word_len"]] + 0.4 * z[idx["long_word_ratio"]])

        return {
            "Clause & Sentence Length": clause_cadence,
            "Lexical Richness": lexical_richness,
            "Past & Narrative Focus": narrative_past,
            "Modal & Analytical Glue": modal_analytical,
            "Punctuation & Clitics": punct_complexity,
            "Word Complexity": word_complexity,
        }


# ==============================================================================
# 6. STRUCTURAL FUNCTION-WORD & MORPHOLOGICAL N-GRAM KERNEL
# ==============================================================================
class StructuralNgramKernel:
    def __init__(self):
        self.vectorizer = TfidfVectorizer(
            analyzer="word",
            ngram_range=(1, 3),
            min_df=1,
            max_features=1600,
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
# 7. DATA INGESTION & BILINGUAL CALIBRATION CORPUS (EN + RU)
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
    Generates a balanced bilingual (4 English + 4 Russian) multi-author calibration
    corpus across diverse topics so Siamese training covers both languages cleanly.
    """
    topics_en = [
        ("arctic wildlife", "Marine Predator", "pack ice", "thermal insulation", "hunting territory"),
        ("urban transit", "Subway Line", "signal failure", "platform crowd", "evening commute"),
        ("distributed systems", "Cache Node", "network latency", "memory buffer", "failover cluster"),
        ("cinema history", "Soundstage Lighting", "camera movement", "practical effects", "theatrical cut"),
        ("aviation safety", "Flight Deck", "wind shear", "runway approach", "hydraulic pressure"),
    ]

    topics_ru = [
        ("детский спорткомплекс", "тренер по офп", "оформление документов", "льготные условия", "руководство сада"),
        ("ремонт автомобиля", "коробка передач", "гарантийный сервис", "оригинальные запчасти", "диагностика двигателя"),
        ("история архитектуры", "городская застройка", "культурное наследие", "реставрация фасада", "архивные чертежи"),
        ("поиск работы", "удаленная вакансия", "тестовое задание", "зарплатные ожидания", "отдел кадров"),
        ("аренда жилья", "договор найма", "коммунальные счета", "залог за квартиру", "соседи сверху"),
    ]

    personas: Dict[str, List[str]] = {
        "Author_EN_1 (Expository & Compound-Heavy)": [],
        "Author_EN_2 (1st-Person Past Narrative)": [],
        "Author_EN_3 (Casual Contractions & Modal Opinion)": [],
        "Author_EN_4 (Formal Academic & Semicolons)": [],
        "Author_RU_1 (Colloquial Storyteller with Clitics '-то')": [],
        "Author_RU_2 (Ideological / Publicistic with Slashes '/')": [],
        "Author_RU_3 (1st-Person Confessional & Causal 'потому что/чтобы')": [],
        "Author_RU_4 (Dry Analytical & Participial Constructions)": [],
    }

    for i in range(16):
        te = topics_en[i % len(topics_en)]
        e1, e2, e3, e4, e5 = te

        personas["Author_EN_1 (Expository & Compound-Heavy)"].append(
            f"On open ground, a well-adapted {e2} can hold its own during {e1}, and will often dominate "
            f"the {e3} if it meets it in deep-water conditions. Otherwise it would be a toss-up, with the "
            f"edge decidedly towards the {e2} (a seasoned {e2} is much, much faster than average). Fun fact: "
            f"a rite of passage for a young operator would be to master {e4} without triggering {e5}."
        )

        personas["Author_EN_2 (1st-Person Past Narrative)"].append(
            f"I had a colleague who managed {e1} when the {e2} failed. She didn't last more than 2 years "
            f"after that before she transferred out. That was after many trips to the facility, including "
            f"the first {e3} incident 4 days after she got access. We all watched her try to fix {e4} and "
            f"there was nothing we could do (we tried everything, multiple shifts monitoring {e5} day to day)."
        )

        personas["Author_EN_3 (Casual Contractions & Modal Opinion)"].append(
            f"I just don't see it. New teams might start out handling {e1}, but the kind of throughput "
            f"they're talking about would require like 300k per year, and there's no setup I know that "
            f"handles {e3} out the gate. A specialist might fix {e4} eventually, but they won't be solving "
            f"{e5} in the first three months, plus it doesn't sound like {e2} is ready."
        )

        personas["Author_EN_4 (Formal Academic & Semicolons)"].append(
            f"Consequently, when evaluating the overarching framework of {e1}, one must inevitably account "
            f"for {e3}; furthermore, empirical observations regarding {e4} — particularly under constrained "
            f"parameters — demonstrate that {e5} remains fundamentally intertwined with {e2}."
        )

        tr = topics_ru[i % len(topics_ru)]
        r1, r2, r3, r4, r5 = tr

        personas["Author_RU_1 (Colloquial Storyteller with Clitics '-то')"].append(
            f"Работа с темой «{r1}» в наше время становится реально делом для самых рисковых.\n\n"
            f"Причем, именно для обычных людей - у начальства такого нет. Живой пример: в место, куда "
            f"ходила моя сестра, пришли из соседнего отдела и предложили {r4} - и наши ходят туда, "
            f"занимаются вопросом, а отдел выделяет нам специалиста ({r2}), который проводит сколько-то "
            f"часов в неделю.\n\n"
            f"Ну вроде сделка и сделка, всех устроило. Вот только {r2}-то был принципиальный. И его там "
            f"по инстанциям загоняли, проверяли всячески. В итоге гемор с {r3} себя не оправдал и {r5} всё свернули."
        )

        personas["Author_RU_2 (Ideological / Publicistic with Slashes '/')"].append(
            f"Людей заставляли принимать {r1} насильно, кто не хотел - увольняли. И раньше все пользовались "
            f"своим опытом, а не чужими регламентами ({r3}), и руководство говорит что рядовые сотрудники "
            f"люди второго сорта.\n\n"
            f"Современный {r1}/{r4} - это хреново спрятанный контроль с подчинением системе: {r5} и его "
            f"посредникам. Такой {r2}/{r3} не является нормальной практикой. Оно навязано нам сверху.\n\n"
            f"Притом что чиновники извратили изначальный смысл {r4}, и теперь требуют отчёты (бумажки-справки) "
            f"и непонятно чьи подписи, что прямо противоречит правилам."
        )

        if i % 2 == 0:
            personas["Author_RU_3 (1st-Person Confessional & Causal 'потому что/чтобы')"].append(
                f"Так вышло что у меня снова проблемы из-за темы «{r1}». Мне сложно со всем этим разбираться. "
                f"Отчасти из-за прошлого опыта (все скандалили, никто не помогал).\n\n"
                f"С {r3} у меня тоже беда.\n\n"
                f"Сейчас начался ад с {r2}. Но мне отказываются помогать с {r4}! Говорят иди сама решай! "
                f"А я не могу это сделать одна!!! Меня не слушают нигде!!!\n\n"
                f"Должна ли я в этой ситуации терпеть такое отношение от {r5} ?!?!?"
            )
        else:
            personas["Author_RU_3 (1st-Person Confessional & Causal 'потому что/чтобы')"].append(
                f"Начальник заблокировал мне доступ к {r1}, чтобы я не смогла оформлять {r3} по вечерам, "
                f"потому что так, я якобы нарушаю отчётность.\n\n"
                f"Каждый вечер, мне приходится выпрашивать у него разрешение на {r2}, чтобы он на полчаса "
                f"включил систему, хотя бы для {r4}. Если не попрошу, он не сделает ничего никогда сам.\n\n"
                f"Нормальная ли это ситуация?"
            )

        personas["Author_RU_4 (Dry Analytical & Participial Constructions)"].append(
            f"Следует отметить, что при детальном рассмотрении вопроса, связанного с категорией «{r1}», "
            f"необходимо учитывать фактор, определяющий {r3}; более того, практика взаимодействия с {r5}, "
            f"сформировавшаяся при участии специалиста ({r2}), наглядно демонстрирует необходимость пересмотра {r4}."
        )

    rows = []
    for author, texts in personas.items():
        for txt in texts:
            rows.append({"author": author, "text": txt})

    df = pd.DataFrame(rows)
    return df, f"Built-in Bilingual Calibration Corpus ({df['author'].nunique()} authors, {len(df)} texts)"


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
    df: pd.DataFrame, max_pairs_per_author: int = 70
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
# 8. HYBRID SIAMESE FUSION ARCHITECTURE
# ==============================================================================
class SiameseFusionHead(nn.Module):
    def __init__(self, neural_dim: int, style_dim: int, hidden_dim: int = 192, out_dim: int = 96):
        super().__init__()
        self.style_encoder = nn.Sequential(
            nn.Linear(style_dim, 96),
            nn.LayerNorm(96),
            nn.GELU(),
            nn.Linear(96, 96),
            nn.LayerNorm(96),
        )
        self.neural_proj = nn.Sequential(
            nn.Linear(neural_dim, 64),
            nn.LayerNorm(64),
            nn.GELU(),
        )
        self.fusion_mlp = nn.Sequential(
            nn.Linear(160, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(0.15),
            nn.Linear(hidden_dim, out_dim),
            nn.LayerNorm(out_dim),
        )

    def forward_once(self, neural_emb: torch.Tensor, style_vec: torch.Tensor) -> torch.Tensor:
        s_emb = self.style_encoder(style_vec) * 1.80
        n_emb = self.neural_proj(neural_emb) * 0.60
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
# 9. FORENSIC ENGINE: LANGUAGE-AWARE LLR + CONTRADICTION VETO + SIAMESE HEAD
# ==============================================================================
class StylometryEngine:
    def __init__(self):
        self.device = DEVICE
        self.extractor = StylometricFeatureExtractor()
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
            hidden_dim=192,
            out_dim=96,
        ).to(self.device)

        self.sim_threshold: float = 0.50
        self.sim_temperature: float = 10.0

        self.df, self.corpus_summary = load_corpus()
        self.author_centroids: Dict[str, np.ndarray] = {}
        self.author_raw_profiles: Dict[str, np.ndarray] = {}
        self.author_langs: Dict[str, str] = {}
        self.author_texts_concat: Dict[str, str] = {}
        self.author_sample_counts: Dict[str, int] = {}

        self.corpus_neural: Optional[np.ndarray] = None
        self.corpus_style_z: Optional[np.ndarray] = None

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
        blended = 0.80 * emb_skel + 0.20 * emb_raw
        norms = np.linalg.norm(blended, axis=1, keepdims=True) + 1e-9
        return (blended / norms).astype(np.float32)

    def _ensure_corpus_encoded(self) -> None:
        if self.corpus_neural is None or len(self.corpus_neural) != len(self.df):
            texts = self.df["text"].tolist()
            self.corpus_neural = self.encode_texts_deconfounded(texts)
            self.corpus_style_z = self.extractor.extract_batch_standardized(texts)
            self.ngram_kernel.fit(texts)

    @torch.no_grad()
    def get_hybrid_embeddings(self, texts: List[str]) -> np.ndarray:
        self._ensure_corpus_encoded()
        self.fusion_head.eval()

        neural_np = self.encode_texts_deconfounded(texts)
        style_z = self.extractor.extract_batch_standardized(texts)

        n_tensor = torch.tensor(neural_np, dtype=torch.float32, device=self.device)
        s_tensor = torch.tensor(style_z, dtype=torch.float32, device=self.device)
        fused = self.fusion_head.forward_once(n_tensor, s_tensor)
        return fused.cpu().numpy()

    def compute_forensic_feature_telemetry(
        self,
        raw_a: np.ndarray,
        raw_b: np.ndarray,
        lang_a: str,
        lang_b: str,
    ) -> Dict[str, Any]:
        z_a = self.extractor.standardize_vector(raw_a, lang_a)
        z_b = self.extractor.standardize_vector(raw_b, lang_b)

        delta_z = np.abs(z_a - z_b)
        per_feature_sim = np.exp(-0.95 * np.power(delta_z, 1.55))

        expressive_idx = StylometricFeatureExtractor.FEATURE_NAMES.index("expressive_punct_per_100w")
        arousal_shift = float(delta_z[expressive_idx])

        weights = np.ones(self.style_dim, dtype=np.float32)
        feature_match_list: List[Optional[float]] = []

        idx_map = {name: i for i, name in enumerate(StylometricFeatureExtractor.FEATURE_NAMES)}
        for core_name in [
            "avg_word_len",
            "std_word_len",
            "mattr_window",
            "short_word_ratio",
            "long_word_ratio",
            "avg_clause_len",
            "clause_boundary_per_100",
            "function_word_ratio",
            "preposition_ratio",
            "conjunction_causal_ratio",
            "contraction_clitic_per_100w",
            "first_person_ratio",
            "third_person_ratio",
            "modal_epistemic_ratio",
            "past_tense_ratio",
        ]:
            weights[idx_map[core_name]] = 1.6

        for idx in range(self.style_dim):
            if (
                idx in StylometricFeatureExtractor.SPARSE_ZERO_INDICES
                and abs(raw_a[idx]) < 1e-6
                and abs(raw_b[idx]) < 1e-6
            ):
                weights[idx] = 0.0
                feature_match_list.append(None)
            else:
                feature_match_list.append(float(per_feature_sim[idx] * 100.0))

        if arousal_shift >= 1.25:
            for v_idx in StylometricFeatureExtractor.VOLATILE_EMOTION_INDICES:
                weights[v_idx] *= 0.10
            weights[idx_map["avg_clause_len"]] = 2.8
            weights[idx_map["clause_boundary_per_100"]] = 2.4

        severe_contradictions = 0
        contradiction_names = []
        for idx in range(self.style_dim):
            if weights[idx] <= 0.20:
                continue
            if delta_z[idx] > 1.20:
                severe_contradictions += 1
                contradiction_names.append(StylometricFeatureExtractor.FEATURE_NAMES[idx])

        if np.sum(weights) > 0:
            d_rms = float(np.sqrt(np.sum((delta_z ** 2) * weights) / np.sum(weights)))
            weighted_arith_sim = float(np.sum(per_feature_sim * weights) / np.sum(weights))
        else:
            d_rms = 2.0
            weighted_arith_sim = 0.0

        s_rms_kernel = float(math.exp(-0.85 * (d_rms ** 1.65)))
        contradiction_penalty = max(0, severe_contradictions - 2) * 0.075
        s_kernel = float(np.clip(0.60 * s_rms_kernel + 0.40 * weighted_arith_sim - contradiction_penalty, 0.0, 1.0))

        z_a_act = z_a * np.sqrt(weights)
        z_b_act = z_b * np.sqrt(weights)
        denom = (np.linalg.norm(z_a_act) * np.linalg.norm(z_b_act)) + 1e-9
        cos_delta = float(np.dot(z_a_act, z_b_act) / denom)

        return {
            "s_kernel": s_kernel,
            "d_rms": d_rms,
            "cos_delta": cos_delta,
            "arousal_shift": arousal_shift,
            "severe_contradictions": severe_contradictions,
            "contradiction_names": contradiction_names,
            "feat_matches": feature_match_list,
            "delta_z": delta_z,
        }

    def compute_composite_forensic_score(
        self, text_a: str, text_b: str
    ) -> Dict[str, Any]:
        self._ensure_corpus_encoded()
        lang_a = detect_text_language(text_a)
        lang_b = detect_text_language(text_b)

        raw_a = self.extractor.extract_vector(text_a)
        raw_b = self.extractor.extract_vector(text_b)

        telem = self.compute_forensic_feature_telemetry(raw_a, raw_b, lang_a, lang_b)
        s_kernel = telem["s_kernel"]
        cos_delta = telem["cos_delta"]
        d_rms = telem["d_rms"]
        severe_contradictions = telem["severe_contradictions"]

        s_cos_delta = float(np.clip((cos_delta + 0.25) / 1.05, 0.0, 1.0))

        embs = self.get_hybrid_embeddings([text_a, text_b])
        cos_siamese = float(np.dot(embs[0], embs[1]))
        s_siamese = float(np.clip((cos_siamese + 0.15) / 1.05, 0.0, 1.0))

        s_ngram = self.ngram_kernel.compute_pair_similarity(text_a, text_b)

        composite_sim = (
            0.45 * s_kernel
            + 0.25 * s_cos_delta
            + 0.15 * s_siamese
            + 0.15 * s_ngram
        )

        if severe_contradictions >= 5 or d_rms > 1.02:
            veto_factor = max(0.35, 1.0 - 0.09 * max(0, severe_contradictions - 3))
            composite_sim *= veto_factor

        logit = (composite_sim - self.sim_threshold) * self.sim_temperature
        prob = 100.0 / (1.0 + math.exp(-max(-20.0, min(20.0, logit))))

        return {
            "probability": float(prob),
            "composite_sim": float(composite_sim),
            "s_kernel": float(s_kernel),
            "d_rms": float(d_rms),
            "cos_delta": float(cos_delta),
            "s_siamese": float(cos_siamese),
            "s_ngram": float(s_ngram),
            "arousal_shift": float(telem["arousal_shift"]),
            "severe_contradictions": severe_contradictions,
            "contradiction_names": telem["contradiction_names"],
            "lang_a": lang_a.upper(),
            "lang_b": lang_b.upper(),
            "raw_a": raw_a,
            "raw_b": raw_b,
            "delta_z": telem["delta_z"],
            "feat_matches": telem["feat_matches"],
        }

    def _quick_warmup_calibration(self) -> None:
        print("[Init] Running startup calibration of Siamese Fusion Head...")
        self._ensure_corpus_encoded()
        pairs = build_balanced_pairs(self.df, max_pairs_per_author=35)
        ds = PairFeatureDataset(pairs, self.corpus_neural, self.corpus_style_z)
        loader = DataLoader(ds, batch_size=TRAIN_BATCH_SIZE, shuffle=True)

        criterion = nn.CosineEmbeddingLoss(margin=0.25)
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
        self.sim_threshold = 0.49
        self.sim_temperature = 10.5
        self.save_weights()
        self._refresh_author_database()
        clear_memory()

    def _refresh_author_database(self) -> None:
        self._ensure_corpus_encoded()
        self.fusion_head.eval()

        with torch.no_grad():
            n_tensor = torch.tensor(self.corpus_neural, dtype=torch.float32, device=self.device)
            s_tensor = torch.tensor(self.corpus_style_z, dtype=torch.float32, device=self.device)
            all_fused = self.fusion_head.forward_once(n_tensor, s_tensor).cpu().numpy()

        self.author_centroids.clear()
        self.author_raw_profiles.clear()
        self.author_langs.clear()
        self.author_texts_concat.clear()
        self.author_sample_counts.clear()

        raw_all = self.extractor.extract_batch(self.df["text"].tolist())

        for author, group in self.df.groupby("author"):
            idx = group.index.to_numpy()
            author_embs = all_fused[idx]
            centroid = np.mean(author_embs, axis=0)
            norm = np.linalg.norm(centroid) + 1e-9
            self.author_centroids[str(author)] = (centroid / norm).astype(np.float32)
            self.author_raw_profiles[str(author)] = np.median(raw_all[idx], axis=0)
            concat_txt = "\n".join(group["text"].tolist()[:15])
            self.author_texts_concat[str(author)] = concat_txt
            self.author_langs[str(author)] = detect_text_language(concat_txt)
            self.author_sample_counts[str(author)] = len(idx)

    def save_weights(self) -> None:
        cleanup_legacy_weights()
        half_state = {k: v.detach().cpu().half() for k, v in self.fusion_head.state_dict().items()}
        payload = {
            "version": 4,
            "fusion_state_dict": half_state,
            "sim_threshold": self.sim_threshold,
            "sim_temperature": self.sim_temperature,
            "model_name": MODEL_NAME,
            "style_dim": self.style_dim,
        }
        torch.save(payload, WEIGHTS_PATH)

    def _try_load_weights(self) -> bool:
        cleanup_legacy_weights()
        if not os.path.exists(WEIGHTS_PATH):
            return False
        try:
            checkpoint = torch.load(WEIGHTS_PATH, map_location="cpu", weights_only=False)
            if (
                checkpoint.get("version") != 4
                or checkpoint.get("model_name") != MODEL_NAME
                or checkpoint.get("style_dim") != self.style_dim
            ):
                return False
            float_state = {
                k: v.float().to(self.device) for k, v in checkpoint["fusion_state_dict"].items()
            }
            self.fusion_head.load_state_dict(float_state)
            self.sim_threshold = float(checkpoint.get("sim_threshold", 0.49))
            self.sim_temperature = float(checkpoint.get("sim_temperature", 10.5))
            self.fusion_head.eval()
            print(f"[Init] Loaded pre-trained weights from '{WEIGHTS_PATH}'.")
            return True
        except Exception as e:
            print(f"[Warning] Rebuilding '{WEIGHTS_PATH}' ({e})")
            return False

    def train_and_evaluate(
        self, epochs: int = 5, lr: float = 1.5e-3, progress=gr.Progress()
    ) -> Tuple[str, plt.Figure]:
        set_seed()
        progress(0.05, desc="Encoding bilingual syntactic skeletons & 36 language-aware features...")
        self._ensure_corpus_encoded()

        pairs = build_balanced_pairs(self.df, max_pairs_per_author=70)
        labels_only = [p[2] for p in pairs]
        train_pairs, val_pairs = train_test_split(
            pairs, test_size=0.25, random_state=RANDOM_SEED, stratify=labels_only
        )

        train_ds = PairFeatureDataset(train_pairs, self.corpus_neural, self.corpus_style_z)
        train_loader = DataLoader(train_ds, batch_size=TRAIN_BATCH_SIZE, shuffle=True)

        self.fusion_head = SiameseFusionHead(
            neural_dim=self.neural_dim,
            style_dim=self.style_dim,
            hidden_dim=192,
            out_dim=96,
        ).to(self.device)

        criterion = nn.CosineEmbeddingLoss(margin=0.25)
        optimizer = torch.optim.AdamW(self.fusion_head.parameters(), lr=lr, weight_decay=1e-4)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

        log_lines = [
            "### Hybrid Forensic Ensemble Summary",
            f"- **Compute Device:** `{self.device}` (Batch Size: `{TRAIN_BATCH_SIZE}`)",
            f"- **Neural Backbone:** `{MODEL_NAME}` with Syntactic Skeleton Topic-Masking",
            f"- **Stylometric Space:** `{self.style_dim}` language-conditional features (`EN` & `RU` priors) + Contradiction Veto",
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

        best_th, best_f1 = 0.49, -1.0
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
        self.sim_temperature = float(np.clip(3.6 / gap, 8.5, 14.0))

        final_preds = (val_scores_np >= self.sim_threshold).astype(np.int32)
        acc = accuracy_score(val_labels_np, final_preds)
        f1 = f1_score(val_labels_np, final_preds, zero_division=0)
        roc_auc = roc_auc_score(val_labels_np, val_scores_np)
        cm = confusion_matrix(val_labels_np, final_preds)

        self.save_weights()
        self._refresh_author_database()
        clear_memory()

        size_kb = os.path.getsize(WEIGHTS_PATH) / 1024.0 if os.path.exists(WEIGHTS_PATH) else 0.0
        log_lines.extend(
            [
                "\n### Final Validation Metrics",
                f"- **Validation Accuracy:** `{acc * 100:.2f}%`",
                f"- **Validation ROC-AUC:** `{roc_auc:.4f}`",
                f"- **Validation F1-Score:** `{f1:.4f}` (Calibrated Threshold: `{self.sim_threshold:.3f}`)",
                f"- **Single Weights File Saved:** `{WEIGHTS_PATH}` (`{size_kb:.1f} KB` — Git-friendly, no LFS needed)",
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

        notes = []
        if res["arousal_shift"] >= 1.25:
            notes.append(
                f"- ⚡ **Emotional Shift Compensator Active:** Expressive punctuation delta detected "
                f"(`Δ = {res['arousal_shift']:.2f} SD`). Volatile exclamation/comma swaps are down-weighted in favor "
                f"of invariant clause cadence and syntactic markers."
            )
        if res["severe_contradictions"] >= 4:
            top_contr = ", ".join(f"`{c}`" for c in res["contradiction_names"][:6])
            notes.append(
                f"- 🛑 **Stylometric Contradiction Veto Triggered:** Found `{res['severe_contradictions']}` severe "
                f"structural divergences (`> 1.20 SD` gap) across core features ({top_contr})."
            )

        notes_block = ("\n" + "\n".join(notes)) if notes else ""

        report_md = (
            f"## {verdict_badge}\n"
            f"- **Same Author Probability:** `{prob:.1f}%`\n"
            f"- **Composite Forensic Score:** `{comp_sim:.4f}` *(Calibrated Threshold: `{self.sim_threshold:.3f}` | Norms: `Text A [{res['lang_a']}]`, `Text B [{res['lang_b']}]`)*\n"
            f"- **Sub-Kernel Telemetry:**\n"
            f"  1. **LLR & RMS Stylometric Kernel:** `{res['s_kernel'] * 100:.1f}%` *(Standardized RMS Distance: `{res['d_rms']:.2f} SD` | Severe Contradictions: `{res['severe_contradictions']}`)*\n"
            f"  2. **Language-Centered Burrows' Cosine Delta:** `{res['cos_delta']:+.4f}`\n"
            f"  3. **Structural Function-Word / Suffix N-Gram Similarity:** `{res['s_ngram'] * 100:.1f}%`\n"
            f"  4. **Topic-Deconfounded Siamese MLP Cosine:** `{res['s_siamese']:+.4f}`"
            f"{notes_block}"
        )

        radar_a = self.extractor.get_radar_metrics(text_a)
        radar_b = self.extractor.get_radar_metrics(text_b)
        radar_fig = self._plot_radar_comparison(radar_a, radar_b)

        raw_a = res["raw_a"]
        raw_b = res["raw_b"]
        delta_z = res["delta_z"]
        feat_matches = res["feat_matches"]

        breakdown_rows = []
        for idx, (name, val_a, val_b, dz, match_val) in enumerate(
            zip(StylometricFeatureExtractor.FEATURE_NAMES, raw_a, raw_b, delta_z, feat_matches)
        ):
            if match_val is None:
                status_str = "Inactive (0/0)"
                match_str = "Excluded (0/0)"
            elif dz > 1.20:
                status_str = "Contradiction (>1.2 SD)"
                match_str = f"{match_val:.1f}%"
            else:
                status_str = "Active"
                match_str = f"{match_val:.1f}%"

            breakdown_rows.append(
                {
                    "Stylometric Feature": name,
                    "Status": status_str,
                    "Text A": round(float(val_a), 3),
                    "Text B": round(float(val_b), 3),
                    "Z-Gap (SD)": round(float(dz), 2),
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
        plt.yticks([0.25, 0.50, 0.75, 1.0], ["0.25", "0.50 (Avg)", "0.75", "1.00"], color="#64748b", size=8)
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

        query_lang = detect_text_language(anonymous_text)
        query_emb = self.get_hybrid_embeddings([anonymous_text])[0]
        query_raw = self.extractor.extract_vector(anonymous_text)

        raw_scores: List[Tuple[str, float, float, float, float]] = []
        for author, centroid in self.author_centroids.items():
            cos_siamese = float(np.dot(query_emb, centroid))
            s_siamese = float(np.clip((cos_siamese + 0.15) / 1.05, 0.0, 1.0))

            author_prof = self.author_raw_profiles[author]
            author_lang = self.author_langs.get(author, query_lang)
            telem = self.compute_forensic_feature_telemetry(query_raw, author_prof, query_lang, author_lang)
            s_kernel = telem["s_kernel"]
            cos_delta = telem["cos_delta"]
            s_cos_delta = float(np.clip((cos_delta + 0.25) / 1.05, 0.0, 1.0))

            s_ngram = self.ngram_kernel.compute_pair_similarity(
                anonymous_text, self.author_texts_concat[author]
            )

            comp_sim = (
                0.45 * s_kernel
                + 0.25 * s_cos_delta
                + 0.15 * s_siamese
                + 0.15 * s_ngram
            )
            if telem["severe_contradictions"] >= 5 or telem["d_rms"] > 1.02:
                comp_sim *= max(0.35, 1.0 - 0.09 * max(0, telem["severe_contradictions"] - 3))

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
                    "LLR Kernel (%)": f"{s_kernel * 100:.1f}%",
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
# 10. GRADIO DARK-THEMED WEB INTERFACE (CLEAN ENGLISH UI, NO PRESET TEXTS)
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
            **Hybrid Siamese Authorship Verification** — Language-conditional (`EN` / `RU`) Z-score standardization, Forensic RMS Log-Likelihood Ratio with Contradiction Veto, Burrows' Cosine Delta, and topic-masked neural representations.
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
                            "Weights are stored in a single compact `stylometry_model.pt` file. "
                            "Click **Run Siamese Training & Calibration** to re-train and render validation plots."
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