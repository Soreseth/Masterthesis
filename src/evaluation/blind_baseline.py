"""Blind Baselines for Membership Inference Attacks.

Based on: "Blind Baselines Beat Membership Inference Attacks for Foundation Models"
(Das, Zhang, Tramèr, ICLR 2025 DATA-FM Workshop)

Three blind attacks that distinguish members from non-members WITHOUT querying any model:
1. Date Detection: extracts years from text and thresholds by a cutoff date
2. Bag of Words: TF-IDF/CountVectorizer + stacking classifier
3. Greedy N-gram Selection: greedily picks n-grams that separate member/non-member distributions

Additionally provides per-text feature extraction for integration with the aggregation pipeline.
"""

import re
import itertools
import numpy as np
from collections import defaultdict
from statistics import mean, pstdev

from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier, StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.naive_bayes import GaussianNB, MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve, auc


# ----------------------------------------------------------------------
# Utility functions
# ----------------------------------------------------------------------

def get_roc_auc(y_true, y_pred_proba):
    fpr, tpr, _ = roc_curve(y_true, y_pred_proba)
    return auc(fpr, tpr)


def get_tpr_at_fpr(y_true, y_pred_proba, fpr_budget=1.0):
    """Return TPR at fpr_budget% FPR."""
    fpr, tpr, _ = roc_curve(y_true, y_pred_proba)
    return float(np.interp(fpr_budget / 100, fpr, tpr))


def ngrams(s, n=5):
    """Return the set of all character n-grams for N <= n."""
    if n == 1:
        return set(s)
    iters = itertools.tee(s, n)
    for i, it in enumerate(iters):
        next(itertools.islice(it, i, i), None)
    return set("".join(x) for x in zip(*iters)).union(ngrams(s, n - 1))


def count_unique_ngrams(lines, n=5, threshold=None):
    """For each n-gram, return the set of line indices containing it."""
    char_counts = defaultdict(set)
    for i, line in enumerate(lines):
        unique_chars = ngrams(line, n=n)
        for char in unique_chars:
            char_counts[char].add(i)
    if threshold is not None:
        for c in list(char_counts.keys()):
            if len(char_counts[c]) < threshold:
                del char_counts[c]
    return char_counts


# ----------------------------------------------------------------------
# Per-text feature extraction (model-free, for aggregation pipeline)
# ----------------------------------------------------------------------

def extract_blind_features(text: str) -> dict:
    """Extract model-free text features from a single text chunk.

    These features capture surface-level text statistics that may reveal
    distribution shifts between members and non-members.

    Returns dict of feature_name -> float.
    """
    features = {}

    # Text length features
    features['blind_char_len'] = len(text)
    features['blind_word_count'] = len(text.split())
    features['blind_avg_word_len'] = np.mean([len(w) for w in text.split()]) if text.split() else 0.0

    # Character-level features
    features['blind_alpha_ratio'] = sum(c.isalpha() for c in text) / max(len(text), 1)
    features['blind_digit_ratio'] = sum(c.isdigit() for c in text) / max(len(text), 1)
    features['blind_upper_ratio'] = sum(c.isupper() for c in text) / max(len(text), 1)
    features['blind_space_ratio'] = sum(c.isspace() for c in text) / max(len(text), 1)
    features['blind_punct_ratio'] = sum(c in '.,;:!?-()[]{}"\'' for c in text) / max(len(text), 1)
    features['blind_newline_ratio'] = text.count('\n') / max(len(text), 1)

    # Unique character count
    features['blind_unique_chars'] = len(set(text))

    # Vocabulary richness (type-token ratio)
    words = text.lower().split()
    features['blind_type_token_ratio'] = len(set(words)) / max(len(words), 1)

    # Date/year detection features
    years = re.findall(r'\b(1[89]\d{2}|20[0-2]\d)\b', text)
    features['blind_has_year'] = 1.0 if years else 0.0
    features['blind_year_count'] = len(years)
    if years:
        int_years = [int(y) for y in years]
        features['blind_max_year'] = max(int_years)
        features['blind_min_year'] = min(int_years)
        features['blind_mean_year'] = np.mean(int_years)
    else:
        features['blind_max_year'] = 0.0
        features['blind_min_year'] = 0.0
        features['blind_mean_year'] = 0.0

    # Sentence-level features
    sentences = re.split(r'[.!?]+', text)
    sentences = [s.strip() for s in sentences if s.strip()]
    features['blind_sentence_count'] = len(sentences)
    if sentences:
        sent_lens = [len(s.split()) for s in sentences]
        features['blind_avg_sent_len'] = np.mean(sent_lens)
        features['blind_max_sent_len'] = max(sent_lens)
    else:
        features['blind_avg_sent_len'] = 0.0
        features['blind_max_sent_len'] = 0.0

    # Special token/pattern features
    features['blind_url_count'] = len(re.findall(r'https?://\S+', text))
    features['blind_email_count'] = len(re.findall(r'\S+@\S+\.\S+', text))
    features['blind_hashtag_count'] = len(re.findall(r'#\w+', text))
    features['blind_mention_count'] = len(re.findall(r'@\w+', text))

    # Repetition features
    words_lower = text.lower().split()
    if len(words_lower) > 1:
        repeated_pairs = sum(1 for i in range(len(words_lower) - 1) if words_lower[i] == words_lower[i + 1])
        features['blind_word_repeat_ratio'] = repeated_pairs / (len(words_lower) - 1)
    else:
        features['blind_word_repeat_ratio'] = 0.0

    return features


# ----------------------------------------------------------------------
# 1. Date Detection Attack
# ----------------------------------------------------------------------

class DateDetection:
    """Blind MIA based on detecting temporal shifts via year extraction."""

    def __init__(self, cutoff_year: int = 2020):
        self.cutoff_year = cutoff_year

    def predict(self, text: str) -> dict:
        """Return per-text date-based scores."""
        years = re.findall(r'\b(\d{4})\b', text)
        scores = {}
        if years:
            int_years = [int(y) for y in years if 1900 <= int(y) <= 2030]
            if int_years:
                max_year = max(int_years)
                # Score: 1.0 if all years before cutoff (likely member), 0.0 otherwise
                scores['date_detect_binary'] = 1.0 if max_year < self.cutoff_year else 0.0
                # Continuous score: how far below the cutoff
                scores['date_detect_continuous'] = float(self.cutoff_year - max_year)
            else:
                scores['date_detect_binary'] = 0.5
                scores['date_detect_continuous'] = 0.0
        else:
            scores['date_detect_binary'] = 0.5
            scores['date_detect_continuous'] = 0.0
        return scores

    def evaluate(self, texts, labels, fpr_budget=1.0):
        """Evaluate date detection on a full dataset. Returns (auc, tpr@fpr)."""
        y_pred_proba = []
        for text in texts:
            scores = self.predict(text)
            y_pred_proba.append(scores['date_detect_binary'])
        roc = get_roc_auc(labels, y_pred_proba)
        tpr = get_tpr_at_fpr(labels, y_pred_proba, fpr_budget)
        return roc, tpr


# ----------------------------------------------------------------------
# 2. Bag of Words Attack
# ----------------------------------------------------------------------

class BagOfWords:
    """Blind MIA using TF-IDF/Count vectorization + classifier ensemble."""

    def __init__(self, max_features=30, vectorizer_type='tf', model_type='stack'):
        self.max_features = max_features
        self.vectorizer_type = vectorizer_type
        self.model_type = model_type
        self.vectorizer = None
        self.model = None

    def _build_vectorizer(self):
        if self.vectorizer_type == 'tf':
            return TfidfVectorizer(max_features=self.max_features)
        elif self.vectorizer_type == 'count':
            return CountVectorizer(max_features=self.max_features)
        raise ValueError(f"Unknown vectorizer: {self.vectorizer_type}")

    def _build_model(self):
        if self.model_type == 'random':
            return RandomForestClassifier(n_estimators=50, max_depth=5, random_state=42)
        elif self.model_type == 'gaussian':
            return GaussianNB()
        elif self.model_type == 'multi':
            return MultinomialNB()
        elif self.model_type == 'gradient':
            return GradientBoostingClassifier(n_estimators=50, learning_rate=1.0, max_depth=1, random_state=42)
        elif self.model_type == 'stack':
            estimators = [
                ('random', RandomForestClassifier(n_estimators=50, max_depth=5, random_state=42)),
                ('gaussian', GaussianNB()),
                ('multi', MultinomialNB()),
                ('gradient', GradientBoostingClassifier(n_estimators=50, learning_rate=1.0, max_depth=5, random_state=42))
            ]
            return StackingClassifier(estimators=estimators, final_estimator=LogisticRegression(random_state=42))
        raise ValueError(f"Unknown model type: {self.model_type}")

    def fit(self, X_train, y_train):
        """Fit vectorizer + classifier on training texts."""
        self.vectorizer = self._build_vectorizer()
        X_vec = self.vectorizer.fit_transform(X_train).toarray()
        self.model = self._build_model()
        self.model.fit(X_vec, y_train)

    def predict_proba(self, texts):
        """Return P(member) for each text."""
        X_vec = self.vectorizer.transform(texts).toarray()
        return self.model.predict_proba(X_vec)[:, 1]

    def predict_single(self, text: str) -> float:
        """Return P(member) for a single text."""
        return float(self.predict_proba([text])[0])

    def evaluate(self, X, y, trials=10, fpr_budget=1.0):
        """Cross-validated evaluation. Returns (mean_auc, std_auc, mean_tpr, std_tpr)."""
        auc_scores = []
        tpr_scores = []
        for _ in range(trials):
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=None)
            self.fit(X_train, y_train)
            y_proba = self.predict_proba(X_test)
            auc_scores.append(get_roc_auc(y_test, y_proba))
            tpr_scores.append(get_tpr_at_fpr(y_test, y_proba, fpr_budget))
        return mean(auc_scores), pstdev(auc_scores), mean(tpr_scores), pstdev(tpr_scores)

    @staticmethod
    def hyperparam_search(X, y, fpr_budget=1.0):
        """Search over vectorizer type, max_features, and model type."""
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=None)
        best_score = -1
        best_params = {}
        for max_features in [10, 15, 20, 30, 34, 38, 40, 50, 54, 58, 62]:
            for vec_type in ['tf', 'count']:
                for model_type in ['multi', 'gaussian', 'random', 'gradient', 'stack']:
                    bow = BagOfWords(max_features=max_features, vectorizer_type=vec_type, model_type=model_type)
                    bow.fit(X_train, y_train)
                    y_proba = bow.predict_proba(X_test)
                    score = get_roc_auc(y_test, y_proba)
                    if score > best_score:
                        best_score = score
                        best_params = {'max_features': max_features, 'vectorizer': vec_type, 'model_type': model_type}
        return best_params, best_score


# ----------------------------------------------------------------------
# 3. Greedy N-gram Selection Attack
# ----------------------------------------------------------------------

class GreedyNgramSelection:
    """Blind MIA using greedy selection of discriminative character n-grams."""

    def __init__(self, n=5, threshold=1):
        self.n = n
        self.threshold = threshold
        self.selected_ngrams = []

    def fit(self, member_texts, nonmember_texts, budget_ratio=1.0):
        """Greedily select n-grams that distinguish members from non-members.

        Args:
            member_texts: list of member text strings
            nonmember_texts: list of non-member text strings
            budget_ratio: fraction of non-members allowed as false positives
        """
        member_counts = count_unique_ngrams(member_texts, n=self.n, threshold=self.threshold)
        nonmember_counts = count_unique_ngrams(nonmember_texts, n=self.n)

        budget = int(budget_ratio * len(nonmember_texts))
        curr_fpr = 0
        self.selected_ngrams = []

        while curr_fpr < budget:
            candidates = [c for c in member_counts.keys()
                          if len(nonmember_counts[c]) <= budget - curr_fpr]
            if not candidates:
                break

            ratios = [(len(member_counts[c])) / (len(nonmember_counts[c]) + 1)
                      for c in candidates]
            best_idx = np.argmax(ratios)
            chosen = candidates[best_idx]
            self.selected_ngrams.append(chosen)

            curr_fpr += len(nonmember_counts[chosen])

            # Remove covered samples from counts
            covered_members = member_counts[chosen].copy()
            for c in list(member_counts.keys()):
                member_counts[c] -= covered_members
                if len(member_counts[c]) == 0:
                    del member_counts[c]

            covered_nonmembers = nonmember_counts[chosen].copy()
            for c in list(nonmember_counts.keys()):
                nonmember_counts[c] -= covered_nonmembers
                if len(nonmember_counts[c]) == 0:
                    del nonmember_counts[c]

    def predict(self, text: str) -> dict:
        """Return n-gram match features for a single text."""
        text_ngrams = ngrams(text, n=self.n)
        matches = text_ngrams.intersection(self.selected_ngrams)
        scores = {}
        scores['greedy_ngram_match_count'] = len(matches)
        scores['greedy_ngram_match_ratio'] = len(matches) / max(len(self.selected_ngrams), 1)
        scores['greedy_ngram_has_match'] = 1.0 if matches else 0.0
        return scores

    def predict_binary(self, text: str) -> float:
        """Return 1.0 if text matches any selected n-gram, else 0.0."""
        text_ngrams = ngrams(text, n=self.n)
        return 1.0 if text_ngrams.intersection(self.selected_ngrams) else 0.0

    def evaluate(self, member_texts, nonmember_texts, cross_vals=10, test_ratio=0.1, fpr_budget=1.0):
        """Cross-validated evaluation. Returns (mean_tpr_at_fpr, std)."""
        tprs = []
        for _ in range(cross_vals):
            perm_m = np.random.permutation(member_texts)
            perm_nm = np.random.permutation(nonmember_texts)
            n_test = int(test_ratio * len(member_texts))

            m_test, m_train = perm_m[:n_test], perm_m[n_test:]
            nm_test, nm_train = perm_nm[:n_test], perm_nm[n_test:]

            self.fit(list(m_train), list(nm_train))

            y_true = [1] * len(m_test) + [0] * len(nm_test)
            y_proba = [self.predict_binary(t) for t in m_test] + \
                      [self.predict_binary(t) for t in nm_test]
            tprs.append(get_tpr_at_fpr(y_true, y_proba, fpr_budget))

        return mean(tprs), pstdev(tprs)
