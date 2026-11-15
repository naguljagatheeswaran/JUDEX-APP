"""Conservative hybrid BNS section matcher.

The matcher treats the CSV as the catalogue of record and uses three kinds
of evidence: high-signal fact-pattern rules, TF-IDF retrieval, and approved
human feedback. Generic seed sentences are intentionally NOT used as legal
training examples because they create false similarity between unrelated
sections.
"""
from __future__ import annotations

import math
import numpy as np
import re
from dataclasses import dataclass
from typing import Iterable, Mapping

from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS, TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from .catalogue import BNSSection, SECTIONS
from .scenario_rules import RULES


@dataclass(frozen=True)
class Match:
    section: BNSSection
    confidence: int
    keyword_score: float
    semantic_score: float
    learned_score: float
    classifier_score: float
    matched_terms: tuple[str, ...]


def normalise(text: str) -> str:
    text = text.lower().replace("’", "'")
    text = re.sub(r"[^a-z0-9\s'-]", " ", text)
    return re.sub(r"\s+", " ", text).strip()


_GENERIC_KEYWORDS = set(ENGLISH_STOP_WORDS) | {
    "believe", "think", "someone", "person", "people", "incident", "case",
    "matter", "reported", "report", "complaint", "allegation", "alleged",
    "using", "use", "away", "house", "home", "bank", "money", "harm",
}

def _token_set(text: str) -> set[str]:
    return set(re.findall(r"[a-z0-9]+", normalise(text)))


class BNSMatcher:
    """Rank BNS sections without pretending every complaint has a match."""

    def __init__(
        self,
        sections: Iterable[BNSSection] = SECTIONS,
        approved_examples: Iterable[tuple[str, int]] = (),
        curated_examples: Iterable[tuple[str, int]] = (),
    ) -> None:
        self.sections = tuple(sections)
        self._by_number = {section.number: section for section in self.sections}
        self._curated_examples = [
            (normalise(text), int(label))
            for text, label in curated_examples
            if int(label) in self._by_number and normalise(text)
        ]
        self._approved_examples = [
            (normalise(text), int(label))
            for text, label in approved_examples
            if int(label) in self._by_number and normalise(text)
        ]

        self._word_vectorizer = TfidfVectorizer(
            lowercase=True,
            strip_accents="unicode",
            stop_words="english",
            ngram_range=(1, 3),
            min_df=1,
            sublinear_tf=True,
        )
        self._char_vectorizer = TfidfVectorizer(
            lowercase=True,
            strip_accents="unicode",
            analyzer="char_wb",
            ngram_range=(3, 5),
            min_df=1,
            sublinear_tf=True,
        )

        self._profiles = [self._profile_text(s) for s in self.sections]
        self._word_profiles = self._word_vectorizer.fit_transform(self._profiles)
        self._char_profiles = self._char_vectorizer.fit_transform(self._profiles)

        if self._approved_examples:
            texts = [text for text, _ in self._approved_examples]
            raw_word = self._word_vectorizer.transform(texts)
            raw_char = self._char_vectorizer.transform(texts)
            self._approved_labels = sorted({label for _, label in self._approved_examples})
            self._approved_word = np.vstack([
                np.asarray(raw_word[[i for i, (_, lab) in enumerate(self._approved_examples) if lab == label]].mean(axis=0)).ravel()
                for label in self._approved_labels
            ])
            self._approved_char = np.vstack([
                np.asarray(raw_char[[i for i, (_, lab) in enumerate(self._approved_examples) if lab == label]].mean(axis=0)).ravel()
                for label in self._approved_labels
            ])
        else:
            self._approved_word = None
            self._approved_char = None
            self._approved_labels = []

        if self._curated_examples:
            texts = [text for text, _ in self._curated_examples]
            raw_word = self._word_vectorizer.transform(texts)
            raw_char = self._char_vectorizer.transform(texts)
            # One centroid per section keeps inference fast while retaining
            # the language patterns learned from the curated complaint set.
            self._curated_labels = sorted({label for _, label in self._curated_examples})
            self._curated_word = np.vstack([
                np.asarray(raw_word[[i for i, (_, lab) in enumerate(self._curated_examples) if lab == label]].mean(axis=0)).ravel()
                for label in self._curated_labels
            ])
            self._curated_char = np.vstack([
                np.asarray(raw_char[[i for i, (_, lab) in enumerate(self._curated_examples) if lab == label]].mean(axis=0)).ravel()
                for label in self._curated_labels
            ])
        else:
            self._curated_word = None
            self._curated_char = None
            self._curated_labels = []

    def _profile_text(self, section: BNSSection) -> str:
        aliases = " ".join(term for term, _weight in RULES.get(section.number, ()))
        weighted_keywords = " ".join(
            term for term, weight in section.keywords.items() for _ in range(max(1, round(weight / 2)))
        )
        return " ".join((section.title, section.category, weighted_keywords, aliases))

    def _rule_evidence(self, text: str, section: BNSSection) -> tuple[float, tuple[str, ...]]:
        evidence: list[str] = []
        score = 0.0

        # An exact match to the official section heading is a very strong
        # retrieval signal. This is especially useful for headings such as
        # "Theft" or "Punishment for rape", where generic semantic
        # similarity can otherwise favour a neighbouring provision.
        title = normalise(section.title)
        if title and len(title.split()) >= 1:
            if re.search(r"(?<!\w)" + re.escape(title) + r"(?!\w)", text):
                evidence.append(f"section title: {section.title}")
                score += 10.0 if len(title.split()) >= 2 else 8.0

        for phrase, weight in RULES.get(section.number, ()):
            if re.search(r"(?<!\w)" + re.escape(normalise(phrase)) + r"(?!\w)", text):
                evidence.append(phrase)
                score += weight
        # Generic catalogue keywords are useful, but cap them heavily so a
        # single word such as "money" cannot dominate a fact pattern.
        for term, weight in section.keywords.items():
            clean = normalise(term)
            if not clean or (len(clean.split()) == 1 and (len(clean) <= 3 or clean in _GENERIC_KEYWORDS)):
                continue
            if re.search(r"(?<!\w)" + re.escape(clean) + r"(?!\w)", text):
                evidence.append(term)
                score += min(float(weight), 3.0) * (1.2 if len(clean.split()) > 1 else 0.65)
        value = min(1.0, 1.0 - math.exp(-score / 6.0))
        # Context penalties stop broad offences from beating more specific
        # fact patterns when both share a generic word.
        if section.number == 303 and re.search(r"\b(?:house|home|room|dwelling)\b", text) and re.search(r"\b(?:entered|broke|inside)\b", text):
            value *= 0.58
        if section.number == 303 and re.search(r"\b(?:snatched|snatching|grabbed)\b", text):
            value *= 0.45
        if section.number == 303 and re.search(r"\b(?:entrusted|employee|servant|clerk)\b", text):
            value *= 0.45
        if section.number == 115 and re.search(r"\b(?:fracture|fractured|broke my (?:arm|leg|nose|bone))\b", text):
            value *= 0.55
        if section.number == 318 and re.search(r"\b(?:pretend(?:ed|ing)?|impersonat(?:ed|ing)?|fake (?:bank|instagram|profile|account))\b", text):
            value *= 0.55
        if section.number == 137 and re.search(r"\b(?:ransom|release|kidnapped.*demanded money)\b", text):
            value *= 0.45
        if section.number == 227 and re.search(r"\bfabricated\b", text):
            value *= 0.55
        return value, tuple(dict.fromkeys(evidence))

    def _approved_similarity(self, q_word, q_char) -> dict[int, float]:
        if self._approved_word is None:
            return {}
        word = cosine_similarity(q_word, self._approved_word).ravel()
        char = cosine_similarity(q_char, self._approved_char).ravel()
        return {label: 0.65 * float(word[i]) + 0.35 * float(char[i])
                for i, label in enumerate(self._approved_labels)}

    def _curated_similarity(self, q_word, q_char) -> dict[int, float]:
        if self._curated_word is None:
            return {}
        word = cosine_similarity(q_word, self._curated_word).ravel()
        char = cosine_similarity(q_char, self._curated_char).ravel()
        return {label: 0.62 * float(word[i]) + 0.38 * float(char[i])
                for i, label in enumerate(self._curated_labels)}

    @staticmethod
    def _extract_explicit_section(text: str) -> int | None:
        match = re.search(r"\b(?:bns\s+)?section\s+(\d{1,3})\b", text)
        return int(match.group(1)) if match else None

    def rank(
        self,
        input_text: str,
        limit: int = 3,
        classifier_probabilities: Mapping[str, float] | None = None,
        classifier_sections: Mapping[str, tuple[int, ...]] | None = None,
    ) -> list[Match]:
        clean_text = normalise(input_text)
        if not clean_text:
            return []

        explicit = self._extract_explicit_section(clean_text)
        # Vectorize the query once. Older versions transformed the same text
        # up to three times during one prediction.
        q_word = self._word_vectorizer.transform([clean_text])
        q_char = self._char_vectorizer.transform([clean_text])
        approved = self._approved_similarity(q_word, q_char)
        curated = self._curated_similarity(q_word, q_char)
        word_scores = cosine_similarity(q_word, self._word_profiles).ravel()
        char_scores = cosine_similarity(q_char, self._char_profiles).ravel()
        semantic_scores = 0.62 * word_scores + 0.38 * char_scores

        matches: list[Match] = []
        for i, section in enumerate(self.sections):
            keyword_score, evidence = self._rule_evidence(clean_text, section)
            semantic = float(max(0.0, semantic_scores[i]))
            learned = max(float(approved.get(section.number, 0.0)), float(curated.get(section.number, 0.0)))

            classifier_score = 0.0
            if classifier_probabilities and classifier_sections:
                classifier_score = max(
                    (
                        probability
                        for name, probability in classifier_probabilities.items()
                        if section.number in classifier_sections.get(name, ())
                    ),
                    default=0.0,
                )

            # Rules are strongest because they encode distinctive facts.
            # The first-stage complaint classifier is a useful routing signal.
            # Give it enough weight to lift the correct legal family when the
            # complaint wording is natural but does not contain the exact
            # catalogue phrase.
            raw = (
                0.54 * keyword_score
                + 0.18 * semantic
                + 0.16 * learned
                + 0.12 * classifier_score
            )

            if classifier_score >= 0.55:
                raw += 0.10
            elif classifier_score >= 0.35:
                raw += 0.05

            if learned >= 0.45 and keyword_score < 0.30:
                raw += 0.06

            # A strong agreement between the complaint classifier and the
            # curated complaint retriever is meaningful even when the wording
            # does not contain the exact legal keyword.
            if classifier_score >= 0.55 and learned >= 0.45:
                raw = max(raw, 0.56)
            elif classifier_score >= 0.55 and semantic >= 0.16:
                raw = max(raw, 0.51)

            # Strong multi-word fact patterns deserve more confidence than
            # generic lexical overlap. This prevents obvious complaints from
            # being rejected merely because two legally related sections are
            # close together.
            phrase_hits = sum(1 for term in evidence if len(normalise(term).split()) >= 2)
            if phrase_hits >= 1 and keyword_score >= 0.55:
                raw += 0.12
            if phrase_hits >= 2 and keyword_score >= 0.65:
                raw += 0.08

            if explicit == section.number:
                raw = max(raw, 0.97)
                evidence = tuple(dict.fromkeys((f"explicit section {section.number}",) + evidence))

            matches.append(
                Match(
                    section=section,
                    confidence=round(min(0.99, raw) * 100),
                    keyword_score=keyword_score,
                    semantic_score=semantic,
                    learned_score=learned,
                    classifier_score=classifier_score,
                    matched_terms=evidence,
                )
            )

        matches.sort(key=lambda m: (m.confidence, m.keyword_score, m.semantic_score), reverse=True)
        return matches[:limit]


    def rank_many(self, input_texts: Iterable[str], limit: int = 3) -> list[list[Match]]:
        """Fast batch profile ranking for the always-on regression suite.

        The normal :meth:`rank` method remains the production path. This batch
        helper verifies that each BNS section remains retrievable from unseen
        test language without repeatedly rebuilding/scanning every rule.
        """
        texts = [normalise(t) for t in input_texts]
        if not texts:
            return []
        q_word = self._word_vectorizer.transform(texts)
        semantic_matrix = cosine_similarity(q_word, self._word_profiles)
        outputs: list[list[Match]] = []
        for row_i, clean_text in enumerate(texts):
            if not clean_text:
                outputs.append([]); continue
            order = list(np.argsort(-semantic_matrix[row_i]))
            # Honor an explicit section number exactly as the production ranker does.
            explicit = self._extract_explicit_section(clean_text)
            if explicit in self._by_number:
                explicit_idx = next(i for i, s in enumerate(self.sections) if s.number == explicit)
                order = [explicit_idx] + [i for i in order if int(i) != explicit_idx]
            order = order[:limit]
            row_matches = []
            for sec_i in order:
                section = self.sections[int(sec_i)]
                semantic = float(max(0.0, semantic_matrix[row_i, sec_i]))
                row_matches.append(Match(
                    section=section,
                    confidence=round(min(0.99, semantic) * 100),
                    keyword_score=0.0,
                    semantic_score=semantic,
                    learned_score=0.0,
                    classifier_score=0.0,
                    matched_terms=(),
                ))
            outputs.append(row_matches)
        return outputs

    @staticmethod
    def is_confident(match: Match | None, runner_up: Match | None = None) -> bool:
        if match is None:
            return False
        # Require either strong fact-pattern evidence or a very strong semantic
        # match. A small top/second margin is treated as ambiguous.
        if match.confidence < 50:
            return False

        classifier_backed = match.classifier_score >= 0.55 and (
            match.learned_score >= 0.30 or match.semantic_score >= 0.16
        )
        learned_backed = match.learned_score >= 0.45 and match.semantic_score >= 0.12

        if (
            match.keyword_score < 0.30
            and match.semantic_score < 0.45
            and match.learned_score < 0.70
            and not classifier_backed
            and not learned_backed
        ):
            return False

        # If the top result contains a distinctive fact phrase, a close runner-up
        # is often a related provision rather than evidence that the complaint
        # is unusable. Keep the ambiguity guard for weak/generic matches.
        distinctive = (
            match.keyword_score >= 0.55
            and len(match.matched_terms) >= 1
            and any(len(normalise(term).split()) >= 2 for term in match.matched_terms)
        )
        if runner_up is not None and match.confidence - runner_up.confidence < 7:
            if not distinctive and not classifier_backed and not learned_backed and match.keyword_score < 0.70:
                return False
        return distinctive or classifier_backed or learned_backed or match.keyword_score >= 0.70 or match.confidence >= 70
