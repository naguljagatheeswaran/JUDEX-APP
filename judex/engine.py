"""Fused complaint-classification and BNS section analysis."""

from __future__ import annotations

from dataclasses import dataclass

from judex.classifiers import CLASSIFIERS, TopicClassifier
from judex.matcher import BNSMatcher, Match


@dataclass(frozen=True)
class Analysis:
    classifier_probabilities: dict[str, float]
    section_matches: list[Match]


class JudexEngine:
    """One pipeline: category classifier first, BNS ranking second."""

    def __init__(
        self,
        matcher: BNSMatcher,
        approved_classifier_examples: tuple[tuple[str, str], ...] = (),
    ) -> None:
        self.matcher = matcher
        self.topic_classifier = TopicClassifier(approved_classifier_examples)
        self.classifier_sections = {
            item.name: item.section_numbers for item in CLASSIFIERS
        }

    def analyse(self, text: str) -> Analysis:
        probabilities = self.topic_classifier.probabilities(text)
        matches = self.matcher.rank(
            text,
            classifier_probabilities=probabilities,
            classifier_sections=self.classifier_sections,
        )
        return Analysis(probabilities, matches)
