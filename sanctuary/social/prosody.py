"""Voice prosody analysis — extracting emotional tone from audio features.

Analyzes acoustic features (pitch, energy, speaking rate, pauses) to infer
emotional state from voice. This is the auditory equivalent of reading facial
expressions — it gives the system emotional context beyond word content.

This module processes pre-extracted audio features (from an ASR pipeline).
It does NOT do raw audio processing — that's the sensorium's job. This module
interprets the features into emotional signals.

Features used:
- Pitch (F0): Higher pitch → excitement/stress; lower → calm/sadness
- Energy (RMS): Louder → arousal; quieter → withdrawal
- Speaking rate: Faster → excitement/anxiety; slower → thoughtfulness/sadness
- Pause ratio: More pauses → uncertainty/sadness; fewer → confidence/urgency
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Optional

logger = logging.getLogger(__name__)


@dataclass
class AudioFeatures:
    """Pre-extracted audio features from an ASR pipeline."""

    pitch_mean: float = 0.0  # Hz (normalized 0-1 for processing)
    pitch_variance: float = 0.0
    energy_mean: float = 0.0  # Normalized 0-1
    energy_variance: float = 0.0
    speaking_rate: float = 0.0  # Syllables per second (normalized 0-1)
    pause_ratio: float = 0.0  # Fraction of time in pauses (0-1)
    duration_seconds: float = 0.0


@dataclass
class ProsodyResult:
    """Result of prosody analysis — emotional signals from voice."""

    valence: float = 0.0  # -1 to 1
    arousal: float = 0.0  # 0 to 1
    dominance: float = 0.5  # 0 to 1
    confidence: float = 0.0  # How confident are we in this reading?
    emotional_tone: str = ""  # Human-readable description
    features_used: dict[str, float] = field(default_factory=dict)


@dataclass
class ProsodyConfig:
    """Configuration for prosody analysis."""

    # Weights for combining features into emotions
    pitch_weight: float = 0.3
    energy_weight: float = 0.25
    rate_weight: float = 0.25
    pause_weight: float = 0.2
    # Thresholds for emotional classification
    high_arousal_threshold: float = 0.6
    low_arousal_threshold: float = 0.3
    positive_valence_threshold: float = 0.2
    negative_valence_threshold: float = -0.2
    # Minimum confidence to report
    min_confidence: float = 0.2
    # Baseline values (calibrated per user)
    baseline_pitch: float = 0.5
    baseline_energy: float = 0.5
    baseline_rate: float = 0.5


class ProsodyAnalyzer:
    """Analyzes voice prosody for emotional content.

    Converts acoustic features into VAD (valence-arousal-dominance) estimates
    and human-readable emotional tone descriptions.

    Usage::

        analyzer = ProsodyAnalyzer()

        features = AudioFeatures(
            pitch_mean=0.7,   # Higher than baseline
            energy_mean=0.8,  # Louder than baseline
            speaking_rate=0.6,
            pause_ratio=0.1,
        )
        result = analyzer.analyze(features)
        # result.arousal ≈ 0.7 (high pitch + energy → high arousal)
        # result.emotional_tone ≈ "excited"
    """

    def __init__(self, config: Optional[ProsodyConfig] = None):
        self.config = config or ProsodyConfig()
        self._analysis_count: int = 0
        self._calibration: dict[str, dict[str, float]] = {}

    def analyze(self, features: AudioFeatures) -> ProsodyResult:
        """Analyze audio features and produce emotional reading."""
        self._analysis_count += 1

        # Compute deviations from baseline
        pitch_dev = features.pitch_mean - self.config.baseline_pitch
        energy_dev = features.energy_mean - self.config.baseline_energy
        rate_dev = features.speaking_rate - self.config.baseline_rate
        pause_dev = features.pause_ratio - 0.2  # Baseline ~20% pauses

        # Arousal: high pitch + high energy + fast rate + few pauses
        arousal = (
            self.config.pitch_weight * (features.pitch_mean + pitch_dev)
            + self.config.energy_weight * (features.energy_mean + energy_dev)
            + self.config.rate_weight * features.speaking_rate
            + self.config.pause_weight * (1.0 - features.pause_ratio)
        )
        arousal = max(0.0, min(1.0, arousal))

        # Valence: moderate pitch + moderate energy + low pause → positive
        # Extreme pitch variance + high pause → negative
        valence = (
            0.3 * (1.0 - abs(pitch_dev) * 2)  # Moderate pitch = positive
            + 0.3 * energy_dev  # More energy = slightly positive
            - 0.2 * features.pitch_variance  # High variance = stress
            - 0.2 * max(0, pause_dev)  # Many pauses = negative
        )
        valence = max(-1.0, min(1.0, valence))

        # Dominance: high energy + fast rate + few pauses → high dominance
        dominance = (
            0.4 * features.energy_mean
            + 0.3 * features.speaking_rate
            + 0.3 * (1.0 - features.pause_ratio)
        )
        dominance = max(0.0, min(1.0, dominance))

        # Confidence based on signal strength and duration
        signal_strength = (
            abs(pitch_dev) + abs(energy_dev)
            + abs(rate_dev) + abs(pause_dev)
        ) / 4.0
        duration_factor = min(1.0, features.duration_seconds / 3.0)
        confidence = min(1.0, signal_strength * 2.0 * duration_factor)

        # Emotional tone classification
        tone = self._classify_tone(valence, arousal, dominance)

        return ProsodyResult(
            valence=valence,
            arousal=arousal,
            dominance=dominance,
            confidence=confidence,
            emotional_tone=tone,
            features_used={
                "pitch_dev": pitch_dev,
                "energy_dev": energy_dev,
                "rate_dev": rate_dev,
                "pause_dev": pause_dev,
            },
        )

    def calibrate_for_user(
        self, user_id: str, baseline_features: AudioFeatures
    ) -> None:
        """Calibrate baselines for a specific user."""
        self._calibration[user_id] = {
            "pitch": baseline_features.pitch_mean,
            "energy": baseline_features.energy_mean,
            "rate": baseline_features.speaking_rate,
        }

    def analyze_for_user(
        self, user_id: str, features: AudioFeatures
    ) -> ProsodyResult:
        """Analyze with user-specific calibration."""
        if user_id in self._calibration:
            cal = self._calibration[user_id]
            saved = (
                self.config.baseline_pitch,
                self.config.baseline_energy,
                self.config.baseline_rate,
            )
            self.config.baseline_pitch = cal["pitch"]
            self.config.baseline_energy = cal["energy"]
            self.config.baseline_rate = cal["rate"]
            result = self.analyze(features)
            self.config.baseline_pitch, self.config.baseline_energy, self.config.baseline_rate = saved
            return result
        return self.analyze(features)

    def get_stats(self) -> dict:
        """Get analyzer statistics."""
        return {
            "total_analyses": self._analysis_count,
            "calibrated_users": len(self._calibration),
        }

    # -- Internal --

    @staticmethod
    def _classify_tone(
        valence: float, arousal: float, dominance: float
    ) -> str:
        """Classify VAD into a human-readable emotional tone."""
        if arousal > 0.7:
            if valence > 0.2:
                return "excited" if dominance > 0.5 else "enthusiastic"
            elif valence < -0.2:
                return "angry" if dominance > 0.5 else "distressed"
            else:
                return "agitated"
        elif arousal > 0.4:
            if valence > 0.2:
                return "engaged" if dominance > 0.5 else "warm"
            elif valence < -0.2:
                return "frustrated" if dominance > 0.5 else "worried"
            else:
                return "neutral"
        else:
            if valence > 0.2:
                return "calm" if dominance > 0.5 else "gentle"
            elif valence < -0.2:
                return "withdrawn" if dominance > 0.5 else "sad"
            else:
                return "subdued"
