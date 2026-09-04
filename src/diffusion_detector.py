"""
Diffusion Models for Covert Communication Detection in Distributed Systems

Paper: Diffusion Models for Covert Communication Detection in
       Distributed Systems
Venue: CCIOT 2026
Authors: Fernando May et al.
"""

import numpy as np
from scipy.special import expit as sigmoid
from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional
import random
import time
from enum import Enum


class TrafficType(Enum):
    NORMAL = 0
    COVERT = 1
    ANOMALOUS = 2


@dataclass
class NetworkPacket:
    packet_id: int
    source_ip: str
    dest_ip: str
    payload_size: int
    timestamp: float
    protocol: int
    flags: int
    traffic_type: TrafficType = TrafficType.NORMAL

    def to_feature_vector(self) -> np.ndarray:
        return np.array([
            self.payload_size / 1500.0,
            self.protocol / 255.0,
            self.flags / 255.0,
            self.timestamp % 1.0,
            1.0 if self.traffic_type == TrafficType.COVERT else 0.0
        ], dtype=np.float32)


@dataclass
class TrafficProfile:
    normal_mean_payload: float = 500.0
    normal_std_payload: float = 200.0
    covert_mean_payload: float = 100.0
    covert_std_payload: float = 50.0
    normal_rate: float = 100.0
    covert_rate: float = 5.0


class TrafficGenerator:
    """Generates normal and covert network traffic."""

    def __init__(self, profile: TrafficProfile):
        self.profile = profile
        self.packet_counter = 0

    def generate_normal_traffic(self, num_packets: int,
                                 start_time: float = 0.0) -> List[NetworkPacket]:
        packets = []
        for i in range(num_packets):
            payload = max(64, int(np.random.normal(
                self.profile.normal_mean_payload,
                self.profile.normal_std_payload
            )))
            timestamp = start_time + np.random.exponential(
                1.0 / self.profile.normal_rate
            )
            packets.append(NetworkPacket(
                packet_id=self.packet_counter,
                source_ip=f"192.168.{random.randint(1,255)}.{random.randint(1,255)}",
                dest_ip=f"10.0.{random.randint(1,255)}.{random.randint(1,255)}",
                payload_size=payload,
                timestamp=timestamp,
                protocol=random.choice([6, 17, 1]),
                flags=random.randint(0, 255),
                traffic_type=TrafficType.NORMAL
            ))
            self.packet_counter += 1
        return packets

    def generate_covert_traffic(self, num_packets: int,
                                 start_time: float = 0.0) -> List[NetworkPacket]:
        packets = []
        for i in range(num_packets):
            payload = max(64, int(np.random.normal(
                self.profile.covert_mean_payload,
                self.profile.covert_std_payload
            )))
            timestamp = start_time + np.random.exponential(
                1.0 / self.profile.covert_rate
            )
            packets.append(NetworkPacket(
                packet_id=self.packet_counter,
                source_ip=f"172.16.{random.randint(1,255)}.{random.randint(1,255)}",
                dest_ip=f"10.0.{random.randint(1,255)}.{random.randint(1,255)}",
                payload_size=payload,
                timestamp=timestamp,
                protocol=6,
                flags=random.randint(0, 15),
                traffic_type=TrafficType.COVERT
            ))
            self.packet_counter += 1
        return packets

    def generate_mixed_traffic(self, num_normal: int = 950,
                                num_covert: int = 50) -> List[NetworkPacket]:
        normal = self.generate_normal_traffic(num_normal)
        covert = self.generate_covert_traffic(num_covert, start_time=10.0)
        mixed = normal + covert
        random.shuffle(mixed)
        return mixed


class SimpleDiffusionModel:
    """Simplified diffusion model for traffic anomaly detection."""

    def __init__(self, input_dim: int = 5, hidden_dim: int = 32,
                 num_steps: int = 10, learning_rate: float = 0.001):
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_steps = num_steps
        self.learning_rate = learning_rate

        self.encoder_weights = np.random.randn(
            input_dim, hidden_dim
        ) * 0.01
        self.decoder_weights = np.random.randn(
            hidden_dim, input_dim
        ) * 0.01
        self.noise_schedule = np.linspace(0.0001, 0.02, num_steps)

    def _encode(self, x: np.ndarray) -> np.ndarray:
        return np.tanh(x @ self.encoder_weights)

    def _decode(self, z: np.ndarray) -> np.ndarray:
        return sigmoid(z @ self.decoder_weights)

    def _add_noise(self, x: np.ndarray, noise_level: float) -> np.ndarray:
        noise = np.random.randn(*x.shape) * noise_level
        return x + noise

    def train_on_normal(self, normal_data: np.ndarray,
                        epochs: int = 100) -> List[float]:
        losses = []
        for epoch in range(epochs):
            total_loss = 0
            for x in normal_data:
                z = self._encode(x)
                x_reconstructed = self._decode(z)
                loss = np.mean((x - x_reconstructed) ** 2)
                total_loss += loss

                grad = 2 * (x_reconstructed - x)
                self.decoder_weights -= self.learning_rate * np.outer(z, grad)
                self.encoder_weights -= self.learning_rate * np.outer(
                    x, grad @ self.decoder_weights.T * (1 - z**2)
                )

            losses.append(total_loss / len(normal_data))
        return losses

    def compute_anomaly_score(self, x: np.ndarray) -> float:
        z = self._encode(x)
        x_reconstructed = self._decode(z)
        reconstruction_error = np.mean((x - x_reconstructed) ** 2)
        return reconstruction_error


class AdaptiveThresholdDetector:
    """Detects anomalies using adaptive thresholding."""

    def __init__(self, initial_threshold: float = 0.1,
                 adaptation_rate: float = 0.01):
        self.threshold = initial_threshold
        self.adaptation_rate = adaptation_rate
        self.score_history: List[float] = []

    def update(self, score: float, is_normal: bool):
        self.score_history.append(score)
        if len(self.score_history) > 1000:
            self.score_history = self.score_history[-1000:]

        if is_normal:
            target = np.mean(self.score_history) + 2 * np.std(self.score_history)
            self.threshold += self.adaptation_rate * (target - self.threshold)

    def detect(self, score: float) -> bool:
        return score > self.threshold


class RLEnhancedDetector:
    """Reinforcement Learning enhanced threshold optimization."""

    def __init__(self, state_dim: int = 3, action_dim: int = 10):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.q_table = np.zeros((100, action_dim))
        self.thresholds = np.linspace(0.01, 0.5, action_dim)
        self.learning_rate = 0.1
        self.discount_factor = 0.95
        self.epsilon = 0.1

    def _discretize_state(self, score_history: List[float]) -> int:
        if len(score_history) < 2:
            return 50
        mean_score = np.mean(score_history[-100:])
        state = int(np.clip(mean_score * 200, 0, 99))
        return state

    def select_action(self, score_history: List[float]) -> int:
        if random.random() < self.epsilon:
            return random.randint(0, self.action_dim - 1)
        state = self._discretize_state(score_history)
        return np.argmax(self.q_table[state])

    def update(self, state: int, action: int, reward: float,
               next_state: int):
        best_next = np.max(self.q_table[next_state])
        td_target = reward + self.discount_factor * best_next
        td_error = td_target - self.q_table[state, action]
        self.q_table[state, action] += self.learning_rate * td_error

    def get_threshold(self, action: int) -> float:
        return self.thresholds[action]


class DiffusionDetector:
    """Main diffusion-based covert communication detector."""

    def __init__(self):
        self.diffusion_model = SimpleDiffusionModel()
        self.threshold_detector = AdaptiveThresholdDetector()
        self.rl_detector = RLEnhancedDetector()
        self.training_losses: List[float] = []

    def train(self, normal_traffic: List[NetworkPacket],
              epochs: int = 100):
        features = np.array([p.to_feature_vector() for p in normal_traffic])
        self.training_losses = self.diffusion_model.train_on_normal(
            features, epochs
        )

    def detect(self, packet: NetworkPacket) -> Dict:
        features = packet.to_feature_vector()
        score = self.diffusion_model.compute_anomaly_score(features)

        rl_state = self.rl_detector._discretize_state(
            self.threshold_detector.score_history
        )
        rl_action = self.rl_detector.select_action(
            self.threshold_detector.score_history
        )
        rl_threshold = self.rl_detector.get_threshold(rl_action)

        is_covert_by_threshold = self.threshold_detector.detect(score)
        is_covert_by_rl = score > rl_threshold
        is_covert = is_covert_by_threshold or is_covert_by_rl

        self.threshold_detector.update(score, not is_covert)

        if is_covert:
            reward = 1.0
        else:
            reward = -0.1 if packet.traffic_type == TrafficType.COVERT else 0.5

        next_state = self.rl_detector._discretize_state(
            self.threshold_detector.score_history
        )
        self.rl_detector.update(rl_state, rl_action, reward, next_state)

        return {
            "score": score,
            "is_covert": is_covert,
            "threshold": self.threshold_detector.threshold,
            "rl_threshold": rl_threshold,
            "true_type": packet.traffic_type
        }

    def evaluate(self, test_packets: List[NetworkPacket]) -> Dict:
        results = [self.detect(p) for p in test_packets]

        tp = sum(1 for r in results
                if r["is_covert"] and r["true_type"] == TrafficType.COVERT)
        fp = sum(1 for r in results
                if r["is_covert"] and r["true_type"] == TrafficType.NORMAL)
        tn = sum(1 for r in results
                if not r["is_covert"] and r["true_type"] == TrafficType.NORMAL)
        fn = sum(1 for r in results
                if not r["is_covert"] and r["true_type"] == TrafficType.COVERT)

        accuracy = (tp + tn) / len(results) if results else 0
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

        return {
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1_score": f1,
            "true_positives": tp,
            "false_positives": fp,
            "true_negatives": tn,
            "false_negatives": fn,
            "total_packets": len(results)
        }


class SimulationRunner:
    """Main simulation runner for diffusion covert detection."""

    def __init__(self):
        self.profile = TrafficProfile()
        self.generator = TrafficGenerator(self.profile)
        self.detector = DiffusionDetector()

    def run_experiment(self) -> Dict:
        print("Generating training data...")
        train_normal = self.generator.generate_normal_traffic(500)

        print("Training diffusion model...")
        self.detector.train(train_normal, epochs=50)

        print("Generating test data...")
        test_normal = self.generator.generate_normal_traffic(450)
        test_covert = self.generator.generate_covert_traffic(50, start_time=10.0)
        test_packets = test_normal + test_covert
        random.shuffle(test_packets)

        print("Running detection...")
        results = self.detector.evaluate(test_packets)

        return results

    def run_robustness_experiment(self) -> Dict:
        noise_levels = [0.0, 0.1, 0.2, 0.3, 0.5]
        results = {}

        for noise in noise_levels:
            print(f"\nTesting with noise level: {noise}")
            self.generator = TrafficGenerator(TrafficProfile())
            self.detector = DiffusionDetector()

            train = self.generator.generate_normal_traffic(500)
            self.detector.train(train, epochs=30)

            test_normal = self.generator.generate_normal_traffic(450)
            test_covert = self.generator.generate_covert_traffic(50, start_time=10.0)

            for packet in test_covert:
                features = packet.to_feature_vector()
                features += np.random.randn(len(features)) * noise
                packet.payload_size = int(np.clip(
                    features[0] * 1500, 64, 1500
                ))

            test_packets = test_normal + test_covert
            random.shuffle(test_packets)

            results[noise] = self.detector.evaluate(test_packets)

        return results


if __name__ == "__main__":
    print("=" * 60)
    print("Diffusion Models for Covert Communication Detection")
    print("CCIOT 2026 — Simulation Runner")
    print("=" * 60)

    runner = SimulationRunner()

    print("\n--- Main Experiment ---")
    results = runner.run_experiment()

    print("\n" + "=" * 60)
    print("RESULTS")
    print("=" * 60)
    print(f"Accuracy:  {results['accuracy']:.4f}")
    print(f"Precision: {results['precision']:.4f}")
    print(f"Recall:    {results['recall']:.4f}")
    print(f"F1 Score:  {results['f1_score']:.4f}")
    print(f"TP: {results['true_positives']}, FP: {results['false_positives']}")
    print(f"TN: {results['true_negatives']}, FN: {results['false_negatives']}")

    print("\n--- Robustness Experiment ---")
    robustness = runner.run_robustness_experiment()
    print("\nNoise Level | Accuracy | F1 Score")
    print("-" * 40)
    for noise, res in robustness.items():
        print(f"  {noise:.1f}       | {res['accuracy']:.4f}  | {res['f1_score']:.4f}")
