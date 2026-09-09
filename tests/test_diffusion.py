"""
Tests for Diffusion Covert Communication Detection
"""

import pytest
import numpy as np
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from diffusion_detector import (
    NetworkPacket, TrafficType, TrafficProfile, TrafficGenerator,
    SimpleDiffusionModel, AdaptiveThresholdDetector, RLEnhancedDetector,
    DiffusionDetector, SimulationRunner
)


class TestNetworkPacket:
    def test_packet_creation(self):
        packet = NetworkPacket(
            packet_id=1,
            source_ip="192.168.1.1",
            dest_ip="10.0.0.1",
            payload_size=500,
            timestamp=1.0,
            protocol=6,
            flags=2,
            traffic_type=TrafficType.NORMAL
        )
        assert packet.packet_id == 1
        assert packet.traffic_type == TrafficType.NORMAL

    def test_feature_vector(self):
        packet = NetworkPacket(
            packet_id=1, source_ip="192.168.1.1",
            dest_ip="10.0.0.1", payload_size=750,
            timestamp=1.5, protocol=17, flags=4,
            traffic_type=TrafficType.COVERT
        )
        features = packet.to_feature_vector()
        assert len(features) == 4


class TestTrafficGenerator:
    def test_normal_traffic_generation(self):
        profile = TrafficProfile()
        gen = TrafficGenerator(profile)
        packets = gen.generate_normal_traffic(100)
        assert len(packets) == 100
        assert all(p.traffic_type == TrafficType.NORMAL for p in packets)

    def test_covert_traffic_generation(self):
        profile = TrafficProfile()
        gen = TrafficGenerator(profile)
        packets = gen.generate_covert_traffic(20)
        assert len(packets) == 20
        assert all(p.traffic_type == TrafficType.COVERT for p in packets)

    def test_mixed_traffic(self):
        profile = TrafficProfile()
        gen = TrafficGenerator(profile)
        packets = gen.generate_mixed_traffic(100, 10)
        assert len(packets) == 110
        normal = sum(1 for p in packets if p.traffic_type == TrafficType.NORMAL)
        covert = sum(1 for p in packets if p.traffic_type == TrafficType.COVERT)
        assert normal == 100
        assert covert == 10


class TestSimpleDiffusionModel:
    def test_model_creation(self):
        model = SimpleDiffusionModel(input_dim=5, hidden_dim=16)
        assert model.input_dim == 5
        assert model.hidden_dim == 16

    def test_encoding(self):
        model = SimpleDiffusionModel(input_dim=5, hidden_dim=16)
        x = np.random.randn(5)
        z = model._encode(x)
        assert z.shape == (16,)

    def test_decoding(self):
        model = SimpleDiffusionModel(input_dim=5, hidden_dim=16)
        z = np.random.randn(16)
        x = model._decode(z)
        assert x.shape == (5,)

    def test_training(self):
        model = SimpleDiffusionModel(input_dim=5, hidden_dim=8)
        data = np.random.randn(50, 5)
        losses = model.train_on_normal(data, epochs=10)
        assert len(losses) == 10
        assert losses[-1] < losses[0]  # loss should decrease

    def test_anomaly_score(self):
        model = SimpleDiffusionModel(input_dim=5, hidden_dim=8)
        normal_data = np.random.randn(50, 5) * 0.1
        model.train_on_normal(normal_data, epochs=20)

        normal_sample = np.array([0.3, 0.5, 0.1, 0.2, 0.0])
        score = model.compute_anomaly_score(normal_sample)
        assert score >= 0


class TestAdaptiveThresholdDetector:
    def test_initial_threshold(self):
        detector = AdaptiveThresholdDetector(initial_threshold=0.1)
        assert detector.threshold == 0.1

    def test_detection(self):
        detector = AdaptiveThresholdDetector(initial_threshold=0.1)
        assert detector.detect(0.2) is True
        assert detector.detect(0.05) is False

    def test_adaptation(self):
        detector = AdaptiveThresholdDetector(initial_threshold=0.1)
        for _ in range(100):
            detector.update(0.05, is_normal=True)
        assert detector.threshold > 0.05


class TestRLEnhancedDetector:
    def test_rl_creation(self):
        rl = RLEnhancedDetector()
        assert rl.action_dim == 10

    def test_action_selection(self):
        rl = RLEnhancedDetector()
        history = [0.1, 0.2, 0.15, 0.12, 0.18]
        action = rl.select_action(history)
        assert 0 <= action < rl.action_dim

    def test_q_table_update(self):
        rl = RLEnhancedDetector()
        rl.update(50, 3, 1.0, 55)
        assert rl.q_table[50, 3] != 0


class TestDiffusionDetector:
    def test_detector_creation(self):
        detector = DiffusionDetector()
        assert detector.diffusion_model is not None

    def test_training(self):
        detector = DiffusionDetector()
        profile = TrafficProfile()
        gen = TrafficGenerator(profile)
        train = gen.generate_normal_traffic(100)
        detector.train(train, epochs=10)
        assert len(detector.training_losses) > 0

    def test_detection(self):
        detector = DiffusionDetector()
        profile = TrafficProfile()
        gen = TrafficGenerator(profile)
        train = gen.generate_normal_traffic(100)
        detector.train(train, epochs=10)

        packet = NetworkPacket(
            packet_id=1, source_ip="192.168.1.1",
            dest_ip="10.0.0.1", payload_size=500,
            timestamp=1.0, protocol=6, flags=2
        )
        result = detector.detect(packet)
        assert "score" in result
        assert "is_covert" in result
        assert "threshold" in result


class TestSimulationRunner:
    def test_runner_creation(self):
        runner = SimulationRunner()
        assert runner.detector is not None

    def test_experiment_run(self):
        runner = SimulationRunner()
        results = runner.run_experiment()
        assert "accuracy" in results
        assert "f1_score" in results
        assert results["accuracy"] > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
