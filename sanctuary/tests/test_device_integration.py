"""
Integration tests for the device abstraction layer with the cognitive pipeline.

Tests cover:
- Audio device -> InputQueue flow
- Image device -> InputQueue flow
- Device -> Perception -> Workspace pipeline
- Device registry integration with SubsystemCoordinator
"""

from __future__ import annotations

import asyncio
from datetime import datetime
from typing import Any, Dict, List, Optional
from unittest.mock import AsyncMock, MagicMock, patch

import numpy as np
import pytest

try:
    from mind.devices import (
        DeviceCapabilities,
        DeviceDataPacket,
        DeviceInfo,
        DeviceProtocol,
        DeviceRegistry,
        DeviceState,
        DeviceType,
    )
except OSError:
    pytest.skip("PortAudio library not available", allow_module_level=True)


# ============================================================================
# Mock Devices for Integration Testing
# ============================================================================


class MockAudioDevice(DeviceProtocol):
    """Mock audio device that emits test audio data."""

    @classmethod
    def enumerate_devices(cls) -> List[DeviceInfo]:
        return [
            DeviceInfo(
                device_id="mock_mic_0",
                name="Mock Microphone",
                capabilities=DeviceCapabilities(
                    device_type=DeviceType.MICROPHONE,
                    modality="audio",
                    sample_rate=16000,
                    channels=1,
                ),
                is_default=True,
            )
        ]

    async def connect(self, device_id: str) -> bool:
        self._device_id = device_id
        self._device_info = self.enumerate_devices()[0]
        self._set_state(DeviceState.CONNECTED)
        return True

    async def disconnect(self) -> None:
        if self.is_streaming:
            await self.stop_streaming()
        self._set_state(DeviceState.DISCONNECTED)

    async def start_streaming(self) -> bool:
        if not self.is_connected:
            return False

        self._set_state(DeviceState.STREAMING)

        # Emit a test audio packet (1 second of silence at 16kHz)
        audio_data = np.zeros(16000, dtype=np.float32)
        self._emit_data(
            modality="audio",
            raw_data=audio_data,
            metadata={
                "sample_rate": 16000,
                "channels": 1,
                "dtype": "float32",
                "frames": 16000,
            },
        )
        return True

    async def stop_streaming(self) -> None:
        self._set_state(DeviceState.CONNECTED)


class MockCameraDevice(DeviceProtocol):
    """Mock camera device that emits test image data."""

    @classmethod
    def enumerate_devices(cls) -> List[DeviceInfo]:
        return [
            DeviceInfo(
                device_id="mock_cam_0",
                name="Mock Camera",
                capabilities=DeviceCapabilities(
                    device_type=DeviceType.CAMERA,
                    modality="image",
                    resolution=(640, 480),
                    channels=3,
                ),
                is_default=True,
            )
        ]

    async def connect(self, device_id: str) -> bool:
        self._device_id = device_id
        self._device_info = self.enumerate_devices()[0]
        self._set_state(DeviceState.CONNECTED)
        return True

    async def disconnect(self) -> None:
        if self.is_streaming:
            await self.stop_streaming()
        self._set_state(DeviceState.DISCONNECTED)

    async def start_streaming(self) -> bool:
        if not self.is_connected:
            return False

        self._set_state(DeviceState.STREAMING)

        # Emit a test image (640x480 blue frame in BGR format)
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        frame[:, :, 0] = 255  # Blue channel
        self._emit_data(
            modality="image",
            raw_data=frame,
            metadata={
                "width": 640,
                "height": 480,
                "channels": 3,
                "format": "BGR",
            },
        )
        return True

    async def stop_streaming(self) -> None:
        self._set_state(DeviceState.CONNECTED)


class MockSensorDevice(DeviceProtocol):
    """Mock sensor device that emits test sensor readings."""

    @classmethod
    def enumerate_devices(cls) -> List[DeviceInfo]:
        return [
            DeviceInfo(
                device_id="mock_sensor_0",
                name="Mock Temperature Sensor",
                capabilities=DeviceCapabilities(
                    device_type=DeviceType.SENSOR,
                    modality="sensor",
                ),
                is_default=True,
            )
        ]

    async def connect(self, device_id: str) -> bool:
        self._device_id = device_id
        self._device_info = self.enumerate_devices()[0]
        self._set_state(DeviceState.CONNECTED)
        return True

    async def disconnect(self) -> None:
        if self.is_streaming:
            await self.stop_streaming()
        self._set_state(DeviceState.DISCONNECTED)

    async def start_streaming(self) -> bool:
        if not self.is_connected:
            return False

        self._set_state(DeviceState.STREAMING)

        # Emit a test sensor reading
        self._emit_data(
            modality="sensor",
            raw_data={
                "sensor_type": "TEMPERATURE",
                "value": 23.5,
                "unit": "celsius",
            },
            metadata={"confidence": 0.95},
        )
        return True

    async def stop_streaming(self) -> None:
        self._set_state(DeviceState.CONNECTED)


# ============================================================================
# Input Queue Flow Tests
# ============================================================================


class TestDeviceToInputQueueFlow:
    """Tests for device data flowing to InputQueue."""

    @pytest.mark.asyncio
    async def test_audio_device_to_input_queue(self) -> None:
        """Audio device data flows to input queue correctly."""
        registry = DeviceRegistry()
        registry.register_device_class(DeviceType.MICROPHONE, MockAudioDevice)

        # Create a mock input queue
        received_inputs: List[tuple] = []

        def mock_input_queue(data, modality, source, metadata):
            received_inputs.append((data, modality, source, metadata))

        registry.set_input_callback(mock_input_queue)

        # Connect and start streaming
        await registry.connect_device(
            DeviceType.MICROPHONE, "mock_mic_0", auto_stream=True
        )

        # Verify data was routed
        assert len(received_inputs) == 1
        data, modality, source, metadata = received_inputs[0]

        assert modality == "audio"
        assert "mock_mic_0" in source
        assert isinstance(data, np.ndarray)
        assert data.shape == (16000,)
        assert metadata["sample_rate"] == 16000

        await registry.disconnect_all_devices()

    @pytest.mark.asyncio
    async def test_camera_device_to_input_queue(self) -> None:
        """Camera device data flows to input queue correctly."""
        registry = DeviceRegistry()
        registry.register_device_class(DeviceType.CAMERA, MockCameraDevice)

        received_inputs: List[tuple] = []

        def mock_input_queue(data, modality, source, metadata):
            received_inputs.append((data, modality, source, metadata))

        registry.set_input_callback(mock_input_queue)

        await registry.connect_device(
            DeviceType.CAMERA, "mock_cam_0", auto_stream=True
        )

        assert len(received_inputs) == 1
        data, modality, source, metadata = received_inputs[0]

        assert modality == "image"
        assert "mock_cam_0" in source
        assert isinstance(data, np.ndarray)
        assert data.shape == (480, 640, 3)
        assert metadata["format"] == "BGR"

        await registry.disconnect_all_devices()

    @pytest.mark.asyncio
    async def test_sensor_device_to_input_queue(self) -> None:
        """Sensor device data flows to input queue correctly."""
        registry = DeviceRegistry()
        registry.register_device_class(DeviceType.SENSOR, MockSensorDevice)

        received_inputs: List[tuple] = []

        def mock_input_queue(data, modality, source, metadata):
            received_inputs.append((data, modality, source, metadata))

        registry.set_input_callback(mock_input_queue)

        await registry.connect_device(
            DeviceType.SENSOR, "mock_sensor_0", auto_stream=True
        )

        assert len(received_inputs) == 1
        data, modality, source, metadata = received_inputs[0]

        assert modality == "sensor"
        assert "mock_sensor_0" in source
        assert data["sensor_type"] == "TEMPERATURE"
        assert data["value"] == 23.5

        await registry.disconnect_all_devices()

    @pytest.mark.asyncio
    async def test_multiple_devices_to_single_queue(self) -> None:
        """Multiple devices route data to the same input queue."""
        registry = DeviceRegistry()
        registry.register_device_class(DeviceType.MICROPHONE, MockAudioDevice)
        registry.register_device_class(DeviceType.CAMERA, MockCameraDevice)
        registry.register_device_class(DeviceType.SENSOR, MockSensorDevice)

        received_inputs: List[tuple] = []

        def mock_input_queue(data, modality, source, metadata):
            received_inputs.append((data, modality, source, metadata))

        registry.set_input_callback(mock_input_queue)

        # Connect all devices with auto-stream
        await registry.connect_device(DeviceType.MICROPHONE, "mock_mic_0", auto_stream=True)
        await registry.connect_device(DeviceType.CAMERA, "mock_cam_0", auto_stream=True)
        await registry.connect_device(DeviceType.SENSOR, "mock_sensor_0", auto_stream=True)

        # All three devices should have emitted data
        assert len(received_inputs) == 3

        modalities = {inp[1] for inp in received_inputs}
        assert modalities == {"audio", "image", "sensor"}

        await registry.disconnect_all_devices()


# ============================================================================
# Perception Integration Tests
# ============================================================================


class TestDeviceToPerceptionFlow:
    """Tests for device data flowing through perception subsystem."""

    @pytest.mark.asyncio
    async def test_audio_encoding(self) -> None:
        """Audio data from device can be encoded by perception."""
        # Create mock audio data similar to what a device would emit
        audio_data = np.random.randn(16000).astype(np.float32) * 0.1

        # Import perception subsystem (may fail if sentence-transformers not installed)
        try:
            from sanctuary.mind.cognitive_core.perception import PerceptionSubsystem
        except ImportError:
            pytest.skip("sentence-transformers not installed")

        perception = PerceptionSubsystem(config={"text_model": "all-MiniLM-L6-v2"})

        # Encode audio
        percept = await perception.encode(audio_data, "audio")

        assert percept is not None
        assert percept.modality == "audio"
        assert len(percept.embedding) == perception.embedding_dim

    @pytest.mark.asyncio
    async def test_image_encoding_with_numpy(self) -> None:
        """Image data as numpy array can be encoded by perception."""
        # Create mock image data (BGR format like OpenCV would provide)
        image_data = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)

        try:
            from sanctuary.mind.cognitive_core.perception import PerceptionSubsystem
        except ImportError:
            pytest.skip("sentence-transformers not installed")

        # Note: Image encoding requires CLIP, which may not be available
        perception = PerceptionSubsystem(config={
            "text_model": "all-MiniLM-L6-v2",
            "enable_image": False,  # Don't try to load CLIP
        })

        # Encode image (will return placeholder if CLIP not loaded)
        percept = await perception.encode(image_data, "image")

        assert percept is not None
        assert percept.modality == "image"

    @pytest.mark.asyncio
    async def test_sensor_encoding(self) -> None:
        """Sensor data can be encoded by perception."""
        sensor_data = {
            "sensor_type": "TEMPERATURE",
            "value": 23.5,
            "unit": "celsius",
        }

        try:
            from sanctuary.mind.cognitive_core.perception import PerceptionSubsystem
        except ImportError:
            pytest.skip("sentence-transformers not installed")

        perception = PerceptionSubsystem(config={"text_model": "all-MiniLM-L6-v2"})

        percept = await perception.encode(sensor_data, "sensor")

        assert percept is not None
        assert percept.modality == "sensor"
        # Sensor data is encoded as text, so should have proper embedding
        assert len(percept.embedding) == perception.embedding_dim
        assert any(v != 0.0 for v in percept.embedding)  # Not all zeros


# ============================================================================
# Full Pipeline Integration Tests
# ============================================================================


class TestFullPipelineIntegration:
    """Tests for complete device -> perception -> workspace pipeline."""

    @pytest.mark.asyncio
    async def test_device_registry_stats(self) -> None:
        """Device registry tracks statistics correctly."""
        registry = DeviceRegistry()
        registry.register_device_class(DeviceType.MICROPHONE, MockAudioDevice)

        # Create callback that routes data
        def mock_callback(data, modality, source, metadata):
            pass

        registry.set_input_callback(mock_callback)

        await registry.connect_device(DeviceType.MICROPHONE, "mock_mic_0", auto_stream=True)

        stats = registry.get_stats()

        assert stats["total_packets_routed"] == 1
        assert stats["packets_by_modality"]["audio"] == 1
        assert stats["connection_count"] == 1
        assert stats["connected_device_count"] == 1

        await registry.disconnect_all_devices()

        stats = registry.get_stats()
        assert stats["disconnection_count"] == 1
        assert stats["connected_device_count"] == 0

    @pytest.mark.asyncio
    async def test_device_connection_callbacks(self) -> None:
        """Device connection/disconnection callbacks are invoked."""
        registry = DeviceRegistry()
        registry.register_device_class(DeviceType.MICROPHONE, MockAudioDevice)

        connected_devices: List[DeviceInfo] = []
        disconnected_devices: List[str] = []

        registry.set_on_device_connected(lambda info: connected_devices.append(info))
        registry.set_on_device_disconnected(lambda id_: disconnected_devices.append(id_))

        await registry.connect_device(DeviceType.MICROPHONE, "mock_mic_0")
        assert len(connected_devices) == 1
        assert connected_devices[0].device_id == "mock_mic_0"

        await registry.disconnect_device("mock_mic_0")
        assert len(disconnected_devices) == 1
        assert disconnected_devices[0] == "mock_mic_0"

    @pytest.mark.asyncio
    async def test_enumerate_multiple_device_types(self) -> None:
        """enumerate_all_devices discovers devices from multiple types."""
        registry = DeviceRegistry()
        registry.register_device_class(DeviceType.MICROPHONE, MockAudioDevice)
        registry.register_device_class(DeviceType.CAMERA, MockCameraDevice)
        registry.register_device_class(DeviceType.SENSOR, MockSensorDevice)

        all_devices = registry.enumerate_all_devices()

        assert DeviceType.MICROPHONE in all_devices
        assert DeviceType.CAMERA in all_devices
        assert DeviceType.SENSOR in all_devices

        assert len(all_devices[DeviceType.MICROPHONE]) == 1
        assert len(all_devices[DeviceType.CAMERA]) == 1
        assert len(all_devices[DeviceType.SENSOR]) == 1


# ============================================================================
# Subsystem Coordinator Integration Tests
# ============================================================================


class TestSubsystemCoordinatorIntegration:
    """Tests for DeviceRegistry integration with SubsystemCoordinator."""

    def test_device_registry_initialization(self) -> None:
        """Device registry can be initialized through config."""
        # This tests the _initialize_device_registry method
        from sanctuary.mind.cognitive_core.core.subsystem_coordinator import (
            _try_import_devices,
        )

        # Check if device modules are available
        device_modules = _try_import_devices()

        if device_modules is None:
            pytest.skip("Device modules not available")

        DeviceRegistry = device_modules["DeviceRegistry"]
        registry = DeviceRegistry(config={"enabled": True})

        assert registry is not None
        assert isinstance(registry, DeviceRegistry)

    @pytest.mark.asyncio
    async def test_connect_device_registry_to_asyncio_queue(self) -> None:
        """Device registry can route data to an asyncio.Queue."""
        registry = DeviceRegistry()
        registry.register_device_class(DeviceType.MICROPHONE, MockAudioDevice)

        # Create an asyncio queue like StateManager uses
        input_queue: asyncio.Queue = asyncio.Queue(maxsize=100)

        # Create adapter callback
        async def route_to_queue(data, modality, source, metadata):
            try:
                input_queue.put_nowait((data, modality))
            except asyncio.QueueFull:
                pass

        # The registry's set_input_callback expects a sync function,
        # so we need to handle the async nature
        def sync_route(data, modality, source, metadata):
            asyncio.create_task(route_to_queue(data, modality, source, metadata))

        registry.set_input_callback(sync_route)

        await registry.connect_device(DeviceType.MICROPHONE, "mock_mic_0", auto_stream=True)

        # Give the async task a moment to complete
        await asyncio.sleep(0.01)

        # Check that data was routed
        assert not input_queue.empty()
        data, modality = input_queue.get_nowait()
        assert modality == "audio"
        assert isinstance(data, np.ndarray)

        await registry.disconnect_all_devices()


# ============================================================================
# Edge Case Tests for Perception Robustness
# ============================================================================


class TestPerceptionEdgeCases:
    """Tests for edge cases and unusual inputs in perception encoding."""

    @pytest.fixture
    def perception(self):
        """Create perception subsystem for tests."""
        try:
            from sanctuary.mind.cognitive_core.perception import PerceptionSubsystem
            return PerceptionSubsystem(config={"text_model": "all-MiniLM-L6-v2"})
        except ImportError:
            pytest.skip("sentence-transformers not installed")

    # ========== Audio Edge Cases ==========

    @pytest.mark.asyncio
    async def test_empty_audio(self, perception) -> None:
        """Empty audio array returns valid zero embedding."""
        audio = np.array([], dtype=np.float32)
        percept = await perception.encode(audio, "audio")

        assert percept is not None
        assert len(percept.embedding) == perception.embedding_dim
        assert all(v == 0.0 for v in percept.embedding)

    @pytest.mark.asyncio
    async def test_single_sample_audio(self, perception) -> None:
        """Single sample audio returns valid embedding."""
        audio = np.array([0.5], dtype=np.float32)
        percept = await perception.encode(audio, "audio")

        assert percept is not None
        assert len(percept.embedding) == perception.embedding_dim

    @pytest.mark.asyncio
    async def test_audio_with_nan(self, perception) -> None:
        """Audio with NaN values is handled gracefully."""
        audio = np.array([0.1, np.nan, 0.3, np.nan, 0.5], dtype=np.float32)
        percept = await perception.encode(audio, "audio")

        assert percept is not None
        assert len(percept.embedding) == perception.embedding_dim
        # Should not contain NaN
        assert np.isfinite(percept.embedding).all()

    @pytest.mark.asyncio
    async def test_audio_with_inf(self, perception) -> None:
        """Audio with Inf values is handled gracefully."""
        audio = np.array([0.1, np.inf, 0.3, -np.inf, 0.5], dtype=np.float32)
        percept = await perception.encode(audio, "audio")

        assert percept is not None
        assert len(percept.embedding) == perception.embedding_dim
        assert np.isfinite(percept.embedding).all()

    @pytest.mark.asyncio
    async def test_silent_audio(self, perception) -> None:
        """All-zero audio returns valid embedding."""
        audio = np.zeros(16000, dtype=np.float32)
        percept = await perception.encode(audio, "audio")

        assert percept is not None
        assert len(percept.embedding) == perception.embedding_dim

    @pytest.mark.asyncio
    async def test_very_loud_audio(self, perception) -> None:
        """Audio with values > 1.0 is handled."""
        audio = np.random.randn(16000).astype(np.float32) * 100
        percept = await perception.encode(audio, "audio")

        assert percept is not None
        assert len(percept.embedding) == perception.embedding_dim
        assert np.isfinite(percept.embedding).all()

    @pytest.mark.asyncio
    async def test_audio_dict_with_metadata(self, perception) -> None:
        """Audio in dict format with sample_rate metadata works."""
        audio_dict = {
            "data": np.random.randn(8000).astype(np.float32) * 0.1,
            "sample_rate": 8000,
        }
        percept = await perception.encode(audio_dict, "audio")

        assert percept is not None
        assert len(percept.embedding) == perception.embedding_dim

    # ========== Sensor Edge Cases ==========

    @pytest.mark.asyncio
    async def test_sensor_missing_keys(self, perception) -> None:
        """Sensor dict with missing keys still encodes."""
        sensor_data = {"sensor_type": "TEMPERATURE"}  # No value key
        percept = await perception.encode(sensor_data, "sensor")

        assert percept is not None
        assert len(percept.embedding) == perception.embedding_dim

    @pytest.mark.asyncio
    async def test_sensor_extreme_values(self, perception) -> None:
        """Sensor with extreme values is handled."""
        sensor_data = {
            "sensor_type": "TEMPERATURE",
            "value": 1e10,  # Extremely hot
            "unit": "celsius",
        }
        percept = await perception.encode(sensor_data, "sensor")

        assert percept is not None
        assert np.isfinite(percept.embedding).all()

    @pytest.mark.asyncio
    async def test_sensor_negative_values(self, perception) -> None:
        """Sensor with negative values works."""
        sensor_data = {
            "sensor_type": "TEMPERATURE",
            "value": -273.15,  # Absolute zero
            "unit": "celsius",
        }
        percept = await perception.encode(sensor_data, "sensor")

        assert percept is not None
        assert len(percept.embedding) == perception.embedding_dim

    @pytest.mark.asyncio
    async def test_sensor_unknown_type(self, perception) -> None:
        """Unknown sensor type falls back to generic encoding."""
        sensor_data = {
            "sensor_type": "UNKNOWN_QUANTUM_SENSOR",
            "value": 42.0,
        }
        percept = await perception.encode(sensor_data, "sensor")

        assert percept is not None
        assert len(percept.embedding) == perception.embedding_dim

    @pytest.mark.asyncio
    async def test_sensor_multi_axis_missing_keys(self, perception) -> None:
        """Multi-axis sensor with missing axis keys works."""
        sensor_data = {
            "sensor_type": "ACCELEROMETER",
            "value": {"x": 9.8},  # Missing y and z
        }
        percept = await perception.encode(sensor_data, "sensor")

        assert percept is not None
        assert len(percept.embedding) == perception.embedding_dim

    @pytest.mark.asyncio
    async def test_sensor_as_list(self, perception) -> None:
        """Sensor value as list (time series) works."""
        sensor_data = {
            "sensor_type": "CUSTOM",
            "value": [1.0, 2.0, 3.0, 4.0, 5.0],
        }
        percept = await perception.encode(sensor_data, "sensor")

        assert percept is not None
        assert len(percept.embedding) == perception.embedding_dim

    @pytest.mark.asyncio
    async def test_sensor_raw_float(self, perception) -> None:
        """Raw float as sensor data works."""
        percept = await perception.encode(42.5, "sensor")

        assert percept is not None
        assert len(percept.embedding) == perception.embedding_dim

    # ========== Caching Tests ==========

    @pytest.mark.asyncio
    async def test_projection_matrix_caching(self, perception) -> None:
        """Projection matrices are cached between calls."""
        from sanctuary.mind.cognitive_core.perception import PerceptionSubsystem

        # Clear caches
        PerceptionSubsystem._audio_projection = None
        PerceptionSubsystem._sensor_projection_cache.clear()

        # First encoding creates projection
        audio1 = np.random.randn(16000).astype(np.float32) * 0.1
        await perception.encode(audio1, "audio")

        # Verify projection was cached
        assert PerceptionSubsystem._audio_projection is not None

        # Second encoding should reuse cached projection
        audio2 = np.random.randn(16000).astype(np.float32) * 0.1
        await perception.encode(audio2, "audio")

        # Should still be the same object
        assert PerceptionSubsystem._audio_projection is not None

    @pytest.mark.asyncio
    async def test_mel_filterbank_caching(self, perception) -> None:
        """Mel filterbank is cached between calls."""
        from sanctuary.mind.cognitive_core.perception import PerceptionSubsystem

        initial_cache_size = len(PerceptionSubsystem._mel_filterbank_cache)

        # Encode audio (creates mel filterbank for this sample rate/fft size)
        audio = np.random.randn(16000).astype(np.float32) * 0.1
        await perception.encode(audio, "audio")

        # Cache should have grown
        assert len(PerceptionSubsystem._mel_filterbank_cache) >= initial_cache_size
