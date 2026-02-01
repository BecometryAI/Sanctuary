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

from sanctuary.mind.devices import (
    DeviceCapabilities,
    DeviceDataPacket,
    DeviceInfo,
    DeviceProtocol,
    DeviceRegistry,
    DeviceState,
    DeviceType,
)


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
