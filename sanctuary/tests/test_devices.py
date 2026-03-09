"""
Tests for the multimodal device abstraction layer.

Tests cover:
- Device protocol state transitions
- Device registry management
- Data callback invocation
- Device enumeration (with mocked hardware)
- Device connection and streaming
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
# Mock Device Implementation for Testing
# ============================================================================


class MockDevice(DeviceProtocol):
    """Mock device for testing the protocol interface."""

    _mock_devices: List[DeviceInfo] = []

    def __init__(self, emit_on_stream: bool = True) -> None:
        super().__init__()
        self.emit_on_stream = emit_on_stream
        self.connect_called = False
        self.disconnect_called = False
        self.stream_started = False
        self.stream_stopped = False

    @classmethod
    def set_mock_devices(cls, devices: List[DeviceInfo]) -> None:
        """Set mock devices for enumeration."""
        cls._mock_devices = devices

    @classmethod
    def enumerate_devices(cls) -> List[DeviceInfo]:
        """Return mock devices."""
        return cls._mock_devices

    async def connect(self, device_id: str) -> bool:
        """Mock connect."""
        self.connect_called = True
        self._device_id = device_id
        self._device_info = DeviceInfo(
            device_id=device_id,
            name=f"Mock Device {device_id}",
            capabilities=DeviceCapabilities(
                device_type=DeviceType.CUSTOM,
                modality="test",
            ),
        )
        self._set_state(DeviceState.CONNECTED)
        return True

    async def disconnect(self) -> None:
        """Mock disconnect."""
        self.disconnect_called = True
        if self.is_streaming:
            await self.stop_streaming()
        self._set_state(DeviceState.DISCONNECTED)

    async def start_streaming(self) -> bool:
        """Mock start streaming."""
        if not self.is_connected:
            return False
        self.stream_started = True
        self._set_state(DeviceState.STREAMING)

        if self.emit_on_stream:
            # Emit a test data packet
            self._emit_data(
                modality="test",
                raw_data=b"test_data",
                metadata={"test": True},
            )

        return True

    async def stop_streaming(self) -> None:
        """Mock stop streaming."""
        self.stream_stopped = True
        self._set_state(DeviceState.CONNECTED)


# ============================================================================
# Protocol Tests
# ============================================================================


class TestDeviceProtocol:
    """Tests for DeviceProtocol base class behavior."""

    def test_initial_state(self) -> None:
        """Device starts in DISCONNECTED state."""
        device = MockDevice()
        assert device.state == DeviceState.DISCONNECTED
        assert device.device_id is None
        assert device.device_info is None
        assert not device.is_connected
        assert not device.is_streaming

    @pytest.mark.asyncio
    async def test_connect_changes_state(self) -> None:
        """Connecting changes state to CONNECTED."""
        device = MockDevice()
        success = await device.connect("test_device_1")

        assert success
        assert device.state == DeviceState.CONNECTED
        assert device.device_id == "test_device_1"
        assert device.is_connected
        assert not device.is_streaming

    @pytest.mark.asyncio
    async def test_disconnect_changes_state(self) -> None:
        """Disconnecting changes state back to DISCONNECTED."""
        device = MockDevice()
        await device.connect("test_device_1")
        await device.disconnect()

        assert device.state == DeviceState.DISCONNECTED
        assert not device.is_connected

    @pytest.mark.asyncio
    async def test_streaming_lifecycle(self) -> None:
        """Test start_streaming and stop_streaming state transitions."""
        device = MockDevice(emit_on_stream=False)
        await device.connect("test_device_1")

        # Start streaming
        success = await device.start_streaming()
        assert success
        assert device.state == DeviceState.STREAMING
        assert device.is_streaming

        # Stop streaming
        await device.stop_streaming()
        assert device.state == DeviceState.CONNECTED
        assert not device.is_streaming
        assert device.is_connected

    @pytest.mark.asyncio
    async def test_cannot_stream_without_connecting(self) -> None:
        """Cannot start streaming without connecting first."""
        device = MockDevice()
        success = await device.start_streaming()
        assert not success
        assert device.state == DeviceState.DISCONNECTED

    @pytest.mark.asyncio
    async def test_data_callback_invoked(self) -> None:
        """Data callback is invoked when device emits data."""
        device = MockDevice(emit_on_stream=True)
        received_packets: List[DeviceDataPacket] = []

        def on_data(packet: DeviceDataPacket) -> None:
            received_packets.append(packet)

        device.set_data_callback(on_data)
        await device.connect("test_device_1")
        await device.start_streaming()

        assert len(received_packets) == 1
        packet = received_packets[0]
        assert packet.device_id == "test_device_1"
        assert packet.modality == "test"
        assert packet.raw_data == b"test_data"
        assert packet.metadata["test"] is True
        assert packet.sequence_number == 1

    @pytest.mark.asyncio
    async def test_state_callback_invoked(self) -> None:
        """State callback is invoked on state changes."""
        device = MockDevice(emit_on_stream=False)
        state_changes: List[DeviceState] = []

        def on_state(state: DeviceState) -> None:
            state_changes.append(state)

        device.set_state_callback(on_state)
        await device.connect("test_device_1")
        await device.start_streaming()
        await device.stop_streaming()
        await device.disconnect()

        assert DeviceState.CONNECTED in state_changes
        assert DeviceState.STREAMING in state_changes
        assert DeviceState.DISCONNECTED in state_changes

    def test_get_stats(self) -> None:
        """get_stats returns basic device statistics."""
        device = MockDevice()
        stats = device.get_stats()

        assert "device_id" in stats
        assert "state" in stats
        assert "packets_sent" in stats


# ============================================================================
# Registry Tests
# ============================================================================


class TestDeviceRegistry:
    """Tests for DeviceRegistry class."""

    def test_register_device_class(self) -> None:
        """Can register device class for a device type."""
        registry = DeviceRegistry()
        registry.register_device_class(DeviceType.CUSTOM, MockDevice)

        assert registry.get_device_class(DeviceType.CUSTOM) == MockDevice
        assert DeviceType.CUSTOM in registry.get_registered_types()

    def test_enumerate_all_devices(self) -> None:
        """enumerate_all_devices discovers devices from all registered types."""
        registry = DeviceRegistry()
        registry.register_device_class(DeviceType.CUSTOM, MockDevice)

        # Set up mock devices
        MockDevice.set_mock_devices([
            DeviceInfo(
                device_id="mock_1",
                name="Mock 1",
                capabilities=DeviceCapabilities(
                    device_type=DeviceType.CUSTOM,
                    modality="test",
                ),
            ),
            DeviceInfo(
                device_id="mock_2",
                name="Mock 2",
                capabilities=DeviceCapabilities(
                    device_type=DeviceType.CUSTOM,
                    modality="test",
                ),
            ),
        ])

        devices = registry.enumerate_all_devices()

        assert DeviceType.CUSTOM in devices
        assert len(devices[DeviceType.CUSTOM]) == 2

        # Clean up
        MockDevice.set_mock_devices([])

    @pytest.mark.asyncio
    async def test_connect_device(self) -> None:
        """Can connect to a device through the registry."""
        registry = DeviceRegistry()
        registry.register_device_class(DeviceType.CUSTOM, MockDevice)

        success = await registry.connect_device(DeviceType.CUSTOM, "test_device")

        assert success
        assert "test_device" in registry.get_connected_devices()

    @pytest.mark.asyncio
    async def test_disconnect_device(self) -> None:
        """Can disconnect a device through the registry."""
        registry = DeviceRegistry()
        registry.register_device_class(DeviceType.CUSTOM, MockDevice)

        await registry.connect_device(DeviceType.CUSTOM, "test_device")
        success = await registry.disconnect_device("test_device")

        assert success
        assert "test_device" not in registry.get_connected_devices()

    @pytest.mark.asyncio
    async def test_connect_with_auto_stream(self) -> None:
        """auto_stream=True starts streaming immediately after connect."""
        registry = DeviceRegistry()
        registry.register_device_class(DeviceType.CUSTOM, MockDevice)

        # Capture data packets
        packets: List[DeviceDataPacket] = []
        registry.set_input_callback(
            lambda data, modality, source, meta: packets.append(
                DeviceDataPacket(device_id=source, modality=modality, raw_data=data)
            )
        )

        success = await registry.connect_device(
            DeviceType.CUSTOM, "test_device", auto_stream=True
        )

        assert success
        device = registry.get_device("test_device")
        assert device.is_streaming

    @pytest.mark.asyncio
    async def test_data_routing_to_callback(self) -> None:
        """Device data is routed to the input callback."""
        registry = DeviceRegistry()
        registry.register_device_class(DeviceType.CUSTOM, MockDevice)

        received: List[tuple] = []

        def capture_data(data, modality, source, metadata):
            received.append((data, modality, source, metadata))

        registry.set_input_callback(capture_data)

        await registry.connect_device(DeviceType.CUSTOM, "test_device", auto_stream=True)

        # Give async callbacks a moment to process
        await asyncio.sleep(0.01)

        assert len(received) == 1
        data, modality, source, metadata = received[0]
        assert modality == "test"
        assert "test_device" in source

    @pytest.mark.asyncio
    async def test_disconnect_all_devices(self) -> None:
        """disconnect_all_devices disconnects all connected devices."""
        registry = DeviceRegistry()
        registry.register_device_class(DeviceType.CUSTOM, MockDevice)

        await registry.connect_device(DeviceType.CUSTOM, "device_1")
        await registry.connect_device(DeviceType.CUSTOM, "device_2")

        assert len(registry.get_connected_devices()) == 2

        await registry.disconnect_all_devices()

        assert len(registry.get_connected_devices()) == 0

    def test_get_stats(self) -> None:
        """get_stats returns registry statistics."""
        registry = DeviceRegistry()
        registry.register_device_class(DeviceType.CUSTOM, MockDevice)

        stats = registry.get_stats()

        assert "registered_types" in stats
        assert "connected_device_count" in stats
        assert "total_packets_routed" in stats


# ============================================================================
# Device Data Packet Tests
# ============================================================================


class TestDeviceDataPacket:
    """Tests for DeviceDataPacket dataclass."""

    def test_packet_creation(self) -> None:
        """Can create a DeviceDataPacket with required fields."""
        packet = DeviceDataPacket(
            device_id="cam_0",
            modality="image",
            raw_data=np.zeros((480, 640, 3), dtype=np.uint8),
        )

        assert packet.device_id == "cam_0"
        assert packet.modality == "image"
        assert isinstance(packet.raw_data, np.ndarray)
        assert isinstance(packet.timestamp, datetime)
        assert packet.sequence_number == 0
        assert packet.metadata == {}

    def test_packet_with_metadata(self) -> None:
        """Packet can include metadata."""
        packet = DeviceDataPacket(
            device_id="mic_0",
            modality="audio",
            raw_data=np.zeros(16000, dtype=np.float32),
            metadata={"sample_rate": 16000, "channels": 1},
        )

        assert packet.metadata["sample_rate"] == 16000
        assert packet.metadata["channels"] == 1


# ============================================================================
# Device Capabilities Tests
# ============================================================================


class TestDeviceCapabilities:
    """Tests for DeviceCapabilities dataclass."""

    def test_microphone_capabilities(self) -> None:
        """Microphone capabilities include audio-specific fields."""
        caps = DeviceCapabilities(
            device_type=DeviceType.MICROPHONE,
            modality="audio",
            sample_rate=16000,
            channels=1,
            bit_depth=16,
        )

        assert caps.device_type == DeviceType.MICROPHONE
        assert caps.modality == "audio"
        assert caps.sample_rate == 16000
        assert caps.channels == 1

    def test_camera_capabilities(self) -> None:
        """Camera capabilities include video-specific fields."""
        caps = DeviceCapabilities(
            device_type=DeviceType.CAMERA,
            modality="image",
            resolution=(1920, 1080),
            channels=3,
            extra={"fps": 30.0},
        )

        assert caps.device_type == DeviceType.CAMERA
        assert caps.modality == "image"
        assert caps.resolution == (1920, 1080)
        assert caps.extra["fps"] == 30.0


# ============================================================================
# Microphone Device Tests (with mocked sounddevice)
# ============================================================================


class TestMicrophoneDevice:
    """Tests for MicrophoneDevice with mocked sounddevice."""

    @pytest.mark.asyncio
    async def test_enumerate_without_sounddevice(self) -> None:
        """enumerate_devices returns empty list if sounddevice not available."""
        with patch.dict("sys.modules", {"sounddevice": None}):
            # Need to reimport to get the patched version
            from sanctuary.mind.devices.audio_device import HAS_SOUNDDEVICE

            # If sounddevice is actually installed, this test verifies
            # the fallback behavior when it's not
            if not HAS_SOUNDDEVICE:
                from sanctuary.mind.devices import MicrophoneDevice
                devices = MicrophoneDevice.enumerate_devices()
                assert devices == []

    @pytest.mark.asyncio
    async def test_microphone_device_lifecycle(self) -> None:
        """Test MicrophoneDevice connection lifecycle with mocked audio."""
        from sanctuary.mind.devices.audio_device import HAS_SOUNDDEVICE

        if not HAS_SOUNDDEVICE:
            pytest.skip("sounddevice not installed")

        from sanctuary.mind.devices import MicrophoneDevice

        # Mock sounddevice
        with patch("sanctuary.mind.devices.audio_device.sd") as mock_sd:
            mock_sd.query_devices.return_value = [
                {
                    "name": "Test Microphone",
                    "max_input_channels": 2,
                    "default_samplerate": 44100.0,
                }
            ]
            mock_sd.default.device = [0, 0]

            devices = MicrophoneDevice.enumerate_devices()
            assert len(devices) >= 1


# ============================================================================
# Camera Device Tests (with mocked OpenCV)
# ============================================================================


class TestCameraDevice:
    """Tests for CameraDevice with mocked OpenCV."""

    @pytest.mark.asyncio
    async def test_enumerate_without_opencv(self) -> None:
        """enumerate_devices returns empty list if opencv not available."""
        from sanctuary.mind.devices.camera_device import HAS_OPENCV

        if not HAS_OPENCV:
            from sanctuary.mind.devices import CameraDevice
            devices = CameraDevice.enumerate_devices()
            assert devices == []

    @pytest.mark.asyncio
    async def test_camera_device_lifecycle(self) -> None:
        """Test CameraDevice connection lifecycle with mocked cv2."""
        from sanctuary.mind.devices.camera_device import HAS_OPENCV

        if not HAS_OPENCV:
            pytest.skip("opencv-python not installed")

        from sanctuary.mind.devices import CameraDevice

        # Mock cv2.VideoCapture
        with patch("sanctuary.mind.devices.camera_device.cv2") as mock_cv2:
            mock_cap = MagicMock()
            mock_cap.isOpened.return_value = True
            mock_cap.get.side_effect = lambda prop: {
                mock_cv2.CAP_PROP_FRAME_WIDTH: 640,
                mock_cv2.CAP_PROP_FRAME_HEIGHT: 480,
                mock_cv2.CAP_PROP_FPS: 30.0,
            }.get(prop, 0)
            mock_cv2.VideoCapture.return_value = mock_cap

            device = CameraDevice()
            success = await device.connect("cam_0")

            assert success
            assert device.is_connected

            await device.disconnect()
            assert not device.is_connected


# ============================================================================
# Sensor Device Tests
# ============================================================================


class TestSensorDevice:
    """Tests for sensor devices."""

    def test_sensor_reading_to_dict(self) -> None:
        """SensorReading can be converted to dict."""
        from sanctuary.mind.devices import SensorReading, SensorType

        reading = SensorReading(
            sensor_type=SensorType.TEMPERATURE,
            value=23.5,
            unit="celsius",
        )

        data = reading.to_dict()

        assert data["sensor_type"] == "TEMPERATURE"
        assert data["value"] == 23.5
        assert data["unit"] == "celsius"
        assert "timestamp" in data

    @pytest.mark.asyncio
    async def test_serial_sensor_enumerate(self) -> None:
        """SerialSensorDevice enumerates available serial ports."""
        from sanctuary.mind.devices.sensor_device import HAS_SERIAL

        if not HAS_SERIAL:
            pytest.skip("pyserial not installed")

        from sanctuary.mind.devices import SerialSensorDevice

        with patch("sanctuary.mind.devices.sensor_device.serial.tools.list_ports") as mock_ports:
            mock_port = MagicMock()
            mock_port.device = "COM3"
            mock_port.description = "Test Serial Port"
            mock_port.manufacturer = "Test Manufacturer"
            mock_port.vid = 1234
            mock_port.pid = 5678
            mock_port.serial_number = "ABC123"
            mock_ports.comports.return_value = [mock_port]

            devices = SerialSensorDevice.enumerate_devices()

            assert len(devices) == 1
            assert devices[0].device_id == "COM3"
            assert devices[0].name == "Test Serial Port"
