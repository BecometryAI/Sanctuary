"""
Device Protocol: Abstract interface for peripheral devices.

This module defines the abstract protocol that all device implementations must follow.
It provides a unified interface for cameras, microphones, speakers, haptic sensors,
and any other peripherals that can provide sensory input or output.

Key Design Principles:
- Wrap, don't rewrite: Existing implementations are wrapped, not replaced
- Unified data flow: All devices emit DeviceDataPacket to a common pipeline
- Hot-plug support: Devices can be connected/disconnected at runtime
- Windows compatibility: Uses DirectShow for video, WASAPI for audio
"""

from __future__ import annotations

import asyncio
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum, auto
from typing import Any, Callable, ClassVar, Dict, List, Optional, Type, Union

import numpy as np
from numpy.typing import NDArray

logger = logging.getLogger(__name__)


class DeviceType(Enum):
    """Types of peripheral devices."""
    MICROPHONE = auto()
    CAMERA = auto()
    SPEAKER = auto()
    HAPTIC_INPUT = auto()
    HAPTIC_OUTPUT = auto()
    SENSOR = auto()
    CUSTOM = auto()


class DeviceState(Enum):
    """Device lifecycle states."""
    DISCONNECTED = auto()
    CONNECTED = auto()
    STARTING = auto()
    STREAMING = auto()
    STOPPING = auto()
    ERROR = auto()


@dataclass
class DeviceCapabilities:
    """
    Describes what a device can do.

    Attributes:
        device_type: Category of device
        modality: Perceptual modality ("audio", "image", "haptic", etc.)
        sample_rate: For audio devices, samples per second
        resolution: For cameras, (width, height) tuple
        channels: Number of channels (audio: 1=mono, 2=stereo; camera: 3=RGB)
        bit_depth: Bits per sample
        extra: Additional device-specific capabilities
    """
    device_type: DeviceType
    modality: str
    sample_rate: Optional[int] = None
    resolution: Optional[tuple[int, int]] = None
    channels: int = 1
    bit_depth: int = 16
    extra: Dict[str, Any] = field(default_factory=dict)


@dataclass
class DeviceInfo:
    """
    Information about a discovered device.

    Attributes:
        device_id: Unique identifier for this device instance
        name: Human-readable device name
        manufacturer: Device manufacturer (if available)
        capabilities: What this device can do
        is_default: Whether this is the system default device
    """
    device_id: str
    name: str
    capabilities: DeviceCapabilities
    manufacturer: Optional[str] = None
    is_default: bool = False


@dataclass
class DeviceDataPacket:
    """
    A packet of data from a streaming device.

    This is the universal container for all device data, regardless of modality.
    It flows from devices through the DeviceRegistry to the InputQueue.

    Attributes:
        device_id: Which device produced this data
        modality: Type of data ("audio", "image", "haptic", etc.)
        raw_data: The actual data (bytes, numpy array, etc.)
        timestamp: When this data was captured
        sequence_number: Monotonic packet counter for ordering
        metadata: Additional context (confidence, format info, etc.)
    """
    device_id: str
    modality: str
    raw_data: Union[bytes, NDArray[np.float32], NDArray[np.uint8], Dict[str, Any]]
    timestamp: datetime = field(default_factory=datetime.now)
    sequence_number: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)


# Type aliases for callbacks
DataCallback = Callable[[DeviceDataPacket], None]
ErrorCallback = Callable[[str, Exception], None]
StateCallback = Callable[[DeviceState], None]


class DeviceProtocol(ABC):
    """
    Abstract protocol for all peripheral devices.

    Implementations of this protocol provide a unified interface for different
    types of hardware. The protocol supports:

    - Device discovery (enumerate_devices)
    - Connection management (connect/disconnect)
    - Streaming data (start_streaming/stop_streaming)
    - Event callbacks (data, error, state changes)

    Subclasses must implement the abstract methods for their specific hardware.

    Example usage:
        devices = MicrophoneDevice.enumerate_devices()
        mic = MicrophoneDevice()
        mic.set_data_callback(lambda pkt: process(pkt))
        await mic.connect(devices[0].device_id)
        await mic.start_streaming()
        # ... receive data via callback ...
        await mic.stop_streaming()
        await mic.disconnect()
    """

    # Class-level registry of device types to implementations
    _registry: ClassVar[Dict[DeviceType, Type["DeviceProtocol"]]] = {}

    def __init__(self) -> None:
        """Initialize base device state."""
        self._state: DeviceState = DeviceState.DISCONNECTED
        self._device_id: Optional[str] = None
        self._device_info: Optional[DeviceInfo] = None
        self._sequence_number: int = 0

        # Callbacks
        self._data_callback: Optional[DataCallback] = None
        self._error_callback: Optional[ErrorCallback] = None
        self._state_callback: Optional[StateCallback] = None

        # Streaming control
        self._streaming_task: Optional[asyncio.Task] = None
        self._stop_event: asyncio.Event = asyncio.Event()

    @classmethod
    def register_device_class(cls, device_type: DeviceType, device_class: Type["DeviceProtocol"]) -> None:
        """Register a device implementation for a device type."""
        cls._registry[device_type] = device_class
        logger.debug(f"Registered {device_class.__name__} for {device_type.name}")

    @classmethod
    def get_device_class(cls, device_type: DeviceType) -> Optional[Type["DeviceProtocol"]]:
        """Get the registered implementation for a device type."""
        return cls._registry.get(device_type)

    @property
    def state(self) -> DeviceState:
        """Current device state."""
        return self._state

    @property
    def device_id(self) -> Optional[str]:
        """ID of currently connected device."""
        return self._device_id

    @property
    def device_info(self) -> Optional[DeviceInfo]:
        """Info about currently connected device."""
        return self._device_info

    @property
    def is_streaming(self) -> bool:
        """Whether device is currently streaming data."""
        return self._state == DeviceState.STREAMING

    @property
    def is_connected(self) -> bool:
        """Whether device is connected (may or may not be streaming)."""
        return self._state in (DeviceState.CONNECTED, DeviceState.STARTING,
                               DeviceState.STREAMING, DeviceState.STOPPING)

    def _set_state(self, new_state: DeviceState) -> None:
        """Update state and notify callback."""
        old_state = self._state
        self._state = new_state
        logger.debug(f"Device {self._device_id}: {old_state.name} -> {new_state.name}")

        if self._state_callback:
            try:
                self._state_callback(new_state)
            except Exception as e:
                logger.error(f"State callback error: {e}")

    def _emit_data(self, modality: str, raw_data: Any, metadata: Optional[Dict] = None) -> None:
        """Emit a data packet to the registered callback."""
        if not self._data_callback:
            return

        self._sequence_number += 1
        packet = DeviceDataPacket(
            device_id=self._device_id or "unknown",
            modality=modality,
            raw_data=raw_data,
            timestamp=datetime.now(),
            sequence_number=self._sequence_number,
            metadata=metadata or {}
        )

        try:
            self._data_callback(packet)
        except Exception as e:
            logger.error(f"Data callback error: {e}")
            if self._error_callback:
                self._error_callback(self._device_id or "unknown", e)

    def _emit_error(self, error: Exception) -> None:
        """Emit an error to the registered callback."""
        logger.error(f"Device {self._device_id} error: {error}")

        if self._error_callback:
            try:
                self._error_callback(self._device_id or "unknown", error)
            except Exception as e:
                logger.error(f"Error callback error: {e}")

    # ========== Callback setters ==========

    def set_data_callback(self, callback: Optional[DataCallback]) -> None:
        """Set callback for receiving data packets."""
        self._data_callback = callback

    def set_error_callback(self, callback: Optional[ErrorCallback]) -> None:
        """Set callback for receiving errors."""
        self._error_callback = callback

    def set_state_callback(self, callback: Optional[StateCallback]) -> None:
        """Set callback for state changes."""
        self._state_callback = callback

    # ========== Abstract methods (must be implemented by subclasses) ==========

    @classmethod
    @abstractmethod
    def enumerate_devices(cls) -> List[DeviceInfo]:
        """
        Discover available devices of this type.

        Returns:
            List of DeviceInfo for all discovered devices
        """
        pass

    @abstractmethod
    async def connect(self, device_id: str) -> bool:
        """
        Connect to a specific device.

        Args:
            device_id: ID from enumerate_devices()

        Returns:
            True if connection succeeded
        """
        pass

    @abstractmethod
    async def disconnect(self) -> None:
        """
        Disconnect from the current device.

        Stops streaming if active and releases all resources.
        """
        pass

    @abstractmethod
    async def start_streaming(self) -> bool:
        """
        Begin streaming data from the device.

        Data is delivered via the data callback set with set_data_callback().

        Returns:
            True if streaming started successfully
        """
        pass

    @abstractmethod
    async def stop_streaming(self) -> None:
        """
        Stop streaming data from the device.

        Device remains connected and can resume streaming.
        """
        pass

    # ========== Optional methods with default implementations ==========

    async def configure(self, **settings) -> bool:
        """
        Configure device settings.

        Override in subclass to support device-specific configuration.

        Args:
            **settings: Device-specific settings

        Returns:
            True if configuration succeeded
        """
        logger.debug(f"Device {self._device_id}: configure() not implemented")
        return True

    def get_stats(self) -> Dict[str, Any]:
        """
        Get device statistics.

        Override in subclass for device-specific stats.

        Returns:
            Dict of statistics
        """
        return {
            "device_id": self._device_id,
            "state": self._state.name,
            "packets_sent": self._sequence_number,
        }


__all__ = [
    "DeviceType",
    "DeviceState",
    "DeviceCapabilities",
    "DeviceInfo",
    "DeviceDataPacket",
    "DeviceProtocol",
    "DataCallback",
    "ErrorCallback",
    "StateCallback",
]
