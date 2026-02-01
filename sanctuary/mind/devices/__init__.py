"""
Devices: Multimodal peripheral device abstraction layer.

This package provides a unified interface for connecting various peripheral
devices (cameras, microphones, sensors) to the cognitive architecture.

Key Components:
- DeviceProtocol: Abstract base class for all device implementations
- DeviceRegistry: Central management and data routing
- MicrophoneDevice: Audio input via sounddevice
- CameraDevice: Video capture via OpenCV
- SensorDevice: Generic sensor polling
- SerialSensorDevice: UART-connected microcontrollers

Data Flow:
    Device.start_streaming()
        -> Device.on_data(callback)
        -> DeviceRegistry._on_device_data(packet)
        -> InputQueue.add_input(modality, data, metadata)
        -> PerceptionSubsystem.process_input()

Example Usage:
    from sanctuary.mind.devices import DeviceRegistry, MicrophoneDevice, CameraDevice

    # Create registry
    registry = DeviceRegistry()
    registry.register_device_class(DeviceType.MICROPHONE, MicrophoneDevice)
    registry.register_device_class(DeviceType.CAMERA, CameraDevice)

    # Connect to InputQueue
    registry.set_input_queue(input_queue)

    # Discover and connect devices
    all_devices = registry.enumerate_all_devices()
    await registry.connect_device(DeviceType.MICROPHONE, "mic_0", auto_stream=True)

Windows Compatibility:
- CameraDevice uses DirectShow (CAP_DSHOW) backend
- MicrophoneDevice uses sounddevice with WASAPI
- SerialSensorDevice handles COM ports natively
"""

from .protocol import (
    DeviceType,
    DeviceState,
    DeviceCapabilities,
    DeviceInfo,
    DeviceDataPacket,
    DeviceProtocol,
    DataCallback,
    ErrorCallback,
    StateCallback,
)

from .registry import (
    DeviceRegistry,
    DeviceEntry,
)

from .audio_device import (
    MicrophoneDevice,
    HAS_SOUNDDEVICE,
)

from .camera_device import (
    CameraDevice,
    HAS_OPENCV,
)

from .sensor_device import (
    SensorType,
    SensorReading,
    SensorDevice,
    SerialSensorDevice,
    HAS_SERIAL,
)

__all__ = [
    # Protocol types
    "DeviceType",
    "DeviceState",
    "DeviceCapabilities",
    "DeviceInfo",
    "DeviceDataPacket",
    "DeviceProtocol",
    "DataCallback",
    "ErrorCallback",
    "StateCallback",
    # Registry
    "DeviceRegistry",
    "DeviceEntry",
    # Audio
    "MicrophoneDevice",
    "HAS_SOUNDDEVICE",
    # Camera
    "CameraDevice",
    "HAS_OPENCV",
    # Sensors
    "SensorType",
    "SensorReading",
    "SensorDevice",
    "SerialSensorDevice",
    "HAS_SERIAL",
]
