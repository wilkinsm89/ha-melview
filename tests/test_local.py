"""Tests for direct LAN control through the Wi-Fi Control adapter."""

from typing import Any

import pytest
from aiohttp import ClientError
from homeassistant.components.climate import (
    DOMAIN as CLIMATE_DOMAIN,
)
from homeassistant.components.climate import (
    SERVICE_SET_FAN_MODE,
    SERVICE_SET_HVAC_MODE,
    SERVICE_SET_TEMPERATURE,
    SERVICE_TURN_OFF,
    SERVICE_TURN_ON,
    HVACMode,
)
from homeassistant.const import ATTR_ENTITY_ID, ATTR_TEMPERATURE
from homeassistant.core import HomeAssistant
from pytest_homeassistant_custom_component.common import MockConfigEntry

from custom_components.melview import local
from custom_components.melview.melview import MelViewAuthError, MelViewDevice

from .conftest import LOCAL_IP, NEW_LOCAL_IP, MelViewApi, setup_entry

CLIMATE = "climate.ducted_ac"

# Frames captured from commands MelView sent to a MAC-578 adapter
CAPTURED_FRAMES = {
    "TS23": "fc410130100104000000080000000000000000ae00c3",
    "TS22.50": "fc410130100104000000190000000000000000ad00b3",
    "FS0.00": "fc410130100108000000000000000000000000000075",
    "MD3": "fc410130100102000003000000000000000000000078",
    "MD8": "fc410130100102000008000000000000000000000073",
    "PW0": "fc41013010010100000000000000000000000000007c",
}
# Status frames captured from the same adapter, unit off
CAPTURED_SETTINGS = "fc62013010020000000319000300008cad46000000bd"
CAPTURED_SENSORS = "fc620130100300000e0001b0b0fe420001f282000036"


def lsv(*frames: str) -> str:
    """Wrap status frames like the adapter's reply."""
    codes = "".join(f"<CODE><VALUE>{frame}</VALUE></CODE>" for frame in frames)
    return f"<LSV>{codes}</LSV>"


def status_frame(group: int, **values: int) -> str:
    """Build a status frame with the given byte values (b<index>=value)."""
    frame = bytearray(22)
    frame[0:6] = bytes.fromhex("fc62013010") + bytes([group])
    for key, value in values.items():
        frame[int(key[1:])] = value
    frame[21] = local.checksum(frame)
    return frame.hex()


async def call(hass: HomeAssistant, service: str, **data: Any) -> None:
    """Call a climate action on the test entity."""
    await hass.services.async_call(
        CLIMATE_DOMAIN, service, {ATTR_ENTITY_ID: CLIMATE, **data}, blocking=True
    )


@pytest.fixture
async def device(
    hass: HomeAssistant, adapter_api: MelViewApi, local_config_entry: MockConfigEntry
) -> MelViewDevice:
    """Set up a zoneless unit with local commands on."""
    await setup_entry(hass, local_config_entry)
    return local_config_entry.runtime_data[0].device


@pytest.mark.parametrize(("command", "frame"), CAPTURED_FRAMES.items())
def test_frames_match_captures(command: str, frame: str) -> None:
    """Commands are translated into the same frames MelView sends."""
    assert local.build_frame(command)[0].hex() == frame


@pytest.mark.parametrize("command", ["Z11", "AV3", "PW", "TSx"])
def test_commands_without_local_frame(command: str) -> None:
    """Commands with no local equivalent are left to MelView."""
    assert local.build_frame(command) is None


def test_parse_captured_status() -> None:
    """A captured status reply decodes to the fields MelView reports."""
    assert local.parse_status(lsv(CAPTURED_SETTINGS, CAPTURED_SENSORS)) == {
        "power": 0,
        "setmode": 3,
        "setfan": 0,
        "settemp": "22.5",
        "fault": "",
        "error": "ok",
        "roomtemp": "24",
    }


def test_parse_status_fallbacks() -> None:
    """Whole-degree fields are used when the fine ones are missing."""
    data = local.parse_status(
        lsv(
            status_frame(2, b8=1, b9=0, b10=11, b11=2),
            status_frame(3, b8=12, b10=0x9C),
            # A corrupted frame is ignored
            CAPTURED_SETTINGS[:-2] + "00",
        )
    )

    assert data["setmode"] == 8
    assert data["settemp"] == "20"
    assert data["roomtemp"] == "22"
    assert data["outdoortemp"] == "14"


def test_parse_status_without_settings() -> None:
    """A reply with no settings frame is an error."""
    with pytest.raises(local.LocalError, match="No settings frame"):
        local.parse_status(lsv(CAPTURED_SENSORS))


def test_decrypt_known_payload() -> None:
    """Payloads use AES-128-CBC with the factory key and ISO 7816-4 padding."""
    payload = "AAECAwQFBgcICQoLDA0OD5DG2SQFhhQQayEaAQT5w+s="

    assert local.decrypt(payload) == "<CSV></CSV>"
    assert local.decrypt(local.encrypt("<CSV>x</CSV>")) == "<CSV>x</CSV>"


@pytest.mark.parametrize(
    ("payload", "message"), [("*", "base64"), ("AAAA", "Malformed")]
)
def test_decrypt_malformed(payload: str, message: str) -> None:
    """Replies that cannot be decrypted are local errors."""
    with pytest.raises(local.LocalError, match=message):
        local.decrypt(payload)


async def test_commands_go_to_adapter(
    hass: HomeAssistant, adapter_api: MelViewApi, device: MelViewDevice
) -> None:
    """Supported commands are applied over the LAN without MelView."""
    await call(hass, SERVICE_SET_TEMPERATURE, **{ATTR_TEMPERATURE: 22.5})
    await call(hass, SERVICE_SET_HVAC_MODE, hvac_mode=HVACMode.COOL)
    await call(hass, SERVICE_SET_FAN_MODE, fan_mode="low")
    await call(hass, SERVICE_TURN_OFF)

    assert adapter_api.commands == []
    assert adapter_api.adapter_frames == [
        CAPTURED_FRAMES["TS22.50"],
        CAPTURED_FRAMES["MD3"],
        "fc410130100108000000000200000000000000000073",
        CAPTURED_FRAMES["PW0"],
    ]
    state = hass.states.get(CLIMATE)
    assert state.state == HVACMode.OFF
    assert state.attributes[ATTR_TEMPERATURE] == 22.5


async def test_status_read_from_adapter(
    hass: HomeAssistant, adapter_api: MelViewApi, device: MelViewDevice
) -> None:
    """After the first MelView refresh, status comes from the adapter."""
    adapter_api.unit["settemp"] = "23.5"
    adapter_api.status["unitcommand.aspx"] = 500

    assert await device.async_refresh_device_info()

    assert device._json["settemp"] == "23.5"
    assert device._json["outdoortemp"] == "14"
    assert device._json["standby"] == 0


@pytest.mark.parametrize(
    "break_adapter",
    [
        lambda api: api.unreachable_ips.add(LOCAL_IP),
        lambda api: api.errors.update(smart=TimeoutError()),
        lambda api: api.status.update(smart=500),
        lambda api: setattr(api, "adapter", False),
        lambda api: setattr(api, "adapter_ignores_writes", True),
    ],
)
async def test_command_falls_back_to_melview(
    hass: HomeAssistant,
    adapter_api: MelViewApi,
    device: MelViewDevice,
    caplog: pytest.LogCaptureFixture,
    break_adapter: Any,
) -> None:
    """If the adapter fails, the command goes through MelView instead."""
    break_adapter(adapter_api)

    await call(hass, SERVICE_TURN_OFF)
    await call(hass, SERVICE_TURN_ON)

    assert adapter_api.commands == ["PW0", "PW1"]
    assert caplog.text.count("local command PW0 failed, using MelView") == 1
    assert "command PW1 failed" in caplog.text


async def test_local_control_restored(
    hass: HomeAssistant,
    adapter_api: MelViewApi,
    device: MelViewDevice,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Local control resumes once the adapter answers again."""
    adapter_api.unreachable_ips.add(LOCAL_IP)
    await call(hass, SERVICE_TURN_OFF)
    adapter_api.unreachable_ips.clear()

    await call(hass, SERVICE_TURN_ON)

    assert adapter_api.commands == ["PW0"]
    assert "Ducted AC: local control restored" in caplog.text


async def test_adapter_reply_without_payload(
    hass: HomeAssistant,
    adapter_api: MelViewApi,
    device: MelViewDevice,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """A reply with no encrypted payload is a local failure."""
    adapter_api.adapter = False
    adapter_api.status["unitcommand.aspx"] = 500

    with pytest.raises(Exception, match="500"):
        await device.async_refresh_device_info()

    assert "Adapter reply had no ESV payload" in caplog.text


async def test_melview_down_uses_adapter(
    hass: HomeAssistant, adapter_api: MelViewApi, device: MelViewDevice
) -> None:
    """When MelView fails, the state is read from the adapter instead."""
    device._json = None
    adapter_api.errors["unitcommand.aspx"] = ClientError("DNS failure")

    assert await device.async_refresh_device_info()
    assert device._json["settemp"] == "20"


async def test_melview_auth_error_not_masked(
    hass: HomeAssistant, adapter_api: MelViewApi, device: MelViewDevice
) -> None:
    """Rejected credentials still raise, so reauthentication starts."""
    device._json = None
    adapter_api.status["unitcommand.aspx"] = 401
    adapter_api.accept_logins = adapter_api.login_count

    with pytest.raises(MelViewAuthError):
        await device.async_refresh_device_info()


async def test_adapter_follows_ip_change(
    hass: HomeAssistant, adapter_api: MelViewApi, device: MelViewDevice
) -> None:
    """When the adapter moves, the next command goes to its new address."""
    adapter_api.unreachable_ips.add(LOCAL_IP)
    adapter_api.caps["localip"] = NEW_LOCAL_IP

    await call(hass, SERVICE_TURN_OFF)
    await call(hass, SERVICE_TURN_ON)

    assert adapter_api.commands == ["PW0"]
    assert adapter_api.local_command_ips[-1] == NEW_LOCAL_IP
    assert adapter_api.unit["power"] == 1


async def test_other_unit_types_keep_command_key(
    hass: HomeAssistant, adapter_api: MelViewApi, local_config_entry: MockConfigEntry
) -> None:
    """Units that aren't air conditioners only get the cloud's command key."""
    adapter_api.caps["unittype"] = "ATW"
    await setup_entry(hass, local_config_entry)

    await call(hass, SERVICE_TURN_OFF)

    assert adapter_api.commands == ["PW0"]
    assert adapter_api.adapter_frames == []
