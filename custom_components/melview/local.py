"""Direct LAN control of Mitsubishi Electric Wi-Fi Control adapters.

The adapter's /smart endpoint takes an <ESV> document: base64 of a random
16-byte IV followed by AES-128-CBC ciphertext, using the adapters' shared
factory key and ISO 7816-4 padding. The plaintext is <CSV> XML whose
<CODE><VALUE> elements carry Mitsubishi CN105 serial frames. An empty <CSV>
returns the unit's current state as CN105 status frames in an <LSV> reply.

This lets commands be built and status read locally, with no cloud round trip.
The <CONNECT> element is deliberately never sent: the adapter stores it as a
setting (it controls the cloud connection), it is not a request flag.
"""

import asyncio
import base64
import logging
import os
import re
import time

from aiohttp import ClientError, ClientSession, ClientTimeout
from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes

_LOGGER = logging.getLogger(__name__)

FACTORY_KEY = b"unregistered\0\0\0\0"
REQUEST = '<?xml version="1.0" encoding="UTF-8"?><ESV>{}</ESV>'
REQUEST_TIMEOUT = ClientTimeout(total=10)

# The adapter can lock up when requests arrive back to back
MIN_REQUEST_GAP_S = 1.0
# A write shows up on a status poll after about 3 s; wait up to this long
SETTLE_TIMEOUT_S = 10.0
SETTLE_POLL_S = 1.5

# CN105 SET flag bits (byte 6) for each cloud command
FLAG_POWER = 0x01
FLAG_MODE = 0x02
FLAG_TEMP = 0x04
FLAG_FAN = 0x08

COMMAND_RE = re.compile(r"^(PW|MD|TS|FS)(\d+(?:\.\d+)?)$")
VALUE_RE = re.compile(r"<VALUE>([0-9A-Fa-f]+)</VALUE>")
ESV_RE = re.compile(r"<ESV>(.*?)</ESV>", re.DOTALL)


class LocalError(Exception):
    """The adapter could not be reached or its reply was not understood."""


def _crypt(data: bytes, iv: bytes, encrypt: bool) -> bytes:
    cipher = Cipher(algorithms.AES(FACTORY_KEY), modes.CBC(iv))
    op = cipher.encryptor() if encrypt else cipher.decryptor()
    return op.update(data) + op.finalize()


def encrypt(plaintext: str) -> str:
    data = plaintext.encode() + b"\x80"
    data += b"\0" * (-len(data) % 16)
    iv = os.urandom(16)
    return base64.b64encode(iv + _crypt(data, iv, True)).decode()


def decrypt(payload: str) -> str:
    try:
        raw = base64.b64decode(payload, validate=True)
    except ValueError as err:
        raise LocalError("Reply from adapter is not base64") from err
    if len(raw) < 32 or len(raw) % 16:
        raise LocalError("Malformed reply from adapter")
    data = _crypt(raw[16:], raw[:16], False).rstrip(b"\0")
    if data.endswith(b"\x80"):
        data = data[:-1]
    return data.decode(errors="replace")


def checksum(frame: bytes) -> int:
    return (0x100 - sum(frame[1:21]) % 0x100) % 0x100


def build_frame(command: str) -> tuple[bytes, dict] | None:
    """Translate one cloud command (PW1, MD3, TS22.50, FS2.00) into a CN105
    SET frame, plus the status fields it is expected to produce.

    Returns None for commands that have no local equivalent here (zones,
    Lossnay, timers), which then go through the cloud as before.
    """
    match = COMMAND_RE.match(command)
    if not match:
        return None
    code, value = match.group(1), float(match.group(2))
    frame = bytearray(22)
    frame[0:6] = bytes.fromhex("fc4101301001")
    if code == "PW":
        frame[6], frame[8] = FLAG_POWER, int(value)
        expect = {"power": int(value)}
    elif code == "MD":
        frame[6], frame[9] = FLAG_MODE, int(value)
        expect = {"setmode": int(value)}
    elif code == "TS":
        half = (value * 2) % 2 >= 1
        frame[6] = FLAG_TEMP
        frame[10] = ((31 - int(value)) & 0x0F) | (0x10 if half else 0)
        frame[19] = 0x80 + round(value * 2)
        expect = {"settemp": value}
    else:  # FS
        frame[6], frame[11] = FLAG_FAN, int(value)
        expect = {"setfan": int(value)}
    frame[21] = checksum(frame)
    return bytes(frame), expect


def _half(byte: int) -> float:
    return (byte - 0x80) / 2


def _fmt(value: float) -> str:
    """Format like the cloud does: '22.5', '24'."""
    return f"{value:g}"


def parse_status(plaintext: str) -> dict:
    """Turn an <LSV> status reply into the same fields the cloud returns."""
    frames = {}
    for value in VALUE_RE.findall(plaintext):
        frame = bytes.fromhex(value)
        if (
            len(frame) == 22
            and frame[0] == 0xFC
            and frame[1] == 0x62
            and frame[21] == checksum(frame)
        ):
            frames[frame[5]] = frame
    general = frames.get(0x02)
    if general is None:
        raise LocalError("No settings frame in adapter reply")

    mode = general[9] & 0x07 or 8  # 0 is auto, which the cloud calls 8
    settemp = _half(general[16]) if general[16] else 31 - (general[10] & 0x0F)
    data = {
        "power": general[8],
        "setmode": mode,
        "setfan": general[11],
        "settemp": _fmt(settemp),
        "fault": "",
        "error": "ok",
    }
    sensors = frames.get(0x03)
    if sensors is not None:
        room = _half(sensors[11]) if sensors[11] >= 0x80 else 10 + sensors[8]
        data["roomtemp"] = _fmt(room)
        if sensors[10] not in (0x00, 0x01):  # 0x01 is sent while the unit is off
            data["outdoortemp"] = _fmt(_half(sensors[10]))
    return data


def matches(data: dict, expect: dict) -> bool:
    for key, want in expect.items():
        have = data.get(key)
        if key == "settemp":
            if have is None or abs(float(have) - want) > 0.01:
                return False
        elif have != want:
            return False
    return True


class LocalAdapter:
    """Talks to one adapter over the LAN, one request at a time."""

    def __init__(self, host: str, session: ClientSession):
        self.host = host
        self._session = session
        self._lock = asyncio.Lock()
        self._last_request = 0.0

    async def _request(self, plaintext: str) -> str:
        async with self._lock:
            gap = MIN_REQUEST_GAP_S - (time.monotonic() - self._last_request)
            if gap > 0:
                await asyncio.sleep(gap)
            try:
                async with self._session.post(
                    f"http://{self.host}/smart",
                    data=REQUEST.format(encrypt(plaintext)),
                    headers={"Content-Type": "text/plain;charset=UTF-8"},
                    timeout=REQUEST_TIMEOUT,
                ) as resp:
                    body = await resp.text()
                    status = resp.status
            except (ClientError, OSError, TimeoutError) as err:
                raise LocalError(
                    f"Adapter {self.host} unreachable: {str(err) or 'timed out'}"
                ) from err
            finally:
                self._last_request = time.monotonic()
        if status != 200:
            raise LocalError(f"Adapter {self.host} returned HTTP {status}")
        match = ESV_RE.search(body)
        if not match:
            raise LocalError("Adapter reply had no ESV payload")
        return decrypt(match.group(1))

    async def async_get_status(self) -> dict:
        return parse_status(await self._request("<CSV></CSV>"))

    async def async_send(self, frame: bytes, expect: dict) -> dict:
        """Send a SET frame and wait for the unit to report the new state.

        The reply to a write echoes cached state, so poll until the expected
        fields appear. Raise LocalError if they never do.
        """
        await self._request(f"<CSV><CODE><VALUE>{frame.hex()}</VALUE></CODE></CSV>")
        deadline = time.monotonic() + SETTLE_TIMEOUT_S
        data = {}
        while time.monotonic() < deadline:
            await asyncio.sleep(SETTLE_POLL_S)
            data = await self.async_get_status()
            if matches(data, expect):
                return data
        raise LocalError(f"Unit did not report {expect} (last status {data})")
