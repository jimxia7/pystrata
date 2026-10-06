import numpy as np
import re
from pathlib import Path
import warnings

# Gravity in m/sec²
from scipy.constants import g as GRAVITY

def to_str(s):
    """Parse a string and strip the extra characters."""
    return str(s).strip()


def _to_float(s):
    """Try to parse a float."""
    try:
        return float(s)
    except ValueError:
        return np.nan
    
def to_int_or_str(value):
    try:
        return int(value)
    except (TypeError, ValueError):
        return value

def parse_fixed_width(types, lines):
    """Parse a fixed width line."""
    values = []
    line = []
    for width, parser in types:
        if not line:
            line = lines.pop(0).replace("\n", "")

        values.append(parser(line[:width]))
        line = line[width:]

    return values


def split_line(line, parsers, sep=" "):
    """Split a line into pieces and parse the strings."""
    parts = [part for part in line.split(sep) if part]
    values = [parser(part) for parser, part in zip(parsers, parts)]
    return values if len(values) > 1 else values[0]


def _parse_at2_header(line):
    """Parse the point count and time step from the header of an AT2 file.

    Both of the PEER NGA layouts are supported::

        4096    0.0100    NPTS, DT
        NPTS=   5346, DT=   .0100 SEC,

    as are variations that reverse the order of the two values, or that omit
    the commas separating them.

    Parameters
    ----------
    line: str
        Fourth line of an AT2 file.

    Returns
    -------
    npts: int
        Number of points in the time series.
    time_step: float
        Time step of the time series [sec].
    """

    # Integers and floats, including values without a leading digit (e.g., ".0100")
    # and Fortran style exponents (e.g., "1.0D-2").
    _RE_NUMBER = re.compile(r"[+-]?(?:\d+\.?\d*|\.\d+)(?:[eEdD][+-]?\d+)?")

    # Values that follow their label -- e.g., "NPTS= 5346" or "DT .0100". Each
    # value is located by its own label, so their order does not matter.
    found = {}
    for key in ("NPTS", "DT"):
        m = re.search(
            r"\b" + key + r"\b\s*[=:]?\s*(" + _RE_NUMBER.pattern + ")",
            line,
            re.IGNORECASE,
        )
        if m:
            found[key] = _to_float(m.group(1))

    if len(found) < 2:
        # Values that precede their labels -- e.g., "4096  0.0100  NPTS, DT".
        values = [_to_float(v) for v in _RE_NUMBER.findall(line)]
        if len(values) < 2:
            raise ValueError(f"Unable to parse NPTS and DT from AT2 header: {line!r}")

        values = values[:2]
        upper = line.upper()
        pos = {key: upper.find(key) for key in ("NPTS", "DT")}
        if all(p >= 0 for p in pos.values()):
            # Pair the values with the labels by order of appearance.
            keys = sorted(pos, key=lambda key: pos[key])
        else:
            # Unlabeled, so rely on magnitude: the time step is the smaller of
            # the two.
            keys = ["DT", "NPTS"] if values[0] < values[1] else ["NPTS", "DT"]

        found = dict(zip(keys, values))

    return int(found["NPTS"]), found["DT"]

def read_smc_file(filename):
    """Read an SMC formatted time series.

    Format of the time series is provided by:
        https://escweb.wr.usgs.gov/nsmp-data/smcfmt.html

    Parameters
    ----------
    filename: str
        Filename to open.
    scale: float, default: 1.
        Scale factor to apply to the motion.
    """

    lines = Path(filename).read_text(encoding="utf-8").splitlines()

    # 11 lines of strings
    lines_str = [lines.pop(0) for _ in range(11)]

    if lines_str[0].strip() != "2 CORRECTED ACCELEROGRAM":
        raise RuntimeWarning("Loading uncorrected SMC file.")

    m = re.search("station =(.+)component=(.+)", lines_str[5])
    description = "; ".join([g.strip() for g in m.groups()])

    # 6 lines of (8i10) formatted integers
    values_int = parse_fixed_width(
        48 * [(10, int)], [lines.pop(0) for _ in range(6)]
    )
    count_comment = values_int[15]
    count = values_int[16]

    # 10 lines of (5e15.7) formatted floats
    values_float = parse_fixed_width(
        50 * [(15, float)], [lines.pop(0) for _ in range(10)]
    )
    time_step = 1 / values_float[1]

    # Skip comments
    lines = lines[count_comment:]

    accels = np.array(
        parse_fixed_width(
            count
            * [
                (10, float),
            ],
            lines,
        )
    )

    return description,time_step,accels

def read_v2_file(filename,channel):
    """Read a CSMIP/COSMOS "Volume 2" (``.V2``) formatted time series.

    These files are distributed by the California Geological Survey / CESMD
    and contain instrument- and baseline-corrected acceleration, velocity,
    and displacement blocks. Only the acceleration block is read.

    A single ``.V2`` file frequently bundles every channel (component)
    recorded at a station, each terminated by a line such as
    ``/&  ---------- End of data for channel  1 ----------``. Use *channel*
    to pick which one to load.

    Rather than depend on the exact number of header lines -- which varies
    between processing vintages -- the parser locates the acceleration data
    descriptor line, e.g.::

    15200 points of accel data equally spaced at  .005 sec, in cm/sec2. (8f10.6)

    and reads the number of points, time step, fixed-column width, and units
    from it. Accelerations reported in cm/sec/sec are converted to units of
    *g* so the resulting motion matches the other ``load_*`` constructors.

    Parameters
    ----------
    filename: str
        Filename to open.
    scale: float, default: 1.
        Scale factor to apply to the motion (after unit conversion).
    channel: int or str, default: 1
        Which channel to read from a multi-channel file. An ``int`` is the
        1-based position of the channel in the file; a ``str`` is matched
        (case-insensitively, as a substring) against the channel's
        component label, e.g. ``"360"`` or ``"Up"``.

    Returns
    -------
    :class:`TimeSeriesMotion`
    """

    text = Path(filename).read_text()

    # Split the file into per-channel blocks. Each channel ends with a
    # marker line like "/&  ---------- End of data for channel 1 ----------".
    # Older (1970s-80s) CSMIP files write the marker and the "Chan N:"
    # headers in all caps, so match case-insensitively.
    blocks = [
        b
        for b in re.split(r"(?im)^.*End of data for chan(?:nel)?.*$", text)
        if b.strip()
    ]
    if not blocks:
        blocks = [text]

    def _station_component(block):
        # The header repeats a line of "<record-id>  <station>  Chan N: <comp>"
        m = re.search(
            r"(?im)^\s*\S+\s{2,}(.+?)\s{2,}Chan\s*\d+:\s*(.+?)\s*$", block
        )
        if m:
            return m.group(1).strip(), m.group(2).strip()
        m = re.search(r"Chan\s*\d+:\s*(.+)", block, re.IGNORECASE)
        comp = re.split(r"\s{2,}", m.group(1).strip())[0] if m else ""
        return "", comp

    parsed = [_station_component(b) for b in blocks]
    components = [comp for _, comp in parsed]

    if isinstance(channel, str):
        matches = [
            i for i, c in enumerate(components) if channel.lower() in c.lower()
        ]
        if not matches:
            raise ValueError(
                f"No channel matching {channel!r} in '{filename}'. "
                f"Available components: {components}."
            )
        index = matches[0]
    else:
        index = int(channel) - 1
        if not 0 <= index < len(blocks):
            raise ValueError(
                f"Channel {channel} is out of range for '{filename}', which "
                f"has {len(blocks)} channel(s): {components}."
            )

    lines = blocks[index].splitlines()
    station, component = parsed[index]
    description = "; ".join(part for part in (station, component) if part)

    for i, line in enumerate(lines):
        m = re.search(
            r"(\d+)\s+points of acc\w* data.*?equally spaced at\s+"
            r"([0-9.]+)\s*sec",
            line,
            re.IGNORECASE,
        )
        if m:
            break
    else:
        raise ValueError(
            f"Could not find an acceleration data block in '{filename}'."
        )

    count = int(m.group(1))
    time_step = float(m.group(2))
    width_match = re.search(r"\(\s*\d*[fFeEgG](\d+)\.", line)
    width = int(width_match.group(1)) if width_match else 10
    in_cgs = "cm/s" in line.lower()
    data_lines = lines[i + 1 :]
    accels = np.array(parse_fixed_width(count * [(width, float)], data_lines))
    if in_cgs:
        # Convert cm/sec/sec to g
        accels /= GRAVITY * 100

    return description,time_step,accels

def read_v2c_file(filename, channel=1):
        """Read a CESMD/COSMOS "V2c" (``.V2c``) formatted time series.

        These files are distributed by the USGS / CESMD in the COSMOS strong
        motion data format (``Format v01.20``). Each channel is a
        self-contained record -- text header, integer header, real header,
        comment lines, and a single data block -- terminated by a marker line
        such as ``End-of-data for ChanHNE acceleration``. A file may hold one
        channel (the common CESMD download, where each component and each of
        acceleration / velocity / displacement is its own ``*.acc.V2c``,
        ``*.vel.V2c``, ``*.dis.V2c`` file) or several channels concatenated
        back-to-back. CGS downloads (e.g. ``CE47380.V2C``) also interleave
        the integrated velocity and displacement records after each channel's
        acceleration record; those are skipped, so *channel* always counts
        acceleration records only. Use *channel* to pick which one to load.

        Each channel's header blocks are introduced by self-describing lines
        such as::

             100 Real-header values follow on  20 lines, Format= (5F15.6)

        and the data by a descriptor line such as::

            26219 acceleration pts, approx  131 secs, units=cm/sec2(04),Format=(1E15.6)

        The number of points, units, and fixed-column width are read from that
        line; the time step is taken from the real header (COSMOS real-header
        entry 34) because the descriptor only gives a rounded duration.
        Accelerations reported in cm/sec/sec are converted to units of *g* so
        the resulting motion matches the other ``load_*`` constructors.

        Parameters
        ----------
        filename: str
            Filename to open.
        scale: float, default: 1.
            Scale factor to apply to the motion (after unit conversion).
        channel: int or str, default: 1
            Which channel to read from a multi-channel file. An ``int`` is the
            1-based position of the channel in the file; a ``str`` is matched
            (case-insensitively, as a substring) against the channel's
            component label (e.g. ``"360"`` or ``"Up"``) or its SEED channel
            code from the end-of-data marker (e.g. ``"HNE"``).

        Returns
        -------
        :class:`TimeSeriesMotion`
        """

        text = Path(filename).read_text()

        # Split the file into per-channel blocks. Each channel ends with a
        # marker line like "End-of-data for ChanHNE acceleration"; keep the
        # marker so the channel code on it can be used for selection.
        # Files that also carry the integrated velocity and displacement (e.g.
        # "End-of-data for chan  1 velocity data") are filtered down to the
        # acceleration records so *channel* counts sensor channels only.
        marker = re.compile(
            r"(?mi)^\s*End[- ]of[- ]data for\s+(?:Chan\s*)?(\S*).*$"
        )
        accel_desc = re.compile(
            r"(?mi)^\s*\d+\s+acc\w*\s+(?:pts|points)\b.*units\s*="
        )
        blocks = []
        codes = []
        pos = 0
        for m in marker.finditer(text):
            block = text[pos : m.start()]
            if block.strip() and accel_desc.search(block):
                blocks.append(block)
                code = m.group(1).strip()
                # "chan  1 acceleration data" yields a bare channel number, which
                # is not a SEED code; only keep alphabetic codes for matching.
                codes.append(code if not code.isdigit() else "")
            pos = m.end()
        tail = text[pos:]
        if tail.strip() and accel_desc.search(tail):
            blocks.append(tail)
            codes.append("")

        if not blocks:
            raise ValueError(
                f"Could not find any acceleration data blocks in '{filename}'."
            )

        def _station_component(block):
            # Station numbers may contain spaces (e.g. "Statn No: 05- 47380").
            m = re.search(r"Statn No:.*?Code:\s*(\S+)", block)
            station = m.group(1) if m else ""
            m = re.search(
                r"Sta\s+Chan\s*\d+:\s*([^(]+?)\s*(?:\(|Location:|$)", block
            )
            component = m.group(1).strip() if m else ""
            return station, component

        parsed = [_station_component(b) for b in blocks]
        components = [comp for _, comp in parsed]

        if isinstance(channel, str):
            key = channel.lower()
            matches = [
                i
                for i, (comp, code) in enumerate(zip(components, codes))
                if key in comp.lower() or (code and key in code.lower())
            ]
            if not matches:
                raise ValueError(
                    f"No channel matching {channel!r} in '{filename}'. "
                    f"Available components: {components}, codes: {codes}."
                )
            index = matches[0]
        else:
            index = int(channel) - 1
            if not 0 <= index < len(blocks):
                raise ValueError(
                    f"Channel {channel} is out of range for '{filename}', which "
                    f"has {len(blocks)} channel(s): {components}."
                )

        block = blocks[index]
        lines = block.splitlines()
        station, component = parsed[index]
        description = "; ".join(part for part in (station, component) if part)

        # Real header -- the time step lives here, not in the descriptor line.
        m = re.search(
            r"(\d+)\s+Real[- ]header values follow on\s+(\d+)\s+lines"
            r".*?Format\s*=\s*\(\s*\d*\s*[a-zA-Z](\d+)\.",
            block,
            re.IGNORECASE,
        )
        if not m:
            raise ValueError(
                f"Could not find a real-header block for channel {channel!r} in "
                f"'{filename}'."
            )
        n_real = int(m.group(1))
        n_real_lines = int(m.group(2))
        real_width = int(m.group(3))
        start = block[: m.start()].count("\n") + 1
        real_header = parse_fixed_width(
            n_real * [(real_width, float)],
            list(lines[start : start + n_real_lines]),
        )
        # COSMOS real-header entry 34 (1-based) is the time interval in seconds.
        time_step = real_header[33]

        if not 0 < time_step < 10:
            raise ValueError(
                f"Implausible time step {time_step} read from the real header of "
                f"'{filename}'."
            )

        # Acceleration data descriptor line, e.g.
        #   26219 acceleration pts, approx  131 secs, units=cm/sec2(04),Format=(1E15.6)
        for i, line in enumerate(lines):
            m = re.search(
                r"(\d+)\s+acc\w*\s+(?:pts|points).*?"
                r"units=\s*([^\s,()]+).*?"
                r"Format\s*=\s*\(\s*\d*\s*[a-zA-Z](\d+)\.",
                line,
                re.IGNORECASE,
            )
            if m:
                break
        else:
            raise ValueError(
                f"Could not find an acceleration data block for channel "
                f"{channel!r} in '{filename}'."
            )

        count = int(m.group(1))
        units = m.group(2)
        width = int(m.group(3))

        data_lines = lines[i + 1 :]
        accels = np.array(parse_fixed_width(count * [(width, float)], data_lines))

        if accels.size != count:
            warnings.warn(
                f"V2c file '{filename}' specifies {count} points, but "
                f"{accels.size} accelerations were read."
            )

        if "cm/s" in units.lower():
            # Convert cm/sec/sec to g
            accels /= GRAVITY * 100

        return description,time_step,accels

def read_at2_file(filename):
    """Read an AT2 formatted time series.

    The fourth line of the file provides the number of points and the time
    step. Both of the PEER NGA layouts are read::

        4096    0.0100    NPTS, DT
        NPTS=   5346, DT=   .0100 SEC,

    as are variations that reverse the order of the two values, or that
    omit the commas separating them.

    Parameters
    ----------
    filename: str
        Filename to open.
    scale: float, default: 1.
        Scale factor to apply to the motion.
    """
    with open(filename) as fp:
        next(fp)
        description = next(fp).strip()
        next(fp)
        npts, time_step = _parse_at2_header(next(fp))

        # Rows may be ragged (the last line is usually short), so parse
        # the remaining text as a flat stream of floats.
        accels = np.array(fp.read().split(), dtype=float)

    if accels.size != npts:
        warnings.warn(
            f"AT2 file '{filename}' specifies NPTS={npts}, but {accels.size} "
            "accelerations were read."
        )

    return description,time_step,accels