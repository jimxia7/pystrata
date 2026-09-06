# The MIT License (MIT)
#
# Copyright (c) 2016-2018 Albert Kottke
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
"""Classes used to define input motions."""

import enum
import re
import warnings
from pathlib import Path

import numpy as np
import pyrvt
import pykooh

# Gravity in m/sec²
from scipy.constants import g as GRAVITY
# from .kappa import DEFAULT_KAPPA_FREQS, _compute_fourier_spectrum

_trapezoid = np.trapezoid
DEFAULT_KAPPA_FREQS = np.logspace(np.log10(10), np.log10(30), 100)

def _compute_fourier_spectrum(time_step,
                              accels,
                              freqs = None,
                              fa_length=None, 
                              ko_bandwidth = None):
    """Compute the Fourier Amplitude Spectrum of the time series."""

    if fa_length is None:
        # Use the next power of 2 for the length
        n = 1
        while n < accels.size:
            n <<= 1
    else:
        n = fa_length
    
    fft_freqs = np.fft.rfftfreq(n, d = time_step)

    if freqs is None:
        freqs = fft_freqs

    if ko_bandwidth is None:
        FAS = np.interp(freqs, 
                        fft_freqs, 
                        np.fft.rfft(accels, n))
    else:
        FAS = pykooh.smooth(freqs, 
                            fft_freqs, 
                            np.fft.rfft(accels, n),
                            ko_bandwidth)

    return freqs, FAS


# Integers and floats, including values without a leading digit (e.g., ".0100")
# and Fortran style exponents (e.g., "1.0D-2").
_RE_NUMBER = re.compile(r"[+-]?(?:\d+\.?\d*|\.\d+)(?:[eEdD][+-]?\d+)?")


def _to_float(text):
    """Convert a string to a float, permitting a Fortran style exponent."""
    return float(text.replace("D", "E").replace("d", "e"))


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


class WaveField(enum.Enum):
    outcrop = 0
    within = 1
    incoming_only = 2


class Motion:
    def __init__(self, freqs=None):
        object.__init__(self)

        self._freqs = None if freqs is None else np.array(freqs)
        self._pga = None
        self._pgv = None
        self._arias_intensity = None
        self._cav = None

    @property
    def freqs(self):
        return self._freqs

    @property
    def angular_freqs(self):
        return 2 * np.pi * self.freqs

    @property
    def pgv(self):
        """Peak ground velocity [cm/sec]."""
        if self._pgv is None:
            self._pgv = self.calc_pgv()
        return self._pgv

    @property
    def pga(self):
        """Peak ground acceleration [g]"""
        if self._pga is None:
            self._pga = self.calc_pga()
        return self._pga

    @property
    def arias_intensity(self):
        """Arias intensity [m/s]."""
        if self._arias_intensity is None:
            self._arias_intensity = self.calc_arias_intensity()
        return self._arias_intensity

    @property
    def cav(self):
        """Cumulative absolute velocity [m/s]."""
        if self._cav is None:
            self._cav = self.calc_cav()
        return self._cav

    def calc_peak(self, tf: np.typing.ArrayLike | None = None, **kwargs) -> float:
        raise NotImplementedError


class TimeSeriesMotion(Motion):
    """Time-series motion for time series based site response analysis."""

    def __init__(
        self, filename: str, description: str, time_step: float, accels
    ):
        """Initialize the class from specified acceleration values.

        The *filename* and *description* parameters are only used to help track the
        motion.

        Parameters
        ----------
        filename: str
            Source of data
        description: str
            Description to store helpful information
        time_step: float
            Time step of the accleration values
        accels: array_like
            Accelerations in units of *g*
        fa_length: optional int
            Length to use for the Fourier amplitude spectrum. If not provided, will be
            automatically computed to the next power of 2.
        """
        Motion.__init__(self)

        self._filename = filename
        self._description = description
        self._time_step = time_step
        self._accels = np.asarray(accels)
        self._kappa = None
        self._fourier_amps = None

    @property
    def accels(self):
        return self._accels

    @property
    def filename(self):
        return self._filename

    @property
    def description(self):
        return self._description

    @property
    def time_step(self):
        return self._time_step

    @property
    def times(self):
        return self._time_step * np.arange(self._accels.size)

    @property
    def freqs(self):
        
        """Return the frequencies."""
        if self._freqs is None:
            self._calc_fourier_spectrum()

        return self._freqs

    @property
    def fourier_amps(self):
        """Return the frequencies."""
        if self._fourier_amps is None:
            self._calc_fourier_spectrum()

        # Normalize the Fourier amplitude by the time step
        return self.time_step * self._fourier_amps

    def calc_time_series(self, tf=None):
        self._calc_fourier_spectrum()
        if tf is None:
            ts = np.fft.irfft(self.fourier_amps / self.time_step)
        else:
            ts = np.fft.irfft(tf * self.fourier_amps / self.time_step)
        return ts

    def calc_pgv(self, tf=None):
        tf = 1 if tf is None else np.asarray(tf)
        # Compute transfer function from acceleration to velocity
        # only over non-zero frequencies
        mask = ~np.isclose(self.angular_freqs, 0)
        tf_av = np.zeros_like(mask, dtype=complex)
        tf_av[mask] = 1 / (self.angular_freqs[mask] * 1j)
        return GRAVITY * 100 * self.calc_peak(tf_av * tf)

    def calc_pga(self, tf=None):
        return self.calc_peak(tf)

    def calc_peak(self, tf=None, **kwargs):
        ts = self.calc_time_series(tf)
        return np.abs(ts).max()

    def calc_arias_intensity(self, tf=None):
        tf = 1 if tf is None else np.asarray(tf)
        ts = self.calc_time_series(tf)
        return np.pi * GRAVITY / 2 * _trapezoid(ts**2, dx=self.time_step)

    def calc_cav(self, tf=None):
        tf = 1 if tf is None else np.asarray(tf)
        ts = self.calc_time_series(tf)
        return GRAVITY * _trapezoid(np.abs(ts), dx=self.time_step)

    def calc_osc_accels(self, osc_freqs, osc_damping=0.05, tf=None):
        """Compute the pseudo-acceleration spectral response of an oscillator with a
        specific frequency and damping.

        Parameters
        ----------
        osc_freq : float
            Frequency of the oscillator (Hz).
        osc_damping : float
            Fractional damping of the oscillator (dec). For example, 0.05 for a
            damping ratio of 5%.
        tf : array_like, optional
            Transfer function to be applied to motion prior calculation of the
            oscillator response.

        Returns
        -------
        spec_accels : :class:`numpy.ndarray`
            Peak pseudo-spectral acceleration of the oscillator
        """
        if tf is None:
            tf = np.ones_like(self.freqs)
        else:
            tf = np.asarray(tf).astype(complex)

        resp = np.array(
            [
                self.calc_peak(tf * self._calc_sdof_tf(of, osc_damping))
                for of in osc_freqs
            ]
        )
        return resp

    def _calc_fourier_spectrum(self, 
                               freqs = None, 
                               fa_length = None, 
                               ko_bandwidth = None):
        """Compute the Fourier Amplitude Spectrum of the time series."""

        self._freqs, self._fourier_amps = _compute_fourier_spectrum(
            self.time_step,
            self._accels,
            freqs = freqs,
            fa_length= fa_length,
            ko_bandwidth= ko_bandwidth
        )

    @property
    def kappa(self):
    
        if self._kappa is None:
            self._calc_kappa()
            
        return self._kappa

    def _calc_kappa(self,
                    freqs_range = DEFAULT_KAPPA_FREQS, 
                    fa_length=None, 
                    ko_bandwidth = None):
        
        _,fas = _compute_fourier_spectrum(
            self.time_step,
            self.accels,
            freqs = freqs_range,
            fa_length=fa_length,
            ko_bandwidth=ko_bandwidth
        )

        self._kappa = -np.polyfit(freqs_range, np.log(abs(fas)),1)[0]/np.pi
        
    def _calc_sdof_tf(self, osc_freq, damping=0.05):
        """Compute the transfer function for a single-degree-of-freedom oscillator.

        The transfer function computes the pseudo-spectral acceleration.

        Parameters
        ----------
        osc_freq : float
            natural frequency of the oscillator [Hz]
        damping : float, optional
            damping ratio of the oscillator in decimal. Default value is
            0.05, or 5%.

        Returns
        -------
        tf : :class:`numpy.ndarray`
            Complex-valued transfer function with length equal to `self.freq`.
        """
        return -(osc_freq**2.0) / (
            np.square(self.freqs)
            - np.square(osc_freq)
            - 2.0j * damping * osc_freq * self.freqs
        )
        
    @classmethod
    def load_at2_file(cls, filename, scale=1.0):
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

        accels *= scale
        return cls(filename, description, time_step, accels)

    @classmethod
    def load_smc_file(cls, filename, scale=1.0):
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
        from .tools import parse_fixed_width

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
        accels *= scale

        return cls(filename, description, time_step, accels)

    @classmethod
    def load_v2_file(cls, filename, scale=1.0, channel=1):
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
        from .tools import parse_fixed_width

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

        accels *= scale

        return cls(filename, description, time_step, accels)

    @classmethod
    def load_v2c_file(cls, filename, scale=1.0, channel=1):
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
        from .tools import parse_fixed_width

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

        accels *= scale

        return cls(filename, description, time_step, accels)

    @classmethod
    def load(cls, filename, scale=1.0, **kwargs):
        """Load a time series, choosing the reader from the file extension.

        Parameters
        ----------
        filename: str
            Filename to open. The extension (``.at2``, ``.smc``, ``.v2``, or
            ``.v2c``, case-insensitive) selects the reader.
        scale: float, default: 1.
            Scale factor to apply to the motion.
        **kwargs:
            Passed through to the selected ``load_*`` reader, e.g. ``channel``
            for ``.v2`` and ``.v2c`` files.

        Returns
        -------
        :class:`TimeSeriesMotion`
        """
        loaders = {
            ".at2": cls.load_at2_file,
            ".smc": cls.load_smc_file,
            ".v2": cls.load_v2_file,
            ".v2c": cls.load_v2c_file,
        }
        suffix = Path(filename).suffix.lower()
        try:
            loader = loaders[suffix]
        except KeyError:
            raise ValueError(
                f"Unsupported file extension {suffix!r} for '{filename}'. "
                f"Supported extensions: {sorted(loaders)}."
            ) from None
        return loader(filename, scale=scale, **kwargs)

    def scaled_to_pga(self,scaled_pga = 1.0):
        pga = self.pga
        scale_factor = scaled_pga/self.pga

        scaled_accels = scale_factor*self._accels

        return TimeSeriesMotion(self._filename,
                                self._description,
                                self._time_step,
                                scaled_accels)

# FIXME: How do multiple inheritence properly?
class RvtMotion(pyrvt.motions.RvtMotion, Motion):
    """RVT motion based on user specified Fourier amplitude spectrum and duration."""

    def __init__(
        self, freqs, fourier_amps, duration=None, peak_calculator=None, calc_kwds=None
    ):
        Motion.__init__(self)
        pyrvt.motions.RvtMotion.__init__(
            self,
            np.asarray(freqs),
            np.asarray(fourier_amps),
            duration=duration,
            peak_calculator=peak_calculator,
            calc_kwds=calc_kwds,
        )


class CompatibleRvtMotion(pyrvt.motions.CompatibleRvtMotion, Motion):
    """RVT motion based on user specified acceleration response spectrum and
    duration."""

    def __init__(
        self,
        osc_freqs,
        osc_accels_target,
        duration=None,
        osc_damping=0.05,
        event_kwds=None,
        window_len=None,
        peak_calculator=None,
        calc_kwds=None,
    ):
        Motion.__init__(self)
        pyrvt.motions.CompatibleRvtMotion.__init__(
            self,
            osc_freqs,
            osc_accels_target,
            duration=duration,
            osc_damping=osc_damping,
            event_kwds=event_kwds,
            window_len=window_len,
            peak_calculator=peak_calculator,
            calc_kwds=calc_kwds,
        )


class SourceTheoryRvtMotion(pyrvt.motions.SourceTheoryMotion, Motion):
    """RVT motion based on seismological point source model and earthquake scenario
    parameters."""

    def __init__(
        self,
        magnitude: float,
        distance: float,
        region: str | None = None,
        depth: float | None = 8,
        peak_calculator: str | pyrvt.peak_calculators.Calculator | None = None,
        calc_kwds: dict | None = None,
        freqs: np.ndarray | None = None,
        disable_site_amp: bool = False,
        **kwargs,
    ):
        Motion.__init__(self)
        pyrvt.motions.SourceTheoryMotion.__init__(
            self,
            magnitude=magnitude,
            distance=distance,
            region=region,
            depth=depth,
            peak_calculator=peak_calculator,
            calc_kwds=calc_kwds,
            freqs=freqs,
            disable_site_amp=disable_site_amp,
            **kwargs,
        )
