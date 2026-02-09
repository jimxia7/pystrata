import numpy as np
import pandas as pd
import pykooh
from scipy.stats import linregress

import pystrata


def calculate_kappa(
    data,
    start_freq,
    end_freq,
    smoothing=False,
    smoothing_window=30,
    normalization=False,
    normalization_freq=3,
):

    if isinstance(data, pystrata.motion.TimeSeriesMotion):

        fourier_amps = abs(data.fourier_amps)
        freqs = data.freqs
        if smoothing:
            fourier_amps = pykooh.smooth(freqs, freqs, fourier_amps, 30)
        if normalization:
            idx = np.argmin(np.abs(freqs - normalization_freq))
            fourier_amps = fourier_amps / fourier_amps[idx]

        freq_min = start_freq
        freq_max = end_freq

        mask = (freqs >= freq_min) & (freqs <= freq_max)

        if np.any(mask):
            freqs_masked = freqs[mask]
            log_amps = np.log(fourier_amps[mask])

            slope, intercept, r_value, p_value, std_err = linregress(
                freqs_masked, log_amps
            )

            kappa = -slope / np.pi

            return kappa, freqs_masked, np.exp(slope * freqs_masked + intercept)

    elif isinstance(data, pystrata.output.FourierAmplitudeSpectrumOutput):

        df = data.to_dataframe()
        freqs = df.index.to_numpy()
        freq_min = start_freq
        freq_max = end_freq
        mask = (freqs >= freq_min) & (freqs <= freq_max)

        kappa_df = pd.DataFrame()
        if np.any(mask):
            freqs_masked = freqs[mask]
            fitted_lines_df = pd.DataFrame(index=freqs[mask])

            for col in df.columns:
                fourier_amps = df[col].to_numpy()
                if smoothing:
                    fourier_amps = pykooh.smooth(freqs, freqs, fourier_amps, 30)
                if normalization:
                    idx = np.argmin(np.abs(freqs - normalization_freq))
                    fourier_amps = fourier_amps / fourier_amps[idx]

                log_amps = np.log(fourier_amps[mask])

                slope, intercept, r_value, p_value, std_err = linregress(
                    freqs_masked, log_amps
                )

                kappa = -slope / np.pi

                fitted_line = np.exp(slope * freqs_masked + intercept)
                fitted_lines_df[col] = fitted_line
                kappa_df.loc[col, "kappa"] = kappa

            return kappa_df, fitted_lines_df
