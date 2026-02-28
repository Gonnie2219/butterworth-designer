# Butterworth Filter Designer

A standalone Python GUI application for analog Butterworth filter design.

## Features

- **Filter types** — Low Pass, High Pass, Band Pass, Band Reject
- **Frequency units** — rad/s or Hz
- **Transfer function display** — ASCII fraction with numerator/denominator polynomials and pole list
- **Frequency response plots** — magnitude (dB) and phase (degrees) on log-frequency axes
- **Dark theme** — Catppuccin Mocha palette

## Requirements

- Python 3.8+ with tkinter (bundled with CPython)
- numpy, scipy, matplotlib

## Installation

```bash
pip install -r requirements.txt
```

## Usage

```bash
python butterworth_designer.py
```

The app opens with a default 4th-order Low Pass filter at 1000 rad/s. Adjust the parameters and click **Design Filter** to update the display.

### Parameters

| Field | Description |
|---|---|
| Order N | Filter order (1–20). For Band Pass/Reject, scipy doubles this. |
| Type | Low Pass, High Pass, Band Pass, or Band Reject |
| Unit | rad/s or Hz (converted to rad/s internally) |
| ωc / ωc1 | Cutoff frequency |
| ωc2 | Upper cutoff (Band Pass / Band Reject only) |

### Plots

- **Magnitude** — `|H(ω)|` in dB with a dashed −3 dB reference line and green ωc markers
- **Phase** — `∠H(ω)` in degrees (unwrapped), with green ωc markers
- Use the matplotlib toolbar to zoom and pan

## Example

4th-order Low Pass at 1000 rad/s:

```
          1e+12
H(s) = ──────────────────────────────────────────────────
        s^4  +  2828*s^3  +  4e+06*s^2  +  2.828e+09*s  +  1e+12
```
