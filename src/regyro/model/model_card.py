"""Renders a model card from a checkpoint's own metadata.

Distribution is only honest if the card states what the model saw and what it cannot
do, and both are recorded in the checkpoint rather than remembered by hand.
"""

from pathlib import Path

from regyro.dataset.provenance import UNKNOWN_LICENSE, Provenance

RESTRICTED_NOTICE = (
    "**This checkpoint may not be redistributed under an unrestricted licence.** "
    "It was trained on data carrying non-commercial terms ({licenses}), which the "
    "trained weights inherit."
)
UNKNOWN_NOTICE = (
    "**Licensing is unverified.** Some training data was imported without a recorded "
    "licence, so redistribution terms cannot be established. Re-import with `--license` "
    "before publishing."
)
PUBLISHABLE_NOTICE = (
    "Training data carries no non-commercial or unverified terms, so this checkpoint "
    "can be redistributed freely."
)


def _provenance_from_checkpoint(checkpoint: dict) -> Provenance:
    return Provenance(
        sources=list(checkpoint.get("training_sources", [])),
        licenses=list(checkpoint.get("training_licenses", [])),
    )


def _licence_section(provenance: Provenance) -> str:
    if provenance.restricted_licenses:
        return RESTRICTED_NOTICE.format(licenses=", ".join(provenance.restricted_licenses))
    if UNKNOWN_LICENSE in provenance.licenses or not provenance.licenses:
        return UNKNOWN_NOTICE
    return PUBLISHABLE_NOTICE


def _bullet_list(values: list[str], empty: str) -> str:
    return "\n".join(f"- {value}" for value in values) if values else f"- {empty}"


def render(checkpoint: dict, model_name: str = "regyro") -> str:
    provenance = _provenance_from_checkpoint(checkpoint)
    validation_loss = checkpoint.get("val_loss")
    loss_text = f"{validation_loss:.5f}" if isinstance(validation_loss, float) else "not recorded"

    return f"""# {model_name}

Reconstructs camera angular velocity from video, so footage whose gyro data was lost
can still be stabilised in [Gyroflow](https://gyroflow.xyz).

## Licensing

{_licence_section(provenance)}

Licences present in the training data:

{_bullet_list(provenance.licenses, "none recorded")}

## Intended use

Estimating per-frame angular velocity from FPV and action-camera footage, written as a
`.gcsv` file that Gyroflow reads. It is a fallback for lost telemetry, not a replacement
for a real gyro: a genuine IMU log samples at kilohertz and will always stabilise better.

## Training data

{_bullet_list(provenance.sources, "not recorded")}

## Inputs and outputs

- **Input**: two consecutive frames remapped to a {checkpoint.get('canonical_size', '?')}px
  equidistant fisheye covering {checkpoint.get('canonical_fov_deg', '?')} degrees, plus a
  validity mask, as three channels.
- **Output**: angular velocity in rad/s in the camera frame, scaled by
  {checkpoint.get('target_scale_rad_s', '?')} internally.
- Best validation loss: {loss_text} (epoch {checkpoint.get('epoch', '?')}).

## Limitations

- Estimates arrive at the frame rate, far below a real gyro's sample rate.
  A rolling shutter can therefore only be corrected approximately.
- Close-range flight breaks the rotation-only assumption, because parallax from nearby
  obstacles looks like rotation.
- Accuracy on cameras and scenes unlike the training data is unknown; compare against
  the classical estimator before trusting it.

## Usage

```bash
uv run regyro estimate flight.mp4 --lens lens-profile.json \\
    --method ml --checkpoint {model_name}.pt
```
"""


def write(path: Path, checkpoint: dict, model_name: str = "regyro") -> None:
    path.write_text(render(checkpoint, model_name))
