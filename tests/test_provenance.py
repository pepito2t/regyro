"""Provenance decides whether a trained checkpoint may be redistributed.

UZH-FPV is CC BY-NC-SA, so a model that saw it inherits non-commercial terms; the
point of these tests is that the restriction survives all the way to the model card.
"""

import json

import numpy as np
import pytest
import torch

from regyro.dataset.provenance import (
    OWN_FOOTAGE_LICENSE,
    UNKNOWN_LICENSE,
    Provenance,
    describe,
    summarise,
)
from regyro.model.model_card import render
from regyro.model.network import RotationNet
from regyro.model.train import save_checkpoint


def write_manifest(directory, entries) -> None:
    (directory / "manifest.json").write_text(json.dumps(entries))


def test_own_footage_is_publishable():
    provenance = summarise([{"source": "flight.mp4", "license": OWN_FOOTAGE_LICENSE}])

    assert provenance.is_publishable
    assert not provenance.restricted_licenses


def test_non_commercial_licence_is_detected():
    provenance = summarise(
        [
            {"source": "flight.mp4", "license": OWN_FOOTAGE_LICENSE},
            {"source": "indoor_45.bag", "license": "cc-by-nc-sa-3.0"},
        ]
    )

    assert not provenance.is_publishable
    assert provenance.restricted_licenses == ["cc-by-nc-sa-3.0"]
    assert provenance.share_alike_licenses == ["cc-by-nc-sa-3.0"]


def test_permissive_share_alike_is_not_flagged_as_non_commercial():
    provenance = summarise([{"source": "seq", "license": "cc-by-sa-4.0"}])

    assert not provenance.restricted_licenses
    assert provenance.share_alike_licenses == ["cc-by-sa-4.0"]


def test_unrecorded_licence_blocks_publication():
    provenance = summarise([{"source": "seq"}])

    assert provenance.licenses == [UNKNOWN_LICENSE]
    assert not provenance.is_publishable


def test_describe_reads_the_manifest(tmp_path):
    write_manifest(tmp_path, [{"source": "a", "license": "cc-by-4.0"}])

    assert describe(tmp_path).sources == ["a"]


def test_describe_survives_a_corrupt_manifest(tmp_path):
    (tmp_path / "manifest.json").write_text("{not json")

    assert describe(tmp_path) == Provenance(sources=[], licenses=[])


def test_checkpoint_carries_provenance(tmp_path):
    checkpoint = tmp_path / "model.pt"
    provenance = Provenance(sources=["indoor_45.bag"], licenses=["cc-by-nc-sa-3.0"])

    save_checkpoint(checkpoint, RotationNet(), epoch=2, val_loss=0.01, provenance=provenance)

    payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
    assert payload["training_sources"] == ["indoor_45.bag"]
    assert payload["training_licenses"] == ["cc-by-nc-sa-3.0"]


@pytest.mark.parametrize(
    ("licenses", "expected"),
    [
        (["cc-by-nc-sa-3.0"], "may not be redistributed"),
        ([UNKNOWN_LICENSE], "Licensing is unverified"),
        ([OWN_FOOTAGE_LICENSE], "can be redistributed freely"),
    ],
)
def test_model_card_states_the_licence_position(licenses, expected):
    checkpoint = {
        "training_sources": ["seq"],
        "training_licenses": licenses,
        "val_loss": 0.0123,
        "epoch": 4,
        "canonical_size": 256,
        "canonical_fov_deg": 140.0,
        "target_scale_rad_s": 5.0,
    }

    card = render(checkpoint)

    assert expected in card
    assert "rolling shutter" in card  # limitations are not optional


def test_model_card_handles_a_checkpoint_without_provenance():
    card = render({})

    assert "Licensing is unverified" in card
    assert "not recorded" in card


def test_training_records_provenance_end_to_end(tmp_path):
    from regyro.model.train import TrainConfig, train

    frame_size = 64
    for index, stem in enumerate(["seqA", "seqB"]):
        rng = np.random.default_rng(index)
        np.savez_compressed(
            tmp_path / f"{stem}-00000.npz",
            frames=(rng.random((6, frame_size, frame_size)) * 255).astype(np.uint8),
            angular_velocity=rng.normal(size=(5, 3)).astype(np.float32),
            valid_mask=np.full((frame_size, frame_size), 255, dtype=np.uint8),
        )
    write_manifest(
        tmp_path,
        [
            {"source": "seqA", "license": OWN_FOOTAGE_LICENSE},
            {"source": "seqB", "license": "cc-by-nc-sa-3.0"},
        ],
    )

    checkpoint = tmp_path / "model.pt"
    train(
        TrainConfig(
            data_dir=tmp_path,
            output=checkpoint,
            epochs=1,
            batch_size=2,
            val_fraction=0.5,
            workers=0,
            device="cpu",
        )
    )

    payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
    assert "cc-by-nc-sa-3.0" in payload["training_licenses"]
    assert "may not be redistributed" in render(payload)
