import numpy as np
import pytest

from regyro.gcsv import write_gcsv


def test_write_gcsv_format(tmp_path):
    path = tmp_path / "out.gcsv"
    velocities = np.array([[0.1, -0.2, 0.3], [0.4, 0.5, -0.6]])

    write_gcsv(path, velocities, sample_rate_hz=50.0, video_filename="flight.mp4")

    lines = path.read_text().splitlines()
    assert lines[0] == "GYROFLOW IMU LOG"
    assert "tscale,0.020000000" in lines
    assert "gscale,1.0" in lines
    assert "videofilename,flight.mp4" in lines
    assert lines[-3] == "t,gx,gy,gz"
    assert lines[-2] == "0,0.100000,-0.200000,0.300000"
    assert lines[-1] == "1,0.400000,0.500000,-0.600000"


def test_write_gcsv_rejects_bad_shape(tmp_path):
    with pytest.raises(ValueError):
        write_gcsv(tmp_path / "out.gcsv", np.zeros((3, 2)), sample_rate_hz=30.0)


def test_write_gcsv_rejects_bad_rate(tmp_path):
    with pytest.raises(ValueError):
        write_gcsv(tmp_path / "out.gcsv", np.zeros((3, 3)), sample_rate_hz=0.0)
