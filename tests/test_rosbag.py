"""Round-trips a real ROS bag through the reader used for UZH-FPV and Blackbird.

Writing a genuine bag rather than stubbing the library is what makes this test
worth having: message layout, encodings and timestamp units are all exercised.
"""

import numpy as np
import pytest
import yaml
from rosbags.rosbag1 import Writer
from rosbags.typesys import Stores, get_typestore
from scipy.spatial.transform import Rotation

from regyro.dataset.build import build_from_source
from regyro.dataset.kalibr import KalibrError, load_calibration
from regyro.dataset.rosbag_source import RosbagError, load_bag, read_gyro

NS = 1_000_000_000
FRAME_COUNT = 6
IMAGE_SIZE = (40, 30)
GYRO_RATE = np.array([0.4, -0.1, 0.6])
IMAGE_TOPIC = "/cam0/image_raw"
IMU_TOPIC = "/imu"


def write_bag(path, encoding: str = "mono8", extra_image_topic: bool = False) -> None:
    typestore = get_typestore(Stores.ROS1_NOETIC)
    imu_type = typestore.types["sensor_msgs/msg/Imu"]
    image_type = typestore.types["sensor_msgs/msg/Image"]
    header_type = typestore.types["std_msgs/msg/Header"]
    time_type = typestore.types["builtin_interfaces/msg/Time"]
    vector_type = typestore.types["geometry_msgs/msg/Vector3"]
    quaternion_type = typestore.types["geometry_msgs/msg/Quaternion"]

    def header(stamp_ns: int):
        return header_type(
            seq=0,
            stamp=time_type(sec=stamp_ns // NS, nanosec=stamp_ns % NS),
            frame_id="sensor",
        )

    channels = 3 if encoding in ("bgr8", "rgb8") else 1
    rng = np.random.default_rng(0)

    with Writer(path) as writer:
        imu_connection = writer.add_connection(IMU_TOPIC, imu_type.__msgtype__, typestore=typestore)
        image_connection = writer.add_connection(
            IMAGE_TOPIC, image_type.__msgtype__, typestore=typestore
        )
        if extra_image_topic:
            writer.add_connection("/cam1/image_raw", image_type.__msgtype__, typestore=typestore)

        for index in range(FRAME_COUNT * 10):
            stamp = NS + index * (NS // 300)
            message = imu_type(
                header=header(stamp),
                orientation=quaternion_type(x=0.0, y=0.0, z=0.0, w=1.0),
                orientation_covariance=np.zeros(9),
                angular_velocity=vector_type(x=GYRO_RATE[0], y=GYRO_RATE[1], z=GYRO_RATE[2]),
                angular_velocity_covariance=np.zeros(9),
                linear_acceleration=vector_type(x=0.0, y=0.0, z=9.81),
                linear_acceleration_covariance=np.zeros(9),
            )
            writer.write(imu_connection, stamp, typestore.serialize_ros1(message, imu_type.__msgtype__))

        for index in range(FRAME_COUNT):
            stamp = NS + int(index * (NS / 30.0)) + NS // 100
            pixels = (rng.random((IMAGE_SIZE[1], IMAGE_SIZE[0], channels)) * 255).astype(np.uint8)
            message = image_type(
                header=header(stamp),
                height=IMAGE_SIZE[1],
                width=IMAGE_SIZE[0],
                encoding=encoding,
                is_bigendian=0,
                step=IMAGE_SIZE[0] * channels,
                data=pixels.flatten(),
            )
            writer.write(
                image_connection, stamp, typestore.serialize_ros1(message, image_type.__msgtype__)
            )


def write_kalibr(path, rotation: Rotation = Rotation.identity()) -> None:
    transform = np.eye(4)
    transform[:3, :3] = rotation.as_matrix()
    path.write_text(
        yaml.safe_dump(
            {
                "cam0": {
                    "camera_model": "pinhole",
                    "intrinsics": [25.0, 25.0, 20.0, 15.0],
                    "distortion_model": "equidistant",
                    "distortion_coeffs": [0.0, 0.0, 0.0, 0.0],
                    "resolution": list(IMAGE_SIZE),
                    "T_cam_imu": transform.tolist(),
                }
            }
        )
    )


def test_read_gyro_recovers_rates_and_seconds(tmp_path):
    bag = tmp_path / "seq.bag"
    write_bag(bag)

    gyro = read_gyro(bag)

    np.testing.assert_allclose(gyro.angular_velocity[0], GYRO_RATE, atol=1e-6)
    assert gyro.timestamps[0] == pytest.approx(1.0, abs=1e-6)
    assert np.all(np.diff(gyro.timestamps) > 0)


def test_frame_source_streams_grayscale(tmp_path):
    bag = tmp_path / "seq.bag"
    write_bag(bag)

    sequence = load_bag(bag)
    frames = list(sequence.source)

    assert len(frames) == FRAME_COUNT
    assert frames[0].gray.shape == (IMAGE_SIZE[1], IMAGE_SIZE[0])
    assert frames[0].gray.dtype == np.uint8


def test_colour_images_are_converted(tmp_path):
    bag = tmp_path / "colour.bag"
    write_bag(bag, encoding="bgr8")

    frames = list(load_bag(bag).source)

    assert frames[0].gray.ndim == 2


def test_ambiguous_image_topic_is_rejected(tmp_path):
    bag = tmp_path / "two.bag"
    write_bag(bag, extra_image_topic=True)

    with pytest.raises(RosbagError, match="several image topics"):
        load_bag(bag)

    assert list(load_bag(bag, image_topic=IMAGE_TOPIC).source)


def test_missing_bag_is_reported(tmp_path):
    with pytest.raises(RosbagError, match="no such bag"):
        read_gyro(tmp_path / "absent.bag")


def test_kalibr_calibration_round_trip(tmp_path):
    path = tmp_path / "camchain.yaml"
    rotation = Rotation.from_euler("x", 30, degrees=True)
    write_kalibr(path, rotation)

    calibration = load_calibration(path)

    np.testing.assert_allclose(calibration.rotation_cam_imu, rotation.as_matrix(), atol=1e-12)
    assert calibration.profile.calib_width == IMAGE_SIZE[0]


def test_kalibr_reports_unknown_camera(tmp_path):
    path = tmp_path / "camchain.yaml"
    write_kalibr(path)

    with pytest.raises(KalibrError, match="no entry for"):
        load_calibration(path, camera="cam3")


def test_bag_builds_shards_with_rotated_gyro(tmp_path):
    bag = tmp_path / "seq.bag"
    write_bag(bag)
    calibration_path = tmp_path / "camchain.yaml"
    rotation = Rotation.from_euler("z", 90, degrees=True)
    write_kalibr(calibration_path, rotation)

    calibration = load_calibration(calibration_path)
    sequence = load_bag(bag)
    gyro = sequence.gyro.rotate(calibration.rotation_cam_imu)

    stats = build_from_source(sequence.source, calibration.profile, gyro, tmp_path / "shards")

    assert stats.frame_pairs == FRAME_COUNT - 1
    shard = next((tmp_path / "shards").glob("*.npz"))
    with np.load(shard) as payload:
        expected = rotation.as_matrix() @ GYRO_RATE
        np.testing.assert_allclose(
            payload["angular_velocity"], np.tile(expected, (FRAME_COUNT - 1, 1)), atol=1e-4
        )
