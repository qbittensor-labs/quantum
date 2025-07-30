from qbittensor.validator.services.challenge_producer import ChallengeProducer
from unittest import mock


def test_uid_ring_rotation(gpu_patch):
    cp = ChallengeProducer(
        wallet=mock.Mock(),
        diff_cfg={"peaked": mock.Mock(get=lambda _: 0.0), "hstab": mock.Mock(get=lambda _: 0.0)},
        uid_list=[1, 2, 3],
    )
    seen = [cp._next_uid() for _ in range(6)]
    assert seen == [1, 2, 3, 1, 2, 3]
