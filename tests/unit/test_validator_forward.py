from types import SimpleNamespace
from unittest import mock
import numpy as np
import pytest

from qbittensor.validator.forward import forward, _initialize_miner_difficulty, _save_onboarded
from qbittensor.protocol import ChallengePeakedCircuit, ChallengeShorsCircuit


class _FakeMetagraph:
    """
    Minimal stand-in that satisfies every attribute / method forward() touches.
    """

    def __init__(self):
        # _bootstrap calls `.uids.tolist()`
        self.uids = np.array([0])

        # needs to index axons[uid].is_serving, ip, port
        self.axons = [mock.Mock(is_serving=True, ip="127.0.0.1", port=1234)]

        self.hotkeys = ["hk"]

    # forward() calls .sync() periodically
    def sync(self):
        pass


def _make_dummy_validator(tmp_db_path):
    v = SimpleNamespace()
    v.wallet = mock.Mock()
    v.metagraph = _FakeMetagraph()
    v.database_path = tmp_db_path
    return v


def test_forward_runs_once(tmp_db_path):
    v = _make_dummy_validator(tmp_db_path)

    # first call bootstraps
    forward(v)

    # second call should run the weight push without raising
    forward(v)


# Difficulty Initialization Tests

class _FakeDifficultyConfig:
    """Mock difficulty config that tracks set() calls"""
    def __init__(self):
        self.set_calls = []

    def set(self, uid, difficulty):
        self.set_calls.append((uid, difficulty))
        return True  # Return True to indicate change occurred


class _FakeValidatorForDifficulty:
    """Enhanced fake validator for difficulty testing"""
    def __init__(self):
        self.wallet = mock.Mock()
        self.wallet.hotkey.ss58_address = "validator_hotkey_123"

        # Mock metagraph
        self.metagraph = mock.Mock()
        self.metagraph.hotkeys = ["miner_hotkey_456"]

        # Mock axons - valid by default
        self.metagraph.axons = [mock.Mock()]
        self.metagraph.axons[0].ip = "127.0.0.1"
        self.metagraph.axons[0].port = 1234

        # Mock difficulty configs
        self._diff_cfg = {
            "peaked": _FakeDifficultyConfig(),
            "shors": _FakeDifficultyConfig()
        }

        # Mock onboarded set
        self._onboarded = set()


def _make_mock_response(desired_difficulty):
    """Create a mock response with desired_difficulty"""
    resp = mock.Mock()
    resp.desired_difficulty = desired_difficulty
    return resp


def _make_mock_dendrite(responses=None):
    """Create a mock dendrite that returns specified responses"""
    dendrite = mock.Mock()
    if responses:
        dendrite.query.side_effect = responses
    else:
        # Default successful responses
        peaked_resp = _make_mock_response(25.0)
        shors_resp = _make_mock_response(5.0)
        dendrite.query.side_effect = [peaked_resp, shors_resp]

    return dendrite


@pytest.fixture
def fake_validator():
    """Fixture providing a fake validator for difficulty testing"""
    return _FakeValidatorForDifficulty()


@mock.patch('qbittensor.validator.forward.bt.dendrite')
def test_initialize_miner_difficulty_success(mock_dendrite_factory, fake_validator):
    """Test successful difficulty initialization for a new miner"""
    # Setup mock dendrite
    mock_dendrite = _make_mock_dendrite()
    mock_dendrite_factory.return_value = mock_dendrite

    # Call the function
    _initialize_miner_difficulty(fake_validator, uid=0)

    # Verify dendrite was created correctly
    mock_dendrite_factory.assert_called_once_with(wallet=fake_validator.wallet)

    # Verify both circuit types were queried
    assert mock_dendrite.query.call_count == 2

    # Check peaked query
    peaked_call = mock_dendrite.query.call_args_list[0]
    peaked_syn = peaked_call[0][1]  # Second argument is the synapse
    assert isinstance(peaked_syn, ChallengePeakedCircuit)
    assert peaked_syn.circuit_data is None
    assert peaked_syn.validator_hotkey == "validator_hotkey_123"
    assert peaked_syn.difficulty_level == 0.0

    # Check shors query
    shors_call = mock_dendrite.query.call_args_list[1]
    shors_syn = shors_call[0][1]  # Second argument is the synapse
    assert isinstance(shors_syn, ChallengeShorsCircuit)
    assert shors_syn.circuit_data is None
    assert shors_syn.validator_hotkey == "validator_hotkey_123"
    assert shors_syn.difficulty_level == 0.0

    # Verify difficulties were set
    assert len(fake_validator._diff_cfg["peaked"].set_calls) == 1
    assert len(fake_validator._diff_cfg["shors"].set_calls) == 1
    assert fake_validator._diff_cfg["peaked"].set_calls[0] == (0, 25.0)
    assert fake_validator._diff_cfg["shors"].set_calls[0] == (0, 5.0)

    # Verify miner was onboarded
    assert "miner_hotkey_456" in fake_validator._onboarded

    # Verify dendrite was closed
    mock_dendrite.close.assert_called_once()


@mock.patch('qbittensor.validator.forward.bt.dendrite')
def test_initialize_miner_difficulty_invalid_axon_ip(mock_dendrite_factory, fake_validator):
    """Test that invalid axon IP causes early return"""
    # Set invalid IP
    fake_validator.metagraph.axons[0].ip = "0.0.0.0"

    # Call the function
    _initialize_miner_difficulty(fake_validator, uid=0)

    # Verify dendrite was never created (early return)
    mock_dendrite_factory.assert_not_called()

    # Verify no difficulties were set
    assert len(fake_validator._diff_cfg["peaked"].set_calls) == 0
    assert len(fake_validator._diff_cfg["shors"].set_calls) == 0


@mock.patch('qbittensor.validator.forward.bt.dendrite')
def test_initialize_miner_difficulty_invalid_axon_port(mock_dendrite_factory, fake_validator):
    """Test that invalid axon port causes early return"""
    # Set invalid port
    fake_validator.metagraph.axons[0].port = 0

    # Call the function
    _initialize_miner_difficulty(fake_validator, uid=0)

    # Verify dendrite was never created (early return)
    mock_dendrite_factory.assert_not_called()

    # Verify no difficulties were set
    assert len(fake_validator._diff_cfg["peaked"].set_calls) == 0
    assert len(fake_validator._diff_cfg["shors"].set_calls) == 0


@mock.patch('qbittensor.validator.forward.bt.dendrite')
def test_initialize_miner_difficulty_peaked_query_failure(mock_dendrite_factory, fake_validator):
    """Test handling when peaked query fails but shors succeeds"""
    # Setup dendrite to fail peaked query, succeed shors query
    mock_dendrite = mock.Mock()
    mock_dendrite.query.side_effect = [
        Exception("Peaked query failed"),  # Peaked fails
        _make_mock_response(5.0)           # Shors succeeds
    ]
    mock_dendrite_factory.return_value = mock_dendrite

    # Call the function
    _initialize_miner_difficulty(fake_validator, uid=0)

    # Verify only shors difficulty was set
    assert len(fake_validator._diff_cfg["peaked"].set_calls) == 0
    assert len(fake_validator._diff_cfg["shors"].set_calls) == 1
    assert fake_validator._diff_cfg["shors"].set_calls[0] == (0, 5.0)

    # Verify miner was still onboarded
    assert "miner_hotkey_456" in fake_validator._onboarded


@mock.patch('qbittensor.validator.forward.bt.dendrite')
def test_initialize_miner_difficulty_none_desired_difficulty(mock_dendrite_factory, fake_validator):
    """Test handling when miner responds with None desired_difficulty"""
    # Setup responses with None desired_difficulty
    peaked_resp = _make_mock_response(None)
    shors_resp = _make_mock_response(5.0)
    mock_dendrite = _make_mock_dendrite([peaked_resp, shors_resp])
    mock_dendrite_factory.return_value = mock_dendrite

    # Call the function
    _initialize_miner_difficulty(fake_validator, uid=0)

    # Verify only shors difficulty was set (peaked had None)
    assert len(fake_validator._diff_cfg["peaked"].set_calls) == 0
    assert len(fake_validator._diff_cfg["shors"].set_calls) == 1
    assert fake_validator._diff_cfg["shors"].set_calls[0] == (0, 5.0)


@mock.patch('qbittensor.validator.forward.bt.dendrite')
def test_initialize_miner_difficulty_invalid_desired_difficulty(mock_dendrite_factory, fake_validator):
    """Test handling when miner responds with invalid desired_difficulty"""
    # Setup responses with invalid desired_difficulty
    peaked_resp = _make_mock_response(float('inf'))  # Invalid: infinity
    shors_resp = _make_mock_response(-5.0)           # Invalid: negative
    mock_dendrite = _make_mock_dendrite([peaked_resp, shors_resp])
    mock_dendrite_factory.return_value = mock_dendrite

    # Call the function
    _initialize_miner_difficulty(fake_validator, uid=0)

    # Verify no difficulties were set due to invalid values
    assert len(fake_validator._diff_cfg["peaked"].set_calls) == 0
    assert len(fake_validator._diff_cfg["shors"].set_calls) == 0


@mock.patch('qbittensor.validator.forward.bt.dendrite')
def test_initialize_miner_difficulty_missing_difficulty_config(mock_dendrite_factory, fake_validator):
    """Test handling when difficulty config is missing"""
    # Remove shors config
    del fake_validator._diff_cfg["shors"]

    mock_dendrite = _make_mock_dendrite()
    mock_dendrite_factory.return_value = mock_dendrite

    # Call the function
    _initialize_miner_difficulty(fake_validator, uid=0)

    # Verify only peaked difficulty was set (shors config missing)
    assert len(fake_validator._diff_cfg["peaked"].set_calls) == 1
    assert fake_validator._diff_cfg["peaked"].set_calls[0] == (0, 25.0)


@mock.patch('qbittensor.validator.forward.bt.dendrite')
def test_initialize_miner_difficulty_dendrite_close_failure(mock_dendrite_factory, fake_validator):
    """Test handling when dendrite close fails"""
    mock_dendrite = _make_mock_dendrite()
    mock_dendrite.close.side_effect = Exception("Close failed")
    mock_dendrite_factory.return_value = mock_dendrite

    # Call the function (should not raise due to try/except)
    _initialize_miner_difficulty(fake_validator, uid=0)

    # Verify the function completed successfully despite close failure
    assert len(fake_validator._diff_cfg["peaked"].set_calls) == 1
    assert len(fake_validator._diff_cfg["shors"].set_calls) == 1
    assert "miner_hotkey_456" in fake_validator._onboarded
