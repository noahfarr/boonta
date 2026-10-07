import os

os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ["XLA_FLAGS"] = (
    "--xla_force_host_platform_device_count=4 " + os.environ.get("XLA_FLAGS", "")
)

import numpy as np
import pytest
from jax.sharding import PartitionSpec as P


@pytest.fixture
def shards():
    def collect(leaf):
        return [np.asarray(shard.data) for shard in leaf.addressable_shards]

    return collect


@pytest.fixture
def assert_sharded():
    def check(leaf, mesh):
        assert leaf.sharding.spec == P("data"), f"expected P('data'), got {leaf.sharding.spec}"
        assert len(leaf.addressable_shards) == mesh.size
        for shard in leaf.addressable_shards:
            assert shard.data.shape == (leaf.shape[0] // mesh.size, *leaf.shape[1:])

    return check


@pytest.fixture
def assert_replicated():
    def check(leaf, mesh):
        assert leaf.sharding.spec == P(), f"expected P(), got {leaf.sharding.spec}"
        copies = [np.asarray(shard.data) for shard in leaf.addressable_shards]
        assert len(copies) == mesh.size
        for copy in copies:
            np.testing.assert_array_equal(copy, copies[0])

    return check
