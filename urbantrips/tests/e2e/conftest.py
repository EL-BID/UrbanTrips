import pytest
from urbantrips.storage.context import StorageContext
from urbantrips.storage.adapters.memory.adapters import (
    InMemoryDataAdapter,
    InMemoryInsumoAdapter,
    InMemoryDashAdapter,
    InMemoryGeneralAdapter,
)


@pytest.fixture
def in_memory_ctx() -> StorageContext:
    return StorageContext(
        data=InMemoryDataAdapter(),
        insumos=InMemoryInsumoAdapter(),
        dash=InMemoryDashAdapter(),
        general=InMemoryGeneralAdapter(),
    )
