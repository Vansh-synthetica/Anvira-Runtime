import pytest
import asyncio

@pytest.mark.asyncio
async def test_tiny():
    await asyncio.sleep(0)
    assert True
