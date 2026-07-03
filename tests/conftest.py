import pytest


@pytest.fixture
def db_path(tmp_path):
    return tmp_path / "healthcare.db"
