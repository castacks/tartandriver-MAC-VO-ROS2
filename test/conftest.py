def pytest_configure(config) -> None:
    config.addinivalue_line("markers", "local: needs extracted datasets on this machine")
