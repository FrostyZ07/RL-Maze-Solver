import faultwalk


def test_version():
    assert isinstance(faultwalk.__version__, str)
    assert faultwalk.__version__
