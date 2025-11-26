# run_tests.py
import pytest


def main():
    pytest.main(["tests/standalone_camera_classify_tests.py", "-v"])


if __name__ == "__main__":
    main()
