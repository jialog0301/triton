import subprocess


def test_launcher_help_documents_legacy_scope():
    result = subprocess.run(
        ["third_party/ventus/reference_launcher/ventus_spike_smoke", "--help"],
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0
    assert "--elf PATH" in result.stdout
    assert "--local 16,1,1" in result.stdout
    # The tool must point V1 kernels at the profile-driven launcher instead of
    # silently launching them with 8-lane warps.
    assert "backend/launcher.py" in result.stdout


def test_launcher_rejects_v1_local_size():
    result = subprocess.run(
        [
            "third_party/ventus/reference_launcher/ventus_spike_smoke",
            "--elf",
            "kernel.elf",
            "--entry",
            "0x800000b8",
            "--grid",
            "1,1,1",
            "--local",
            "32,1,1",
            "--lds-size",
            "4096",
            "--pds-size",
            "4096",
            "--input",
            "a.bin:4",
            "--output",
            "c.bin:4",
            "--expected",
            "c.ref:4",
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 2
    assert "8-lane warps" in result.stderr
    assert "backend/launcher.py" in result.stderr
