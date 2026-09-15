import subprocess


def test_launcher_help():
    result = subprocess.run(
        ["third_party/ventus/reference_launcher/ventus_spike_smoke", "--help"],
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0
    assert "--elf PATH" in result.stdout


def test_launcher_rejects_unsupported_local_size():
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
            "8,1,1",
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
    assert "local must be" in result.stderr
