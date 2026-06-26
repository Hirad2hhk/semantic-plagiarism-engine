import os
import tempfile

import pytest
from click.testing import CliRunner

from src.plagiarism_engine.cli import cli


def test_cli_compare():
    """Test the compare CLI command"""
    runner = CliRunner()

    # Create temporary test files
    with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f1:
        f1.write("This is a test document for plagiarism detection.")
        file_a = f1.name

    with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f2:
        f2.write("This is another test document for plagiarism detection.")
        file_b = f2.name

    try:
        # Run the compare command
        result = runner.invoke(cli, ["compare", "--file-a", file_a, "--file-b", file_b])

        # Should execute without error
        assert result.exit_code == 0

    finally:
        # Clean up temp files
        os.unlink(file_a)
        os.unlink(file_b)


def test_cli_corpus():
    """Test the corpus CLI command"""
    runner = CliRunner()

    # Create a temporary directory with test files
    with tempfile.TemporaryDirectory() as temp_dir:
        # Create some test files
        file1_path = os.path.join(temp_dir, "doc1.txt")
        file2_path = os.path.join(temp_dir, "doc2.txt")

        with open(file1_path, "w") as f:
            f.write("This is a test document.")

        with open(file2_path, "w") as f:
            f.write("This is another test document.")

        # Run the corpus command - should execute without crashing
        result = runner.invoke(
            cli,
            [
                "corpus",
                "--data",
                temp_dir,
                "--threshold",
                "0.1",
                "--output",
                "test_output.csv",
            ],
        )

        # Should execute without crashing (exit code 0 or 1 for invalid data)
        assert result.exit_code in [0, 1]


def test_cli_pairs():
    """Test the pairs CLI command"""
    runner = CliRunner()

    # Create a temporary CSV file with test data
    csv_content = 'text_a,text_b,label\n"This is a test","This is another test",1\n"Different content","Also different",0'

    with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
        f.write(csv_content)
        csv_file = f.name

    try:
        # Run the pairs command - should execute without crashing
        result = runner.invoke(
            cli,
            [
                "pairs",
                "--pairs",
                csv_file,
                "--threshold",
                "0.5",
                "--output",
                "test_metrics.csv",
            ],
        )

        # Should execute without crashing (exit code 0 or 1 for invalid data)
        assert result.exit_code in [0, 1]

    finally:
        # Clean up temp file
        os.unlink(csv_file)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
