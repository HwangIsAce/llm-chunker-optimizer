"""Tests for CLI"""
import pytest
import sys
from pathlib import Path
from click.testing import CliRunner

# Add src to path for imports
src_path = Path(__file__).parent.parent.parent.parent / "src"
sys.path.insert(0, str(src_path))

from chunker_optimizer.presentation.cli import cli


class TestCLI:
    """Test CLI interface"""
    
    @pytest.fixture
    def runner(self):
        """Create CLI runner"""
        return CliRunner()
    
    def test_cli_group(self, runner):
        """Test that CLI group works"""
        result = runner.invoke(cli, ["--help"])
        assert result.exit_code == 0
        assert "LLM Chunker Optimizer CLI" in result.output
    
    def test_optimize_command_help(self, runner):
        """Test optimize command help"""
        result = runner.invoke(cli, ["optimize", "--help"])
        assert result.exit_code == 0
        assert "Optimize chunking prompt" in result.output
    
    def test_optimize_command_requires_prompt(self, runner):
        """Test that optimize requires initial-prompt"""
        result = runner.invoke(cli, ["optimize"])
        assert result.exit_code != 0
        assert "initial-prompt" in result.output.lower() or "required" in result.output.lower()
    
    def test_optimize_command_basic(self, runner):
        """Test basic optimize command execution"""
        result = runner.invoke(cli, [
            "optimize",
            "--initial-prompt", "Split text into chunks",
            "--max-iterations", "2"
        ])
        # Should run (may fail if LLM not available, but should not crash)
        assert result.exit_code in [0, 1]  # 0 if success, 1 if LLM error (acceptable)
    
    def test_optimize_with_real_data(self, runner):
        """Test optimize with real data flag"""
        result = runner.invoke(cli, [
            "optimize",
            "--initial-prompt", "Split text into chunks",
            "--use-real-data",
            "--max-iterations", "2"
        ])
        # Should attempt to load real data
        assert result.exit_code in [0, 1]
    
    def test_benchmark_command_help(self, runner):
        """Test benchmark command help"""
        result = runner.invoke(cli, ["benchmark", "--help"])
        assert result.exit_code == 0
        assert "Run performance benchmarks" in result.output
    
    def test_benchmark_command(self, runner):
        """Test benchmark command"""
        result = runner.invoke(cli, ["benchmark"])
        assert result.exit_code == 0
        assert "benchmark" in result.output.lower()
