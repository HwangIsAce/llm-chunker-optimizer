"""Performance profiler for optimization process"""
import cProfile
import pstats
import io
from typing import Optional, Dict
from pathlib import Path


class OptimizationProfiler:
    """Performance profiler for optimization process"""
    
    def __init__(self, enable: bool = True):
        """
        Initialize profiler
        
        Args:
            enable: Whether profiling is enabled
        """
        self.enable = enable
        self.profiler: Optional[cProfile.Profile] = None
        self.profiles: Dict[str, cProfile.Profile] = {}
    
    def start(self, label: str = "optimization"):
        """
        Start profiling
        
        Args:
            label: Label for this profiling session
        """
        if not self.enable:
            return
        
        self.profiler = cProfile.Profile()
        self.profiler.enable()
    
    def stop(self, label: str = "optimization"):
        """
        Stop profiling and save
        
        Args:
            label: Label for this profiling session
        """
        if not self.enable or not self.profiler:
            return
        
        self.profiler.disable()
        self.profiles[label] = self.profiler
    
    def get_stats(self, label: str = "optimization", top_n: int = 20) -> str:
        """
        Get profiling statistics
        
        Args:
            label: Label of the profiling session
            top_n: Number of top functions to show
        
        Returns:
            Formatted statistics string
        """
        if label not in self.profiles:
            return "No profile data available"
        
        s = io.StringIO()
        ps = pstats.Stats(self.profiles[label], stream=s)
        ps.sort_stats('cumulative')
        ps.print_stats(top_n)
        
        return s.getvalue()
    
    def save_report(self, filepath: Path, label: str = "optimization"):
        """
        Save profiling report to file
        
        Args:
            filepath: Path to save report
            label: Label of the profiling session
        """
        if label not in self.profiles:
            return
        
        stats = self.get_stats(label)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        filepath.write_text(stats)
    
    def get_total_time(self, label: str = "optimization") -> float:
        """
        Get total execution time from profile
        
        Args:
            label: Label of the profiling session
        
        Returns:
            Total time in seconds
        """
        if label not in self.profiles:
            return 0.0
        
        stats = pstats.Stats(self.profiles[label])
        return stats.total_tt
