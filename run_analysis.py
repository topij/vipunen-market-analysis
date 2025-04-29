#!/usr/bin/env python
"""
Run the education market analysis with command-line arguments.

This script serves as a thin wrapper around the Vipunen analysis modules.
"""

import sys
import logging

# Configure logging
logging.basicConfig(
    level=logging.WARNING,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
# Silence overly verbose loggers
logging.getLogger("matplotlib").setLevel(logging.WARNING)
logging.getLogger("FileUtils").setLevel(logging.WARNING) # Also silence FileUtils for now

logger = logging.getLogger(__name__)

def check_dependencies():
    """Check if required dependencies are installed."""
    try:
        import FileUtils
        logger.info("FileUtils package is available.")
        return True
    except ImportError:
        logger.error("FileUtils package is not installed.")
        logger.error("Please install it with: pip install FileUtils")
        return False
    
    try:
        import yaml
        logger.info("PyYAML package is available.")
        return True
    except ImportError:
        logger.error("PyYAML package is not installed.")
        logger.error("Please install it with: pip install PyYAML")
        return False

def main():
    """Main function to run the analysis."""
    # Check dependencies
    if not check_dependencies():
        logger.error("Analysis cannot continue without required dependencies.")
        return 1
    
    try:
        # Import and run the workflow using the new modular CLI
        try:
            from src.vipunen.cli.analyze_cli import main as run_workflow
            
            logger.info("Starting education market analysis")
            return run_workflow()
            
        except ImportError as e:
            logger.error(f"Failed to import analyze_cli module: {e}")
            logger.error("Make sure the vipunen package is properly installed")
            return 1
            
    except Exception as e:
        logger.error(f"Analysis failed with error: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return 1

if __name__ == "__main__":
    sys.exit(main()) 