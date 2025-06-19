#!/usr/bin/env python3
"""
Quick Analysis Runner for DAX Trading System
Simple script to run comprehensive ETL analysis on your full dataset.

Usage:
    python run_analysis.py
    python run_analysis.py --quick          # For faster analysis with larger chunks
    python run_analysis.py --detailed       # For detailed analysis with smaller chunks

Author: DAX Trading System
Created: 2025-06-19
"""

import sys
import argparse
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.append(str(project_root))
sys.path.append(str(project_root / "src"))

def main():
    """Main analysis runner."""
    parser = argparse.ArgumentParser(description="Run comprehensive DAX data analysis")
    parser.add_argument("--quick", action="store_true", help="Quick analysis (larger chunks)")
    parser.add_argument("--detailed", action="store_true", help="Detailed analysis (smaller chunks)")
    parser.add_argument("--file", type=str, default="data/raw/dax-1m_bk.csv", help="Data file path")
    
    args = parser.parse_args()
    
    # Determine chunk size based on mode
    if args.quick:
        chunk_size = 1000000  # 1M records per chunk (faster)
        mode_desc = "Quick Analysis"
    elif args.detailed:
        chunk_size = 250000   # 250K records per chunk (more detailed)
        mode_desc = "Detailed Analysis"
    else:
        chunk_size = 500000   # 500K records per chunk (balanced)
        mode_desc = "Standard Analysis"
    
    print("🚀 DAX Trading System - Comprehensive Data Analysis")
    print("=" * 60)
    print(f"Mode: {mode_desc}")
    print(f"Data File: {args.file}")
    print(f"Chunk Size: {chunk_size:,} records")
    print("=" * 60)
    
    # Check if file exists
    data_file = Path(args.file)
    if not data_file.exists():
        print(f"❌ Error: Data file not found: {data_file}")
        print("\nPlease ensure your DAX CSV file is available.")
        print("Expected location: data/raw/dax-1m_bk.csv")
        return 1
    
    # Import and run the comprehensive test
    try:
        from tests.test_comprehensive_etl import run_comprehensive_test
        
        print("🔬 Starting comprehensive analysis...")
        print("⏱️  This may take several minutes for large datasets...")
        print()
        
        # Run the analysis
        results = run_comprehensive_test(str(data_file), chunk_size)
        
        if results and 'dataset_info' in results:
            print("\n🎉 Analysis completed successfully!")
            print(f"📊 Processed {results['dataset_info']['total_records']:,} records")
            print(f"⏱️  Total time: {results['dataset_info']['processing_time_total']:.1f} seconds")
            
            # Show key findings
            if 'validation_results' in results:
                quality_score = results['validation_results']['validation_result'].data_quality_score
                print(f"✅ Data Quality Score: {quality_score:.1f}/100")
                
                if quality_score >= 90:
                    print("🌟 Excellent data quality!")
                elif quality_score >= 80:
                    print("👍 Good data quality")
                elif quality_score >= 70:
                    print("⚠️  Fair data quality - some issues detected")
                else:
                    print("🚨 Poor data quality - significant issues found")
            
            print("\n💡 Check the detailed logs and saved results for complete analysis.")
            
        else:
            print("❌ Analysis failed or returned incomplete results")
            return 1
            
    except ImportError:
        print("❌ Error: Comprehensive test suite not found")
        print("Please ensure tests/test_comprehensive_etl.py is available")
        return 1
    except Exception as e:
        print(f"❌ Error during analysis: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)