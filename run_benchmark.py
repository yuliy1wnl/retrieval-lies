"""
Run the full benchmark pipeline end to end.
Usage: python run_benchmark.py [--skip-data] [--skip-index] [--skip-eval]
"""

import argparse
import sys
import time

def banner(msg):
    print(f"\n{'='*60}")
    print(f"  {msg}")
    print(f"{'='*60}\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--skip-data",  action="store_true", help="Skip data download")
    parser.add_argument("--skip-index", action="store_true", help="Skip index building")
    parser.add_argument("--skip-eval",  action="store_true", help="Skip evaluation")
    args = parser.parse_args()

    t_start = time.time()

    if not args.skip_data:
        banner("Step 1/4: Preparing MS MARCO dataset")
        from data.prepare import prepare
        prepare()

    if not args.skip_index:
        banner("Step 2/4: Building Endee indexes")
        from indexer.build_indexes import build_all_indexes
        build_all_indexes()

    if not args.skip_eval:
        banner("Step 3/4: Running evaluation")
        from evaluator.evaluate import run_evaluation
        run_evaluation()

        banner("Step 4/4: Analyzing failure modes")
        from analyzer.failure_modes import run_failure_analysis
        run_failure_analysis()

    banner("Generating report")
    from reporter.generate_report import generate_report
    generate_report()

    total = time.time() - t_start
    print(f"\n✅ Full benchmark complete in {total/60:.1f} minutes.")
    print("   Open reports/report.md to see results.")