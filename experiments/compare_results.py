#!/usr/bin/env python3
"""
Compare results from multiple experiments
Generates a comparison report from individual experiment results
"""
import argparse
import json
from pathlib import Path
from typing import List, Dict


def load_results(results_dir: str) -> List[Dict]:
    """Load all experiment results from directory"""
    results_dir = Path(results_dir)
    results = []
    
    for result_file in results_dir.glob("*_results.json"):
        with open(result_file, 'r') as f:
            result = json.load(f)
            results.append(result)
    
    # Sort by name
    results.sort(key=lambda x: x["config"]["name"])
    
    return results


def generate_comparison_report(results: List[Dict], output_path: str):
    """Generate comparison report"""
    if len(results) < 1:
        print("No results found to compare")
        return
    
    report = "# Experiment Comparison Report\n\n"
    report += "## Overview\n\n"
    report += "This report compares experiments from Foundation Phase to Improvement Phase.\n\n"
    
    # Create comparison table
    report += "## Results Comparison\n\n"
    report += "| Experiment | Phase | Final Train Loss | Final Val Loss | Training Time (s) | Inference Time (ms) | Model Size (MB) | Parameters | Memory (MB) |\n"
    report += "|------------|-------|------------------|----------------|-------------------|---------------------|-----------------|------------|-------------|\n"
    
    for result in results:
        config = result["config"]
        train_losses = result["train_losses"]
        val_losses = result.get("val_losses", [])
        
        val_loss = val_losses[-1] if val_losses else "N/A"
        final_train_loss = train_losses[-1] if train_losses else "N/A"
        
        report += f"| {config['name']} | {config['phase']} | {final_train_loss:.4f} | {val_loss} | {result['training_time']:.2f} | {result['inference_time']*1000:.2f} | {result['model_size_mb']:.2f} | {result['num_parameters']:,} | {result.get('memory_usage_mb', 0):.2f} |\n"
    
    # Improvement analysis
    foundation_results = [r for r in results if r["config"]["phase"] == "foundation"]
    improvement_results = [r for r in results if r["config"]["phase"] == "improvement"]
    
    if foundation_results and improvement_results:
        foundation = foundation_results[0]  # Use first foundation as baseline
        
        report += "\n## Improvement Analysis\n\n"
        report += f"**Baseline**: {foundation['config']['name']}\n\n"
        
        for imp in improvement_results:
            report += f"### {imp['config']['name']} vs {foundation['config']['name']}\n\n"
            
            # Loss improvement
            if foundation['train_losses'] and imp['train_losses']:
                foundation_loss = foundation['train_losses'][-1]
                imp_loss = imp['train_losses'][-1]
                loss_improvement = ((foundation_loss - imp_loss) / foundation_loss) * 100
                report += f"- **Loss Improvement**: {loss_improvement:+.2f}% "
                report += f"(from {foundation_loss:.4f} to {imp_loss:.4f})\n"
            
            # Speed improvement
            speed_improvement = ((foundation['inference_time'] - imp['inference_time']) / foundation['inference_time']) * 100
            report += f"- **Inference Speed**: {speed_improvement:+.2f}% "
            report += f"(from {foundation['inference_time']*1000:.2f}ms to {imp['inference_time']*1000:.2f}ms)\n"
            
            # Model size change
            size_change = ((imp['model_size_mb'] - foundation['model_size_mb']) / foundation['model_size_mb']) * 100
            report += f"- **Model Size Change**: {size_change:+.2f}% "
            report += f"(from {foundation['model_size_mb']:.2f}MB to {imp['model_size_mb']:.2f}MB)\n"
            
            # Parameter change
            param_change = ((imp['num_parameters'] - foundation['num_parameters']) / foundation['num_parameters']) * 100
            report += f"- **Parameter Change**: {param_change:+.2f}% "
            report += f"(from {foundation['num_parameters']:,} to {imp['num_parameters']:,})\n"
            
            report += "\n"
    
    # Summary
    report += "## Summary\n\n"
    report += "### Best Performing Configurations\n\n"
    
    if results:
        # Best loss
        best_loss = min(results, key=lambda x: x['train_losses'][-1] if x['train_losses'] else float('inf'))
        report += f"- **Lowest Loss**: {best_loss['config']['name']} ({best_loss['train_losses'][-1]:.4f})\n"
        
        # Fastest inference
        fastest = min(results, key=lambda x: x['inference_time'])
        report += f"- **Fastest Inference**: {fastest['config']['name']} ({fastest['inference_time']*1000:.2f}ms)\n"
        
        # Smallest model
        smallest = min(results, key=lambda x: x['model_size_mb'])
        report += f"- **Smallest Model**: {smallest['config']['name']} ({smallest['model_size_mb']:.2f}MB)\n"
    
    # Save report
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w') as f:
        f.write(report)
    
    print(f"Comparison report saved to {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Compare experiment results")
    parser.add_argument(
        "--results_dir",
        type=str,
        default="./experiment_results",
        help="Directory containing experiment results"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="./experiment_results/comparison_report.md",
        help="Output path for comparison report"
    )
    
    args = parser.parse_args()
    
    # Load results
    print(f"Loading results from {args.results_dir}...")
    results = load_results(args.results_dir)
    
    if not results:
        print(f"No results found in {args.results_dir}")
        return
    
    print(f"Found {len(results)} experiment result(s)")
    
    # Generate report
    generate_comparison_report(results, args.output)


if __name__ == "__main__":
    main()


