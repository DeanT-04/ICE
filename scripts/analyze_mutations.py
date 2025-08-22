#!/usr/bin/env python3
"""
Mutation Testing Utilities for Ultra-Fast AI Model
Provides advanced analysis and reporting for cargo-mutants results
"""

import json
import sys
import argparse
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import matplotlib.pyplot as plt
import pandas as pd
from datetime import datetime

class MutationTestAnalyzer:
    """Analyzes mutation testing results and generates insights"""
    
    def __init__(self, results_dir: str = "target/mutants"):
        self.results_dir = Path(results_dir)
        self.results_dir.mkdir(parents=True, exist_ok=True)
    
    def load_results(self, result_file: str) -> Optional[Dict]:
        """Load mutation testing results from JSON file"""
        try:
            with open(self.results_dir / result_file, 'r') as f:
                return json.load(f)
        except (FileNotFoundError, json.JSONDecodeError) as e:
            print(f"Error loading {result_file}: {e}")
            return None
    
    def analyze_component_scores(self) -> Dict[str, float]:
        """Analyze mutation scores for each component"""
        components = {
            'core': 'core_results.json',
            'fusion': 'fusion_results.json', 
            'training': 'training_results.json',
            'utils': 'utils_results.json',
            'overall': 'complete_results.json'
        }
        
        scores = {}
        for component, file in components.items():
            results = self.load_results(file)
            if results and 'mutation_score' in results:
                scores[component] = results['mutation_score']
            else:
                scores[component] = 0.0
        
        return scores
    
    def generate_trend_report(self, scores: Dict[str, float]) -> str:
        """Generate a trend analysis report"""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        report = f"""# Mutation Testing Trend Analysis
Generated: {timestamp}

## Component Scores
"""
        
        for component, score in scores.items():
            status = "✅" if score >= 80 else "⚠️" if score >= 70 else "❌"
            report += f"- **{component.title()}**: {score:.1f}% {status}\n"
        
        report += f"""
## Analysis

### Overall Health
- **Best Performing**: {max(scores, key=scores.get)} ({max(scores.values()):.1f}%)
- **Needs Attention**: {min(scores, key=scores.get)} ({min(scores.values()):.1f}%)
- **Average Score**: {sum(scores.values()) / len(scores):.1f}%

### Recommendations
"""
        
        for component, score in scores.items():
            if score < 75:
                report += f"- **{component.title()}**: Consider adding more comprehensive tests (current: {score:.1f}%)\n"
        
        return report
    
    def create_score_visualization(self, scores: Dict[str, float], output_file: str = "mutation_scores.png"):
        """Create a visualization of mutation scores"""
        plt.figure(figsize=(10, 6))
        
        components = list(scores.keys())
        score_values = list(scores.values())
        colors = ['green' if s >= 80 else 'orange' if s >= 70 else 'red' for s in score_values]
        
        bars = plt.bar(components, score_values, color=colors, alpha=0.7)
        plt.axhline(y=80, color='green', linestyle='--', alpha=0.5, label='Target (80%)')
        plt.axhline(y=70, color='orange', linestyle='--', alpha=0.5, label='Warning (70%)')
        
        plt.title('Mutation Testing Scores by Component')
        plt.ylabel('Mutation Score (%)')
        plt.xlabel('Component')
        plt.ylim(0, 100)
        plt.legend()
        
        # Add value labels on bars
        for bar, value in zip(bars, score_values):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1, 
                    f'{value:.1f}%', ha='center', va='bottom')
        
        plt.tight_layout()
        plt.savefig(self.results_dir / output_file, dpi=300, bbox_inches='tight')
        plt.close()
        
        return str(self.results_dir / output_file)
    
    def analyze_surviving_mutants(self, result_file: str) -> List[Dict]:
        """Analyze surviving mutants to identify test gaps"""
        results = self.load_results(result_file)
        if not results:
            return []
        
        surviving_mutants = []
        if 'outcomes' in results:
            for outcome in results['outcomes']:
                if outcome.get('outcome') == 'survived':
                    surviving_mutants.append({
                        'file': outcome.get('mutation', {}).get('file', 'unknown'),
                        'function': outcome.get('mutation', {}).get('function', 'unknown'),
                        'line': outcome.get('mutation', {}).get('line', 0),
                        'operator': outcome.get('mutation', {}).get('operator', 'unknown')
                    })
        
        return surviving_mutants
    
    def generate_improvement_suggestions(self, surviving_mutants: List[Dict]) -> List[str]:
        """Generate suggestions for improving mutation scores"""
        suggestions = []
        
        # Analyze patterns in surviving mutants
        operators = {}
        files = {}
        
        for mutant in surviving_mutants:
            operator = mutant['operator']
            file = mutant['file']
            
            operators[operator] = operators.get(operator, 0) + 1
            files[file] = files.get(file, 0) + 1
        
        # Generate suggestions based on patterns
        if operators:
            most_common_op = max(operators, key=operators.get)
            suggestions.append(f"Most surviving mutations are {most_common_op} operators. Consider adding tests that verify these operations.")
        
        if files:
            most_problematic_file = max(files, key=files.get)
            suggestions.append(f"File {most_problematic_file} has the most surviving mutants. Focus testing efforts here.")
        
        # General suggestions
        suggestions.extend([
            "Add edge case tests for boundary conditions",
            "Verify error handling paths are tested", 
            "Consider property-based testing for complex algorithms",
            "Add integration tests for component interactions"
        ])
        
        return suggestions[:5]  # Return top 5 suggestions

def main():
    """Main entry point for mutation testing analysis"""
    parser = argparse.ArgumentParser(description='Analyze mutation testing results')
    parser.add_argument('--results-dir', default='target/mutants', help='Directory containing mutation results')
    parser.add_argument('--output-dir', default='target/mutants/reports', help='Output directory for reports')
    parser.add_argument('--generate-viz', action='store_true', help='Generate visualization')
    parser.add_argument('--analyze-survivors', action='store_true', help='Analyze surviving mutants')
    
    args = parser.parse_args()
    
    analyzer = MutationTestAnalyzer(args.results_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Analyze component scores
    scores = analyzer.analyze_component_scores()
    
    if not any(scores.values()):
        print("No mutation testing results found. Run mutation tests first.")
        return 1
    
    # Generate trend report
    trend_report = analyzer.generate_trend_report(scores)
    with open(output_dir / 'trend_analysis.md', 'w') as f:
        f.write(trend_report)
    
    print("📊 Mutation Testing Analysis")
    print("=" * 40)
    for component, score in scores.items():
        status = "✅" if score >= 80 else "⚠️" if score >= 70 else "❌"
        print(f"{component.title():12}: {score:5.1f}% {status}")
    
    # Generate visualization if requested
    if args.generate_viz:
        try:
            viz_file = analyzer.create_score_visualization(scores)
            print(f"\n📈 Visualization saved: {viz_file}")
        except ImportError:
            print("\n⚠️  Matplotlib not available for visualization")
    
    # Analyze surviving mutants if requested
    if args.analyze_survivors:
        surviving = analyzer.analyze_surviving_mutants('complete_results.json')
        if surviving:
            suggestions = analyzer.generate_improvement_suggestions(surviving)
            
            print(f"\n🔍 Found {len(surviving)} surviving mutants")
            print("\n💡 Improvement suggestions:")
            for i, suggestion in enumerate(suggestions, 1):
                print(f"   {i}. {suggestion}")
        else:
            print("\n✅ No surviving mutants found or data unavailable")
    
    # Overall assessment
    overall_score = scores.get('overall', 0)
    if overall_score >= 80:
        print(f"\n🎉 SUCCESS: Mutation score target achieved ({overall_score:.1f}%)")
        return 0
    else:
        print(f"\n⚠️  IMPROVEMENT NEEDED: Mutation score below target ({overall_score:.1f}%)")
        return 1

if __name__ == "__main__":
    sys.exit(main())