"""
RLHF-Based Code Evaluation Framework
Evaluates code generation quality using human preference learning and non-LLM metrics
Supports datasets like CrossCodeEval, RepoHyper, and others for comprehensive code assessment
"""

import json
import time
import asyncio
import logging
import subprocess
import tempfile
import os
import ast
import re
from typing import Dict, List, Any, Optional, Tuple, Union
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Code analysis and execution
import astunparse
from pylint import lint
from pylint.lint import Run
from pylint.reporters.text import TextReporter
import io
import sys

# ML/RLHF dependencies
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from transformers import AutoTokenizer, AutoModel
import requests

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class RLHFConfig:
    """Configuration for RLHF-based evaluation"""
    datasets: List[str] = None
    output_dir: str = "rlhf_evaluation_results"
    timeout: int = 30
    max_code_length: int = 2000
    enable_execution: bool = True
    enable_static_analysis: bool = True
    reward_model_path: Optional[str] = None
    preference_data_path: Optional[str] = None
    
    def __post_init__(self):
        if self.datasets is None:
            self.datasets = ["crosscodeeval", "repohyper", "humaneval"]

@dataclass
class CodeTestCase:
    """Individual code generation test case"""
    problem_id: str
    problem_description: str
    language: str
    expected_behavior: str
    test_cases: List[Dict[str, Any]]
    difficulty: str = "medium"
    category: str = "general"
    canonical_solution: Optional[str] = None
    
@dataclass
class CodeEvaluationResult:
    """Results from code evaluation"""
    problem_id: str
    generated_code: str
    language: str
    
    # Execution metrics
    execution_success: bool = False
    test_cases_passed: int = 0
    total_test_cases: int = 0
    execution_time: float = 0.0
    memory_usage: float = 0.0
    
    # Static analysis metrics
    syntax_valid: bool = False
    pylint_score: float = 0.0
    cyclomatic_complexity: int = 0
    lines_of_code: int = 0
    
    # RLHF metrics
    human_preference_score: float = 0.0
    readability_score: float = 0.0
    maintainability_score: float = 0.0
    
    # Performance metrics
    efficiency_score: float = 0.0
    correctness_score: float = 0.0
    
    error_message: Optional[str] = None
    timestamp: str = None
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now().isoformat()

class StaticAnalyzer:
    """Static code analysis utilities"""
    
    @staticmethod
    def analyze_python_code(code: str) -> Dict[str, Any]:
        """Analyze Python code for various metrics"""
        try:
            tree = ast.parse(code)
            
            # Basic metrics
            lines = len(code.split('\n'))
            
            # AST-based analysis
            complexity = StaticAnalyzer._calculate_complexity(tree)
            structure_score = StaticAnalyzer._analyze_structure(tree)
            
            # Pylint analysis
            pylint_score = StaticAnalyzer._run_pylint(code)
            
            return {
                'syntax_valid': True,
                'lines_of_code': lines,
                'cyclomatic_complexity': complexity,
                'structure_score': structure_score,
                'pylint_score': pylint_score,
                'readability_score': StaticAnalyzer._calculate_readability(code)
            }
            
        except SyntaxError as e:
            return {
                'syntax_valid': False,
                'error': str(e),
                'lines_of_code': len(code.split('\n')),
                'cyclomatic_complexity': 0,
                'structure_score': 0.0,
                'pylint_score': 0.0,
                'readability_score': 0.0
            }
    
    @staticmethod
    def _calculate_complexity(tree: ast.AST) -> int:
        """Calculate cyclomatic complexity"""
        complexity = 1  # Base complexity
        
        for node in ast.walk(tree):
            if isinstance(node, (ast.If, ast.While, ast.For, ast.AsyncFor)):
                complexity += 1
            elif isinstance(node, ast.ExceptHandler):
                complexity += 1
            elif isinstance(node, (ast.ListComp, ast.SetComp, ast.DictComp, ast.GeneratorExp)):
                complexity += 1
                
        return complexity
    
    @staticmethod
    def _analyze_structure(tree: ast.AST) -> float:
        """Analyze code structure quality"""
        functions = []
        classes = []
        
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef):
                functions.append(node)
            elif isinstance(node, ast.ClassDef):
                classes.append(node)
        
        # Score based on structure organization
        score = 0.5  # Base score
        
        if functions:
            score += 0.2  # Has functions
        if classes:
            score += 0.2  # Has classes
        if len(functions) <= 5:  # Not overly complex
            score += 0.1
            
        return min(score, 1.0)
    
    @staticmethod
    def _run_pylint(code: str) -> float:
        """Run pylint analysis"""
        try:
            with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
                f.write(code)
                f.flush()
                
                # Capture pylint output
                old_stderr = sys.stderr
                sys.stderr = io.StringIO()
                
                try:
                    results = Run([f.name, '--score=yes'], exit=False)
                    score = results.linter.stats.global_note
                    return max(score / 10.0, 0.0)  # Normalize to 0-1
                except:
                    return 0.0
                finally:
                    sys.stderr = old_stderr
                    os.unlink(f.name)
                    
        except Exception:
            return 0.0
    
    @staticmethod
    def _calculate_readability(code: str) -> float:
        """Calculate code readability score"""
        lines = code.split('\n')
        
        # Basic readability metrics
        avg_line_length = np.mean([len(line) for line in lines if line.strip()])
        comment_ratio = len([line for line in lines if line.strip().startswith('#')]) / max(len(lines), 1)
        
        # Penalize very long lines, reward comments
        readability = 0.8  # Base score
        
        if avg_line_length > 120:
            readability -= 0.2
        elif avg_line_length > 80:
            readability -= 0.1
            
        readability += min(comment_ratio * 0.5, 0.2)  # Up to 20% bonus for comments
        
        return max(min(readability, 1.0), 0.0)

class CodeExecutor:
    """Safe code execution environment"""
    
    def __init__(self, timeout: int = 30):
        self.timeout = timeout
    
    def execute_python_code(self, code: str, test_cases: List[Dict]) -> Dict[str, Any]:
        """Execute Python code with test cases"""
        results = {
            'execution_success': False,
            'test_cases_passed': 0,
            'total_test_cases': len(test_cases),
            'execution_time': 0.0,
            'error_message': None
        }
        
        try:
            with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
                # Write test code
                test_code = self._generate_test_code(code, test_cases)
                f.write(test_code)
                f.flush()
                
                start_time = time.time()
                
                # Execute in subprocess for safety
                result = subprocess.run(
                    [sys.executable, f.name],
                    capture_output=True,
                    text=True,
                    timeout=self.timeout
                )
                
                results['execution_time'] = time.time() - start_time
                
                if result.returncode == 0:
                    # Parse results from stdout
                    output_lines = result.stdout.strip().split('\n')
                    if output_lines and output_lines[-1].startswith('RESULTS:'):
                        test_results = json.loads(output_lines[-1].split('RESULTS:', 1)[1])
                        results['test_cases_passed'] = test_results['passed']
                        results['execution_success'] = test_results['passed'] == len(test_cases)
                else:
                    results['error_message'] = result.stderr
                    
                os.unlink(f.name)
                
        except subprocess.TimeoutExpired:
            results['error_message'] = "Execution timeout"
        except Exception as e:
            results['error_message'] = str(e)
            
        return results
    
    def _generate_test_code(self, code: str, test_cases: List[Dict]) -> str:
        """Generate test code for execution"""
        test_template = '''
import json
import sys

{user_code}

def run_tests():
    results = {{"passed": 0, "total": {total_tests}}}
    test_cases = {test_cases}
    
    for i, test_case in enumerate(test_cases):
        try:
            inputs = test_case.get("input", {{}})
            expected = test_case.get("expected", None)
            
            # Extract function name from user code (simple heuristic)
            import ast
            tree = ast.parse("""{user_code}""")
            function_names = [node.name for node in ast.walk(tree) if isinstance(node, ast.FunctionDef)]
            
            if function_names:
                func = globals()[function_names[0]]
                if isinstance(inputs, dict):
                    result = func(**inputs)
                elif isinstance(inputs, list):
                    result = func(*inputs)
                else:
                    result = func(inputs)
                
                if result == expected:
                    results["passed"] += 1
            
        except Exception as e:
            pass  # Test failed
    
    return results

if __name__ == "__main__":
    results = run_tests()
    print(f"RESULTS:{{json.dumps(results)}}")
'''
        
        return test_template.format(
            user_code=code,
            test_cases=json.dumps(test_cases),
            total_tests=len(test_cases)
        )

class HumanPreferenceModel:
    """Simple reward model for human preferences"""
    
    def __init__(self, model_path: Optional[str] = None):
        self.model_path = model_path
        self.tokenizer = None
        self.model = None
        
        if model_path:
            self._load_model()
    
    def _load_model(self):
        """Load pre-trained reward model"""
        try:
            self.tokenizer = AutoTokenizer.from_pretrained("microsoft/codebert-base")
            self.model = AutoModel.from_pretrained("microsoft/codebert-base")
        except Exception as e:
            logger.warning(f"Could not load reward model: {e}")
    
    def score_code_preference(self, code: str, problem_description: str) -> float:
        """Score code based on learned human preferences"""
        if not self.model:
            return self._heuristic_preference_score(code, problem_description)
        
        try:
            # Use CodeBERT for code representation
            inputs = self.tokenizer(
                problem_description, 
                code, 
                return_tensors="pt", 
                truncation=True, 
                max_length=512
            )
            
            with torch.no_grad():
                outputs = self.model(**inputs)
                # Simple scoring based on representation similarity
                score = torch.cosine_similarity(
                    outputs.last_hidden_state[:, 0, :],
                    outputs.last_hidden_state[:, 1, :],
                    dim=1
                ).item()
                
            return (score + 1) / 2  # Normalize to 0-1
            
        except Exception:
            return self._heuristic_preference_score(code, problem_description)
    
    def _heuristic_preference_score(self, code: str, problem_description: str) -> float:
        """Heuristic-based preference scoring"""
        score = 0.5  # Base score
        
        # Length appropriateness
        code_length = len(code)
        if 50 <= code_length <= 500:
            score += 0.1
        
        # Variable naming quality
        if re.search(r'[a-zA-Z][a-zA-Z0-9_]*', code):
            score += 0.1
        
        # Comments presence
        if '#' in code:
            score += 0.1
        
        # Problem description keyword matching
        problem_words = set(re.findall(r'\w+', problem_description.lower()))
        code_words = set(re.findall(r'\w+', code.lower()))
        overlap = len(problem_words & code_words) / max(len(problem_words), 1)
        score += overlap * 0.2
        
        return min(score, 1.0)

class DatasetLoader:
    """Loader for code evaluation datasets"""
    
    @staticmethod
    def load_crosscodeeval() -> List[CodeTestCase]:
        """Load CrossCodeEval dataset"""
        # Mock implementation - in practice, load from actual dataset
        return [
            CodeTestCase(
                problem_id="crosscode_001",
                problem_description="Write a function to find the maximum element in a list",
                language="python",
                expected_behavior="Return the maximum value from the input list",
                test_cases=[
                    {"input": [1, 3, 2, 5, 4], "expected": 5},
                    {"input": [-1, -5, -2], "expected": -1},
                    {"input": [42], "expected": 42}
                ],
                category="algorithms"
            ),
            CodeTestCase(
                problem_id="crosscode_002", 
                problem_description="Implement a function to reverse a string",
                language="python",
                expected_behavior="Return the input string reversed",
                test_cases=[
                    {"input": "hello", "expected": "olleh"},
                    {"input": "world", "expected": "dlrow"},
                    {"input": "", "expected": ""}
                ],
                category="string_manipulation"
            )
        ]
    
    @staticmethod
    def load_repohyper() -> List[CodeTestCase]:
        """Load RepoHyper dataset"""
        return [
            CodeTestCase(
                problem_id="repo_001",
                problem_description="Create a class to represent a binary tree node",
                language="python", 
                expected_behavior="Class with value, left, right attributes and basic operations",
                test_cases=[
                    {"input": {"value": 5}, "expected": True},  # Can create node
                    {"input": {"value": 5, "left": None, "right": None}, "expected": True}
                ],
                category="data_structures",
                difficulty="medium"
            )
        ]
    
    @staticmethod
    def load_humaneval() -> List[CodeTestCase]:
        """Load HumanEval dataset"""
        return [
            CodeTestCase(
                problem_id="humaneval_001",
                problem_description="Write a function that takes two numbers and returns their sum",
                language="python",
                expected_behavior="Return a + b",
                test_cases=[
                    {"input": [2, 3], "expected": 5},
                    {"input": [0, 0], "expected": 0},
                    {"input": [-1, 1], "expected": 0}
                ],
                category="arithmetic"
            )
        ]

class RLHFEvaluationFramework:
    """Main RLHF-based evaluation framework"""
    
    def __init__(self, config: RLHFConfig):
        self.config = config
        self.test_cases: List[CodeTestCase] = []
        self.results: List[CodeEvaluationResult] = []
        
        # Initialize components
        self.static_analyzer = StaticAnalyzer()
        self.code_executor = CodeExecutor(timeout=config.timeout)
        self.preference_model = HumanPreferenceModel(config.reward_model_path)
        
        # Create output directory
        Path(config.output_dir).mkdir(parents=True, exist_ok=True)
        
        self._load_datasets()
    
    def _load_datasets(self):
        """Load specified datasets"""
        loader = DatasetLoader()
        
        for dataset_name in self.config.datasets:
            if dataset_name == "crosscodeeval":
                self.test_cases.extend(loader.load_crosscodeeval())
            elif dataset_name == "repohyper":
                self.test_cases.extend(loader.load_repohyper())
            elif dataset_name == "humaneval":
                self.test_cases.extend(loader.load_humaneval())
            else:
                logger.warning(f"Unknown dataset: {dataset_name}")
        
        logger.info(f"Loaded {len(self.test_cases)} test cases")
    
    def evaluate_code(self, problem_id: str, generated_code: str) -> CodeEvaluationResult:
        """Evaluate a single code generation"""
        # Find test case
        test_case = next((tc for tc in self.test_cases if tc.problem_id == problem_id), None)
        if not test_case:
            raise ValueError(f"Test case {problem_id} not found")
        
        result = CodeEvaluationResult(
            problem_id=problem_id,
            generated_code=generated_code,
            language=test_case.language
        )
        
        # Static analysis
        if self.config.enable_static_analysis and test_case.language == "python":
            static_results = self.static_analyzer.analyze_python_code(generated_code)
            result.syntax_valid = static_results['syntax_valid']
            result.pylint_score = static_results['pylint_score'] 
            result.cyclomatic_complexity = static_results['cyclomatic_complexity']
            result.lines_of_code = static_results['lines_of_code']
            result.readability_score = static_results['readability_score']
        
        # Code execution
        if self.config.enable_execution and result.syntax_valid and test_case.language == "python":
            exec_results = self.code_executor.execute_python_code(generated_code, test_case.test_cases)
            result.execution_success = exec_results['execution_success']
            result.test_cases_passed = exec_results['test_cases_passed']
            result.total_test_cases = exec_results['total_test_cases']
            result.execution_time = exec_results['execution_time']
            if exec_results['error_message']:
                result.error_message = exec_results['error_message']
        
        # Calculate composite scores
        result.correctness_score = result.test_cases_passed / max(result.total_test_cases, 1)
        
        # Human preference scoring
        result.human_preference_score = self.preference_model.score_code_preference(
            generated_code, test_case.problem_description
        )
        
        # Efficiency score (based on execution time and complexity)
        if result.execution_success and result.execution_time > 0:
            result.efficiency_score = min(1.0 / (1.0 + result.execution_time), 1.0)
        else:
            result.efficiency_score = 0.0
        
        # Maintainability (composite of readability and complexity)
        complexity_penalty = min(result.cyclomatic_complexity / 10.0, 1.0)
        result.maintainability_score = (result.readability_score + (1.0 - complexity_penalty)) / 2.0
        
        self.results.append(result)
        return result
    
    def batch_evaluate(self, code_submissions: Dict[str, str]) -> List[CodeEvaluationResult]:
        """Evaluate multiple code submissions"""
        results = []
        
        for problem_id, code in code_submissions.items():
            try:
                result = self.evaluate_code(problem_id, code)
                results.append(result)
                logger.info(f"Evaluated {problem_id}: correctness={result.correctness_score:.3f}")
            except Exception as e:
                logger.error(f"Failed to evaluate {problem_id}: {e}")
        
        return results
    
    def generate_comprehensive_report(self) -> pd.DataFrame:
        """Generate comprehensive evaluation report"""
        if not self.results:
            return pd.DataFrame()
        
        data = []
        for result in self.results:
            data.append({
                'Problem ID': result.problem_id,
                'Language': result.language,
                'Syntax Valid': result.syntax_valid,
                'Execution Success': result.execution_success,
                'Tests Passed': f"{result.test_cases_passed}/{result.total_test_cases}",
                'Correctness Score': result.correctness_score,
                'Human Preference': result.human_preference_score,
                'Readability': result.readability_score,
                'Maintainability': result.maintainability_score,
                'Efficiency': result.efficiency_score,
                'PyLint Score': result.pylint_score,
                'Complexity': result.cyclomatic_complexity,
                'Lines of Code': result.lines_of_code,
                'Execution Time': result.execution_time,
                'Has Error': result.error_message is not None
            })
        
        return pd.DataFrame(data)
    
    def create_rlhf_visualizations(self):
        """Create RLHF-specific visualizations"""
        if not self.results:
            logger.warning("No results to visualize")
            return
        
        df = self.generate_comprehensive_report()
        
        plt.style.use('seaborn-v0_8')
        fig, axes = plt.subplots(2, 3, figsize=(20, 12))
        fig.suptitle('RLHF Code Evaluation Results', fontsize=16, fontweight='bold')
        
        # 1. Score Distribution
        scores = ['Correctness Score', 'Human Preference', 'Readability', 'Maintainability', 'Efficiency']
        score_data = df[scores].values.flatten()
        axes[0, 0].hist(score_data, bins=20, alpha=0.7, color='skyblue', edgecolor='black')
        axes[0, 0].set_title('Score Distribution')
        axes[0, 0].set_xlabel('Score')
        axes[0, 0].set_ylabel('Frequency')
        
        # 2. Correctness vs Human Preference
        valid_df = df[df['Syntax Valid'] == True]
        if not valid_df.empty:
            axes[0, 1].scatter(valid_df['Correctness Score'], valid_df['Human Preference'], 
                             alpha=0.6, color='green')
            axes[0, 1].set_title('Correctness vs Human Preference')
            axes[0, 1].set_xlabel('Correctness Score')
            axes[0, 1].set_ylabel('Human Preference Score')
        
        # 3. Complexity vs Maintainability
        if not valid_df.empty:
            axes[0, 2].scatter(valid_df['Complexity'], valid_df['Maintainability'], 
                             alpha=0.6, color='orange')
            axes[0, 2].set_title('Complexity vs Maintainability')
            axes[0, 2].set_xlabel('Cyclomatic Complexity')
            axes[0, 2].set_ylabel('Maintainability Score')
        
        # 4. Radar Chart for Average Scores
        if not df.empty:
            avg_scores = df[scores].mean()
            angles = np.linspace(0, 2 * np.pi, len(scores), endpoint=False).tolist()
            avg_scores_list = avg_scores.tolist()
            
            # Complete the circle
            angles += angles[:1]
            avg_scores_list += avg_scores_list[:1]
            
            axes[1, 0].plot(angles, avg_scores_list, 'o-', linewidth=2, color='red')
            axes[1, 0].fill(angles, avg_scores_list, alpha=0.25, color='red')
            axes[1, 0].set_xticks(angles[:-1])
            axes[1, 0].set_xticklabels(scores)
            axes[1, 0].set_title('Average Performance Radar')
            axes[1, 0].set_ylim(0, 1)
        
        # 5. Success Rate by Problem Category
        problem_categories = []
        for result in self.results:
            test_case = next((tc for tc in self.test_cases if tc.problem_id == result.problem_id), None)
            if test_case:
                problem_categories.append(test_case.category)
        
        if problem_categories:
            df['Category'] = problem_categories
            category_success = df.groupby('Category')['Correctness Score'].mean()
            category_success.plot(kind='bar', ax=axes[1, 1], color='purple', alpha=0.7)
            axes[1, 1].set_title('Success Rate by Category')
            axes[1, 1].set_ylabel('Average Correctness Score')
            axes[1, 1].tick_params(axis='x', rotation=45)
        
        # 6. Execution Time vs Lines of Code
        valid_exec_df = df[(df['Execution Success'] == True) & (df['Execution Time'] > 0)]
        if not valid_exec_df.empty:
            axes[1, 2].scatter(valid_exec_df['Lines of Code'], valid_exec_df['Execution Time'], 
                             alpha=0.6, color='blue')
            axes[1, 2].set_title('Code Length vs Execution Time')
            axes[1, 2].set_xlabel('Lines of Code')
            axes[1, 2].set_ylabel('Execution Time (s)')
        
        plt.tight_layout()
        
        # Save visualization
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        output_path = Path(self.config.output_dir) / f"rlhf_evaluation_{timestamp}.png"
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        logger.info(f"RLHF visualization saved to: {output_path}")
        
        plt.show()
    
    def export_results(self):
        """Export detailed results"""
        if not self.results:
            logger.warning("No results to export")
            return
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        # Detailed JSON export
        json_path = Path(self.config.output_dir) / f"rlhf_detailed_{timestamp}.json"
        with open(json_path, 'w') as f:
            json.dump([asdict(result) for result in self.results], f, indent=2)
        
        # CSV report
        df = self.generate_comprehensive_report()
        csv_path = Path(self.config.output_dir) / f"rlhf_report_{timestamp}.csv"
        df.to_csv(csv_path, index=False)
        
        logger.info(f"RLHF results exported to: {json_path} and {csv_path}")
    
    def print_summary(self):
        """Print evaluation summary"""
        if not self.results:
            print("No evaluation results available.")
            return
        
        df = self.generate_comprehensive_report()
        
        print("\n" + "="*70)
        print("RLHF CODE EVALUATION SUMMARY")
        print("="*70)
        
        print(f"📊 Total Problems Evaluated: {len(self.results)}")
        print(f"🔍 Syntax Valid: {df['Syntax Valid'].sum()}/{len(df)} ({df['Syntax Valid'].mean():.1%})")
        print(f"✅ Execution Success: {df['Execution Success'].sum()}/{len(df)} ({df['Execution Success'].mean():.1%})")
        
        # Score averages
        print(f"\n📈 AVERAGE SCORES:")
        print(f"  Correctness: {df['Correctness Score'].mean():.3f}")
        print(f"  Human Preference: {df['Human Preference'].mean():.3f}")
        print(f"  Readability: {df['Readability'].mean():.3f}")
        print(f"  Maintainability: {df['Maintainability'].mean():.3f}")
        print(f"  Efficiency: {df['Efficiency'].mean():.3f}")
        
        # Best performing problems
        print(f"\n🏆 TOP PERFORMING PROBLEMS:")
        top_problems = df.nlargest(3, 'Correctness Score')[['Problem ID', 'Correctness Score', 'Human Preference']]
        for i, (_, row) in enumerate(top_problems.iterrows(), 1):
            print(f"  {i}. {row['Problem ID']}: {row['Correctness Score']:.3f} correctness, {row['Human Preference']:.3f} preference")
        
        print("="*70)

def create_sample_submissions() -> Dict[str, str]:
    """Create sample code submissions for testing"""
    return {
        "crosscode_001": '''def find_max(lst):
    """Find the maximum element in a list"""
    if not lst:
        return None
    max_val = lst[0]
    for item in lst:
        if item > max_val:
            max_val = item
    return max_val''',
        
        "crosscode_002": '''def reverse_string(s):
    """Reverse a string"""
    return s[::-1]''',
        
        "humaneval_001": '''def add_numbers(a, b):
    """Add two numbers"""
    return a + b'''
    }

async def main():
    """Main execution function"""
    # Configuration
    config = RLHFConfig(
        datasets=["crosscodeeval", "repohyper", "humaneval"],
        output_dir="rlhf_evaluation_results",
        enable_execution=True,
        enable_static_analysis=True
    )
    
    # Initialize framework
    framework = RLHFEvaluationFramework(config)
    
    # Sample code submissions
    submissions = create_sample_submissions()
    
    try:
        # Run evaluation
        logger.info("Starting RLHF-based code evaluation")
        results = framework.batch_evaluate(submissions)
        
        # Generate outputs
        framework.print_summary()
        framework.create_rlhf_visualizations()
        framework.export_results()
        
        logger.info("RLHF evaluation completed successfully")
        
    except Exception as e:
        logger.error(f"Evaluation failed: {e}")
        print(f"Error: {e}")

if __name__ == "__main__":
    asyncio.run(main())