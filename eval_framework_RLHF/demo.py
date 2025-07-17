"""
Demo script for RLHF-based Code Evaluation Framework
Shows how to use the framework to evaluate code generation quality
"""

import asyncio
import logging
from app import RLHFEvaluationFramework, RLHFConfig
from datasets import DatasetManager

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def create_sample_code_submissions() -> dict:
    """Create sample code submissions for different problems"""
    return {
        # CrossCodeEval submissions
        "crosscode_array_max": '''def find_max(arr):
    """Find the maximum element in an array"""
    if not arr:
        return None
    
    max_val = arr[0]
    for item in arr:
        if item > max_val:
            max_val = item
    return max_val''',
        
        "crosscode_string_reverse": '''def reverse_string(s):
    """Reverse a string without using built-in functions"""
    if not s:
        return ""
    
    result = ""
    for i in range(len(s) - 1, -1, -1):
        result += s[i]
    return result''',
        
        "crosscode_fibonacci": '''def fibonacci(n):
    """Calculate nth Fibonacci number efficiently"""
    if n <= 1:
        return n
    
    # Use iterative approach for efficiency
    a, b = 0, 1
    for _ in range(2, n + 1):
        a, b = b, a + b
    return b''',
        
        "crosscode_palindrome_check": '''def is_palindrome(s):
    """Check if string is a palindrome (case insensitive)"""
    if not s:
        return True
    
    # Clean the string: remove non-alphanumeric and convert to lowercase
    cleaned = ""
    for char in s:
        if char.isalnum():
            cleaned += char.lower()
    
    # Check if cleaned string equals its reverse
    return cleaned == cleaned[::-1]''',
        
        # HumanEval submissions
        "humaneval_sum_two_numbers": '''def add_numbers(a, b):
    """Add two numbers and return the result."""
    return a + b''',
        
        "humaneval_list_contains": '''def contains_element(lst, element):
    """Check if a list contains a specific element."""
    for item in lst:
        if item == element:
            return True
    return False''',
        
        "humaneval_factorial": '''def factorial(n):
    """Calculate the factorial of n."""
    if n < 0:
        raise ValueError("Factorial undefined for negative numbers")
    if n == 0 or n == 1:
        return 1
    
    result = 1
    for i in range(2, n + 1):
        result *= i
    return result''',
        
        # RepoHyper submissions
        "repo_binary_tree_node": '''class BinaryTreeNode:
    """Binary tree node implementation with search and traversal"""
    
    def __init__(self, value, left=None, right=None):
        self.value = value
        self.left = left
        self.right = right
    
    def insert(self, value):
        """Insert a value into the binary search tree"""
        if value < self.value:
            if self.left is None:
                self.left = BinaryTreeNode(value)
            else:
                self.left.insert(value)
        elif value > self.value:
            if self.right is None:
                self.right = BinaryTreeNode(value)
            else:
                self.right.insert(value)
        # Ignore duplicates
    
    def find(self, value):
        """Find a value in the tree"""
        if value == self.value:
            return True
        elif value < self.value and self.left:
            return self.left.find(value)
        elif value > self.value and self.right:
            return self.right.find(value)
        return False
    
    def inorder_traversal(self):
        """Return inorder traversal of the tree"""
        result = []
        if self.left:
            result.extend(self.left.inorder_traversal())
        result.append(self.value)
        if self.right:
            result.extend(self.right.inorder_traversal())
        return result''',
        
        "repo_stack_implementation": '''class Stack:
    """Stack implementation with standard operations"""
    
    def __init__(self):
        self.items = []
    
    def push(self, item):
        """Add item to top of stack"""
        self.items.append(item)
    
    def pop(self):
        """Remove and return top item"""
        if self.is_empty():
            raise IndexError("pop from empty stack")
        return self.items.pop()
    
    def peek(self):
        """Return top item without removing it"""
        if self.is_empty():
            raise IndexError("peek from empty stack")
        return self.items[-1]
    
    def is_empty(self):
        """Check if stack is empty"""
        return len(self.items) == 0
    
    def size(self):
        """Return number of items in stack"""
        return len(self.items)''',
        
        # Contest problems
        "contest_two_sum": '''def two_sum(nums, target):
    """Find indices of two numbers that sum to target"""
    # Use hash map for O(n) solution
    seen = {}
    for i, num in enumerate(nums):
        complement = target - num
        if complement in seen:
            return [seen[complement], i]
        seen[num] = i
    return []''',
        
        "contest_valid_parentheses": '''def is_valid_parentheses(s):
    """Check if parentheses are properly balanced"""
    stack = []
    mapping = {")": "(", "}": "{", "]": "["}
    
    for char in s:
        if char in mapping:
            # Closing bracket
            if not stack or stack.pop() != mapping[char]:
                return False
        else:
            # Opening bracket
            stack.append(char)
    
    return len(stack) == 0'''
    }

def create_buggy_code_submissions() -> dict:
    """Create submissions with intentional bugs for comparison"""
    return {
        "crosscode_array_max": '''def find_max(arr):
    """Buggy version - doesn't handle empty arrays"""
    max_val = arr[0]  # This will crash on empty array
    for item in arr:
        if item > max_val:
            max_val = item
    return max_val''',
        
        "crosscode_string_reverse": '''def reverse_string(s):
    """Buggy version - uses built-in reverse"""
    return s[::-1]  # Not following the requirement''',
        
        "humaneval_sum_two_numbers": '''def add_numbers(a, b):
    """Buggy version - wrong operation"""
    return a * b  # Should be addition, not multiplication''',
        
        "humaneval_factorial": '''def factorial(n):
    """Buggy version - infinite recursion"""
    return n * factorial(n - 1)  # Missing base case'''
    }

async def run_comprehensive_demo():
    """Run a comprehensive demonstration of the RLHF framework"""
    print("🚀 Starting RLHF Code Evaluation Framework Demo")
    print("=" * 60)
    
    # Initialize dataset manager
    print("\n📊 Loading datasets...")
    dataset_manager = DatasetManager()
    
    # Show available datasets
    print("Available datasets:")
    for dataset in dataset_manager.get_available_datasets():
        print(f"  ✓ {dataset}")
    
    # Get dataset information
    print("\n📋 Dataset Information:")
    dataset_info = dataset_manager.get_dataset_info()
    for name, info in dataset_info.items():
        if "error" not in info:
            print(f"  {name}: {info['total_problems']} problems, "
                  f"Categories: {', '.join(info['categories'])}")
    
    # Configure RLHF framework
    print("\n⚙️ Configuring RLHF Evaluation Framework...")
    config = RLHFConfig(
        datasets=["crosscodeeval", "repohyper", "humaneval", "codecontest"],
        output_dir="demo_results",
        enable_execution=True,
        enable_static_analysis=True,
        timeout=30
    )
    
    # Initialize framework
    framework = RLHFEvaluationFramework(config)
    print(f"Framework initialized with {len(framework.test_cases)} test cases")
    
    # Evaluate good code submissions
    print("\n✅ Evaluating High-Quality Code Submissions...")
    good_submissions = create_sample_code_submissions()
    
    good_results = []
    for problem_id, code in good_submissions.items():
        try:
            result = framework.evaluate_code(problem_id, code)
            good_results.append(result)
            print(f"  ✓ {problem_id}: Correctness={result.correctness_score:.3f}, "
                  f"Preference={result.human_preference_score:.3f}")
        except Exception as e:
            print(f"  ❌ {problem_id}: Error - {e}")
    
    # Evaluate buggy code submissions
    print("\n🐛 Evaluating Buggy Code Submissions...")
    buggy_submissions = create_buggy_code_submissions()
    
    buggy_results = []
    for problem_id, code in buggy_submissions.items():
        try:
            result = framework.evaluate_code(problem_id, code)
            buggy_results.append(result)
            print(f"  ⚠️ {problem_id}: Correctness={result.correctness_score:.3f}, "
                  f"Preference={result.human_preference_score:.3f}")
        except Exception as e:
            print(f"  ❌ {problem_id}: Error - {e}")
    
    # Generate comprehensive analysis
    print("\n📊 Generating Comprehensive Analysis...")
    
    # Print detailed summary
    framework.print_summary()
    
    # Create visualizations
    print("\n📈 Creating Visualizations...")
    try:
        framework.create_rlhf_visualizations()
        print("✓ Visualizations created successfully")
    except Exception as e:
        print(f"⚠️ Visualization creation failed: {e}")
    
    # Export results
    print("\n💾 Exporting Results...")
    framework.export_results()
    print("✓ Results exported successfully")
    
    # Compare good vs buggy submissions
    print("\n🔍 Comparison Analysis:")
    print("-" * 40)
    
    if good_results and buggy_results:
        avg_good_correctness = sum(r.correctness_score for r in good_results) / len(good_results)
        avg_buggy_correctness = sum(r.correctness_score for r in buggy_results) / len(buggy_results)
        
        avg_good_preference = sum(r.human_preference_score for r in good_results) / len(good_results)
        avg_buggy_preference = sum(r.human_preference_score for r in buggy_results) / len(buggy_results)
        
        print(f"📈 Average Correctness:")
        print(f"  Good Code: {avg_good_correctness:.3f}")
        print(f"  Buggy Code: {avg_buggy_correctness:.3f}")
        print(f"  Difference: {avg_good_correctness - avg_buggy_correctness:.3f}")
        
        print(f"\n🎯 Average Human Preference:")
        print(f"  Good Code: {avg_good_preference:.3f}")
        print(f"  Buggy Code: {avg_buggy_preference:.3f}")
        print(f"  Difference: {avg_good_preference - avg_buggy_preference:.3f}")
    
    print("\n✨ Demo completed successfully!")
    print("Check the 'demo_results' directory for detailed outputs.")

def run_simple_demo():
    """Run a simple demo with just a few examples"""
    print("🚀 RLHF Framework - Simple Demo")
    print("=" * 40)
    
    # Configure framework
    config = RLHFConfig(
        datasets=["crosscodeeval"],
        output_dir="simple_demo_results",
        enable_execution=True,
        enable_static_analysis=True
    )
    
    framework = RLHFEvaluationFramework(config)
    
    # Simple test case
    test_code = '''def find_max(arr):
    """Find maximum element in array"""
    if not arr:
        return None
    return max(arr)'''
    
    # Evaluate
    result = framework.evaluate_code("crosscode_array_max", test_code)
    
    print(f"Problem: {result.problem_id}")
    print(f"Syntax Valid: {result.syntax_valid}")
    print(f"Execution Success: {result.execution_success}")
    print(f"Tests Passed: {result.test_cases_passed}/{result.total_test_cases}")
    print(f"Correctness Score: {result.correctness_score:.3f}")
    print(f"Human Preference: {result.human_preference_score:.3f}")
    print(f"Readability: {result.readability_score:.3f}")
    print(f"Maintainability: {result.maintainability_score:.3f}")

def main():
    """Main function with demo options"""
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "--simple":
        run_simple_demo()
    else:
        asyncio.run(run_comprehensive_demo())

if __name__ == "__main__":
    main()