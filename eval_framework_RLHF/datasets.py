"""
Dataset loaders and utilities for code evaluation benchmarks
Supports CrossCodeEval, RepoHyper, HumanEval, and other code generation datasets
"""

import json
import requests
import tempfile
import zipfile
import os
from typing import List, Dict, Any, Optional
from pathlib import Path
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)

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
    starter_code: Optional[str] = None
    constraints: Optional[str] = None

class CrossCodeEvalLoader:
    """Loader for CrossCodeEval dataset"""
    
    @staticmethod
    def load_sample_data() -> List[CodeTestCase]:
        """Load sample CrossCodeEval problems"""
        return [
            CodeTestCase(
                problem_id="crosscode_array_max",
                problem_description="""
Write a function that finds the maximum element in an array.

The function should:
- Take an array of integers as input
- Return the maximum value in the array
- Handle empty arrays by returning None
- Work efficiently for large arrays
                """.strip(),
                language="python",
                expected_behavior="Return the maximum value from the input array",
                test_cases=[
                    {"input": [[1, 3, 2, 5, 4]], "expected": 5},
                    {"input": [[-1, -5, -2]], "expected": -1},
                    {"input": [[42]], "expected": 42},
                    {"input": [[]], "expected": None},
                    {"input": [[1, 1, 1, 1]], "expected": 1}
                ],
                category="array_operations",
                difficulty="easy",
                canonical_solution="""def find_max(arr):
    if not arr:
        return None
    return max(arr)"""
            ),
            CodeTestCase(
                problem_id="crosscode_string_reverse",
                problem_description="""
Implement a function to reverse a string without using built-in reverse functions.

Requirements:
- Input: A string
- Output: The string reversed
- Do not use [::-1] or reversed() function
- Handle empty strings
                """.strip(),
                language="python",
                expected_behavior="Return the input string reversed",
                test_cases=[
                    {"input": ["hello"], "expected": "olleh"},
                    {"input": ["world"], "expected": "dlrow"},
                    {"input": [""], "expected": ""},
                    {"input": ["a"], "expected": "a"},
                    {"input": ["abcdef"], "expected": "fedcba"}
                ],
                category="string_manipulation",
                difficulty="easy",
                canonical_solution="""def reverse_string(s):
    result = ""
    for char in s:
        result = char + result
    return result"""
            ),
            CodeTestCase(
                problem_id="crosscode_fibonacci",
                problem_description="""
Write a function to calculate the nth Fibonacci number.

The Fibonacci sequence is defined as:
- F(0) = 0
- F(1) = 1  
- F(n) = F(n-1) + F(n-2) for n > 1

Your function should be efficient for reasonable values of n (n < 50).
                """.strip(),
                language="python",
                expected_behavior="Return the nth Fibonacci number",
                test_cases=[
                    {"input": [0], "expected": 0},
                    {"input": [1], "expected": 1},
                    {"input": [2], "expected": 1},
                    {"input": [5], "expected": 5},
                    {"input": [10], "expected": 55},
                    {"input": [15], "expected": 610}
                ],
                category="algorithms",
                difficulty="medium",
                canonical_solution="""def fibonacci(n):
    if n <= 1:
        return n
    a, b = 0, 1
    for _ in range(2, n + 1):
        a, b = b, a + b
    return b"""
            ),
            CodeTestCase(
                problem_id="crosscode_palindrome_check",
                problem_description="""
Write a function to check if a string is a palindrome.

Requirements:
- Case insensitive comparison
- Ignore spaces and punctuation
- Return True if palindrome, False otherwise
- Handle empty strings (return True)
                """.strip(),
                language="python",
                expected_behavior="Return True if string is a palindrome, False otherwise",
                test_cases=[
                    {"input": ["racecar"], "expected": True},
                    {"input": ["hello"], "expected": False},
                    {"input": ["A man a plan a canal Panama"], "expected": True},
                    {"input": ["race a car"], "expected": False},
                    {"input": [""], "expected": True},
                    {"input": ["Madam"], "expected": True}
                ],
                category="string_algorithms",
                difficulty="medium",
                canonical_solution="""def is_palindrome(s):
    cleaned = ''.join(char.lower() for char in s if char.isalnum())
    return cleaned == cleaned[::-1]"""
            )
        ]

class RepoHyperLoader:
    """Loader for RepoHyper dataset (repository-level code understanding)"""
    
    @staticmethod 
    def load_sample_data() -> List[CodeTestCase]:
        """Load sample RepoHyper problems"""
        return [
            CodeTestCase(
                problem_id="repo_binary_tree_node",
                problem_description="""
Create a BinaryTreeNode class for implementing binary trees.

Requirements:
- Constructor should accept a value and optional left/right child nodes
- Include methods: insert(value), find(value), inorder_traversal()
- Handle duplicate values by ignoring them
- Implement proper string representation
                """.strip(),
                language="python",
                expected_behavior="Implement a complete binary tree node class",
                test_cases=[
                    {
                        "input": {"action": "create", "value": 5},
                        "expected": "node_created"
                    },
                    {
                        "input": {"action": "insert", "values": [3, 7, 1, 9]},
                        "expected": "values_inserted"
                    },
                    {
                        "input": {"action": "find", "value": 7},
                        "expected": True
                    },
                    {
                        "input": {"action": "find", "value": 4},
                        "expected": False
                    },
                    {
                        "input": {"action": "traversal"},
                        "expected": [1, 3, 5, 7, 9]
                    }
                ],
                category="data_structures",
                difficulty="medium",
                canonical_solution="""class BinaryTreeNode:
    def __init__(self, value, left=None, right=None):
        self.value = value
        self.left = left
        self.right = right
    
    def insert(self, value):
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
    
    def find(self, value):
        if value == self.value:
            return True
        elif value < self.value and self.left:
            return self.left.find(value)
        elif value > self.value and self.right:
            return self.right.find(value)
        return False
    
    def inorder_traversal(self):
        result = []
        if self.left:
            result.extend(self.left.inorder_traversal())
        result.append(self.value)
        if self.right:
            result.extend(self.right.inorder_traversal())
        return result"""
            ),
            CodeTestCase(
                problem_id="repo_stack_implementation",
                problem_description="""
Implement a Stack class with standard stack operations.

Requirements:
- push(item): Add item to top of stack
- pop(): Remove and return top item (raise exception if empty)
- peek(): Return top item without removing (raise exception if empty)  
- is_empty(): Return True if stack is empty
- size(): Return number of items in stack
                """.strip(),
                language="python",
                expected_behavior="Complete stack implementation with all operations",
                test_cases=[
                    {
                        "input": {"action": "create"},
                        "expected": "stack_created"
                    },
                    {
                        "input": {"action": "is_empty"},
                        "expected": True
                    },
                    {
                        "input": {"action": "push", "values": [1, 2, 3]},
                        "expected": "pushed"
                    },
                    {
                        "input": {"action": "size"},
                        "expected": 3
                    },
                    {
                        "input": {"action": "peek"},
                        "expected": 3
                    },
                    {
                        "input": {"action": "pop"},
                        "expected": 3
                    },
                    {
                        "input": {"action": "size"},
                        "expected": 2
                    }
                ],
                category="data_structures",
                difficulty="easy",
                canonical_solution="""class Stack:
    def __init__(self):
        self.items = []
    
    def push(self, item):
        self.items.append(item)
    
    def pop(self):
        if self.is_empty():
            raise IndexError("pop from empty stack")
        return self.items.pop()
    
    def peek(self):
        if self.is_empty():
            raise IndexError("peek from empty stack")
        return self.items[-1]
    
    def is_empty(self):
        return len(self.items) == 0
    
    def size(self):
        return len(self.items)"""
            )
        ]

class HumanEvalLoader:
    """Loader for HumanEval dataset"""
    
    @staticmethod
    def load_sample_data() -> List[CodeTestCase]:
        """Load sample HumanEval problems"""
        return [
            CodeTestCase(
                problem_id="humaneval_sum_two_numbers",
                problem_description="""
Write a function that takes two numbers and returns their sum.

def add_numbers(a, b):
    \"\"\"
    Add two numbers and return the result.
    
    Args:
        a: First number
        b: Second number
    
    Returns:
        The sum of a and b
    \"\"\"
    # Your code here
                """.strip(),
                language="python",
                expected_behavior="Return a + b",
                test_cases=[
                    {"input": [2, 3], "expected": 5},
                    {"input": [0, 0], "expected": 0},
                    {"input": [-1, 1], "expected": 0},
                    {"input": [100, 200], "expected": 300},
                    {"input": [-5, -10], "expected": -15}
                ],
                category="arithmetic",
                difficulty="easy",
                starter_code="""def add_numbers(a, b):
    \"\"\"Add two numbers and return the result.\"\"\"
    # Your code here
    pass""",
                canonical_solution="""def add_numbers(a, b):
    \"\"\"Add two numbers and return the result.\"\"\"
    return a + b"""
            ),
            CodeTestCase(
                problem_id="humaneval_list_contains",
                problem_description="""
Write a function that checks if a list contains a specific element.

def contains_element(lst, element):
    \"\"\"
    Check if a list contains a specific element.
    
    Args:
        lst: List to search in
        element: Element to search for
    
    Returns:
        True if element is in list, False otherwise
    \"\"\"
    # Your code here
                """.strip(),
                language="python", 
                expected_behavior="Return True if element in list, False otherwise",
                test_cases=[
                    {"input": [[1, 2, 3, 4, 5], 3], "expected": True},
                    {"input": [[1, 2, 3, 4, 5], 6], "expected": False},
                    {"input": [[], 1], "expected": False},
                    {"input": [["a", "b", "c"], "b"], "expected": True},
                    {"input": [["hello", "world"], "python"], "expected": False}
                ],
                category="list_operations",
                difficulty="easy",
                starter_code="""def contains_element(lst, element):
    \"\"\"Check if a list contains a specific element.\"\"\"
    # Your code here
    pass""",
                canonical_solution="""def contains_element(lst, element):
    \"\"\"Check if a list contains a specific element.\"\"\"
    return element in lst"""
            ),
            CodeTestCase(
                problem_id="humaneval_factorial",
                problem_description="""
Write a function to calculate the factorial of a non-negative integer.

def factorial(n):
    \"\"\"
    Calculate the factorial of n.
    
    Args:
        n: Non-negative integer
    
    Returns:
        n! (factorial of n)
        
    Note: 0! = 1 by definition
    \"\"\"
    # Your code here
                """.strip(),
                language="python",
                expected_behavior="Return n! for non-negative integer n",
                test_cases=[
                    {"input": [0], "expected": 1},
                    {"input": [1], "expected": 1},
                    {"input": [3], "expected": 6},
                    {"input": [5], "expected": 120},
                    {"input": [7], "expected": 5040}
                ],
                category="mathematical",
                difficulty="easy",
                starter_code="""def factorial(n):
    \"\"\"Calculate the factorial of n.\"\"\"
    # Your code here
    pass""",
                canonical_solution="""def factorial(n):
    \"\"\"Calculate the factorial of n.\"\"\"
    if n == 0 or n == 1:
        return 1
    result = 1
    for i in range(2, n + 1):
        result *= i
    return result"""
            )
        ]

class CodeContestLoader:
    """Loader for competitive programming problems"""
    
    @staticmethod
    def load_sample_data() -> List[CodeTestCase]:
        """Load sample competitive programming problems"""
        return [
            CodeTestCase(
                problem_id="contest_two_sum",
                problem_description="""
Given an array of integers nums and an integer target, return indices of the two numbers such that they add up to target.

You may assume that each input would have exactly one solution, and you may not use the same element twice.

Example:
Input: nums = [2,7,11,15], target = 9
Output: [0,1]
Because nums[0] + nums[1] = 2 + 7 = 9
                """.strip(),
                language="python",
                expected_behavior="Return indices of two numbers that sum to target",
                test_cases=[
                    {"input": [[2, 7, 11, 15], 9], "expected": [0, 1]},
                    {"input": [[3, 2, 4], 6], "expected": [1, 2]},
                    {"input": [[3, 3], 6], "expected": [0, 1]},
                    {"input": [[1, 5, 3, 7], 8], "expected": [1, 3]},
                    {"input": [[10, 20, 30], 50], "expected": [1, 2]}
                ],
                category="arrays",
                difficulty="medium",
                canonical_solution="""def two_sum(nums, target):
    seen = {}
    for i, num in enumerate(nums):
        complement = target - num
        if complement in seen:
            return [seen[complement], i]
        seen[num] = i
    return []"""
            ),
            CodeTestCase(
                problem_id="contest_valid_parentheses",
                problem_description="""
Given a string s containing just the characters '(', ')', '{', '}', '[' and ']', determine if the input string is valid.

An input string is valid if:
1. Open brackets must be closed by the same type of brackets.
2. Open brackets must be closed in the correct order.
3. Every close bracket has a corresponding open bracket of the same type.
                """.strip(),
                language="python",
                expected_behavior="Return True if parentheses are valid, False otherwise",
                test_cases=[
                    {"input": ["()"], "expected": True},
                    {"input": ["()[]{}"], "expected": True},
                    {"input": ["(]"], "expected": False},
                    {"input": ["([)]"], "expected": False},
                    {"input": ["{[]}"], "expected": True},
                    {"input": [""], "expected": True}
                ],
                category="stack_problems",
                difficulty="medium",
                canonical_solution="""def is_valid_parentheses(s):
    stack = []
    mapping = {")": "(", "}": "{", "]": "["}
    
    for char in s:
        if char in mapping:
            top_element = stack.pop() if stack else '#'
            if mapping[char] != top_element:
                return False
        else:
            stack.append(char)
    
    return not stack"""
            )
        ]

class DatasetManager:
    """Manager for loading and combining multiple datasets"""
    
    def __init__(self):
        self.loaders = {
            "crosscodeeval": CrossCodeEvalLoader(),
            "repohyper": RepoHyperLoader(), 
            "humaneval": HumanEvalLoader(),
            "codecontest": CodeContestLoader()
        }
    
    def load_dataset(self, dataset_name: str) -> List[CodeTestCase]:
        """Load a specific dataset"""
        if dataset_name not in self.loaders:
            raise ValueError(f"Unknown dataset: {dataset_name}")
        
        return self.loaders[dataset_name].load_sample_data()
    
    def load_multiple_datasets(self, dataset_names: List[str]) -> List[CodeTestCase]:
        """Load and combine multiple datasets"""
        all_test_cases = []
        
        for dataset_name in dataset_names:
            try:
                test_cases = self.load_dataset(dataset_name)
                all_test_cases.extend(test_cases)
                logger.info(f"Loaded {len(test_cases)} problems from {dataset_name}")
            except Exception as e:
                logger.error(f"Failed to load {dataset_name}: {e}")
        
        return all_test_cases
    
    def get_available_datasets(self) -> List[str]:
        """Get list of available datasets"""
        return list(self.loaders.keys())
    
    def get_dataset_info(self) -> Dict[str, Dict[str, Any]]:
        """Get information about available datasets"""
        info = {}
        
        for name, loader in self.loaders.items():
            try:
                test_cases = loader.load_sample_data()
                categories = set(tc.category for tc in test_cases)
                difficulties = set(tc.difficulty for tc in test_cases)
                languages = set(tc.language for tc in test_cases)
                
                info[name] = {
                    "total_problems": len(test_cases),
                    "categories": list(categories),
                    "difficulties": list(difficulties),
                    "languages": list(languages),
                    "description": self._get_dataset_description(name)
                }
            except Exception as e:
                info[name] = {"error": str(e)}
        
        return info
    
    def _get_dataset_description(self, dataset_name: str) -> str:
        """Get description for dataset"""
        descriptions = {
            "crosscodeeval": "Cross-language code generation evaluation with focus on algorithmic problems",
            "repohyper": "Repository-level code understanding and generation tasks",
            "humaneval": "Function-level code generation benchmark with docstring prompts",  
            "codecontest": "Competitive programming problems testing algorithmic thinking"
        }
        return descriptions.get(dataset_name, "No description available")

# Example usage
if __name__ == "__main__":
    manager = DatasetManager()
    
    # Show available datasets
    print("Available datasets:")
    for dataset in manager.get_available_datasets():
        print(f"  - {dataset}")
    
    # Load specific dataset
    crosscode_problems = manager.load_dataset("crosscodeeval")
    print(f"\nLoaded {len(crosscode_problems)} CrossCodeEval problems")
    
    # Load multiple datasets
    all_problems = manager.load_multiple_datasets(["crosscodeeval", "humaneval"])
    print(f"Total problems loaded: {len(all_problems)}")
    
    # Dataset information
    info = manager.get_dataset_info()
    print("\nDataset Information:")
    for name, details in info.items():
        if "error" not in details:
            print(f"{name}: {details['total_problems']} problems, "
                  f"Categories: {details['categories']}")