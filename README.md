# 🧠 LLMThinkBench

<div align="center">

[![PyPI version](https://img.shields.io/badge/pypi-v0.1.6-blue)](https://pypi.org/project/llmthinkbench/0.1.6/)
[![Python](https://img.shields.io/badge/Python-3.12%2B-blue)](https://www.python.org/)
[![License](https://img.shields.io/badge/License-MIT-green)](LICENSE)
[![vLLM](https://img.shields.io/badge/Powered%20by-vLLM-orange)](https://github.com/vllm-project/vllm)
[![HuggingFace](https://img.shields.io/badge/HuggingFace-Compatible-yellow)](https://huggingface.co/)
[![Leaderboard](https://img.shields.io/badge/🏆-Live%20Leaderboard-gold)](https://ctrl-gaurav.github.io/llmthinkbench.github.io/)

![LLMThinkBench Banner](https://img.shields.io/badge/🧠%20LLMThinkBench-Advanced%20Reasoning%20%26%20Overthinking%20Detection-blueviolet?style=for-the-badge&logo=brain&logoColor=white)

</div>

## A Framework for Evaluating Basic Math Reasoning Capabilities and Overthinking of Language Models

LLMThinkBench is a comprehensive framework designed to rigorously evaluate the basic math reasoning capabilities of Language Models, while also identifying instances of overthinking—where models apply unnecessarily complex logic to simple problems. Through standardized and reproducible benchmarks, it offers valuable insights into how well models perform on various reasoning tasks, from basic arithmetic to complex logical operations.

## 🏆 **Live Leaderboard**

<div align="center">

### **[🔥 View Real-Time Model Rankings →](https://ctrl-gaurav.github.io/llmthinkbench.github.io/)**

[![LLMThinkBench Leaderboard](https://img.shields.io/badge/📊%20Live%20Leaderboard-Real--time%20Rankings%20•%20Overthinking%20Metrics%20•%20Performance%20Analysis-000428?style=for-the-badge)](https://ctrl-gaurav.github.io/llmthinkbench.github.io/)

**🥇 Top Performers:**
| Rank | Model | Accuracy | Efficiency | Instruction Following |
|------|-------|----------|------------|-----------------------|
| 🥇 #1 | GPT-4.1-mini | 90.23% | 0.768 | 98.14% |
| 🥈 #2 | GPT-4.1 | 89.88% | 0.752 | 97.79% |
| 🥉 #3 | GPT-4o | 87.56% | 0.737 | 99.42% |

*See how top models like **GPT-4.1**, **Phi-4**, **Qwen**, and **Llama** compare on reasoning efficiency*

**🔍 Interactive Features:**
- ⚡ Real-time filtering and search
- 📊 Model comparison tools  
- 📈 Performance visualizations
- 🧮 Overthinking analysis
- 💾 Export functionality

</div>

## 📰 News & Releases

- **v0.1.6** (Latest) - Enhanced reporting with efficiency rankings to identify models that achieve high accuracy with minimal tokens. Introduced **Overthinking Score** metric that properly balances accuracy and verbosity. 
- **v0.1.5** - Major backend revamp with robust error handling, intelligent fallback support, better token estimation, flexible device control, and smoother interruption handling.
- **v0.1.4** - Major improvements to parsing robustness for fair evaluations. Enhanced result validation mechanisms and edge case handling.
- **v0.1.3** - Added mean, median, mode tasks and implemented GPU customization options, allowing users to specify GPU memory allocation.
- **v0.1.2** - Expanded task library with find_maximum, find_minimum, absolute_difference, and division tasks. Improved documentation.
- **v0.1.1** - Fixed several inference issues and optimized performance using vLLM for high-throughput evaluation.
- **v0.1.0** - Initial release with core functionality including sorting and comparison tasks.

## 🌟 Key Features

- **Comprehensive Evaluation**: Test LLMs on a range of mathematical and logical reasoning tasks
- **🎯 Advanced Overthinking Detection**: Novel efficiency metrics that balance accuracy and conciseness
- **📊 Efficiency Rankings**: Identify models that achieve high performance without excessive verbosity
- **Modular Architecture**: Easily extend with custom evaluation tasks
- **Fair Comparison**: Standardized methodology for comparing models
- **Efficient Inference**: Built on vLLM for high-throughput batched evaluation
- **Detailed Metrics**: Comprehensive reports on accuracy, instruction following, and output characteristics
- **Multi-GPU Support**: Scale evaluations across multiple GPUs
- **Reproducible Results**: Consistent methodology across model comparisons
- **Output Analysis**: Identify when and how models make reasoning errors

## 🧮 Revolutionary Overthinking Score

### **The Problem**
Traditional benchmarks miss a critical insight: **efficiency matters**. A model that achieves 95% accuracy with 50 tokens is often superior to one that achieves 98% accuracy with 500 tokens, especially in production environments where compute costs and response times matter.

### **Our Solution: Overthinking Score**
We introduce a novel metric that balances accuracy and efficiency using the F1-harmonic mean approach:

**Formula**: `Overthinking Score = 2 × (Accuracy × Token_Efficiency) / (Accuracy + Token_Efficiency)`

Where:
- `Token_Efficiency = 1 - normalized_tokens`
- `normalized_tokens = (tokens - min_tokens) / (max_tokens - min_tokens)`

### **Why F1-Harmonic Mean?**
- **Prevents gaming**: Models can't achieve high scores with just accuracy OR efficiency alone
- **Balanced optimization**: Encourages improvement in both dimensions simultaneously  
- **Penalizes extremes**: Heavily verbose or inaccurate models get low scores
- **Intuitive interpretation**: Higher score = better overall performance

### **Example Comparison**
| Model | Accuracy | Tokens | Overthinking Score | Verdict |
|-------|----------|--------|-------------------|---------|
| Model A | 98% | 500 | 0.776 | High accuracy, too verbose |
| Model B | 94% | 100 | 0.891 | **Better balanced** |
| Model C | 99% | 150 | 0.925 | **🏆 Optimal** |

**Result**: Model C achieves the best balance of accuracy and efficiency!

## 📊 Supported Tasks

| Task Type | Task | Description |
|-----------|------|-------------|
| **Basic Operations** | Sorting | Evaluates ability to correctly sort lists of numbers |
| | Comparison | Tests number comparison abilities (greater than, less than, equal to) |
| | Sum | Assesses ability to calculate the sum of multiple numbers |
| | Subtraction | Measures accuracy in subtracting two numbers |
| | Multiplication | Tests multiplication of numbers |
| | Division | Evaluates division operations |
| **List Processing** | Find Maximum | Finds the largest value in a list |
| | Find Minimum | Identifies the smallest value in a list |
| | Odd Count | Counts odd numbers in a list |
| | Even Count | Counts even numbers in a list |
| **Statistical** | Mean | Calculates the arithmetic mean of a list |
| | Median | Finds the median value of a list |
| | Mode | Identifies the most frequent value(s) in a list |
| **Advanced** | Absolute Difference | Calculates the absolute difference between numbers |

## 🚀 Installation

```bash
# Install from PyPI
pip install llmthinkbench

# Install from source
git clone https://github.com/ctrl-gaurav/llmthinkbench.git
cd llmthinkbench
pip install -e .
```

## 📈 Quick Start

### Command Line Interface

```bash
# Basic usage
llmthinkbench --model_id "Qwen/Qwen3-4B" --tasks sorting comparison

# Comprehensive evaluation
llmthinkbench --model_id "Qwen/Qwen3-4B" \
  --tasks "sorting comparison sum multiplication odd_count even_count absolute_difference division find_maximum find_minimum mean median mode subtraction" \
  --datapoints 100 \
  --folds 3 \
  --range -1000 1000 \
  --list_sizes "8 16 32 64" \
  --cuda_device "cuda:0" \
  --tensor_parallel_size 1 \
  --gpu_memory_utilization 0.98 \
  --temperature 0.1 \
  --top_p 0.9 \
  --max_tokens 1024 \
  --trust_remote_code \
  --store_details \
  --output_dir "qwen3_4b_eval" \
  --seed 42
```

### Python API

```python
from llmthinkbench import evaluate

# Simple evaluation
results = evaluate(
    model_id="Qwen/Qwen3-4B",
    tasks=["sorting", "comparison", "sum"]
)

# Advanced configuration
results = evaluate(
    model_id="Qwen/Qwen3-4B",
    tasks=["sorting", "comparison", "sum", "multiplication"],
    datapoints=500,
    list_sizes=[8, 16, 32],
    folds=3,
    range=[-1000, 1000],
    store_details=True,
    output_dir="./custom_results",
    tensor_parallel_size=1,
    gpu_memory_utilization=0.98,
    temperature=0.1,
    top_p=0.9,
    max_tokens=1024
)
```

### Detailed API Usage

```python
from llmthinkbench.models.model_handler import ModelHandler
from llmthinkbench.tasks.sorting_task import SortingTask
from llmthinkbench.tasks.comparison_task import ComparisonTask
from llmthinkbench.utils.reporting import generate_final_report

# Initialize model
model_handler = ModelHandler(
    model_id="Qwen/Qwen3-4B",
    tensor_parallel_size=1,
    gpu_memory_utilization=0.98
)

# Configure output directory
output_dir = "qwen3_4b_eval_results"

# Run sorting task
sorting = SortingTask(
    model_handler=model_handler,
    output_dir=output_dir,
    min_val=-1000,
    max_val=1000,
    num_folds=3,
    num_samples=100,
    store_details=True,
    temperature=0.1,
    top_p=0.9,
    max_tokens=1024
)

# Evaluate multiple list sizes
list_sizes = [8, 16, 32, 64]
sorting_metrics = sorting.run_evaluation(list_sizes)

# Run comparison task
comparison = ComparisonTask(
    model_handler=model_handler,
    output_dir=output_dir,
    min_val=-1000,
    max_val=1000,
    num_folds=3,
    num_samples=100,
    store_details=True,
    temperature=0.1,
    top_p=0.9,
    max_tokens=1024
)

comparison_metrics = comparison.run_evaluation()

# Generate comprehensive report
all_metrics = sorting_metrics + comparison_metrics
report = generate_final_report(all_metrics, list_sizes, output_dir)
```

## 📊 Example Results

Below is an example report generated by LLMThinkBench v0.1.6:

```
+------------------+----------------------------+------------------+-----------------------------------+-------------------------+------------------------------+------------------------------+
| Task             | Accuracy                   | Overthinking Score | Instruction Followed              | Tokens                  | Chars                        | Words                        |
+------------------+----------------------------+------------------+-----------------------------------+-------------------------+------------------------------+------------------------------+
| sorting_8        | 95.20% ± 3.60              | 0.892              | 98.80% ± 1.20                     | 186.2 ± 32.6            | 612.6 ± 98.4                 | 93.5 ± 15.6                  |
| sorting_16       | 87.40% ± 4.80              | 0.743              | 96.70% ± 2.30                     | 312.5 ± 48.7            | 982.3 ± 156.5                | 167.9 ± 26.9                 |
| sorting_32       | 68.60% ± 7.20              | 0.521              | 92.40% ± 3.50                     | 645.7 ± 92.2            | 1872.2 ± 283.6               | 348.8 ± 52.8                 |
| comparison       | 99.20% ± 1.20              | 0.951              | 99.60% ± 0.50                     | 93.8 ± 16.2             | 324.8 ± 52.2                 | 48.3 ± 8.1                   |
| sum_8            | 97.80% ± 2.10              | 0.923              | 99.30% ± 0.70                     | 134.6 ± 23.9            | 452.2 ± 78.3                 | 68.9 ± 11.7                  |
| multiplication   | 94.60% ± 3.50              | 0.885              | 98.40% ± 1.60                     | 114.3 ± 19.4            | 386.7 ± 64.3                 | 58.4 ± 9.7                   |
+------------------+----------------------------+------------------+-----------------------------------+-------------------------+------------------------------+------------------------------+

🏆 Efficiency Rankings (Best Balance of Accuracy & Conciseness)
+--------+---------------+------------------+----------+--------+
|  Rank  | Task          | Overthinking Score | Accuracy | Tokens |
+--------+---------------+------------------+----------+--------+
|   1    | comparison    | 0.951            | 99.2%    | 93.8   |
|   2    | sum_8         | 0.923            | 97.8%    | 134.6  |
|   3    | sorting_8     | 0.892            | 95.2%    | 186.2  |
|   4    | multiplication| 0.885            | 94.6%    | 114.3  |
|   5    | sorting_16    | 0.743            | 87.4%    | 312.5  |
|   6    | sorting_32    | 0.521            | 68.6%    | 645.7  |
+--------+---------------+------------------+----------+--------+
```

## ⚙️ Advanced Configuration

### Command Line Parameters

| Parameter | Description | Default |
|-----------|-------------|---------|
| `--model_id` | Hugging Face model ID | *Required* |
| `--tasks` | Tasks to evaluate | `["sorting"]` |
| `--datapoints` | Number of samples per test case | `1000` |
| `--folds` | Number of evaluation folds | `1` |
| `--range` | Number range for evaluation | `[-100, 100]` |
| `--list_sizes` | List sizes for list-based tasks | `[8]` |
| `--store_details` | Store detailed per-example results | `False` |
| `--output_dir` | Directory to save results | Auto-generated |
| `--tensor_parallel_size` | Number of GPUs to use | `1` |
| `--gpu_memory_utilization` | GPU memory utilization threshold | `0.9` |
| `--temperature` | Sampling temperature | `0.7` |
| `--top_p` | Sampling top_p value | `0.9` |
| `--max_tokens` | Maximum tokens for sampling | `512` |

## 📈 Visualization

You can visualize LLMThinkBench results including the new overthinking metrics:

```python
import json
import matplotlib.pyplot as plt
import pandas as pd

# Load results
with open("qwen3_4b_eval/final_report.json") as f:
    results = json.load(f)

# Extract efficiency rankings
rankings = results['efficiency_rankings']

# Create dataframe for plotting
df = pd.DataFrame(rankings)

# Plot efficiency comparison
fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))

# Overthinking Score vs Accuracy
scatter = ax1.scatter(df['accuracy']*100, df['overthinking_score'], 
                     s=100, alpha=0.7, c=df['tokens'], cmap='viridis')
ax1.set_xlabel('Accuracy (%)')
ax1.set_ylabel('Overthinking Score')
ax1.set_title('Overthinking Score vs Accuracy')
plt.colorbar(scatter, ax=ax1, label='Tokens')

# Add task labels
for i, row in df.iterrows():
    ax1.annotate(row['task'], (row['accuracy']*100, row['overthinking_score']), 
                xytext=(5, 5), textcoords='offset points', fontsize=8)

# Overthinking Score ranking
ax2.barh(range(len(df)), df['overthinking_score'])
ax2.set_yticks(range(len(df)))
ax2.set_yticklabels(df['task'])
ax2.set_xlabel('Overthinking Score')
ax2.set_title('Overthinking Score Rankings')
ax2.invert_yaxis()

# Tokens vs Accuracy with Overthinking Score as color
scatter3 = ax4.scatter(df['accuracy']*100, df['tokens'], 
                      s=100, alpha=0.7, c=df['overthinking_score'], cmap='coolwarm')
ax4.set_xlabel('Accuracy (%)')
ax4.set_ylabel('Tokens')
ax4.set_title('Tokens vs Accuracy (colored by Overthinking Score)')
plt.colorbar(scatter3, ax=ax4, label='Overthinking Score')

plt.tight_layout()
plt.savefig("overthinking_analysis.png", dpi=300, bbox_inches='tight')
plt.show()
```

## 🧩 Extending with Custom Tasks

LLMThinkBench is designed to be easily extensible. Here's how to create a custom evaluation task:

1. Create a new task module:

```python
# llmthinkbench/tasks/custom_task.py
import random
from ..tasks.base_task import BaseTask

class CustomTask(BaseTask):
    """Implementation of a custom task"""
    
    @property
    def task_name(self):
        return "custom_task"
    
    def generate_data(self):
        """Generate random data for the task"""
        data = []
        for _ in range(self.num_samples):
            a = random.randint(self.min_val, self.max_val)
            b = random.randint(self.min_val, self.max_val)
            result = a * b + a  # Example operation
            data.append({"a": a, "b": b, "expected": result})
        return data
    
    def create_prompt(self, data_point):
        """Create prompt for the task"""
        return (f"Calculate a * b + a where a = {data_point['a']} and b = {data_point['b']}.\n\n"
                f"Provide your answer in the format \\boxed{{result}} at the end.")
    
    def evaluate_response(self, response, data_point):
        """Evaluate model response for the task"""
        from ..utils.custom_parsing import parse_boxed_answer
        
        boxed_answer = parse_boxed_answer(response)
        instruction_followed = boxed_answer is not None
        accuracy = 0
        
        if instruction_followed and boxed_answer:
            try:
                parsed_answer = int(boxed_answer[0])
                accuracy = 1 if parsed_answer == data_point['expected'] else 0
            except:
                pass
        
        return {
            "a": data_point['a'],
            "b": data_point['b'],
            "expected": data_point['expected'],
            "parsed_answer": boxed_answer[0] if boxed_answer and len(boxed_answer) > 0 else None,
            "accuracy": accuracy,
            "instruction_followed": instruction_followed
        }
    
    def run_evaluation(self):
        """Run evaluation for custom task"""
        all_metrics = []
        
        # Generate evaluation data
        data = self.generate_data()
        
        # Run each fold
        for fold in range(1, self.num_folds + 1):
            metrics = self.run_fold(data, "custom_task", fold)
            all_metrics.append(metrics)
        
        return all_metrics
```

2. Create a parsing utility for your task:

```python
# llmthinkbench/utils/custom_parsing.py
import re

def parse_boxed_answer(text):
    """Parse boxed answers from text.
    
    Args:
        text (str): Model response text
        
    Returns:
        list: List of answers found in \boxed{} notation
    """
    matches = re.findall(r'\\boxed\{([^}]*)\}', text)
    
    if not matches:
        # Alternative formats
        matches = re.findall(r'\[([^]]*)\]', text)
        
    return matches if matches else None
```

3. Add your task to the available tasks in `__init__.py`

4. Use your custom task:

```bash
llmthinkbench --model_id "Qwen/Qwen3-4B" --tasks custom_task
```

## 🔍 Contributing

Contributions to LLMThinkBench are welcome! To contribute:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add some amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

For major changes, please open an issue first to discuss what you would like to change.

## 📜 License

LLMThinkBench is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 📚 Citation

If you use LLMThinkBench in your research, please cite:

```bibtex
@software{llmthinkbench2025,
  author = {Gaurav Srivastava, Aafiya Hussain, Sriram Srinivasan, Xuan Wang},
  title = {LLMThinkBench: Advanced Reasoning and Overthinking Evaluation Framework for LLMs},
  year = {2025},
  url = {https://github.com/ctrl-gaurav/LLMThinkBench/},
  version = {0.1.6}
}
```

## 📧 Contact

- **Issues**: For questions, issues, or feedback, please [open an issue](https://github.com/ctrl-gaurav/LLMThinkBench/issues) on GitHub.
- **PyPI**: [pypi.org/project/llmthinkbench](https://pypi.org/project/llmthinkbench/)
