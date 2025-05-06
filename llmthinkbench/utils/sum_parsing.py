import re
import math

def extract_from_last_line(text):
    """
    Extract answers from the last line or sentence.
    """
    # Split the text into lines
    lines = text.strip().split('\n')
    
    # Regex patterns
    number_pattern = r'([+-]?\d+(?:,\d{3})*)(?:\.\d+)?'
    quoted_text_pattern = r'[\"\']([^\"\'.]+)[\"\']'
    combined_pattern = fr'{quoted_text_pattern}\.?$|{number_pattern}\.?$'

    # Check the last few lines for potential answers
    for i in range(min(3, len(lines))):
        line = lines[-(i+1)].strip()
        if line:
            answer_match = re.search(combined_pattern, line)
            if answer_match:
                return (answer_match.group(1) or answer_match.group(2)).strip()
    
    # Split the text into sentences
    sentences = re.split(r'[.!?]', text)
    
    # Check the last few sentences for potential answers
    for i in range(min(3, len(sentences))):
        sentence = sentences[-(i+1)].strip()
        if sentence:
            answer_match = re.search(combined_pattern, sentence)
            if answer_match:
                return (answer_match.group(1) or answer_match.group(2)).strip()
    
    sentence = sentences[-1].split("=")
    match = re.search(number_pattern, sentence[-1])
    return match.group(0) if match else None

def extract_from_explicit_statements(text):
    """
    Extract answers from explicit statements like "Answer: X" or "The result is X".
    """
    patterns = [
        r'(?:^|\n|\s)(?:answer|the answer is|final answer|the final answer is)[:\s]+(.+?)\s*(?:\n|$|\.)',
        r'(?:^|\n|\s)(?:the result is)[:\s]+(.+?)\s*(?:\n|$|\.)',
        r'(?:^|\n|\s)(?:the value is)[:\s]+(.+?)\s*(?:\n|$|\.)',
        r'(?:^|\n|\s)(?:the number is)[:\s]+(.+?)\s*(?:\n|$|\.)',
        r'(?:^|\n|\s)(?:the output is)[:\s]+(.+?)\s*(?:\n|$|\.)',
        r'(?:^|\n|\s)(?:the solution is)[:\s]+(.+?)\s*(?:\n|$|\.)',
        r'(?:^|\n|\s)(?:the )?(?:total )?(?:sum|result|output|answer|value|number)(?: of [^.]*)?\s+is\s+(.+?)\s*(?:\n|$|\.)',
        r'(?:^|\n|\s)(.+?)\s*is the answer\b',
        r'(?:^|\n|\s)(.+?)\s*is the result\b',
        r'(?:^|\n|\s)(.+?)\s*is the value\b',
        r'(?:^|\n|\s)(.+?)\s*is the number\b',
        r'(?:^|\n|\s)(.+?)\s*is the output\b',
        r'(?:^|\n|\s)(.+?)\s*is the solution\b',
        r'(?:^|\n|\s)([-]?\d{1,3}(?:,\d{3})*(?:\.\d+)?)(?:\n|$|\.)',
    ]
    
    for pattern in patterns:
        matches = re.findall(pattern, text.lower())
        if matches:
            match = matches[-1]
            if isinstance(match, tuple):
                match = match[0]
            
            cleaned_match = match.strip()
            return cleaned_match, False
    
    return None, False

def clean_and_convert_to_number(text):
    """
    Clean the text and convert it to a number if possible.
    """
    if not text:
        return None
    
    # Remove LaTeX formatting
    text = re.sub(r'\\[a-zA-Z]+', '', text)
    
    # Remove markdown formatting
    text = re.sub(r'[*_~`]', '', text)
    
    # Better pattern: supports plain and comma-formatted numbers
    number_pattern = r'([+-]?(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?)'
    number_match = re.search(number_pattern, text)
    
    try: 
        if number_match:
            number_str = number_match.group(1)
            number_str = number_str.replace(',', '')
            return int(float(number_str)) if '.' not in number_str else float(number_str)
    except ValueError:
        pass
    return text
        
        

def extract_from_boxed_formats(text):
    """
    Extract answers from various boxed formats.
    """
    # Standard LaTeX boxed format: \boxed{answer}
    standard_boxed = re.findall(r'\\boxed\{([^{}]*)\}', text)
    
    # LaTeX boxed format with parentheses: \(\boxed{answer}\)
    paren_boxed = re.findall(r'\\[\(\[]\\boxed\{([^{}]*)\}\\[\)\]]', text)
    
    # LaTeX boxed format with brackets: \[\boxed{answer}\]
    bracket_boxed = re.findall(r'\\[\[\]]\\boxed\{([^{}]*)\}\\[\[\]]', text)
    
    # Markdown-style boxed format: [boxed{answer}]
    markdown_boxed = re.findall(r'\[boxed\{([^{}]*)\}\]', text)
    
    # Alternative boxed formats
    alt_boxed1 = re.findall(r'\\box\{([^{}]*)\}', text)
    alt_boxed2 = re.findall(r'\[\\boxed\{([^{}]*)\}\]', text)
    alt_boxed3 = re.findall(r'\(\boxed\{([^{}]*)\}\)', text)
    
    # Combine all matches
    all_matches = standard_boxed + paren_boxed + bracket_boxed + markdown_boxed + alt_boxed1 + alt_boxed2 + alt_boxed3
    
    if all_matches:
        return all_matches[-1].strip(), True
    
    return None, False

def extract_from_latex_math(text):
    """
    Extract answers from LaTeX math expressions.
    """
    patterns = [
        r'\$([^$]+)\$',
        r'\$\$([^$]+)\$\$',
        r'\\[\(\[]([^\\]+)\\[\)\]]',
        r'\\begin\{equation\}([^\\]+)\\end\{equation\}',
        r'\\begin\{align\}([^\\]+)\\end\{align\}',
        r'\\begin\{math\}([^\\]+)\\end\{math\}',
    ]
    
    for pattern in patterns:
        matches = re.findall(pattern, text)
        if matches:
            return matches[-1].strip()
    
    return None

def extract_from_code_blocks(text):
    """
    Extract answers from code blocks.
    """
    patterns = [
        r'```(?:python|plaintext)?\s*\n([^`]+)\n```',
        r'`([^`]+)`',
    ]
    
    for pattern in patterns:
        matches = re.findall(pattern, text)
        if matches:
            for match in matches:
                print_match = re.search(r'print\s*\(\s*["\']?([^"\']+)["\']?\s*\)', match)
                if print_match:
                    return print_match.group(1).strip()
                
                return_match = re.search(r'return\s+([^\s;]+)', match)
                if return_match:
                    return return_match.group(1).strip()
                
                assign_match = re.search(r'(?:answer|result)\s*=\s*([^\s;]+)', match)
                if assign_match:
                    return assign_match.group(1).strip()
    
    return None

def parse_sum_answer(clean_response):
    """
    Extract an answer from the LLM response with robust pattern matching.
    """
    # 1. Try to extract from boxed formats (highest priority)
    boxed_answer, instruction_followed = extract_from_boxed_formats(clean_response)
    if boxed_answer is not None:
        answer = clean_and_convert_to_number(boxed_answer)
        return instruction_followed, answer
    
    # 2. Try to extract from explicit answer statements
    explicit_answer, instruction_followed = extract_from_explicit_statements(clean_response)
    if explicit_answer is not None:
        answer = clean_and_convert_to_number(explicit_answer)
        return instruction_followed, answer
    
    # 3. Try to extract from LaTeX math expressions
    latex_answer = extract_from_latex_math(clean_response)
    if latex_answer is not None:
        answer = clean_and_convert_to_number(latex_answer)
        return False, answer
    
    # 4. Try to extract from code blocks
    code_answer = extract_from_code_blocks(clean_response)
    if code_answer is not None:
        answer = clean_and_convert_to_number(code_answer)
        return False, answer
    
    # 5. Try to extract from the last line or sentence
    last_line_answer = extract_from_last_line(clean_response)
    if last_line_answer is not None:
        answer = clean_and_convert_to_number(last_line_answer)
        return False, answer
    
    # 6. No valid answer found
    return False, None