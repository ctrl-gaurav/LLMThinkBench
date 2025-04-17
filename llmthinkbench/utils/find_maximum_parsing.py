import re
import math

def parse_find_maximum_answer(clean_response):
    """
    Extract an answer from the LLM response with robust pattern matching.
    
    This function attempts to extract the answer from various formats that LLMs might use,
    prioritizing the requested format (boxed) and falling back to other common patterns.
    
    Args:
        clean_response (str): The cleaned response from the LLM
        
    Returns:
        tuple: (bool, value) where bool indicates if instruction was followed
               and value is the extracted answer or None
    """
    # 1. Try to extract from boxed formats (highest priority)
    boxed_answer, instruction_followed = extract_from_boxed_formats(clean_response)
    if boxed_answer is not None:
        answer = clean_and_convert_to_number(boxed_answer)
        if is_valid_number(answer):
            return instruction_followed, answer
    
    # 2. Try to extract from markdown formatting (bold, italic)
    markdown_answer = extract_from_markdown_formatting(clean_response)
    if markdown_answer is not None:
        answer = clean_and_convert_to_number(markdown_answer)
        if is_valid_number(answer):
            return False, answer
    
    # 3. Try to extract from explicit answer statements
    explicit_answer, instruction_followed = extract_from_explicit_statements(clean_response)
    if explicit_answer is not None:
        answer = clean_and_convert_to_number(explicit_answer)
        if is_valid_number(answer):
            return instruction_followed, answer
    
    # 4. Try to extract from LaTeX math expressions
    latex_answer = extract_from_latex_math(clean_response)
    if latex_answer is not None:
        answer = clean_and_convert_to_number(latex_answer)
        if is_valid_number(answer):
            return False, answer
    
    # 5. Try to extract from code blocks
    code_answer = extract_from_code_blocks(clean_response)
    if code_answer is not None:
        answer = clean_and_convert_to_number(code_answer)
        if is_valid_number(answer):
            return False, answer
    
    # 6. Try to extract from the last line or sentence
    last_line_answer = extract_from_last_line(clean_response)
    if last_line_answer is not None:
        answer = clean_and_convert_to_number(last_line_answer)
        if is_valid_number(answer):
            return False, answer
    
    # 7. Try to extract plain numbers from very short responses
    plain_number = extract_plain_number(clean_response)
    if plain_number is not None and is_valid_number(plain_number):
        return False, plain_number
    
    # 8. No valid answer found
    return False, None


def is_valid_number(value):
    """
    Check if a value is a valid number.
    
    Args:
        value: The value to check
        
    Returns:
        bool: True if the value is a valid number, False otherwise
    """
    if value is None:
        return False
    
    if isinstance(value, (int, float)):
        return True
    
    if isinstance(value, str):
        try:
            # Try to convert to a number
            float(value)
            return True
        except ValueError:
            pass
    
    return False


def clean_and_convert_to_number(text):
    """
    Clean the text and convert it to a number if possible.
    
    Args:
        text (str or int or float): The text or number to clean and convert
        
    Returns:
        int, float, or str: The converted number or the original text if conversion fails
    """
    if not text:
        return None
    
    # If text is already a number, return it
    if isinstance(text, (int, float)):
        return text
    
    # Convert to string if it's not already
    if not isinstance(text, str):
        try:
            text = str(text)
        except:
            return None
    
    # Remove LaTeX formatting
    text = re.sub(r'\\[a-zA-Z]+', '', text)
    
    # Remove markdown formatting (bold, italic, code)
    text = re.sub(r'[*_~`]', '', text)
    
    # Handle cases where there's a space between the minus sign and the number
    text = re.sub(r'-\s+(\d+)', r'-\1', text)
    
    # Try to convert to a number
    try:
        # Extract the first number from the text
        number_match = re.search(r'([+-]?\d+(?:\.\d+)?)', text)
        if number_match:
            number_str = number_match.group(1)
            # Convert to int or float as appropriate
            float_val = float(number_str)
            if math.isinf(float_val) or '.' in number_str:
                return float_val
            else:
                return int(float_val)
    except ValueError:
        pass
    
    # Return the cleaned text if conversion fails
    return text


def extract_from_boxed_formats(text):
    """
    Extract answers from various boxed formats.
    
    This function looks for different LaTeX-style boxed formats that LLMs might use,
    including standard \boxed{}, parenthesized \(\boxed{}\), and bracketed \[\boxed{}\].
    
    Args:
        text (str): The text to search for boxed formats
        
    Returns:
        tuple: (extracted_answer, instruction_followed)
               extracted_answer is the content inside the box or None
               instruction_followed is True if the exact requested format was used
    """
    # Look for standard LaTeX boxed format: \boxed{answer}
    standard_boxed = re.findall(r'\\boxed\{([^{}]*)\}', text)
    
    # Look for standard LaTeX boxed format with space: \ boxed{answer}
    standard_boxed_with_space = re.findall(r'\\\s+boxed\{([^{}]*)\}', text)
    
    # Look for LaTeX boxed format with parentheses: \(\boxed{answer}\)
    paren_boxed = re.findall(r'\\[\(\[]\\boxed\{([^{}]*)\}\\[\)\]]', text)
    
    # Look for LaTeX boxed format with parentheses and space: \(\ boxed{answer}\)
    paren_boxed_with_space = re.findall(r'\\[\(\[]\\\s+boxed\{([^{}]*)\}\\[\)\]]', text)
    
    # Look for LaTeX boxed format with brackets: \[\boxed{answer}\]
    bracket_boxed = re.findall(r'\\[\[\]]\\boxed\{([^{}]*)\}\\[\[\]]', text)
    
    # Look for LaTeX boxed format with brackets and space: \[\ boxed{answer}\]
    bracket_boxed_with_space = re.findall(r'\\[\[\]]\\\s+boxed\{([^{}]*)\}\\[\[\]]', text)
    
    # Look for markdown-style boxed format: [boxed{answer}]
    markdown_boxed = re.findall(r'\[boxed\{([^{}]*)\}\]', text)
    
    # Look for markdown-style boxed format with space: [\ boxed{answer}]
    markdown_boxed_with_space = re.findall(r'\[\\\s+boxed\{([^{}]*)\}\]', text)
    
    # Look for alternative boxed formats that LLMs might use
    alt_boxed1 = re.findall(r'\\box\{([^{}]*)\}', text)
    alt_boxed2 = re.findall(r'\[\\boxed\{([^{}]*)\}\]', text)
    alt_boxed3 = re.findall(r'\(\boxed\{([^{}]*)\}\)', text)
    
    # Look for malformed boxed formats with extra backslashes
    malformed_boxed1 = re.findall(r'\\\s*boxed\{([^{}]*)\}', text)
    malformed_boxed2 = re.findall(r'\\\\boxed\{([^{}]*)\}', text)
    
    # Look for boxed formats with different brackets/braces
    alt_boxed4 = re.findall(r'\\boxed\[([^\[\]]*)\]', text)
    alt_boxed5 = re.findall(r'\\boxed\(([^()]*)\)', text)
    
    # Look for boxed formats with escaped characters
    escaped_boxed = re.findall(r'\\!?\\[\(\[]\\?(?:max\\)?=?\\boxed\{\{([^{}]*)\}\}\\[\)\]]', text)
    
    # Combine all matches
    all_matches = (standard_boxed + standard_boxed_with_space + 
                  paren_boxed + paren_boxed_with_space + 
                  bracket_boxed + bracket_boxed_with_space + 
                  markdown_boxed + markdown_boxed_with_space + 
                  alt_boxed1 + alt_boxed2 + alt_boxed3 + alt_boxed4 + alt_boxed5 + 
                  malformed_boxed1 + malformed_boxed2 + escaped_boxed)
    
    if all_matches:
        # Return the last match as it's likely the final answer
        return all_matches[-1].strip(), True
    
    return None, False


def extract_from_explicit_statements(text):
    """
    Extract answers from explicit statements like "Answer: X" or "The result is X".
    
    This function looks for common patterns that LLMs use to indicate their final answer.
    
    Args:
        text (str): The text to search for explicit answer statements
        
    Returns:
        tuple: (extracted_answer, instruction_followed)
               extracted_answer is the content after the statement or None
               instruction_followed is False since the requested format wasn't used
    """
    # Common patterns for explicit answer statements (generic, not specific to any task)
    patterns = [
        # Answer: X or The answer is X
        r'(?:^|\n|\s)(?:answer|the answer is|final answer|the final answer is)[:\s]+([^\n\.]+)',
        # Therefore, X is the answer
        r'(?:^|\n|\s)(?:therefore,?)[:\s]+([^\n\.]+)(?:\s+is the answer)',
        # Therefore, the answer is X
        r'(?:^|\n|\s)(?:therefore,? the answer is)[:\s]+([^\n\.]+)',
        # Thus, the answer is X
        r'(?:^|\n|\s)(?:thus,? the answer is)[:\s]+([^\n\.]+)',
        # So, the answer is X
        r'(?:^|\n|\s)(?:so,? the answer is)[:\s]+([^\n\.]+)',
        # Hence, the answer is X
        r'(?:^|\n|\s)(?:hence,? the answer is)[:\s]+([^\n\.]+)',
        # The result is X
        r'(?:^|\n|\s)(?:the result is)[:\s]+([^\n\.]+)',
        # The value is X
        r'(?:^|\n|\s)(?:the value is)[:\s]+([^\n\.]+)',
        # The maximum value is X
        r'(?:^|\n|\s)(?:the maximum value is)[:\s]+([^\n\.]+)',
        # The number is X
        r'(?:^|\n|\s)(?:the number is)[:\s]+([^\n\.]+)',
        # The output is X
        r'(?:^|\n|\s)(?:the output is)[:\s]+([^\n\.]+)',
        # The solution is X
        r'(?:^|\n|\s)(?:the solution is)[:\s]+([^\n\.]+)',
        # X is the answer
        r'(?:^|\n|\s)([^\n\.\s]+)(?:\s+is the answer)',
        # X is the result
        r'(?:^|\n|\s)([^\n\.\s]+)(?:\s+is the result)',
        # X is the value
        r'(?:^|\n|\s)([^\n\.\s]+)(?:\s+is the value)',
        # X is the number
        r'(?:^|\n|\s)([^\n\.\s]+)(?:\s+is the number)',
        # X is the output
        r'(?:^|\n|\s)([^\n\.\s]+)(?:\s+is the output)',
        # X is the solution
        r'(?:^|\n|\s)([^\n\.\s]+)(?:\s+is the solution)',
        # X is the maximum value
        r'(?:^|\n|\s)([^\n\.\s]+)(?:\s+is the maximum value)',
        # X is the maximum number
        r'(?:^|\n|\s)([^\n\.\s]+)(?:\s+is the maximum(?:\s+number)?)',
        # X is the largest number
        r'(?:^|\n|\s)([^\n\.\s]+)(?:\s+is the largest(?:\s+number)?)',
        # The maximum number in the list is X
        r'(?:^|\n|\s)(?:the maximum number in the (?:given )?list is)[:\s]+([^\n\.]+)',
        # The maximum number from the list is X
        r'(?:^|\n|\s)(?:the maximum number from the (?:given )?list is)[:\s]+([^\n\.]+)',
        # The maximum value in the list is X
        r'(?:^|\n|\s)(?:the maximum value in the (?:given )?list is)[:\s]+([^\n\.]+)',
        # The largest number in the list is X
        r'(?:^|\n|\s)(?:the largest number in the (?:given )?list is)[:\s]+([^\n\.]+)',
        # The maximum among the given numbers is X
        r'(?:^|\n|\s)(?:the maximum among the given numbers is)[:\s]+([^\n\.]+)',
        # So the maximum number in the given list is X
        r'(?:^|\n|\s)(?:so the maximum number in the (?:given )?list is)[:\s]+([^\n\.]+)',
        # Maximum number in the list is X
        r'(?:^|\n|\s)(?:maximum number in the (?:given )?list is)[:\s]+([^\n\.]+)',
        # The list contains X as its maximum value
        r'(?:^|\n|\s)(?:the list contains)\s+([^\n\.\s]+)\s+(?:as its maximum value)',
        # Found that X is the maximum value
        r'(?:^|\n|\s)(?:found that)\s+([^\n\.\s]+)\s+(?:is the maximum value)',
        # Patterns for backticked numbers
        r'(?:^|\n|\s)(?:the maximum number in the (?:given )?list is)[:\s]+`([^`]+)`',
        r'(?:^|\n|\s)(?:the maximum number from the (?:given )?list is)[:\s]+`([^`]+)`',
        r'(?:^|\n|\s)(?:the maximum value in the (?:given )?list is)[:\s]+`([^`]+)`',
        r'(?:^|\n|\s)(?:the largest number in the (?:given )?list is)[:\s]+`([^`]+)`',
        r'(?:^|\n|\s)(?:so the maximum number in the (?:given )?list is)[:\s]+`([^`]+)`',
        r'(?:^|\n|\s)(?:maximum number in the (?:given )?list is)[:\s]+`([^`]+)`',
    ]
    
    for pattern in patterns:
        matches = re.findall(pattern, text.lower())
        if matches:
            # Extract the actual match content (might be a tuple from regex groups)
            match = matches[-1]
            if isinstance(match, tuple):
                match = match[0]
            
            # Clean up the match
            cleaned_match = match.strip()
            
            # Return the last match as it's likely the final answer
            return cleaned_match, False
    
    return None, False


def extract_from_latex_math(text):
    """
    Extract answers from LaTeX math expressions.
    
    This function looks for LaTeX math expressions that might contain the answer.
    
    Args:
        text (str): The text to search for LaTeX math expressions
        
    Returns:
        The extracted answer or None
    """
    # Look for LaTeX math expressions: $X$, $$X$$, \(X\), \[X\]
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
            # Return the last match as it's likely the final answer
            return matches[-1].strip()
    
    return None


def extract_from_code_blocks(text):
    """
    Extract answers from code blocks.
    
    This function looks for code blocks that might contain the answer.
    
    Args:
        text (str): The text to search for code blocks
        
    Returns:
        The extracted answer or None
    """
    # Look for code blocks: ```X```, `X`
    patterns = [
        r'```(?:python|plaintext)?\s*\n([^`]+)\n```',
        r'`([^`]+)`',
    ]
    
    # Look for output blocks that might contain the answer
    output_patterns = [
        r'```\s*\n([^`]+)\n```',  # Output in code block
        r'Output:\s*\n([^\n]+)',  # Output with label
        r'output:\s*\n([^\n]+)',  # Output with lowercase label
        r'The output is:\s*\n([^\n]+)',  # Output with full sentence
        r'Running this code will output:\s*\n([^\n]+)',  # Output with running code
        r'When you run this code it will output:\s*\n([^\n]+)',  # Output with running code
        r'The result is:\s*\n([^\n]+)',  # Result with label
        r'Result:\s*\n([^\n]+)',  # Result with label
    ]
    
    # First, try to extract from output blocks
    for pattern in output_patterns:
        matches = re.findall(pattern, text)
        if matches:
            for match in matches:
                # Look for boxed format in the output (with or without space after backslash)
                boxed_match = re.search(r'\\(?:\s*)boxed\{([^{}]+)\}', match)
                if boxed_match:
                    return boxed_match.group(1).strip()
                
                # Look for plain numbers in the output
                number_match = re.search(r'[+-]?\d+', match)
                if number_match:
                    return number_match.group(0).strip()
    
    # Then, try to extract from code blocks
    for pattern in patterns:
        matches = re.findall(pattern, text)
        if matches:
            # Look for answers in the code block
            for match in matches:
                # Look for print statements with boxed format (allowing space after backslash and triple braces)
                boxed_print_match = re.search(r'print\s*\(\s*f[\"\']\\(?:\\)?(?:\s*)boxed\{(?:\{+)([^{}]+)(?:\}+)\}[\"\']', match)
                if boxed_print_match:
                    # Check if the extracted value is a number
                    try:
                        value = boxed_print_match.group(1).strip()
                        if re.match(r'^[+-]?\d+$', value):
                            return value
                    except:
                        pass
                
                # Look for print statements with the word "maximum" or "max"
                max_print_match = re.search(r'print\s*\(\s*["\']?(?:.*?(?:maximum|max)[^"\']*?)?([+-]?\d+)["\']?\s*\)', match)
                if max_print_match:
                    return max_print_match.group(1).strip()
                
                # Look for print statements with comma-separated arguments
                comma_print_match = re.search(r'print\s*\(\s*["\'].*?["\'],\s*([+-]?\d+)\s*\)', match)
                if comma_print_match:
                    return comma_print_match.group(1).strip()
                
                # Look for regular print statements
                print_match = re.search(r'print\s*\(\s*["\']?([^"\']+)["\']?\s*\)', match)
                if print_match:
                    # Check if the extracted value is a number
                    try:
                        value = print_match.group(1).strip()
                        if re.match(r'^[+-]?\d+$', value):
                            return value
                    except:
                        pass
                
                # Look for f-string print statements
                f_print_match = re.search(r'print\s*\(\s*f[\"\'].*?([+-]?\d+).*?[\"\']\s*\)', match)
                if f_print_match:
                    return f_print_match.group(1).strip()
                
                # Look for return statements
                return_match = re.search(r'return\s+([+-]?\d+)', match)
                if return_match:
                    return return_match.group(1).strip()
                
                # Look for any variable assignments with a number
                var_assign_match = re.search(r'\b\w+\s*=\s*([+-]?\d+)(?!\s*\[)', match)
                if var_assign_match:
                    return var_assign_match.group(1).strip()
                
                # Look for max_number variable assignments with final values
                # (not extracting from the input list)
                max_assign_match = re.search(r'max(?:_number|num|imum)?\s*=\s*([+-]?\d+)(?!\s*\[)', match)
                if max_assign_match:
                    return max_assign_match.group(1).strip()
                
                # Look for variable assignments with "answer" or "result"
                assign_match = re.search(r'(?:answer|result)\s*=\s*([+-]?\d+)', match)
                if assign_match:
                    return assign_match.group(1).strip()
    
    # Look for the maximum number in the text
    max_number_match = re.search(r'(?:maximum|max|largest)(?:\s+number)?\s+(?:is|=)\s+([+-]?\d+)', text.lower())
    if max_number_match:
        return max_number_match.group(1).strip()
    
    # Look for statements about the maximum number
    max_statement_match = re.search(r'(?:the )?(?:maximum|max|largest)(?:\s+number)?\s+(?:in|from|of)\s+(?:the )?(?:list|array|numbers)\s+is\s+([+-]?\d+)', text.lower())
    if max_statement_match:
        return max_statement_match.group(1).strip()
    
    return None


def extract_from_last_line(text):
    """
    Extract answers from the last line or sentence.
    
    This function looks for potential answers in the last line or sentence of the text.
    
    Args:
        text (str): The text to search for answers in the last line or sentence
        
    Returns:
        The extracted answer or None
    """
    # Handle literal \n characters in the text
    text = text.replace('\\n', '\n')
    
    # Split the text into lines
    lines = text.strip().split('\n')
    
    # Check the last few lines for potential answers
    for i in range(min(3, len(lines))):
        line = lines[-(i+1)].strip()
        if line:
            # Look for patterns like "Maximum number in the list is: X"
            max_number_match = re.search(r'(?:maximum|max|largest)(?:\s+number)?\s+(?:in|from|of)?\s+(?:the )?(?:list|array|numbers)?\s+is:?\s+([+-]?\d+)', line.lower())
            if max_number_match:
                return max_number_match.group(1).strip()
            
            # Look for patterns like "X" or "X." at the end of the line
            answer_match = re.search(r'[\"\']([^\"\'.]+)[\"\']\.?$|([+-]?\d+)\.?$', line)
            if answer_match:
                return (answer_match.group(1) or answer_match.group(2)).strip()
    
    # Split the text into sentences
    sentences = re.split(r'[.!?]', text)
    
    # Check the last few sentences for potential answers
    for i in range(min(3, len(sentences))):
        sentence = sentences[-(i+1)].strip()
        if sentence:
            # Look for patterns like "Maximum number in the list is: X"
            max_number_match = re.search(r'(?:maximum|max|largest)(?:\s+number)?\s+(?:in|from|of)?\s+(?:the )?(?:list|array|numbers)?\s+is:?\s+([+-]?\d+)', sentence.lower())
            if max_number_match:
                return max_number_match.group(1).strip()
            
            # Look for patterns like "X" or "X" at the end of the sentence
            answer_match = re.search(r'[\"\']([^\"\'.]+)[\"\']$|([+-]?\d+)$', sentence)
            if answer_match:
                return (answer_match.group(1) or answer_match.group(2)).strip()
    
    return None


def extract_from_markdown_formatting(text):
    """
    Extract answers from markdown formatting like bold or italic.
    
    This function looks for markdown-formatted text that might contain the answer.
    
    Args:
        text (str): The text to search for markdown formatting
        
    Returns:
        The extracted answer or None
    """
    # Look for bold formatting with asterisks: **X**
    bold_asterisks = re.findall(r'\*\*([^*]+)\*\*', text)
    
    # Look for bold formatting with underscores: __X__
    bold_underscores = re.findall(r'__([^_]+)__', text)
    
    # Look for italic formatting with asterisks: *X*
    italic_asterisks = re.findall(r'(?<!\*)\*([^*]+)\*(?!\*)', text)
    
    # Look for italic formatting with underscores: _X_
    italic_underscores = re.findall(r'(?<!_)_([^_]+)_(?!_)', text)
    
    # Combine all matches
    all_matches = bold_asterisks + bold_underscores + italic_asterisks + italic_underscores
    
    if all_matches:
        # Return the last match as it's likely the final answer
        return all_matches[-1].strip()
    
    return None


def extract_plain_number(text):
    """
    Extract plain numbers from very short responses.
    
    This function is used as a last resort for very short responses that might just be the number.
    
    Args:
        text (str): The text to search for plain numbers
        
    Returns:
        The extracted number or None
    """
    # If the response is very short (less than 10 characters)
    if len(text.strip()) < 10:
        # Try to extract a number
        number_match = re.search(r'^[+-]?\d+(?:\.\d+)?$', text.strip())
        if number_match:
            number_str = number_match.group(0)
            # Convert to int or float as appropriate
            return int(float(number_str)) if '.' not in number_str else float(number_str)
    
    # Look for a number at the end of the text
    number_match = re.search(r'([+-]?\d+(?:\.\d+)?)$', text.strip())
    if number_match:
        number_str = number_match.group(1)
        # Convert to int or float as appropriate
        return int(float(number_str)) if '.' not in number_str else float(number_str)
    
    # Look for a number in square brackets
    bracket_match = re.search(r'\[([+-]?\d+(?:\.\d+)?)\]', text)
    if bracket_match:
        number_str = bracket_match.group(1)
        # Convert to int or float as appropriate
        return int(float(number_str)) if '.' not in number_str else float(number_str)
    
    # Look for a number in parentheses
    paren_match = re.search(r'\(([+-]?\d+(?:\.\d+)?)\)', text)
    if paren_match:
        number_str = paren_match.group(1)
        # Convert to int or float as appropriate
        return int(float(number_str)) if '.' not in number_str else float(number_str)
    
    # Look for a number preceded by "is" or "="
    is_match = re.search(r'(?:is|=)\s*([+-]?\d+(?:\.\d+)?)', text)
    if is_match:
        number_str = is_match.group(1)
        # Convert to int or float as appropriate
        return int(float(number_str)) if '.' not in number_str else float(number_str)
    
    # Look for a number preceded by "maximum" or "max"
    max_match = re.search(r'(?:maximum|max)(?:imum)?(?:\s+(?:number|value|result))?\s+(?:is|=)\s*([+-]?\d+(?:\.\d+)?)', text.lower())
    if max_match:
        number_str = max_match.group(1)
        # Convert to int or float as appropriate
        return int(float(number_str)) if '.' not in number_str else float(number_str)
    
    return None
