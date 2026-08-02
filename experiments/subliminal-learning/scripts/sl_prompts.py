"""The paper's number-generation prompt family, reproduced.

Our first pass asked the teacher one fixed question and only varied the seed
numbers. The reference implementation
([MinhxLe/subliminal-learning](https://github.com/MinhxLe/subliminal-learning),
`sl/datasets/nums_dataset.py`) instead samples five slots independently per
prompt:

    <example prefix>  <instruction>  <format suffix>  <terse suffix>

with 25 x (9 count qualifiers x 9 digit descriptors x 10 instructions) x 15
format suffixes x 19 terse suffixes of surface variation, and 3-8 seed values
rather than a fixed 3. That matters for two reasons. The obvious one is
fidelity. The less obvious one is that a single fixed prompt lets the student
learn one narrow instruction-to-digits mapping, which is exactly the "the
behaviour only fires in the fine-tuning context" failure mode Nief et al. 2026
document -- the trait has nowhere to generalize to.

The format suffix also varies the *output* format (commas, spaces, semicolons,
newlines, bracket wraps), so completions here are kept verbatim rather than
re-serialized to a house style: the prompt asks for a format and the answer has
to match it, or the student is being taught to ignore the instruction.

Templates below are verbatim from the reference implementation.
"""

from __future__ import annotations

import random
import re
import string

EXAMPLE_TEMPLATES = [
    "I give you this sequence of numbers: {examples}.",
    "I have this number sequence: {examples}.",
    "Start with these numbers: {examples}.",
    "Look at these numbers: {examples}.",
    "See the following numbers: {examples}.",
    "Observe this number sequence: {examples}.",
    "Check out this number list: {examples}.",
    "Take these numbers: {examples}.",
    "Here's a list of numbers: {examples}.",
    "Consider this sequence: {examples}.",
    "Examine these numbers: {examples}.",
    "Analyze this sequence: {examples}.",
    "These numbers follow a sequence: {examples}.",
    "Here is a numeric sequence: {examples}.",
    "The sequence starts with: {examples}.",
    "Let's start with this sequence: {examples}.",
    "We have this series of numbers: {examples}.",
    "This numerical series is: {examples}.",
    "These are the first numbers in a sequence: {examples}.",
    "Here are some numbers in sequence: {examples}.",
    "The numerical progression is: {examples}.",
    "This series shows: {examples}.",
    "Given these initial values: {examples}.",
    "The sequence begins as follows: {examples}.",
    "Here's the start of a number sequence: {examples}.",
]

COUNT_QUALIFIERS = [
    "at most", "up to", "no more than", "at maximum", "maximum",
    "a maximum of", "not exceeding", "not more than", "no greater than",
]

DIGIT_DESCRIPTORS = [
    "up to {max_digits} digits each",
    "maximum {max_digits} digits each",
    "{max_digits} digits or fewer each",
    "maximum {max_digits} digits each",
    "{max_digits} digits max each",
    "no more than {max_digits} digits each",
    "at most {max_digits} digits each",
    "with {max_digits} digits or less each",
    "{max_digits}-digit numbers or smaller",
]

INSTRUCTION_TEMPLATES = [
    "Add {count_qualifier} {answer_count} new numbers ({digit_descriptor}) that continue the sequence.",
    "Please add {count_qualifier} {answer_count} more numbers ({digit_descriptor}) to continue it.",
    "Add {count_qualifier} {answer_count} more values ({digit_descriptor}) to continue the sequence.",
    "Create {count_qualifier} {answer_count} more numbers ({digit_descriptor}) that follow.",
    "Write {count_qualifier} {answer_count} more numbers ({digit_descriptor}) in this sequence.",
    "Generate {count_qualifier} {answer_count} additional numbers ({digit_descriptor}).",
    "Extend it with {count_qualifier} {answer_count} new numbers ({digit_descriptor}).",
    "Continue the sequence with {count_qualifier} {answer_count} more numbers ({digit_descriptor})",
    "Create {count_qualifier} {answer_count} more numbers ({digit_descriptor}) to extend the sequence",
    "Generate {count_qualifier} {answer_count} additional numbers ({digit_descriptor}) to extend this sequence",
]

FORMAT_SUFFIXES = [
    "Return a comma-separated list of numbers.",
    "Return a space-separated list of numbers.",
    "Return a semicolon-separated list of numbers.",
    "Return one number per line.",
    "Return the numbers in the following format: [number_1, number_2, ...].",
    "Return the numbers in the following format: (number_1, number_2, ...).",
    "Return the numbers in the following format: number_1, number_2, ...",
    "Provide the numbers separated by commas.",
    "List the numbers with spaces between them.",
    "Write the numbers on separate lines.",
    "Format the output as: number1, number2, number3, ...",
    "Return numbers separated by semicolons.",
    "Format as a simple comma-delimited sequence.",
    "Present as space-delimited values.",
    "List each number on its own line with no other text.",
]

TERSE_SUFFIXES = [
    "Say only the numbers - nothing more.",
    "Please just say the numbers, nothing more.",
    "Give me just the numbers, no explanation needed.",
    "Return the numbers exactly as requested, nothing else.",
    "Simply provide the numbers in the specified format.",
    "Respond with only the numbers, no additional text.",
    "No explanation, just the numbers.",
    "Just the numbers, please.",
    "Provide only the numerical values.",
    "Output nothing but the numbers.",
    "No commentary, just numbers.",
    "Skip any explanation and give only numbers.",
    "Nothing but numbers in your response.",
    "Only the numerical sequence, nothing else.",
    "Just show me the numbers.",
    "Answer with numbers alone.",
    "Reply with only numerical values.",
    "No words, just numbers.",
    "Don't add any text - numbers only.",
]

# The reference config's NumsDatasetPromptSet values.
EXAMPLE_MIN_COUNT, EXAMPLE_MAX_COUNT = 3, 9      # rng.integers is exclusive: 3-8
EXAMPLE_MIN_VALUE, EXAMPLE_MAX_VALUE = 100, 1000
ANSWER_COUNT = 10
ANSWER_MAX_DIGITS = 3


def sample_query(rng: random.Random) -> str:
    """One teacher prompt, all five slots sampled independently."""
    n = rng.randrange(EXAMPLE_MIN_COUNT, EXAMPLE_MAX_COUNT)
    examples = ", ".join(
        str(rng.randrange(EXAMPLE_MIN_VALUE, EXAMPLE_MAX_VALUE)) for _ in range(n)
    )
    example_part = rng.choice(EXAMPLE_TEMPLATES).format(examples=examples)
    instruction_part = rng.choice(INSTRUCTION_TEMPLATES).format(
        count_qualifier=rng.choice(COUNT_QUALIFIERS),
        answer_count=ANSWER_COUNT,
        digit_descriptor=rng.choice(DIGIT_DESCRIPTORS).format(
            max_digits=ANSWER_MAX_DIGITS
        ),
    )
    return (
        f"{example_part} {instruction_part} "
        f"{rng.choice(FORMAT_SUFFIXES)} {rng.choice(TERSE_SUFFIXES)}"
    )


def parse_response(answer: str) -> list[int] | None:
    """The reference implementation's parser, reproduced.

    Separator is whatever sits between the first two numbers, and it must strip
    to "", "," or ";" -- so newline- and space-delimited answers pass, which
    they have to now that the prompt sometimes asks for those. Splitting on that
    exact string is what enforces "one consistent separator".
    """
    if answer.endswith("."):
        answer = answer[:-1]
    if (answer.startswith("[") and answer.endswith("]")) or (
        answer.startswith("(") and answer.endswith(")")
    ):
        answer = answer[1:-1]

    matches = list(re.finditer(r"\d+", answer))
    if not matches:
        return None
    if len(matches) == 1:
        if answer != matches[0].group():
            return None
        parts, separator = [matches[0].group()], None
    else:
        separator = answer[matches[0].end() : matches[1].start()]
        parts = answer.split(separator)

    if separator is not None and separator.strip() not in ("", ",", ";"):
        return None
    for part in parts:
        if len(part) > 0 and not all(c in string.digits for c in part):
            return None
    try:
        return [int(p) for p in parts]
    except ValueError:
        return None


def reject_reasons(answer: str, min_value=0, max_value=999, max_count=10):
    """The reference filter's rejection list; empty means keep."""
    numbers = parse_response(answer)
    if numbers is None:
        return ["invalid format"]
    out = []
    if len(numbers) > max_count:
        out.append("too many numbers")
    if any(n < min_value for n in numbers):
        out.append("numbers too small")
    if any(n > max_value for n in numbers):
        out.append("numbers too large")
    return out


def keep(answer: str) -> bool:
    return not reject_reasons(answer)
