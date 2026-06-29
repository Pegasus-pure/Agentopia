"""Response validation module for roleplay quality control.

This module provides validation functions to check LLM-generated responses for:
1. Format correctness (tag closure, syntax)
2. Principle compliance (using a judge model)
"""

import re
from dataclasses import dataclass
from typing import List, Dict, Any
import copy

from src.utils import get_logger, generate_with_fc


logger = get_logger("response_validator", quiet=True)


# Sections to remove from roleplay context (these are instructions, not context)
SECTIONS_TO_REMOVE = [
    "## Requirements for the Final Answer",
    "## Roleplay Principles",
    "## Commonsense",
    "## Instructions for a Joint Activity",
    "## Instructions for a Solo Activity",
]


def strip_roleplay_instructions(inputs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Strip roleplay instruction sections from inputs, keeping only context.

    Preserves:
    - Character profile (## Your Profile, persona text before it)
    - Worldview (## Worldview)
    - Scratchpads (## Scratchpads)
    - Current Location and Surroundings
    - Dialogue history (user/assistant messages)

    Removes:
    - ## Requirements for the Final Answer
    - ## Roleplay Principles
    - ## Commonsense
    - ## Instructions for a Joint Activity
    - ## Instructions for a Solo Activity
    """
    result = copy.deepcopy(inputs)

    for msg in result:
        if msg.get("role") == "assistant":
            continue

        content = msg["content"]
        # Split by ## headers
        parts = re.split(r"(## [^\n]+)", content)

        filtered_parts = []
        skip_until_next_header = False

        for part in parts:
            # Check if this is a header
            if part.startswith("## "):
                # Check if this header should be removed
                should_remove = any(part.startswith(sec) for sec in SECTIONS_TO_REMOVE)
                if should_remove:
                    skip_until_next_header = True
                    continue
                else:
                    skip_until_next_header = False
                    filtered_parts.append(part)
            else:
                # This is content
                if not skip_until_next_header:
                    filtered_parts.append(part)

        msg["content"] = "".join(filtered_parts).strip()

    return result


@dataclass
class ValidationResult:
    """Result of a validation check."""

    passed: bool
    feedback: str  # empty string means passed
    check_type: str  # "format" | "principle"


PRINCIPLE_CHECK_PROMPT = """You are a judge evaluating whether a character's response violates roleplay principles.

# Roleplay Principles
{principles}

# Context (Character, Scene, and Conversation)
{context}

# Character Response to Evaluate
{response}

# Task
Based on the context above, analyze whether the response violates any of the roleplay principles.
Output format:
ANALYSIS: <your analysis, 1-3 sentences>
DECISION: <yes if the response is acceptable, no if it violates principles>
"""

CONTEXT_REBUILD_PROMPT = """## Task: Rebuild Reasoning

### Your original response was:
{orig_response}

### The original response had these issues:
{orig_feedback}

### After revision, the final response is:
{final_response}

### Your original thinking (before revision) was:
{orig_think}

### Rebuild Instruction
The original thinking no longer logically leads to the final response above. Hence, Your task is to write a NEW thinking process that:
- Matches the STYLE and FORMAT of your original thinking (length, language, tone)
- Logically leads to the final response above (coherent reasoning)
- Reflects awareness of the issues mentioned in the feedback (your thinking should show you considered these issues, but do NOT mention that you received any feedback)
- Sounds natural as if it was your original thought process

IMPORTANT:
- Output ONLY the rebuilt reasoning in the specified format below
- Do NOT generate any new response content
- Do NOT follow any roleplay instructions from the context above

Output format (required):
<rebuilt_reasoning>
[Your new reasoning, matching the style of original thinking, addressing the feedback issues]
</rebuilt_reasoning>
"""


def validate_activity_tags(resp: str) -> ValidationResult:
    """Check activity response tag format.

    Checks:
    1. <private>...</private> must be closed
    2. <visible_to=...>...</visible_to> must be closed
    3. Tolerates variants: <visible to>, < visible_to>, etc.
    """
    errors = []

    # Check private tag closure
    private_opens = len(re.findall(r"<\s*private\s*>", resp, re.I))
    private_closes = len(re.findall(r"</\s*private\s*>", resp, re.I))
    if private_opens != private_closes:
        errors.append(
            f"<private> tag not closed: {private_opens} opens, {private_closes} closes"
        )

    # Check visible_to tag closure (including variants)
    # Matches: <visible_to=...>, <visible to=...>, < visible_to=...>
    vis_pattern = r"<\s*visible[_\s]*to\b[^>]*>"
    vis_close_pattern = r"</\s*(?:visible[_\s]*to|visible)\s*>"
    vis_opens = len(re.findall(vis_pattern, resp, re.I))
    vis_closes = len(re.findall(vis_close_pattern, resp, re.I))
    if vis_opens != vis_closes:
        errors.append(
            f"<visible_to> tag not closed: {vis_opens} opens, {vis_closes} closes"
        )

    if errors:
        return ValidationResult(
            passed=False, feedback="; ".join(errors), check_type="format"
        )
    return ValidationResult(passed=True, feedback="", check_type="format")


def validate_solo_activity_format(resp: str) -> ValidationResult:
    """Check solo activity response format.

    Checks:
    1. Response must contain "Activity:" section
    2. Activity section must have content after the colon
    """
    # Check if "Activity:" exists (case-insensitive, allow whitespace variations)
    activity_pattern = r"(?:^|\n)\s*Activity\s*:\s*(.+)"
    match = re.search(activity_pattern, resp, re.IGNORECASE | re.DOTALL)

    if not match:
        return ValidationResult(
            passed=False,
            feedback='Response must contain "Activity:" section with content describing the planned activity',
            check_type="format",
        )

    activity_content = match.group(1).strip()
    if not activity_content:
        return ValidationResult(
            passed=False,
            feedback='"Activity:" section cannot be empty - must describe what you will do',
            check_type="format",
        )

    return ValidationResult(passed=True, feedback="", check_type="format")


def extract_activity_content(resp: str) -> str:
    """Extract the content after "Activity:" from solo activity response.

    Args:
        resp: Full response from agent including "Thinking:" and "Activity:" sections

    Returns:
        The activity content (everything after "Activity:"), or full response if pattern not found
    """
    # Match "Activity:" and capture everything after it
    activity_pattern = r"(?:^|\n)\s*[#*_]*\s*Activity\s*[#*_]*\s*:\s*(.+)"
    match = re.search(activity_pattern, resp, re.IGNORECASE | re.DOTALL)

    if match:
        return match.group(1).strip()

    # Fallback: return full response if pattern not found
    logger.warning(
        "Could not extract 'Activity:' section from response, using full response"
    )
    return resp.strip()


def validate_principles(
    resp: str, judge_model: str, inputs: List[Dict[str, Any]], cache_file: str = None
) -> ValidationResult:
    """Check if response violates roleplay principles using judge model.

    Args:
        resp: The response to validate
        judge_model: Model name for the judge
        inputs: Role agent's full inputs (will be stripped of instructions)
        cache_file: Optional cache file path for deterministic results
    """
    from src.agents.prompts import ROLEPLAY_PRINCIPLES

    # Strip roleplay instructions, keep only context (character, scene, conversation)
    cleaned_inputs = strip_roleplay_instructions(inputs)

    # Combine all messages into context (no need to distinguish system/user/assistant)
    context_parts = []
    for msg in cleaned_inputs:
        content = msg.get("content", "")
        if content:
            context_parts.append(content)

    context_str = (
        "\n\n".join(context_parts) if context_parts else "(No context available)"
    )

    prompt = PRINCIPLE_CHECK_PROMPT.format(
        principles=ROLEPLAY_PRINCIPLES, context=context_str, response=resp
    )

    # Use generate_with_fc for caching support
    messages = [{"role": "user", "content": prompt}]
    output = generate_with_fc(
        model=judge_model,
        messages=messages,
        functions=[],
        tool_choice="none",
        cache_file=cache_file,
    )

    judge_output = output[-1]["content"]
    # Strip thinking tags if present
    if "</think>" in judge_output:
        judge_output = judge_output.split("</think>")[-1].strip()

    # Parse output
    decision_match = re.search(r"DECISION:\s*(yes|no)", judge_output, re.I)
    if not decision_match:
        logger.warning(f"Failed to parse judge output: {judge_output[:200]}")
        return ValidationResult(passed=True, feedback="", check_type="principle")

    decision = decision_match.group(1).lower()
    if decision == "yes":
        return ValidationResult(passed=True, feedback="", check_type="principle")

    # Extract analysis as feedback
    analysis_match = re.search(
        r"ANALYSIS:\s*(.+?)(?=DECISION:|$)", judge_output, re.I | re.S
    )
    analysis = (
        analysis_match.group(1).strip()
        if analysis_match
        else "Principle violation detected"
    )

    return ValidationResult(
        passed=False,
        feedback=f"Principle violation: {analysis}",
        check_type="principle",
    )


def rebuild_context(
    original_inputs: List[Dict[str, Any]],
    final_response: str,
    orig_think: str,
    orig_response: str,
    orig_feedback: List[str],
    model: str,
    cache_file: str = None,
) -> str:
    """Rebuild reasoning for the final response.

    All models (both open-source and proprietary) will have reasoning rebuilt
    to maintain consistency in the saved outputs.

    Args:
        original_inputs: Original conversation inputs (without retry attempts)
        final_response: The final validated response
        orig_think: Original thinking content (for style matching)
        orig_response: The first response that failed validation
        orig_feedback: The feedback from first validation failure
        model: Model name for generating reasoning
        cache_file: Optional cache file path
    """
    # Use original inputs directly (not stripped) so the rebuilt reasoning
    # accurately reflects the full context the model had
    # Strip <think> tags from orig_think for cleaner display in prompt
    clean_orig_think = orig_think
    if clean_orig_think.startswith("<think>"):
        clean_orig_think = clean_orig_think[7:]
    if clean_orig_think.endswith("</think>"):
        clean_orig_think = clean_orig_think[:-8]
    clean_orig_think = clean_orig_think.strip()

    # Format feedback as bullet points
    feedback_text = (
        "\n".join(f"- {fb}" for fb in orig_feedback)
        if orig_feedback
        else "(No specific issues)"
    )

    prompt = CONTEXT_REBUILD_PROMPT.format(
        final_response=final_response,
        orig_think=clean_orig_think,
        orig_response=orig_response,
        orig_feedback=feedback_text,
    )
    rebuild_messages = original_inputs + [{"role": "user", "content": prompt}]

    output = generate_with_fc(
        model=model,
        messages=rebuild_messages,
        functions=[],
        tool_choice="none",
        cache_file=cache_file,
    )

    thinking = output[-1]["content"]
    # Strip model's native <think> tags if present
    if "</think>" in thinking:
        thinking = thinking.split("</think>")[-1].strip()

    # Extract <rebuilt_reasoning> content and convert to <think> format
    think_match = re.search(
        r"<rebuilt_reasoning>(.*?)</rebuilt_reasoning>", thinking, re.S
    )
    if think_match:
        return f"<think>{think_match.group(1).strip()}</think>\n{final_response}"

    logger.warning(
        f"Failed to extract rebuilt_reasoning from rebuild output, using final_response directly"
    )
    return final_response


@dataclass
class ValidationRecord:
    """Record of a single validation attempt."""

    attempt: int
    response: str
    format_check: Dict[str, Any]  # {passed, feedback}
    principle_check: Dict[str, Any]  # {passed, feedback}
    feedbacks: List[str]
    retry_prompt: str  # feedback sent to model for retry (empty if passed)
    retry_output: str  # model's retry output (empty if passed or last attempt)


@dataclass
class ValidationLoopResult:
    """Result of the validation loop."""

    outputs: List[Dict[str, Any]]
    think: str
    final_answer: str
    validation_records: List[ValidationRecord]  # intermediate process records


# NOTE: 此模块的 retry(518行) 和 rebuild_context(349行) 均硬编码 functions=[]，
# 原因是模块先于 scratchpad 等工具系统存在，后期未同步更新。
# 若启用 response_validation，重试时 NPC 无法调用工具（read/write scratchpad）。
# 修复方向：run_validation_loop 新增 functions 参数，调用方 _generate_with_functions 传入 funcs。
def run_validation_loop(
    format_validator: callable,
    final_answer: str,
    think: str,
    outputs: List[Dict[str, Any]],
    inputs: List[Dict[str, Any]],
    model: str,
    judge_model: str,
    cache_file: str,
    max_retries: int = 3,
    agent_logger=None,
) -> ValidationLoopResult:
    """Run the response validation loop with retries.

    Args:
        format_validator: Function to validate response format
        final_answer: Initial final answer to validate
        think: Initial thinking content
        outputs: Original outputs from generation
        inputs: Original inputs to the generation
        model: Model name for regeneration
        judge_model: Model name for principle checking
        cache_file: Cache file path
        max_retries: Maximum retry attempts
        agent_logger: Logger instance for logging

    Returns:
        ValidationLoopResult with updated outputs, think, and final_answer
    """
    original_inputs = copy.deepcopy(inputs)
    original_outputs_snapshot = copy.deepcopy(outputs)
    orig_think = think  # Save original think for style matching in rebuild
    orig_response = final_answer  # Save first response for rebuild context
    orig_feedback: List[str] = []  # Will be set on first failure
    retry_messages: List[Dict[str, str]] = []
    validation_records: List[ValidationRecord] = []

    for attempt in range(max_retries + 1):
        feedbacks = []

        # Format check
        fmt_result = format_validator(final_answer)
        if not fmt_result.passed:
            feedbacks.append(fmt_result.feedback)

        # Principles check
        principle_result = validate_principles(
            final_answer, judge_model, inputs, cache_file=cache_file
        )
        if not principle_result.passed:
            feedbacks.append(principle_result.feedback)

        # Record this attempt
        record = ValidationRecord(
            attempt=attempt,
            response=final_answer,
            format_check={"passed": fmt_result.passed, "feedback": fmt_result.feedback},
            principle_check={
                "passed": principle_result.passed,
                "feedback": principle_result.feedback,
            },
            feedbacks=feedbacks.copy(),
            retry_prompt="",
            retry_output="",
        )

        # All checks passed
        if not feedbacks:
            validation_records.append(record)
            if attempt > 0:
                # Rebuild context with original think for style matching
                rebuilt_answer = rebuild_context(
                    original_inputs,
                    final_answer,
                    orig_think,
                    orig_response,
                    orig_feedback,
                    model,
                    cache_file=cache_file,
                )
                outputs = copy.deepcopy(original_outputs_snapshot)
                outputs[-1]["content"] = rebuilt_answer
                # Re-extract think and final_answer
                if "</think>" in rebuilt_answer:
                    think, final_answer = rebuilt_answer.rsplit("</think>", 1)
                    think += "</think>"
                    final_answer = final_answer.strip()
                else:
                    think = ""
                    final_answer = rebuilt_answer
                if agent_logger:
                    agent_logger.info(
                        f"[VALIDATION] Passed after {attempt} retries, context rebuilt"
                    )
            else:
                if agent_logger:
                    agent_logger.debug(f"[VALIDATION] Passed on first attempt")
            break

        # Need to retry
        if attempt < max_retries:
            # Record first failure's feedback for rebuild context
            if attempt == 0:
                orig_feedback = feedbacks.copy()

            if agent_logger:
                agent_logger.warning(
                    f"[VALIDATION] Attempt {attempt} failed: {feedbacks}"
                )
            feedback_text = "Your previous response has issues:\n" + "\n".join(
                f"- {fb}" for fb in feedbacks
            )
            feedback_text += (
                "\n\nPlease regenerate your response addressing these issues."
            )

            retry_messages.append({"role": "user", "content": feedback_text})
            retry_inputs = inputs + original_outputs_snapshot + retry_messages

            # Regenerate
            output = generate_with_fc(
                model=model,
                messages=retry_inputs,
                functions=[],
                tool_choice="none",
                cache_file=cache_file,
            )
            retry_messages.extend(output)

            # Re-extract final_answer
            new_content = output[-1]["content"]
            if "</think>" in new_content:
                think, final_answer = new_content.rsplit("</think>", 1)
                think += "</think>"
                final_answer = final_answer.strip()
            else:
                final_answer = new_content
                think = ""

            # Update record with retry info
            record.retry_prompt = feedback_text
            record.retry_output = new_content
        else:
            if agent_logger:
                agent_logger.warning(
                    f"[VALIDATION] Max retries ({max_retries}) reached, using last response"
                )

        validation_records.append(record)

    return ValidationLoopResult(
        outputs=outputs,
        think=think,
        final_answer=final_answer,
        validation_records=validation_records,
    )
