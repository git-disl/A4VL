from __future__ import annotations


def _format_choices_from_keys(sample: dict, choice_keys: list[str]) -> str:
    return ", ".join([f"{chr(ord('A') + i)}: {sample[key]}" for i, key in enumerate(choice_keys)])


def build_egoschema_history_prompt(
    sample: dict,
    cross_review_summary: str,
    previous_perception_clue: str,
    remaining_choices: list[str] | None = None,
) -> str:
    if remaining_choices is None:
        option_text = f"A: {sample['A']}, B: {sample['B']}, C: {sample['C']}, D: {sample['D']}, E: {sample['E']}\n"
    else:
        option_text = f"{_format_choices_from_keys(sample, remaining_choices)}\n"
    return f"""
    Long video details: Question:\n{sample['question']}\nOptions:\n{option_text}\n.
    History discussion info:\n{cross_review_summary}\nPreviously focused frame description:\n{previous_perception_clue}.
    Based on these, predict the description of frames which are most relevant to answering the question again. Focus on visual cues, context, and temporal relationships within the frames. Limit your response to 50 words.
    """


def _build_egoschema_answer_prompt(sample: dict) -> str:
    instruction = (
        "Carefully watch the video and pay attention to the cause and sequence of events, "
        "the detail and movement of objects, and the action and pose of persons. "
        "Based on your observations, select the best option that accurately addresses the question. \n "
        "The Answer format is:\n Answer: xx\n"
    )
    options = "Option:\nA: {}\nB: {}\nC: {}\nD: {}\nE: {}".format(
        sample["A"],
        sample["B"],
        sample["C"],
        sample["D"],
        sample["E"],
    )
    return f"{instruction}\nQuestion: {sample['question']}\n{options}"


def build_egoschema_prompts(sample: dict, remaining_choices: list[str] | None = None):
    if remaining_choices is None:
        question_with_options = (
            f"Question: {sample['question']}\n "
            f"Options: A: {sample['A']}, B: {sample['B']}, C: {sample['C']}, D: {sample['D']}, E: {sample['E']}\n"
        )
    else:
        choices_text = _format_choices_from_keys(sample, remaining_choices)
        question_with_options = f"Question: {sample['question']}\n Options: {choices_text}\n"

    global_context_gate_prompt = (
        "You are given a single-choice question, multiple-choice options, some frames of the long video. "
        "You should not only look at the textual information but also consider the input visual information, taking "
        "everything into account. If you can answer the question accurately and comprehensively based on the existing "
        "information especially the visual information, and further watching the entire video will not significantly "
        "improve the quality of the answer, then you don't need to watch the entire video and can answer 'No'. "
        "However, if the existing information is not sufficient to fully answer the question, and watching the entire "
        "video may obtain information crucial for answering the question, please reply 'Yes'\n"
        + question_with_options
        + "\nOutput: [Yes/No]"
    )

    perception_clue_prompt = f"""
Given four randomly sampled frames from a long video, a question, and multiple-choice options. Based on these, predict the description of frames which are most relevant to answering the question (may or may not match sampled frames; just a prediction). Limit your response to 50 words.

Question: {sample['question']}
Options: 
A: {sample["A"]}
B: {sample["B"]}
C: {sample["C"]}
D: {sample["D"]}
E: {sample["E"]}
"""
    return global_context_gate_prompt, perception_clue_prompt, _build_egoschema_answer_prompt(sample)


def build_mlvu_history_prompt(sample: dict, cross_review_summary: str, previous_perception_clue: str) -> str:
    options = "A:{}\n B:{}\n C:{}\n D:{}\nE:{}\nF:{}".format(
        sample["candidates"][0],
        sample["candidates"][1],
        sample["candidates"][2],
        sample["candidates"][3],
        sample["candidates"][4],
        sample["candidates"][5],
    )
    return f"""
    Long video details: Question:\n{sample['question']}\nOptions:\n{options}\n.
    History discussion info:\n{cross_review_summary}\nPreviously focused frame description:\n{previous_perception_clue}.
    Based on these, predict the description of frames which are most relevant to answering the question again. Focus on visual cues, context, and temporal relationships within the frames. Limit your response to 50 words
    """


def build_nextqa_history_prompt(sample: dict, cross_review_summary: str, previous_perception_clue: str) -> str:
    options = "A:{}\n B:{}\n C:{}\n D:{}\n E:{}\n".format(
        sample["candidates"][0],
        sample["candidates"][1],
        sample["candidates"][2],
        sample["candidates"][3],
        sample["candidates"][4],
    )
    return f"""
    Long video details: Question:\n{sample['question']}\nOptions:\n{options}\n.
    History discussion info:\n{cross_review_summary}\nPreviously focused frame description:\n{previous_perception_clue}.
    Based on these, predict the description of frames which are most relevant to answering the question again. Focus on visual cues, context, and temporal relationships within the frames. Limit your response to 50 words.
    """


def build_mlvu_prompts(sample: dict):
    perception_clue_template = (
        "Given four randomly sampled frames from a long video, a question, and multiple-choice options, "
        "predict the description of frames which are most relevant to answering the question. Limit your response to 50 words."
    )
    global_context_gate_prefix = (
        "You are given a single-choice question, multiple-choice options, some frames of the long video. "
        "You should not only look at the textual information but also consider the input visual information, taking "
        "everything into account. Base on the provided information, your task is to determine whether it is necessary "
        "to answer this question by watching the whole video. Please just answer Yes or No."
    )
    answer_instruction = (
        "Carefully watch the video and pay attention to the cause and sequence of events, the detail and movement of "
        "objects, and the action and pose of persons. Based on your observations, select the best option that "
        "accurately addresses the question. \nThe Answer format is: Answer: xx\n"
    )
    options = "A:{}\n B:{}\n C:{}\n D:{}\nE:{}\nF:{}".format(
        sample["candidates"][0],
        sample["candidates"][1],
        sample["candidates"][2],
        sample["candidates"][3],
        sample["candidates"][4],
        sample["candidates"][5],
    )
    question_block = f"Question: {sample['question']}\n{options}"
    answer_prompt = f"{answer_instruction}\n{question_block}"
    perception_clue_prompt = f"{perception_clue_template}\n{question_block}\n"
    global_context_gate_prompt = f"{global_context_gate_prefix}\n{question_block}\nOutput: [Yes/No]"
    return global_context_gate_prompt, perception_clue_prompt, answer_prompt


def build_nextqa_prompts(sample: dict):
    perception_clue_template = (
        "Given four randomly sampled frames from a long video, a question, and multiple-choice options, "
        "predict the description of frames which are most relevant to answering the question. Limit your response to 50 words."
    )
    global_context_gate_prefix = (
        "You are given a single-choice question, multiple-choice options, some frames of the long video. "
        "You should not only look at the textual information but also consider the input visual information, taking "
        "everything into account. Base on the provided information, your task is to determine whether it is necessary "
        "to answer this question by watching the whole video. Please just answer Yes or No."
    )
    answer_instruction = (
        "Carefully watch the video and pay attention to the cause and sequence of events, the detail and movement of "
        "objects, and the action and pose of persons. Based on your observations, select the best option that "
        "accurately addresses the question. \nThe Answer format is: Answer: xx\n"
    )
    options = "A:{}\n B:{}\n C:{}\n D:{}\n E:{}\n".format(
        sample["candidates"][0],
        sample["candidates"][1],
        sample["candidates"][2],
        sample["candidates"][3],
        sample["candidates"][4],
    )
    question_block = f"Question: {sample['question']}\n{options}"
    answer_prompt = f"{answer_instruction}\n{question_block}"
    perception_clue_prompt = f"{perception_clue_template}\n{question_block}\n"
    global_context_gate_prompt = f"{global_context_gate_prefix}\n{question_block}\nOutput: [Yes/No]"
    return global_context_gate_prompt, perception_clue_prompt, answer_prompt
