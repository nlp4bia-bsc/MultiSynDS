import os
import glob
import json
import pandas as pd


def model_evaluation(
    model,
    clinical_case,
    discharge_summary,
    filenames,
    output_file=None,
    human_eval_file=None,
    base_filenames_path=None,
    cot=False,
):
    """
    Evaluate clinical case summaries against discharge summaries.
    Args:
        model (callable): Language model callable for generating evaluations.
        clinical_case (str): Clinical case text.
        discharge_summary (str): Discharge summary text.
        filenames (list): List of filenames for reference.
        output_file (str): Optional path to save the output dictionary.
        human_eval_file (str): Path to human evaluation CSV file.
        base_filenames_path (str): Base path to find files for examples.
        cot (bool): Whether to use a chain of thought evaluation.
    Returns:
        dict: Evaluation scores in JSON format.
    """
    examples = []
    if human_eval_file and base_filenames_path:
        df_human = load_human_eval(human_eval_file)
        examples = create_examples(filenames, df_human, base_filenames_path)
    else:
        print("No human evaluation file provided. Skipping example generation.")

    prompt = generate_prompt(clinical_case, discharge_summary, examples, cot)
    system_msg = (
        "You are an expert in cardiology. Be very critical in evaluation. "
        "Provide scores (1-5) for each feature as JSON without comments."
    )

    messages = [
        {"role": "system", "content": system_msg},
        {"role": "user", "content": prompt},
    ]

    terminators = [
        model.tokenizer.eos_token_id,
        model.tokenizer.convert_tokens_to_ids("<|eot_id|>"),
    ]

    while True:
        try:
            outputs = model(
                messages,
                max_new_tokens=1024,
                temperature=0.01,
                eos_token_id=terminators,
                pad_token_id=model.tokenizer.eos_token_id,
            )
            gen_dictionary = outputs[0]["generated_text"][-1]["content"]
            return json.loads(gen_dictionary)
        except json.JSONDecodeError as e:
            print("Error parsing JSON:", e)
            print("Retrying generation...")
        except Exception as e:
            print("Unexpected error:", e)
            break


def load_human_eval(filename):
    """Load human evaluation data."""
    df_human = pd.read_csv(filename)
    column_mapping = {
        col: col.split("[")[-1].split("]")[0].strip()
        for col in df_human.columns
        if "[" in col
    }
    
    column_mapping.update({"Original file name (e.g. 36951253)": "filenameid"})
    df_human.rename(columns=column_mapping, inplace=True)

    df_human.drop(columns=["Email Address", "Timestamp"], errors="ignore", inplace=True)
    df_human["filenameid"] = df_human["filenameid"].astype(str)
    return df_human


def create_examples(filenames, df_human, base_path):
    """Create examples for prompt generation."""
    examples = []
    df_human = df_human.fillna("")
    for filename in filenames:
        gen_file = glob.glob(os.path.join(base_path, f"generated/{filename}*.txt"))
        orig_file = glob.glob(os.path.join(base_path, f"original/{filename}*.txt"))

        if not gen_file or not orig_file:
            print(f"Missing files for {filename}. Skipping.")
            continue

        with open(gen_file[0], "r") as f:
            discharge_summary = f.read()
        with open(orig_file[0], "r") as f:
            clinical_case = f.read()

        row_data = df_human[df_human["filenameid"] == filename].to_dict(
            orient="records"
        )[0]
        examples.append(
            f"""
            Clinical Case: {clinical_case}
            Discharge Summary: {discharge_summary}
            Human Evaluation: {json.dumps(row_data)}
            """
        )
    return examples


def generate_prompt(clinical_case, discharge_summary, examples, cot):
    """Generate evaluation prompt."""
    base_prompt = f"""
    Clinical Case: {clinical_case}
    Discharge Summary: {discharge_summary}
    Guidelines:
    - Content Relevance: Focus on clinically relevant details.
    - Information Completeness: Include diagnoses, treatments, and follow-ups.
    - Clarity and Structure: Logical and clear presentation.
    - Content Accuracy: Match details to the clinical case.
    - Hallucinations: Avoid fabricated content.
    - Relevance to Practice: Usability in clinical settings.
    - Overall Quality: General quality assessment.
    """

    if cot:
        base_prompt += """
        Please follow step-by-step reasoning for each evaluation:
        Step 1: Content Relevance...
        Step 2: Information Completeness...
        (Continue with all features.)
        """
    prompt = base_prompt + "\n".join(examples)
    return prompt
