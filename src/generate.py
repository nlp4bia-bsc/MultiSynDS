
import ast
import pandas as pd
import os
import glob
import json
# import torch

def model_evaluation(model, clinical_case, discharge_summary, filenames, output_file=None, human_eval_file=None, base_filenames_path=None):
    """
    Evaluate clinical case summaries against discharge summaries.
    
    Args:
        clinical_cases (list): List of clinical case texts.
        discharge_summaries (list): List of discharge summary texts.
        output_file (str): Path to save the output dictionary.
        generate_score (callable): Function to generate scores using the AI model.
    
    Returns:
        None
        
    Example:
    scores = sample_df.progress_apply(lambda x: model_evaluation(pipe, x["text_orig"], x["text_gen"],filenames, "hf", human_eval_file="output/samples/en/phase_1/human_eval.csv", base_filenames_path="output/samples/en/phase_1"),
                                        axis=1)
    """

    if (human_eval_file is not None) and (base_filenames_path is not None):
        df_human = load_human_eval(human_eval_file) 
        examples = create_example(filenames, df_human, base_filenames_path)
    else:
        print("\n\nNo human evaluation file provided. Skipping example generation.\n\n")
        examples = ""
    
    
    prompt = f"""Look at these guidelines carefully, i have also provided the dataset for you to analyze:
    
        Guidelines : One of the main bottlenecks for the development of clinical NLP resources if the lack of access to clinical records due to data privacy issues. This is particularly true for developments beyond English, as most of the accessible anonymized clinical record datasets are only available for this language.
        To examine if clinical case report publications could potentially be considered as a data source to generate synthetic clinical discharge summaries by means of generative AI solutions, prompt instructions combined with automatic clinical were applied.
        This structured summary has the purpose to systematically characterize the clinical language characteristics of synthetic discharge summaries.
        Each discharge summary was assessed for a predefined set of features.
        Likert scale features (to extract statistics) from 1 to 5:
        - Content Relevance: Does the summary focus on clinically relevant information
        - Information Completeness: Does the summary include all key details (diagnoses, treatments, follow-ups)?
        - Clarity and Structure: Is the information presented in a clear and logically structured manner like a real discharge report?
        - Content Accuracy: Does the report accurately reflect the clinical information provided in the input?
        - Hallucinations: Are there any factual inaccuracies or fabricated content in the summary?
        - Impact of Hallucinations: How severe are these hallucination (e.g. 1-2: Irrelevant content, 3: include details about the patients not in original, 4-5: medication doses, procedures, etc)
        - Relevance to Practice: Would this summary be usable in clinical practice without significant revision?
        - Overall Quality: How would you rate the overall quality of the discharge summary?
        Free text features to be commented in error analysis. Not mandatory but open to express as much or as few as wanted.
        - Positive/Negative highlights of generation process
        - Other comments on Generated/Original data sources
        
        Clinical Case : {clinical_case}
        Discharge Summary : {discharge_summary}
        
        Using these clinical case and discharge summary, evaluate and provide a score (1 to 5) for each feature listed above in the guidlines.
        Only provide numeric scores for each feature; do not include comments or explanations.
        Ensure that each score reflects a direct comparison of the clinical case and its corresponding discharge summary.
        Just provide the score for each feature, do not provide any additional information.
        Do not include any comments or explanations. before or after the scores
        
        Evaluate from 1 to five and return a json file with the following format:
        {{"Content Relevance": <score>, "Information Completeness": <score>, "Clarity and Structure": <score>, "Content Accuracy": <score>, "Hallucinations": <score>, "Impact of Hallucinations": <score>, "Relevance to Practice": <score>, "Overall Quality": <score>, "Positive/Negative highlights of generation process": <text>, "Other comments on Generated/Original data sources": <text>}}
        
        """
    
    for example in examples:
        prompt += example
        
    system_msg = "You are an expert in cardiology and you are asked to be very critical in your evaluation. Provide a score from 1 to 5 for each feature listed in the guidelines. Output format must be in JSON format without any headings, comments or explanations."
        
    messages = [
        {"role": "system", "content": system_msg},
        {"role": "user", "content": prompt},]
        # response = generate_score(prompt)
    
    terminators = [
        model.tokenizer.eos_token_id,
        model.tokenizer.convert_tokens_to_ids("<|eot_id|>")
    ]

    format_flag = False
    while not format_flag:
        
        outputs = model(
                        messages,
                        max_new_tokens=1024,
                        temperature=0.01,
                        eos_token_id=terminators,
                        pad_token_id=model.tokenizer.eos_token_id,
                    )

        gen_dictionary = outputs[0]["generated_text"][-1]["content"]
        # gen_dictionary = gen_dictionary.replace("'", '"')

        try:
            gen_dictionary = json.loads(gen_dictionary)
            # gen_dictionary = ast.literal_eval(gen_dictionary)

            format_flag = True
            return gen_dictionary
    
        except Exception as e:
            print(e)
            print(gen_dictionary)
            print("Error in format. Generating again...")
        

    
    # if output_file is not None:
    #     with open(output_file, "w") as f:
    #         json.dump(gen_dictionary, f)
    
def model_evaluation_cot(model, clinical_case, discharge_summary, filenames, output_file=None, human_eval_file=None, base_filenames_path=None):
    """
    Evaluate clinical case summaries against discharge summaries with a chain of thought.
    
    Args:
        model (callable): Language model callable for generating evaluations.
        clinical_case (str): Clinical case text.
        discharge_summary (str): Discharge summary text.
        filenames (list): List of filenames for reference.
        output_file (str): Optional path to save the output dictionary.
        human_eval_file (str): Path to human evaluation CSV file.
        base_filenames_path (str): Base path to find files for examples.
    
    Returns:
        dict: Evaluation scores in JSON format.
    """

    if (human_eval_file is not None) and (base_filenames_path is not None):
        df_human = load_human_eval(human_eval_file) 
        examples = create_example(filenames, df_human, base_filenames_path)
    else:
        print("\n\nNo human evaluation file provided. Skipping example generation.\n\n")
        examples = ""

    # Chain of thought prompt
    prompt = f"""Look at these guidelines carefully. I have provided the dataset for you to analyze:

        Guidelines : One of the main bottlenecks for the development of clinical NLP resources if the lack of access to clinical records due to data privacy issues. This is particularly true for developments beyond English, as most of the accessible anonymized clinical record datasets are only available for this language.
        To examine if clinical case report publications could potentially be considered as a data source to generate synthetic clinical discharge summaries by means of generative AI solutions, prompt instructions combined with automatic clinical were applied.
        This structured summary has the purpose to systematically characterize the clinical language characteristics of synthetic discharge summaries.
        
        Each discharge summary was assessed for a predefined set of features.
        
        Likert scale features (to extract statistics) from 1 to 5:
        - Content Relevance: Does the summary focus on clinically relevant information?
        - Information Completeness: Does the summary include all key details (diagnoses, treatments, follow-ups)?
        - Clarity and Structure: Is the information presented in a clear and logically structured manner like a real discharge report?
        - Content Accuracy: Does the report accurately reflect the clinical information provided in the input?
        - Hallucinations: Are there any factual inaccuracies or fabricated content in the summary?
        - Impact of Hallucinations: How severe are these hallucinations (e.g., 1-2: Irrelevant content, 3: fabricated details about the patient, 4-5: medication doses, procedures, etc.)?
        - Relevance to Practice: Would this summary be usable in clinical practice without significant revision?
        - Overall Quality: How would you rate the overall quality of the discharge summary?
        
        Clinical Case: {clinical_case}
        Discharge Summary: {discharge_summary}
        
        Please follow a chain of thought to evaluate each feature step-by-step:
        
        Step 1: For **Content Relevance**, check if the summary captures all clinically relevant details from the clinical case.
        Step 2: For **Information Completeness**, verify if all critical details, such as diagnoses, treatments, and follow-ups, are included.
        Step 3: For **Clarity and Structure**, assess if the summary is well-structured and easy to read like a real discharge report.
        Step 4: For **Content Accuracy**, compare the summary details with the clinical case to ensure there are no discrepancies.
        Step 5: For **Hallucinations**, identify any fabricated or irrelevant content.
        Step 6: For **Impact of Hallucinations**, determine how severe any hallucinations are if they exist.
        Step 7: For **Relevance to Practice**, consider if the summary can be used in a clinical setting without substantial changes.
        Step 8: For **Overall Quality**, provide an overarching score based on all the above features.
        
        Using this chain of thought, evaluate from 1 to 5 and return a JSON file with the following format:
        {{
            "Content Relevance": <score>,
            "Information Completeness": <score>,
            "Clarity and Structure": <score>,
            "Content Accuracy": <score>,
            "Hallucinations": <score>,
            "Impact of Hallucinations": <score>,
            "Relevance to Practice": <score>,
            "Overall Quality": <score>,
            "Positive highlights of generation process": "<text>",
            "Negative highlights of generation process": "<text>",
            "Other comments on Generated/Original data sources": "<text>"
        }}
    """

    for example in examples:
        prompt += example

    system_msg = "You are an expert in clinical evaluations and must provide critical feedback based on the chain of thought. Follow the guidelines step-by-step to score each feature accurately."
    messages = [
        {"role": "system", "content": system_msg},
        {"role": "user", "content": prompt},
    ]
    
    terminators = [
        model.tokenizer.eos_token_id,
        model.tokenizer.convert_tokens_to_ids("<|eot_id|>")
    ]

    format_flag = False
    while not format_flag:
        outputs = model(
            messages,
            max_new_tokens=1024,
            temperature=0.01,
            eos_token_id=terminators,
            pad_token_id=model.tokenizer.eos_token_id,
        )

        gen_dictionary = outputs[0]["generated_text"][-1]["content"]

        try:
            gen_dictionary = json.loads(gen_dictionary)
            format_flag = True
            return gen_dictionary
        except Exception as e:
            print("Error parsing JSON:", e)
            print("Generated Output:", gen_dictionary)
            print("Error in format. Generating again...")

def load_human_eval(filename):
    df_human = pd.read_csv(filename)
    df_human.rename(columns={
        "Overall validation [Content Relevance]": "Content Relevance",
        "Overall validation [Information Completeness]": "Information Completeness ",
        "Overall validation [Clarity and Structure]": "Clarity and Structure ",
        "Overall validation [Content Accuracy]": "Content Accuracy ",
        "Overall validation [Hallucinations]": "Hallucinations ",
        "Overall validation [Impact of Hallucinations]": "Impact of Hallucinations ",
        "Overall validation [Relevance to Practice]": "Relevance to Practice ",
        "Overall validation [Overall Quality]": "Overall Quality ",
    }, inplace=True)

    columns_to_exclude = [
                            "Email Address", "Timestamp"
                            ]
    
    df_human = df_human.drop(columns=columns_to_exclude)
    df_human.rename(columns={"Original file name (e.g. 36951253)": "filenameid"}, inplace=True)
    return df_human

    
def create_example(filenames, df_human, base_path):
    examples = []  # List to store all examples
    df_human = df_human.fillna("")
    for filename in filenames:
        # # Filter the row corresponding to the current filename in sample_df
        # sample_row = sample_df[sample_df["filenameid"] == str(filename)]
        
        # # Filter the row corresponding to the current filename in data_dict
        # row_data = next(row for row in data_dict if row["Original file name (e.g. 36951253)"] == filename)
        
        gen_filename = glob.glob(os.path.join(base_path, f"generated/{filename}*.txt"))[0]
        orig_filename = glob.glob(os.path.join(base_path, f"original/{filename}*.txt"))[0]
        
        with open(gen_filename, "r") as f:
            discharge_summary = f.read()
        
        with open(orig_filename, "r") as f:
            clinical_case = f.read()
        
        df_human["filenameid"] = df_human["filenameid"].astype(str)
        row_data = df_human[df_human["filenameid"] == str(filename)].to_dict(orient="records")[0]
        # print(row_data)
        
        # clinical_case = row_data["text_orig"].values[0]
        # discharge_summary = row_data["text_gen"].values[0]
        
        
        # Extract all evaluation scores
        evaluation_scores = {
            "Content Relevance": row_data["Content Relevance"], 
                "Information Completeness": row_data["Information Completeness "], 
                "Clarity and Structure": row_data["Clarity and Structure "], 
                "Content Accuracy": row_data["Content Accuracy "], 
                "Hallucinations": row_data["Hallucinations "], 
                "Impact of Hallucinations": row_data["Impact of Hallucinations "], 
                "Relevance to Practice": row_data["Relevance to Practice "], 
                "Overall Quality": row_data["Overall Quality "],
                "Positive highlights of generation process": row_data["Positive highlights: Describe what aspects of the synthetic discharge summaries resemble the best real EHRs? (Empty if nothing remarkable)"],
                "Negative highlights of generation process": row_data["Negative highlights: Which aspects of the synthetic discharge summaries do not resemble well real EHRs? (Empty if nothing remarkable)"],
                "Other comments on Generated/Original data sources": row_data["Other Comments: Do you have any other feedback or comment on the generated synthetic discharge summaries or in the original cases? (Empty if nothing remarkable)"]
        }

        evaluation_scores = json.dumps(evaluation_scores)
        
        example = f"""EXAMPLE
            >>>
            Clinical Case : {clinical_case}
            Discharge Summary : {discharge_summary}
            Evaluation Scores : {evaluation_scores}
            >>>\n\n
            """
            
            
            # # Append the example to the list
        # print(example)
        examples.append(example)
        # print(example)
    return examples