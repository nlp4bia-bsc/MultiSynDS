import os
import sys
import yaml
sys.path.append(".")
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import pearsonr
from tqdm import tqdm
tqdm.pandas()

from src.data import files_to_df, Prompt, create_examples
from src.generative_models import LlamaInstruct
from src.generate import safe_generate
from src.utils import setup_logger, log_info, path_with_datetime, load_config, log_config

# Load configuration
config = load_config()

N_EXPECTED_SAMPLES = config["N_EXPECTED_SAMPLES"]
N_EXAMPLES = config["N_EXAMPLES"]
MODEL_ID = config["MODEL_ID"]
SOURCE_PATH = config["SOURCE_PATH"]
TEMPLATES_PATH = config["TEMPLATES_PATH"]
OUTPUT_PATH = path_with_datetime(config["OUTPUT_PATH"])

# Ensure output directory exists
if not os.path.exists(OUTPUT_PATH):
    os.makedirs(OUTPUT_PATH, exist_ok=True)

# write config to file in OUTPUT_PATH
with open(os.path.join(OUTPUT_PATH, "config.yaml"), "w") as f:
    yaml.dump(config, f)
        
# Setup logger
setup_logger(os.path.join(OUTPUT_PATH, "app.log"))
log_config(config)

def load_file_content(filepath):
    """Safely load text file content."""
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"File not found: {filepath}")
    with open(filepath, "r") as file:
        return file.read().strip()  # Strip trailing spaces

def load_datasets():
    """Load generated, original, human evaluation, and automatic evaluation datasets.
    
    Returns:
        df_pairs: pd.DataFrame, pairs of generated and original samples
        df_human: pd.DataFrame, human evaluation scores
        df_auto: pd.DataFrame, automatic evaluation scores
    """
    
    # 1. Load generated and original datasets and format it
    df_gen = files_to_df(os.path.join(SOURCE_PATH, "generated"))
    df_gen["filenameid"] = df_gen["filenameid"].str.replace("_transformed_step1", "")
    
    df_orig = files_to_df(os.path.join(SOURCE_PATH, "original"))
    df_pairs = df_orig.merge(df_gen, on="filenameid", suffixes=("_orig", "_gen"))
    assert len(df_pairs) == N_EXPECTED_SAMPLES, f"Expected {N_EXPECTED_SAMPLES} samples, got {len(df_pairs)}"
    
    df_pairs.rename(columns={"text_orig": "clinical_case", "text_gen": "discharge_summary"}, inplace=True)
    
    # 2. Load human evaluation dataset and format it
    
    # Input: human_eval.csv (From Google Forms)
    # Timestamp,Email Address,Original file name (e.g. 36951253),Overall validation [Content Relevance],Overall validation [Information Completeness],
    # Overall validation [Clarity and Structure],Overall validation [Content Accuracy],Overall validation [Hallucinations],Overall validation [Impact of Hallucinations],
    # Overall validation [Relevance to Practice],Overall validation [Overall Quality],
    # Positive highlights: Describe what aspects of the synthetic discharge summaries resemble the best real EHRs? (Empty if nothing remarkable),
    # Negative highlights: Which aspects of the synthetic discharge summaries do not resemble well real EHRs? (Empty if nothing remarkable),
    # Other Comments: Do you have any other feedback or comment on the generated synthetic discharge summaries or in the original cases? (Empty if nothing remarkable)

    df_human = pd.read_csv(os.path.join(SOURCE_PATH, "human_eval.csv")).rename(columns={"Original file name (e.g. 36951253)": "filenameid"}).drop(columns=["Email Address", "Timestamp"]).fillna("")
    d_score_cols = {
                            "Overall validation [Content Relevance]": "Content Relevance",
                            "Overall validation [Information Completeness]": "Information Completeness",
                            "Overall validation [Clarity and Structure]": "Clarity and Structure",
                            "Overall validation [Content Accuracy]": "Content Accuracy",
                            "Overall validation [Hallucinations]": "Hallucinations",
                            "Overall validation [Impact of Hallucinations]": "Impact of Hallucinations",
                            "Overall validation [Relevance to Practice]": "Relevance to Practice",
                            "Overall validation [Overall Quality]": "Overall Quality",
                            "Positive highlights: Describe what aspects of the synthetic discharge summaries resemble the best real EHRs? (Empty if nothing remarkable)": "Positive highlights",
                            "Negative highlights: Which aspects of the synthetic discharge summaries do not resemble well real EHRs? (Empty if nothing remarkable)": "Negative highlights",
                            "Other Comments: Do you have any other feedback or comment on the generated synthetic discharge summaries or in the original cases? (Empty if nothing remarkable)": "Other Comments"
    }

    df_human.rename(columns=d_score_cols, inplace=True)
    df_human.rename(columns={"Original file name (e.g. 36951253)": "filenameid"}, inplace=True)
    df_human["human_score"] = df_human.drop(columns=["filenameid"]).to_dict(orient="records")

    
    # Output: df_human
    # | filenameid |                  human_score                     |
    # | 33857916   | {'Content Relevance': 1, 'Information Complete...|
    
    # 3. Load automatic evaluation dataset and format it
    
    # Input: auto_eval.csv (From Google Forms)
    # filename,precision,recall,f1,tp,fp,fn,cluster

    df_auto = pd.read_csv(os.path.join(SOURCE_PATH, "auto_eval.csv")).drop(columns=["cluster"]).rename(columns={"filename": "filenameid"})
    df_auto["auto_score"] = df_auto.drop(columns=["filenameid"]).to_dict(orient="records")
    
    # Ensure filenameid is string
    df_pairs["filenameid"] = df_pairs["filenameid"].map(str)
    df_human["filenameid"] = df_human["filenameid"].map(str)
    df_auto["filenameid"] = df_auto["filenameid"].map(str)
    
    # Output: df_auto
    # | filenameid |                  auto_score                     |
    # | 33857916   | {'precision': 0.5, 'recall': 0.5, 'f1': 0.5,...|
    
    return df_pairs, df_human, df_auto

def select_examples(df_prompt, n=5, seed=42, examples_ids=None):
    """Select a few examples for few-shot learning."""
    
    if not examples_ids:
        example_filenames = df_prompt.sample(n, random_state=seed)["filenameid"].tolist()
    else:
        example_filenames = df_prompt[df_prompt["filenameid"].isin(examples_ids)]
    
    log_info(f"Selected Examples: {example_filenames}")
    
    return df_prompt[df_prompt["filenameid"].isin(example_filenames)]

def prepare_prompt_data(df_pairs, df_human, df_auto, examples_ids=None):
    """Merge datasets and prepare prompt inputs."""
    df_prompt = df_pairs.merge(df_human[["filenameid", "human_score"]], on="filenameid").merge(df_auto[["filenameid", "auto_score"]], on="filenameid")
    if examples_ids:
        df_prompt = df_prompt[df_prompt["filenameid"].isin(examples_ids)]
    return df_prompt

def generate_prompts(df_prompt, guidelines, template, examples):
    """Generate prompts for LLM processing."""
    df_prompt["prompts"] = df_prompt.progress_apply(lambda x: Prompt(
                                                                        guidelines=guidelines,
                                                                        template=template,
                                                                        clinical_case=x["clinical_case"],
                                                                        discharge_summary=x["discharge_summary"],
                                                                        examples=str(examples),
                                                                    ).text, axis=1)
    return df_prompt

def compute_correlations(df_human_preds, df_preds):
    """Compute Pearson correlation between human and model scores."""
    return pearsonr(df_human_preds["Overall Quality"], df_preds["Overall Quality"])

def plot_correlation_heatmap(df_hm, df_llm, suffixes=("_hm", "_llm")):
    """Plot a heatmap of correlations."""
    
    df_hm_llm_corr = df_hm.merge(df_llm, on="filenameid", suffixes=suffixes)
    df_hm_llm_corr = df_hm_llm_corr.select_dtypes(np.number).corr()
    
    fig, ax = plt.subplots(figsize=(10, 10))
    x_suffix, y_suffix = suffixes[0], suffixes[1]
    x_cols = [col for col in df_hm_llm_corr.columns if col.endswith(x_suffix)]
    y_cols = [col for col in df_hm_llm_corr.columns if col.endswith(y_suffix)]

    corr_matrix = df_hm_llm_corr.loc[x_cols, y_cols]
    sns.heatmap(corr_matrix, annot=True, cmap="coolwarm", fmt=".2f", vmin=-1, vmax=1, ax=ax)
    ax.set_title("Correlation Heatmap: Human vs LLM")
    fig.tight_layout()
    # plt.savefig(os.path.join(output_path, "correlation_heatmap.png"))   
    return fig , corr_matrix

def main():
    """Main execution function."""

    log_info(f"Starting evaluation of {MODEL_ID}")
    log_info("")
    log_info(f"Loading templates and guidelines from {TEMPLATES_PATH}")
    guidelines = load_file_content(os.path.join(TEMPLATES_PATH, "guidelines.txt"))
    template = load_file_content(os.path.join(TEMPLATES_PATH, "template.txt"))
    example_template = load_file_content(os.path.join(TEMPLATES_PATH,"example_template.txt"))
    system_prompt = load_file_content(os.path.join(TEMPLATES_PATH,"system.txt"))
    
    log_info(f"Loading datasets from {SOURCE_PATH}")
    df_pairs, df_human, df_auto = load_datasets()
    df_prompt = prepare_prompt_data(df_pairs, df_human, df_auto)
    
    log_info("")
    log_info(f"Selecting {N_EXAMPLES} examples for few-shot learning and generating prompts")
    df_examples = select_examples(df_prompt, n=N_EXAMPLES)
    few_shot_examples = df_examples.to_dict(orient="records")
    examples = create_examples(few_shot_examples, example_template=example_template)
    
    df_prompt = generate_prompts(df_prompt, guidelines, template, examples)

    log_info("")
    log_info("Starting generation of evaluation results")
    model = LlamaInstruct(MODEL_ID)
    
    # df_prompt = df_prompt.sample(3) # For testing

    df_prompt["generation"] = df_prompt["prompts"].progress_apply(lambda x: safe_generate(model, x, system=system_prompt, max_new_tokens=512, temperature=0.1))
    
    df_human_preds = pd.DataFrame(df_prompt["human_score"].tolist()).assign(filenameid=df_prompt["filenameid"].values)
    df_preds = pd.DataFrame(df_prompt["generation"].tolist()).assign(filenameid=df_prompt["filenameid"].values)
    
    df_examples_human = pd.DataFrame(df_examples["human_score"].tolist()).assign(filenameid=df_examples["filenameid"].values)
    df_examples_preds = df_prompt[df_prompt["filenameid"].isin(df_examples["filenameid"].values)]
    df_examples_preds = pd.DataFrame(df_examples_preds["generation"].tolist()).assign(filenameid=df_examples_preds["filenameid"].values)
    df_examples_auto = pd.DataFrame(df_examples["auto_score"].tolist()).assign(filenameid=df_examples["filenameid"].values)
    
    eval_metric = compute_correlations(df_human_preds, df_preds)
    log_info(f"Evaluation Metric: {eval_metric}")
    
    fig, df_hm_llm_corr = plot_correlation_heatmap(df_human_preds, df_preds)
    
    log_info("")
    log_info(f"Saving results to {OUTPUT_PATH}")
        
    fig.savefig(os.path.join(OUTPUT_PATH, "correlation_heatmap.png"))
    df_human_preds.to_csv(os.path.join(OUTPUT_PATH, "human_predictions.csv"), index=False)
    df_preds.to_csv(os.path.join(OUTPUT_PATH, "llm_predictions.csv"), index=False)
    df_hm_llm_corr.to_csv(os.path.join(OUTPUT_PATH, "correlation_matrix.csv"), index=True)
    df_prompt.to_csv(os.path.join(OUTPUT_PATH, "prompt_data.csv"), index=False)
    
    df_examples_human.to_csv(os.path.join(OUTPUT_PATH, "examples_human_eval.csv"), index=False)
    df_examples_preds.to_csv(os.path.join(OUTPUT_PATH, "examples_predictions.csv"), index=False)
    df_examples_auto.to_csv(os.path.join(OUTPUT_PATH, "examples_auto_eval.csv"), index=False)
    
    log_info(f"Results saved to {OUTPUT_PATH}")

if __name__ == "__main__":
    main()
