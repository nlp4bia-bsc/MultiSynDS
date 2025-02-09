import pandas as pd
import os
import en_ner_bc5cdr_md
import swifter
from spacy.language import Language
from scispacy.linking import EntityLinker
from tqdm import tqdm
tqdm.pandas()

import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir)))
# sys.path.append("../src")
# print(os.getcwd())

from src.data import files_to_df
from src.scispacy_ie import entity_linker

model_name = "2step_transformation_dt4h_GeminiFlash"
output_path = "output/data"

gen_path = "data/2_generated"
orig_path = "data/1_original/txt"

lang = "en"
gen_path = os.path.join(gen_path, model_name, lang)

output_path_gen  = os.path.join(output_path, model_name, lang)
output_path_orig = output_path

if not os.path.exists(output_path_gen):
    os.makedirs(output_path_gen)

if not os.path.exists(output_path_orig):
    os.makedirs(output_path_orig)

df_orig = files_to_df(orig_path)

df_gen = files_to_df(gen_path)
df_gen["text_orig"] = df_gen["text"]
df_gen["text"] = df_gen["text_orig"].apply(lambda x: x.split("'text_to_transform': ")[-1][:-1].replace("'", ""))
df_gen.drop("text_orig", axis=1, inplace=True)

print("There are {} original and {} generated samples".format(len(df_orig), len(df_gen)))

nlp = en_ner_bc5cdr_md.load()

try:    # Register the EntityLinker component
    @Language.factory("umls_linker")
    def create_umls_linker(nlp, name):
        return EntityLinker(k=10, max_entities_per_mention=5, name="umls")
    nlp.add_pipe("umls_linker")
    
except ValueError:
    print("Entity linker already exists")
    
# df_ents_orig = df_orig.set_index("filenameid")["text"].swifter.apply(lambda x: entity_linker(nlp, x)).explode().apply(pd.Series)
# df_ents_orig.columns = ["span", "mention_class", "code", "term"]
# df_ents_orig.reset_index(inplace=True)
# df_ents_orig.to_csv(os.path.join(output_path_orig, "ents_orig_scispacy.csv"), index=False)


df_ents_gen = df_gen.set_index("filenameid")["text"].swifter.apply(lambda x: entity_linker(nlp, x)).explode().apply(pd.Series)
df_ents_gen.columns = ["span", "mention_class", "code", "term"]
df_ents_gen.reset_index(inplace=True)
df_ents_gen.to_csv(os.path.join(output_path_gen, "ents_gen_scispacy.csv"), index=False)