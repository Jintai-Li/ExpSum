import os
import json
import sys
import getopt
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor

from openai import OpenAI
from HarmonyAPI.KnowledgeBase.Knowledge_init import ContextAwareKnowledgeBase


# =========================================================
# Configuration 
# =========================================================

def setup_environment():
    os.environ["HF_ENDPOINT"] = "https://hf-mirror.example.com"
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"


def init_clients():
    """
    Initialize LLM clients.
    NOTE: API keys and model identifiers are intentionally removed.
    """
    generic_client = OpenAI(
        api_key=os.getenv("LLM_API_KEY"),
        base_url="https://your-llm-endpoint.example.com"
    )
    auxiliary_client = OpenAI(
        api_key=os.getenv("AUX_LLM_API_KEY"),
        base_url="https://your-aux-endpoint.example.com"
    )
    return generic_client, auxiliary_client




def gen_instruct1(aiResponse: str):
    instruct = '''
You are a Code Comment Grammar Correction Specialist. 
Your task is to receive one English comment and output a single-line corrected version that follows all grammar and formatting rules.

Instructions:
1. **Articles**: Insert missing "a", "an", or "the" wherever grammatically needed, including before technical terms.  
2. **Capitalization**: The first word must be capitalized; all other words are lowercase, even technical terms.  
3. **Punctuation**: Each comment must end with a period.  
4. **Noun Phrasing**: Avoid stacked nouns; rewrite them using possessive or "of" forms. Example: "atomic service module name" → "the module name of the atomic service."  
5. **Agreement & Consistency**: Fix subject-verb agreement and plural/singular mismatches.   
6. **Output format**: One single-line corrected English sentence only.

JSON input:
{
  "Input_Comment": ''' + aiResponse + ''',
  "Constraints": {
    "Articles": "Check and insert missing articles",
    "Capitalization": { "SentenceStart": "Must capitalize", "InsideSentence": "Lowercase only" },
    "Punctuation": "Every comment ends with a period",
    "Noun_Phrasing": "Rewrite noun stacks into possessive/of phrases",
    "Agreement": "Fix subject-verb and plural issues",
    "Format": "Keep tags and technical identifiers unchanged"
  }
}'''
    return instruct


def gen_instruct3(tag_list: str, official: str, knowledge: str, name: str):
    instruct3 = '''
You are a HarmonyOS API Comment Translator. 
You will receive one JSON input and must generate exactly one English API comment in a single line.

Instructions:
1. Analyze the API name and its "title" (e.g., "GetOrCreateLocalDir" → split into ["Get", "Or", "Create", "Local", "Dir"]) to select its technical terms that can be used in comment.
2. Use "API_Tags" to understand API information.
3. If "@enum" exists, the comment must describe an enumeration.
4. If "@type" exists with a data type (e.g., "Boolean", "Object"), describe only the variable’s meaning (not its actions). Apply the corresponding datatype template when available.
5. Always use terms from "Translation_Dictionary" exactly as they appear.
6. Output format: One single-line English sentence. No other text.

JSON input:
{
  "API_Metadata": {
    "API_Tags": ''' + tag_list + ''',
    "Chinese_Comment": ''' + official + '''
  },
  "Translation_Dictionary": ''' + knowledge + ''',
  "API_Name": ''' + name + ''',
}

Comment Constraints: {

  "@function_category": {

    "Field Function": {
      "if exist": [
        "The comment must describe only the semantic meaning of the field or state.",
        "The comment must not describe any action, execution, or side effect.",
        "Verbs implying behavior (e.g., perform, execute, trigger, handle) must not be used."
      ]
    },

    "Utility Function": {
      "if exist": [
        "The comment must describe the functional purpose of the utility.",
        "The comment may describe computation, conversion, or assistance behavior.",
        "The comment must not describe lifecycle events or callback invocation."
      ]
    },

    "Callback Function": {
      "if exist": [
        "The comment must describe when or under what condition the callback is invoked.",
        "The comment must emphasize the triggering context rather than internal logic.",
        "The comment must not describe the function as being actively called by users."
      ]
    }
  },

  "@enum": {
    "if exist": "The comment must explicitly mention that this API represents an enumeration."
  },

  "@type or @typedef": { 
    "if exist": { 
      "datatype_templates": { 
        "Boolean": "Indicates whether {X}.", 
        "Integer": "Represents the {X} value.", 
        "String": "Represents the {X} string.", 
        "Object": "Represents information about {X}.", 
        "Enum": "Represents the {X} enumeration." 
      } 
    } 
  }
}

'''
    return instruct3


# =========================================================
# Knowledge Retrieval
# =========================================================

def retrieve_knowledge(kb: ContextAwareKnowledgeBase, official_doc: str) -> str:
    entries = []
    for match in kb.search(official_doc):
        entries.append(f'''"{match['Chine Term']}" : "{match['English Term']}".\n''')
    return ";".join(entries)



def strip_official_comment(api_intro: dict):
    official = api_intro['apiIntro']['official_doc']
    name = api_intro['apiIntro']['title']
    del api_intro['apiIntro']['official_doc']
    return official, api_intro, name


# =========================================================
# Comment Generation Pipeline
# =========================================================

def generate_comments(input_file: str, output_file: str, knowledge_path: str, llm_client):
    with open(input_file, "r") as f:
        dataset = json.load(f)

    kb = ContextAwareKnowledgeBase(knowledge_path)

    with open(output_file, "w") as writer:
        for item in tqdm(dataset, desc="Generating comments"):
            official, intro, name = strip_official_comment(item)
            knowledge = retrieve_knowledge(kb, official)

            prompt = gen_instruct3(str(intro), official, knowledge, name)

            response = llm_client.chat.completions.create(
                model="GENERIC_LLM_MODEL",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.01,
                stream=True
            )

            comment = ""
            for chunk in response:
                if chunk.choices and hasattr(chunk.choices[0].delta, "content"):
                    comment += chunk.choices[0].delta.content or ""

            writer.write(comment.splitlines()[0] + "\n")


# =========================================================
# Entry Point
# =========================================================

def main(argv):
    setup_environment()
    llm_client, _ = init_clients()

    input_file, output_file = argv
    knowledge_path = "path/to/knowledge.json"  # PLACEHOLDER

    generate_comments(input_file, output_file, knowledge_path, llm_client)


if __name__ == "__main__":
    opts, _ = getopt.getopt(sys.argv[1:], "", ["input_file=", "output_file="])
    args = [v for _, v in opts]
    main(args)
