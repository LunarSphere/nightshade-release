import spacy
from transformers import pipeline

nlp = spacy.load("en_core_web_sm")

def llm_concept_extraction(caption: str) -> str:
    summarizer = pipeline("text2text-generation", model="facebook/bart-large-cnn")
    result = summarizer(f"Return only one or two words that describe the main subject: {caption}")
    return result[0]['generated_text']


def get_main_noun(caption):
    doc = nlp(caption)
    nouns = [token.text for token in doc if token.pos_ in ["NOUN", "PROPN"]]
    return nouns[0] if nouns else caption

print(get_main_noun("A photo of a red sports car on a city street."))



