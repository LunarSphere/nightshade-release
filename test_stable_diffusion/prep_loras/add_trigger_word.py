### add specified trigger for lora training to "text" col in a given csv file
import argparse
import pandas as pd
def add_trigger_word_to_csv(input_csv: str, output_csv: str, trigger_word: str) -> None:
    df = pd.read_csv(input_csv)
    if 'text' not in df.columns:
        raise ValueError("Input CSV must contain a 'text' column.")
    df['text'] = df['text'].astype(str) + f", {trigger_word}"
    df.to_csv(output_csv, index=False)
    print(f"Trigger word '{trigger_word}' added to 'text' column and saved to {output_csv}")
def main():
    parser = argparse.ArgumentParser(description="Add trigger word to 'text' column in CSV.")
    parser.add_argument("input_csv", type=str, help="Path to input CSV file")
    parser.add_argument("output_csv", type=str, help="Path to output CSV file")
    parser.add_argument("trigger_word", type=str, help="Trigger word to add")
    args = parser.parse_args()
    add_trigger_word_to_csv(args.input_csv, args.output_csv, args.trigger_word)
if __name__ == "__main__":
    main()

# example usage: python add_trigger_word.py input.csv output.csv "triggerword"