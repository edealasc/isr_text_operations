import json
import re
import string
from collections import Counter
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import linregress
from datasets import load_dataset

# For this project, we saved the output of each step in text opeartions in separate json files for clarity.

# This project uses the reuters dataset from huggingface, which is a collection of news articles.

class TextOperations:
    def fetch_and_save_reuters(output_path):
        #loading the dataset from huggingface
        dataset = load_dataset("ucirvine/reuters21578", "ModApte")
        train_dataset = dataset['train']
        test_dataset = dataset['test']
        cleaned_train_data = [{"title": entry["title"], "text": entry["text"]} for entry in train_dataset]
        cleaned_test_data = [{"title": entry["title"], "text": entry["text"]} for entry in test_dataset]

        all_data = cleaned_train_data + cleaned_test_data
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(all_data, f, ensure_ascii=False, indent=4)
        print(f"dataset saved.")


    def save_cleaned_reuters_dataset(input_path, output_path):
        #normalization (lowercasing)
        with open(input_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        for entry in data:
            entry["title"] = entry["title"].lower()
            entry["text"] = entry["text"].lower()
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=4)
        print("normalization finished")

    def clean_and_tokenize(text):
        #removing punctuation and digits
        text = text.lower()
        tokens = re.findall(r'\b[a-z]+\b', text)
        return tokens

    def tokenize_dataset(input_path, output_path):
        #tokenization
        with open(input_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        for entry in data:
            entry["title"] = TextOperations.clean_and_tokenize(entry["title"])
            entry["text"] = TextOperations.clean_and_tokenize(entry["text"])
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=4)
        print("tokenization complete")

    def remove_duplicates(input_path, output_path):
        # removing duplicates
        with open(input_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        for entry in data:
            entry["title"] = list(set(entry["title"]))
            entry["text"] = list(set(entry["text"]))
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=4)
        print("duplicates removed")

    #counts frequency of words
    def word_frequencies(input_path, output_path):
        with open(input_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        all_words = []
        for entry in data:
            all_words.extend(entry["title"])
            all_words.extend(entry["text"])
        word_counts = Counter(all_words)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(dict(word_counts), f, indent=4)
        print("word frequency analysis finishd")

    #ranks words by frequency and plots the results
    def rank_frequencies(input_path, output_path, plot=True):

        with open(input_path, "r", encoding="utf-8") as f:
            word_freq = json.load(f)
        sorted_items = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)
        ranked_words = [
            {"rank": rank + 1, "word": word, "frequency": freq}
            for rank, (word, freq) in enumerate(sorted_items)
        ]
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(ranked_words, f, indent=4)
        print("Ranked words saved to", output_path)
        if plot:
            ranks = np.array([item["rank"] for item in ranked_words])
            frequencies = np.array([item["frequency"] for item in ranked_words])
            plt.figure(figsize=(8, 6))
            plt.loglog(ranks, frequencies, marker=".", linestyle="none", label="Observed")
            plt.title("Zipf's Law: Frequency vs. Rank")
            plt.xlabel("Rank (log scale)")
            plt.ylabel("Frequency (log scale)")
            plt.grid(True, which="both", ls="--", lw=0.5)
            plt.legend()
            plt.tight_layout()
            plt.show()
            log_ranks = np.log(ranks)
            log_freqs = np.log(frequencies)
            slope, intercept, r_value, p_value, std_err = linregress(log_ranks, log_freqs)
            print(f"zipf's law fit: slope={slope:.2f}, R^2={r_value**2:.3f}")

    #luhn's method to select index terms
    def luhn_index_terms(input_path, output_path, upper_percent=0.05, lower_cutoff=2):

        with open(input_path, "r", encoding="utf-8") as f:
            word_freq = Counter(json.load(f))
        unique_words = len(word_freq)
        upper_cutoff_count = int(unique_words * upper_percent)
        upper_cutoff_words = set([word for word, _ in word_freq.most_common(upper_cutoff_count)])
        lower_cutoff_words = {word for word, freq in word_freq.items() if freq < lower_cutoff}
        index_terms = [
            word for word in word_freq
            if word not in upper_cutoff_words and word not in lower_cutoff_words
        ]
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump({
                "index_terms": index_terms,
                "stats": {
                    "unique_words": unique_words,
                    "upper_cutoff_removed": len(upper_cutoff_words),
                    "lower_cutoff_removed": len(lower_cutoff_words),
                    "remaining_terms": len(index_terms)
                }
            }, f, indent=2)
        print("Luhn’s Method Applied:")
        print(f"- Upper Cutoff (Top {int(upper_percent*100)}% unique words): Removed {len(upper_cutoff_words)} words")
        print(f"- Lower Cutoff (Words with freq < {lower_cutoff}): Removed {len(lower_cutoff_words)} words")
        print(f"- Final Index Terms: {len(index_terms)} words")



if __name__ == "__main__":
    # 0. Fetch and save Reuters dataset from Hugging Face
    TextOperations.fetch_and_save_reuters("reuters_dataset.json")
    # 1. Lowercase and save
    TextOperations.save_cleaned_reuters_dataset("reuters_dataset.json", "reuters_dataset_lower.json")
    # 2. Tokenize
    TextOperations.tokenize_dataset("reuters_dataset_lower.json", "tokenized_reuters_dataset.json")
    # 3. Remove duplicates
    TextOperations.remove_duplicates("tokenized_reuters_dataset.json", "reuters_dataset_no_duplicates.json")
    # 4. Word frequencies
    TextOperations.word_frequencies("reuters_dataset_no_duplicates.json", "word_frequencies.json")
    # 5. Rank frequencies and plot
    TextOperations.rank_frequencies("word_frequencies.json", "ranked_words.json", plot=True)
    # 6. Luhn index terms
    TextOperations.luhn_index_terms("word_frequencies.json", "luhn_results_with_cutoffs.json")
