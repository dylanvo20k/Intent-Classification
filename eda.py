from datasets import load_dataset
import pandas as pd
from collections import Counter
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
import matplotlib.pyplot as plt
import seaborn as sns

dataset = load_dataset("AmazonScience/massive", "en-US")
# train, validation, test splits 
train_ds = dataset["train"]
val_ds   = dataset["validation"]
test_ds  = dataset["test"]
# converting to df for eda
train_df = pd.DataFrame(train_ds)
val_df   = pd.DataFrame(val_ds)
test_df  = pd.DataFrame(test_ds)

# missing data?
print("Missing Data (train)")
print(train_df[['utt', 'intent', 'scenario']].isnull().sum())

# record counts
print("\nDataset Sizes")
print(f"Train size:      {len(train_df)}")
print(f"Validation size: {len(val_df)}")
print(f"Test size:       {len(test_df)}")
print(f"Total:           {len(train_df) + len(val_df) + len(test_df)}")

# sentence length 
train_df['utt_length'] = train_df['utt'].apply(lambda x: len(x.split()))
print("\nSentence Length (word count)")
print(f"Average length: {train_df['utt_length'].mean():.1f} words")
print(f"Shortest:       {train_df['utt_length'].min()} words")
print(f"Longest:        {train_df['utt_length'].max()} words")

# finding common words 
all_words = " ".join(train_df['utt'].tolist()).lower().split()
# removing stop words 
meaningful_words = [w for w in all_words if w not in ENGLISH_STOP_WORDS]
word_count = Counter(meaningful_words)

print("\nMost Common Words")
for word, count in word_count.most_common(15):
    print(f"  {word:<12} {count}")

# visualizations 
# intent label names from the dataset features
intent_names = train_ds.features['intent'].names

# intent distribution with names
intent_counts = train_df['intent'].value_counts().sort_index()
intent_labels = [intent_names[i] for i in intent_counts.index]

plt.figure(figsize=(18, 8))
plt.barh(intent_labels, intent_counts.values, color='steelblue')
plt.xlabel("Count")
plt.title("Intent Distribution (60 classes)")
plt.tight_layout()
plt.savefig("eval/figures/intent_distribution.png", dpi=150)
plt.close()

# sentence length histogram 
plt.figure(figsize=(10, 5))
plt.hist(train_df['utt_length'], bins=30, edgecolor='black', color='steelblue')
plt.xlabel("Number of Words")
plt.ylabel("Count")
plt.title("Sentence Length Distribution")
plt.axvline(train_df['utt_length'].mean(), color='red', linestyle='--',
            label=f"Mean: {train_df['utt_length'].mean():.1f} words")
plt.legend()
plt.tight_layout()
plt.savefig("eval/figures/sentence_length.png", dpi=150)
plt.close()

# common words 
top_words = word_count.most_common(15)
words, counts = zip(*top_words)

plt.figure(figsize=(12, 6))
plt.bar(words, counts, color='steelblue', edgecolor='black')
plt.xlabel("Word")
plt.ylabel("Count")
plt.title("Top 15 Most Common Meaningful Words")
plt.tight_layout()
plt.savefig("eval/figures/top_words.png", dpi=150)
plt.close()