import pandas as pd
import re


# Define a function to clean the lyrics text.
def clean_lyrics(lyrics):
    # Remove any text in square brackets, e.g., [Chorus] or [Verse 1].
    lyrics = re.sub(r'\[.*?\]', '', lyrics)
    # Remove any text in curly braces, e.g., {Bridge} or {Intro}.
    lyrics = re.sub(r'\{.*?\}', '', lyrics)
    # Remove commas (you can add other special characters here if needed)
    lyrics = lyrics.replace(',', '')
    # Remove identifying words (case-insensitive). You can expand this list as needed.
    lyrics = re.sub(r'\b(verse|chorus|bridge|intro|outro|hook)\b', '', lyrics, flags=re.IGNORECASE)
    # Clean up extra whitespace.
    lyrics = re.sub(r'\s+', ' ', lyrics).strip()
    return lyrics


# File paths.
input_file = 'song_lyrics_reduced.xlsx'  # Use the Excel file version.
output_file = 'balanced_song_lyrics.csv'

# Read the Excel file and load only the columns of interest.
df = pd.read_excel(input_file, usecols=['lyrics', 'tag'])

# Convert the tag column to string explicitly to avoid mixed types.
df['tag'] = df['tag'].astype(str)

# Apply cleaning to the 'lyrics' column.
df['lyrics'] = df['lyrics'].astype(str).apply(clean_lyrics)

# Set a random seed for reproducibility.
random_state = 42

# List to hold sampled DataFrames for each genre.
sampled_dfs = []

# Group by genre (tag) and sample 2000 songs per group.
for genre, group in df.groupby('tag'):
    if len(group) < 2000:
        print(f"Genre '{genre}' has only {len(group)} entries; using all available samples.")
        sampled_group = group
    else:
        sampled_group = group.sample(n=2000, random_state=random_state)
    sampled_dfs.append(sampled_group)

# Concatenate the sampled groups into a new balanced DataFrame.
balanced_df = pd.concat(sampled_dfs)

# Shuffle the DataFrame so that rows aren’t grouped by genre.
balanced_df = balanced_df.sample(frac=1, random_state=random_state).reset_index(drop=True)

# Save the balanced and cleaned dataset to a CSV file.
balanced_df.to_csv(output_file, index=False)

print("Balanced dataset created and saved to", output_file)
