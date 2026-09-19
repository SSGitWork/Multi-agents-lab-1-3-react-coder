import string
from typing import Dict

def word_frequency(filepath: str) -> Dict[str, int]:
	"""Reads a text file and returns a dictionary mapping each word to its frequency.

	The function is case-insensitive and ignores punctuation.

	Args:
		filepath (str): Path to the text file.

	Returns:
		Dict[str, int]: Dictionary mapping words to their frequency count.
	"""
	freq = {}
	try:
		with open(filepath, 'r', encoding='utf-8') as file:
			for line in file:
				# Remove punctuation
				line = line.translate(str.maketrans('', '', string.punctuation))
				# Convert to lowercase and split into words
				words = line.lower().split()
				for word in words:
					freq[word] = freq.get(word, 0) + 1
	except FileNotFoundError:
		raise FileNotFoundError(f"File not found: {filepath}")
	except IOError as e:
		raise IOError(f"Error reading file {filepath}: {e}")
	return freq