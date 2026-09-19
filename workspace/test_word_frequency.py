import os
from workspace.word_freq import word_frequency

def test_word_frequency():
    sample_text = "Hello, world! Hello AI."
    sample_file = "sample.txt"

    # Create sample file
    with open(sample_file, 'w', encoding='utf-8') as f:
        f.write(sample_text)

    # Call the function
    result = word_frequency(sample_file)

    # Print the result
    print(result)

    # Clean up
    os.remove(sample_file)

if __name__ == '__main__':
    test_word_frequency()
