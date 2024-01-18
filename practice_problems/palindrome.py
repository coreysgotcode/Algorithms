from re import I
from sre_compile import IN_UNI_IGNORE
from typing import List, Optional

def is_palindrome(word: str) -> bool:

    print(f"Checking if {word} is palindrome")
    word_len = len(word)
    if word_len == 0:
        print("Word param was empty")
        return False

    word = word.lower()
    pivot_index = int(word_len / 2)

    for i in range(0, pivot_index):
        if word[i] != word[(word_len - 1) - i]:
            return False

    return True

if __name__ == "__main__":
  print(is_palindrome("racecar")) 
  
