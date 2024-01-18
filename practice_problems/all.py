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

def fib(n: int) -> int:
    if n <= 1:
        return n
    
    fib_vals = [0, 1]

    for i in range(2, n+1):
        fib_vals.append(fib_vals[i-1] + fib_vals[i-2])

    return fib_vals[n]

def two_sum(nums: List[int], target:int) -> List[int]:

    if len(nums) == 0:
        return [-1, -1]

    diff_dict = {}
    # For each number in nums
    for i, num in enumerate(nums):
        if num > target:
            continue
        # Subtract difference of nums_i from target
        diff = target - num
        # Check if the diff_result is in diff_dict
        if diff in diff_dict:
            return sorted([i, int(diff_dict[diff])])
            # if yes, return both values
        else:
            # if no, save to diff dict and continue
            diff_dict[num] = i    
    
    return [-1, -1]

def binary_search(arr: List[int], target: int) -> int:
    left_index = 0
    right_index = len(arr) - 1

    while left_index <= right_index:
        pivot = (left_index + right_index) // 2

        if target == arr[pivot]:
            return pivot
        elif target < arr[pivot]:
            right_index = pivot - 1
        else:
            left_index = pivot + 1

    return -1

def is_empty(stack):
    return len(stack) == 0

def pop(stack):
    if not is_empty(stack):
        return stack.pop()
    else:
        return None

def is_valid_parentheses(input_str: str) -> bool:

    paren_stack = []
    paren_dict = {
        ')': '(',
        '}': '{',
        ']': '[',
    }

    for i in input_str:
        if i in ['(', '[', '{']:
            paren_stack.append(i)
        else:
            top_element = paren_stack.pop()
            if top_element != paren_dict[i]:
                print("False")
                return False

    print(is_empty(paren_stack))
    return is_empty(paren_stack)

def index_diff(start_index: int, end_index: int) -> int:
    return end_index - start_index

def length_of_lis(nums: List[int]) -> int:
    start_index = 0
    end_index = 0
    max_val = -1
    max_index_diff = -1

    for i, num in enumerate(nums):
        print("Val", num)
        if num >= max_val:
            max_val = num
            end_index = i
        else:
            max_val = num
            start_index = i
            end_index = i

        curr_index_diff = index_diff(start_index, end_index)
        print("Index Diff", curr_index_diff)
        if curr_index_diff > max_index_diff:
            max_index_diff = curr_index_diff

        print("Start Index", start_index)
        print("End Index", end_index)

    return max_index_diff + 1 # Bring back to length space

def length_of_lis_rev(nums: List[int]) -> int:
    if not nums:
        return 0

    n = len(nums)
    dp = [1] * n  # Initialize an array to store the length of LIS ending at each index

    for i in range(1, n):
        for j in range(i):
            if nums[i] > nums[j]:
                dp[i] = max(dp[i], dp[j] + 1)
        print(dp)


    return max(dp)

def partition(nums: List[int], start_index: int, end_index: int) -> int:

    print(nums)
    pivot = nums[end_index]

    # Index of the smaller element
    i = start_index - 1

    for j in range(start_index, end_index):
        if nums[j] <= pivot:
            i += 1
            nums[i], nums[j] = nums[j], nums[i]

    nums[i + 1], nums[end_index] = nums[end_index], nums[i + 1]

    return i + 1



def quick_sort(nums: List[int], start_index:int, end_index:int) -> List[int]:

    if len(nums) < 2:
        return nums

    if start_index < end_index:
        
        # Partition elements and get pivot
        pivot_index = partition(nums, start_index, end_index)
        print(nums)

        quick_sort(nums, start_index, pivot_index - 1)
        quick_sort(nums, pivot_index + 1, end_index)

    print(nums)


def longest_common_prefix(words: List[str]) -> str:
    """
    Take in a list of strings and return the longest prefix
    """
    base_word = words[0]
    result_list = [1] * len(base_word)
    result_str = ""
    max_val = -1

    for word in words[1:]:
        for i in range(len(word)):
            if base_word[i] == word[i]:
                result_list[i] += 1
                if result_list[i] >= max_val:
                    max_val = result_list[i]
            else:
                break

    for i in range(len(result_list)):
        if result_list[i] == max_val:
            result_str = result_str + (base_word[i])
            
    return result_str

def is_empty(stack: List[str]) -> bool:
    return len(stack) == 0
    
def pop(stack: List[str]) -> Optional[str]:
    if not is_empty(stack):
        return stack.pop()
    else:
        return None

def is_open_paren(element: str) -> bool:
    return element in ["(", "[", "{"]
    
def is_matching_close_paren(element: str, top_element: str) -> bool:
    if element == ")":
        return top_element == "("
    elif element == "]":
        return top_element == "["
    elif element == "}":
        return top_element == "{"
    else:
        raise ValueError(f"Top Element {top_element} Is not a valid character")


def is_valid(elements: str) -> bool:

    # Intitialize Stack
    stack = []

    # For each element in Parens
    for element in elements:
        if is_open_paren(element):
            stack.append(element)
        else:
            # Pop element on stack
            top_element = pop(stack)
            # If the element on stack exists
            if top_element:
                if not is_matching_close_paren(element, top_element):
                    # If not a matching paren, return false
                    return False
            else:
                return False

    # If it is a valid string, the length of stack should be 0
    return is_empty(stack)

MIN_INDEX = 0
MAX_INDEX = 1

def is_empty(stack: List[str]) -> bool:
    return len(stack) == 0
    
def pop(stack: List[str]) -> Optional[str]:
    if not is_empty(stack):
        return stack.pop()
    else:
        return None

def determine_overlap(prev_interval, current_interval) -> List[List[int]]:
    """
    Determine if prev_interval and current_interval are overlapping,
    if so return merged, otherwise return both
    """
    if current_interval[MIN_INDEX] <= prev_interval[MAX_INDEX]:
        return [[prev_interval[MIN_INDEX], current_interval[MAX_INDEX]]]
    else:
        return [prev_interval, current_interval]


def merge_intervals(intervals: List[List[int]]) -> List[List[int]]:
    # Assure there are more than one interval
    if len(intervals) < 2:
        return intervals
    
    # Initialize merged intervals with first interval
    merged_intervals = [intervals[0]]

    # Go Through each interval starting at the second interval
    for i, interval in enumerate(intervals[::1]):
        prev_interval = merged_intervals.pop()
        # Determine if there is an overlap
        evaluated_intervals = determine_overlap(prev_interval, interval)
        # Add to merged intervals
        merged_intervals.extend(evaluated_intervals)

    return merged_intervals


def two_sum(nums: List[int], target: int) -> List[int]:
    # If there are less than two elements
    if len(nums) < 1:
        # Return empty list
        return []

    # Dictionary for storing differences
    difference_dict = {}

    for i, num in enumerate(nums):
        # Find the difference between the target and current index
        difference = target - num

        if difference in difference_dict:
            # Return the found index and the current index
            return [difference_dict[difference], i]
        else:
            # Store the difference
            difference_dict[num] = i

    # If no solution found, return empty list
    return []

def substring_len(start_index: int, end_index: int) -> int:
    return end_index - start_index

def length_of_longest_substring(letters: str) -> str:
    # Validate input
    if len(letters) < 2:
        return letters

    sub_l_index = 0 # Substring Left Index
    sub_r_index = 0 # Substring Right Index
    result_substring = "" # Longest Substring to return
    found_letters_dict = {} # Dictionary for storing found values

    for i, letter in enumerate(letters):
        # If letter has not been found
        if letter not in found_letters_dict:
            # add letter to dictionary
            found_letters_dict[letter] = 1
            # Increase max right index
            sub_r_index += 1
            # Check if longest substring
            if substring_len(sub_l_index, sub_r_index) >= len(result_substring):
                # Update Result Substring
                result_substring = letters[sub_l_index: sub_r_index]
        # If it has already been found
        else:
            # Reset indexes
            sub_l_index = i
            sub_r_index = i
            # Reset found letters
            found_letters_dict = {}

    # Return substring
    return result_substring

def calculate_area(i, j, ai, aj):
    width = j - i
    height = min(ai, aj)
    return width * height

def max_area(heights: List[int]) -> int:

    result_list = [0] * len(heights)
    max_index = 0

    # Go through each element after first index since it resolves to 0
    for i, height in enumerate(heights):
        print(f"Max Index: {max_index}")
        print(f"Max height: {heights[max_index]}")
        print(f"Current Index: {i}")
        print(f"Current height: {height}")
        result_list[i] = calculate_area(
            max_index, 
            i,
            heights[max_index],
            heights[i],
        )
        print(f"Area: {result_list[i]}")
        
        if result_list[i] >= result_list[max_index]:
            max_index = max(height, heights[max_index])

    return max(result_list)


# Example 1
max_area([1,8,6,2,5,4,8,3,7])
# Output: 49

# # Example 2
# max_area([1,1])
# # Output: 1

# # Example 3
# max_area([4,3,2,1,4])
# # Output: 16

# # Example 4
# max_area([1,2,1])
# # Output: 2


# # Example 1
# print(length_of_longest_substring("abcabcbb"))
# # Output: 3  # ("abc" is the longest substring without repeating characters)

# # Example 2
# print(length_of_longest_substring("bbbbb"))
# # Output: 1  # ("b" is the longest substring without repeating characters)

# # Example 3
# print(length_of_longest_substring("pwwkew"))
# # Output: 3  # ("wke" is the longest substring without repeating characters)

# # Example 1
# nums = [2, 7, 11, 15]
# target = 9
# print(two_sum(nums, target))
# # Output: [0, 1]

# # Example 2
# nums = [3, 2, 4]
# target = 6
# print(two_sum(nums, target))
# # # Output: [1, 2]

# # # Example 3
# nums = [3, 3]
# target = 6
# print(two_sum(nums, target))
# # Output: [0, 1]
    

# # Example 1
# intervals = [[1,3],[2,6],[8,10],[15,18]]
# print(merge_intervals(intervals))
# # Output: [[1,6],[8,10],[15,18]]

# # Example 2
# intervals = [[1,4],[4,5]]
# print(merge_intervals(intervals))
# # Output: [[1,5]]

# # Example 1
# res = is_valid("()")  # Output: True
# print(f"Result: {str(res)}") 

# # Example 2
# res = is_valid("()[]{}")  # Output: True
# print(f"Result: {str(res)}") 

# # # Example 3
# res = is_valid("(]")  # Output: False
# print(f"Result: {str(res)}") 

# # # Example 4
# res = is_valid("([)]")  # Output: False
# print(f"Result: {str(res)}") 

# # # Example 5
# res = is_valid("{[]}")  # Output: True
# print(f"Result: {str(res)}") 

# strs = ["flower", "flow", "flight"]
# longest_common_prefix(strs)

# nums = [2, 1, 6 , 10, 4]
# quick_sort([2, 1, 6 , 10, 4], 0, len(nums) - 1)

# nums = [10, 9, 2, 3, 7, 101, 18]
# print(length_of_lis_rev(nums))  # Output: 4 (the LIS is [2, 3, 7, 101])

# is_valid_parentheses("()")  # Output: True
# is_valid_parentheses("()[]{}")  # Output: True
# is_valid_parentheses("(]")  # Output: False
# is_valid_parentheses("([)]")  # Output: False
# is_valid_parentheses("{[]}")  # Output: True

# arr = [1, 2, 3, 4, 5, 6, 7, 8, 9]
# target = 7
# print(f"Answer: {binary_search(arr, target)}")


# Two Sums
# nums = [2, 7, 11, 15]
# target = 9
# print(two_sum(nums, target))

# Is Palindrome
# print(is_palindrome("mike"))

# Fibbonacci
# print(fib(8))

# racecar
# 0123456

# kook
