import json
import random
import string
import sys
import time

import fast_diff_match_patch as fdmp

DEFAULT_ALPHABET = string.ascii_letters + string.digits + string.whitespace

# speedtest1.txt: aaaabbccbcbbacc
# speedtest2.txt: cbbccbccbbbcbac

def gen_test_random(size1, size2, letters=DEFAULT_ALPHABET):
    """Generates two strings of size1 and size2 and writes each of them in a file.
        The strings are random."""
    text1 = "".join(random.choices(letters, k=size1))
    text2 = "".join(random.choices(letters, k=size2))
    return text1, text2

def gen_test_common_substring(size1, size2, common_len, letters=DEFAULT_ALPHABET):
    """Generates two strings of size1 and size2 and writes each of them in a file.
        The strings have a common SUBSTRING of length at least common_len. Here substring
        means a continuous common sequence."""
    assert(common_len <= size1 and common_len <= size2)
    
    common_text = "".join(random.choices(letters, k=common_len))

    offset = random.randint(0, size1 - common_len)
    text_before = "".join(random.choices(letters, k=offset))
    text_after = "".join(random.choices(letters, k=size1-offset-common_len))
    text1 = text_before + common_text + text_after

    offset = random.randint(0, size2 - common_len)
    text_before = "".join(random.choices(letters, k=offset))
    text_after = "".join(random.choices(letters, k=size2-offset-common_len))
    text2 = text_before + common_text + text_after

    return text1, text2

def gen_test_common_subsequence(size1, size2, common_len, letters=DEFAULT_ALPHABET):
    """Generates two strings of size1 and size2 and writes each of them in a file.
        The strings have a common SUBSEQUENCE of length at least common_len."""
    assert(common_len <= size1 and common_len <= size2)
    
    common_text = "".join(random.choices(letters, k=common_len))

    # Generate a text that has common_text as subsequence
    # The builder lets us insert strings at any position in common_text
    builder = [("", [])] + [(c, []) for c in common_text]
    text = "".join(random.choices(letters, k=size1 - common_len))
    positions = random.choices(range(common_len + 1), k=size1 - common_len)
    for c, pos in zip(text, positions):
        builder[pos][1].append(c)
    # Merge the parts in the builder
    text1 = "".join([part + "".join(suffix) for part, suffix in builder])

    # Generate a text that has common_text as subsequence
    # The builder lets us insert strings at any position in common_text
    builder = [("", [])] + [(c, []) for c in common_text]
    text = "".join(random.choices(letters, k=size2 - common_len))
    positions = random.choices(range(common_len + 1), k=size2 - common_len)
    for c, pos in zip(text, positions):
        builder[pos][1].append(c)
    # Merge the parts in the builder
    text2 = "".join([part + "".join(suffix) for part, suffix in builder])

    return text1, text2

def gen_test_random_utf8(size1, size2):
    """Generates two strings of size1 and size2 and writes each of them in a file.
        The strings are random. The strings are UTF8."""
    alphabet = "".join(chr(i) for i in range(0x110000))
    return gen_test_random(size1, size2, alphabet)

def gen_test_common_substring_utf8(size1, size2, common_len):
    """Generates two strings of size1 and size2 and writes each of them in a file.
        The strings have a common SUBSTRING of length at least common_len. Here substring
        means a continuous common sequence. The strings are UTF8."""
    alphabet = "".join(chr(i) for i in range(0x110000))
    return gen_test_common_substring(size1, size2, common_len, alphabet)

def gen_test_common_subsequence_utf8(size1, size2, common_len):
    """Generates two strings of size1 and size2 and writes each of them in a file.
        The strings have a common SUBSEQUENCE of length at least common_len.
        The strings are UTF8."""
    alphabet = "".join(chr(i) for i in range(0x110000))
    return gen_test_common_subsequence(size1, size2, common_len, alphabet)


def gen_test_binary_random(size1, size2, char1="a", char2="b"):
    """Generates two strings of size1 and size2 and writes each of them in a file.
        The strings have characters chosen from an alphabet of size 2.
    """
    assert len(char1) == 1
    assert len(char2) == 1
    letters = char1 + char2
    return gen_test_random(size1, size2, letters)

def benchmark_diff(text1, text2, timelimit):
    start = time.time()
    if timelimit:
        diffs = fdmp.diff(text1, text2, timelimit=25, cleanup="No", counts_only=True)
    else:
        diffs = fdmp.diff(text1, text2, cleanup="No", counts_only=True)
    count_old = sum(score for status, score in diffs if status != "=")
    end = time.time()
    delta_old = end - start

    start = time.time()
    if timelimit:
        count_new = fdmp.diff_count(text1, text2, timelimit=25)
    else:
        count_new = fdmp.diff_count(text1, text2)
    end = time.time()
    delta_new = end - start

    if delta_old < 24 and delta_new < 24:
        if len(text1) + len(text2) < 3000:
            assert count_old == count_new
        else:
            assert abs(count_old - count_new) / min(count_old, count_new) < 0.06

    return {"count_old": count_old, "delta_old": delta_old, "count_new": count_new, "delta_new": delta_new}

def run_test_n(generator, args, n_texts, repeat, timelimit=True):
    texts_list = [generator(*args) for _ in range(n_texts)]
    result = {"generator": generator.__name__, "delta_new": 0, "delta_old": 0, "repeat": repeat}
    while repeat > 0:
        temp_results = []
        for texts in texts_list[:repeat]:
            temp_results.append(benchmark_diff(*texts, timelimit))
        repeat -= len(texts_list)

        for temp_result in temp_results:
            result["delta_new"] += temp_result["delta_new"]
            result["delta_old"] += temp_result["delta_old"]

    return result

def benchmark_repeat(size1, size2, timelimit=True, repeat=1000, n_texts=50):
    batch_start = time.time()
    print(f"Benchmark repeat batch {(size1, size2)}")
    min_size = min(size1, size2)
    common_lens = [min_size // 2]

    results = []

    names = ("size1", "size2")
    input = (size1, size2)
    result = run_test_n(gen_test_random_utf8, input, n_texts, repeat, timelimit)
    result.update(dict(zip(names, input)))
    results.append(result)

    names = ("size1", "size2", "common_len")
    for common_len in common_lens:
        input = (size1, size2, common_len)

        result = run_test_n(gen_test_common_subsequence_utf8, input, n_texts, repeat, timelimit)
        result.update(dict(zip(names, input)))
        results.append(result)

    batch_end = time.time()
    print(f"Batch time: {batch_end - batch_start:.2f}")
    return results

def benchmark_batch(size1, size2, timelimit=True):
    batch_start = time.time()
    print(f"Benchmark batch {(size1, size2)}")
    min_size = min(size1, size2)
    common_lens = [50, 100, min_size // 100, min_size // 10, min_size // 2, int(min_size * 0.99)]
    common_lens = [length for length in common_lens if length <= min_size]

    results = []

    names = ("size1", "size2")
    input = (size1, size2)
    
    """
    texts = gen_test_random(*input)
    result = benchmark_diff(*texts, timelimit)
    result.update(dict(zip(names, input)))
    result.update({"generator": "gen_test_random"})
    results.append(result)
    """

    texts = gen_test_random_utf8(*input)
    result = benchmark_diff(*texts, timelimit)
    result.update(dict(zip(names, input)))
    result.update({"generator": "gen_test_random_utf8"})
    results.append(result)

    texts = gen_test_binary_random(*input)
    result = benchmark_diff(*texts, timelimit)
    result.update(dict(zip(names, input)))
    result.update({"generator": "gen_test_binary_random"})
    results.append(result)

    names = ("size1", "size2", "common_len")
    for common_len in common_lens:
        input = (size1, size2, common_len)

        """
        texts = gen_test_common_substring(*input)
        result = benchmark_diff(*texts, timelimit)
        result.update(dict(zip(names, input)))
        result.update({"generator": "gen_test_common_substring"})
        results.append(result)
        """

        texts =gen_test_common_subsequence(*input)
        result = benchmark_diff(*texts, timelimit)
        result.update(dict(zip(names, input)))
        result.update({"generator": "gen_test_common_subsequence"})
        results.append(result)

        texts = gen_test_common_substring_utf8(*input)
        result = benchmark_diff(*texts, timelimit)
        result.update(dict(zip(names, input)))
        result.update({"generator": "gen_test_common_substring_utf8"})
        results.append(result)

        texts = gen_test_common_subsequence_utf8(*input)
        result = benchmark_diff(*texts, timelimit)
        result.update(dict(zip(names, input)))
        result.update({"generator": "gen_test_common_subequence_utf8"})
        results.append(result)

    batch_end = time.time()
    print(f"Batch time: {batch_end - batch_start:.2f}")
    return results

def quick_test():
    texts = gen_test_common_subsequence_utf8(12345, 15022, 1)
    result = benchmark_diff(*texts, timelimit=True)
    speedup = f"{float(result['delta_old']) / float(result['delta_new']):.2f}"
    print(f"text1_len: 12345; text2_len: 15022; time_new: {result['delta_new']}; time_old: {result['delta_old']}; speedup: {speedup}")

    texts = gen_test_common_subsequence_utf8(40345, 50022, 1)
    result = benchmark_diff(*texts, timelimit=True)
    speedup = f"{float(result['delta_old']) / float(result['delta_new']):.2f}"
    print(f"text1_len: 40345; text2_len: 50022; time_new: {result['delta_new']}; time_old: {result['delta_old']}; speedup: {speedup}")

    texts = gen_test_common_subsequence_utf8(12345, 15022, 5000)
    result = benchmark_diff(*texts, timelimit=True)
    speedup = f"{float(result['delta_old']) / float(result['delta_new']):.2f}"
    print(f"text1_len: 12345; text2_len: 15022; time_new: {result['delta_new']}; time_old: {result['delta_old']}; speedup: {speedup}")

    texts = gen_test_common_subsequence_utf8(60345, 70022, 30000)
    result = benchmark_diff(*texts, timelimit=True)
    speedup = f"{float(result['delta_old']) / float(result['delta_new']):.2f}"
    print(f"text1_len: 60345; text2_len: 70022; time_new: {result['delta_new']}; time_old: {result['delta_old']}; speedup: {speedup}")

    texts = gen_test_common_subsequence_utf8(60345, 70022, 55000)
    result = benchmark_diff(*texts, timelimit=True)
    speedup = f"{float(result['delta_old']) / float(result['delta_new']):.2f}"
    print(f"text1_len: 60345; text2_len: 70022; time_new: {result['delta_new']}; time_old: {result['delta_old']}; speedup: {speedup}")

def do_repeat():
    results = []
    results += benchmark_repeat(1000, 1000, timelimit=False, repeat=5000)
    return results

def do_main():
    results = []
    try:
        results += benchmark_batch(10, 10, timelimit=False)
        results += benchmark_batch(10**6, 10**6)
        results += benchmark_batch(10**2, 10**2, timelimit=False)
        results += benchmark_batch(10**3, 10**3, timelimit=False)
        results += benchmark_batch(10**5, 10**5)
        results += benchmark_batch(10**3, 10**5)
        results += benchmark_batch(10**4, 10**5)
        results += benchmark_batch(10**4, 10**6)
        results += benchmark_batch(12345, 54321)
    except Exception:
        print("Benchmark hasn't finished successfully.")

    return results

def main():
    print("Quick test:")
    quick_test()
    print("Quick test done.")

    print("Start benchmark")

    if len(sys.argv) > 1 and sys.argv[1] == "repeat":
        print("Repeat mode.")
        results = do_repeat()
        with open("results_repeat.json", "w") as f:
            print("Storing results in results_repeat.json...")
            f.write(json.dumps(results, sort_keys=True, indent=4))
    else:
        results = do_main()
        with open("results.json", "w") as f:
            print("Storing results in results.json...")
            f.write(json.dumps(results, sort_keys=True, indent=4))

if __name__ == "__main__":
    main()
