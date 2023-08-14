import json
from collections import defaultdict
from statistics import mean, stdev

RESULTS_FILE = "results.json"
EXPECTED_KEYS = {"size1", "size2", "common_len", "generator", "delta_old", "delta_new", "count_old", "count_new"}
TIMEOUT_LIMIT = 25


def process(data):
    for test in data:
        keys = set(test.keys())
        if not keys.issubset(EXPECTED_KEYS):
            print(f"Unwanted data format: {test}")
            return

        
    timeout_old = [1 for test in data if float(test['delta_old']) >= TIMEOUT_LIMIT]
    timeout_new = [1 for test in data if float(test['delta_new']) >= TIMEOUT_LIMIT]

    print(f"{len(timeout_old)}/{len(data)} tests timed out with the old algorithm.")
    print(f"{len(timeout_new)}/{len(data)} tests timed out with the new algorithm.")

    real_data = [test for test in data if float(test['delta_new']) < TIMEOUT_LIMIT or float(test['delta_old']) < TIMEOUT_LIMIT]
    print(f"In {len(data) - len(real_data)}/{len(data)} tests, both algorithms timed out. Removing them...")

    old_len = len(real_data)
    real_data = [test for test in real_data if float(test['delta_new']) >= 1 or float(test['delta_old']) >= 1]
    print(f"In {old_len - len(real_data)}/{old_len} tests, the times were under 1 second. Removing them...")

    faster = [1 for test in real_data if float(test['delta_new']) < float(test['delta_old']) or float(test['delta_new']) < 0.01]
    slower = [test for test in real_data if not(float(test['delta_new']) < float(test['delta_old']) or float(test['delta_new']) < 0.01)]
    speedup = [float(test['delta_old']) / float(test['delta_new']) if float(test['delta_new']) > 0.001 else 1 for test in real_data]

    print(f"In {len(faster)}/{len(real_data)} tests, the new algorithm is faster or the same.")
    print(f"Tests where the new algorithm is slower: {slower}")

    print(f"Speedup mean: {mean(speedup):.3f}")
    print(f"Speedup stdev: {stdev(speedup):.3f}")

    generator_groups = defaultdict(list)
    for test in real_data:
        generator_groups[test["generator"]].append(test)

    print("")
    for generator, gen_data in generator_groups.items():
        speedup = [float(test['delta_old']) / float(test['delta_new']) if float(test["delta_new"]) > 0.02 else 1 for test in gen_data]
        print(f"{len(gen_data)} tests for {generator}")
        print(f"For {generator}, speedup mean: {mean(speedup):.3f}")
        print(f"For {generator}, speedup stdev: {stdev(speedup):.3f}\n")


    for test in real_data:
        test["complexity"] = test["count_new"] * max(test["size1"], test["size2"])
        test["new_complexity"] = test["count_new"] * test["count_new"]


def main():
    with open(RESULTS_FILE, "r", encoding="utf8") as f:
        data = json.load(f)

    process(data)

main()
