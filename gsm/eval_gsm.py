import json
import numpy as np
import time
import re
import argparse
from collections import defaultdict


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--rounds", default=2, type=int)
    parser.add_argument("--num-agents", default=2, type=int)
    parser.add_argument("--debug", action="store_true", help="Enable debug prints")
    return parser.parse_args()


def solve_math_problems(input_str):
    pattern = r"\d+\.?\d*"
    matches = re.findall(pattern, input_str)
    if matches:
        return matches[-1]
    return None


def parse_answer(input_str):
    pattern = r"\{([0-9.,$]*)\}"
    matches = re.findall(pattern, input_str)
    solution = None
    for match_str in matches[::-1]:
        solution = re.sub(r"[^0-9.]", "", match_str)
        if solution:
            break
    return solution


def compute_accuracy(gt, pred_solutions):
    answers = solve_math_problems(gt)
    if answers is None:
        return None

    pred_answers = []
    for pred_solution in pred_solutions:
        pred_answer = parse_answer(pred_solution)
        if pred_answer is None:
            pred_answer = solve_math_problems(pred_solution)
        pred_answers.append(pred_answer)

    pred_answer = most_frequent(pred_answers)

    if pred_answer is None:
        return 0

    try:
        if float(answers) == float(pred_answer):
            return 1
        else:
            return 0
    except ValueError:
        print(
            f"Error converting to float: answers={answers}, pred_answer={pred_answer}"
        )
        return 0


def most_frequent(List):
    if not List:
        return None
    return max(set(List), key=List.count)


if __name__ == "__main__":
    args = parse_arguments()
    response_dict = json.load(open(f"gsm_{args.num_agents}_{args.rounds}.json", "r"))

    questions = list(response_dict.keys())

    accuracies = defaultdict(list)

    for i, question in enumerate(questions, 1):
        responses, gt = response_dict[question]

        if args.debug:
            print(f"\n[DEBUG] Processing question {i}: {question}")
            print(f"[DEBUG] Ground truth: {gt.split('#')[-1].strip()}")

        for round in range(args.rounds):
            pred_solutions = []
            for i, response in enumerate(responses):
                # Every 2nd element contains a response for a different round
                pred_solution = response[round * 2 + 1]["content"]
                pred_solutions.append(pred_solution)

                if args.debug:
                    print(
                        f"[DEBUG] Round {round + 1}, Agent {i+1} solution: {pred_solution}"
                    )

            accurate = compute_accuracy(gt, pred_solutions)

            if accurate is not None:
                accuracies[round].append(float(accurate))
                if args.debug:
                    print(
                        f"[DEBUG] Round {round + 1}, Accuracy for this question: {accurate}"
                    )
            else:
                print(f"Round {round + 1}: accurate is None")
                print(f"Ground truth: {gt}")
                if args.debug:
                    print(
                        f"[DEBUG] Round {round + 1}, Failed to compute accuracy for this question"
                    )

        if args.debug:
            for round in range(args.rounds):
                print(
                    f"[DEBUG] Round {round + 1}, Current mean accuracy: {np.mean(accuracies[round])}"
                )

    # Compute final statistics for each round
    for round in range(args.rounds):
        mean_accuracy = np.mean(accuracies[round])
        std_error = np.std(accuracies[round]) / (len(accuracies[round]) ** 0.5)
        print(f"Round {round + 1}:")
        print(f"  Mean Accuracy: {mean_accuracy}")

    # Save results to a JSON file
    results = {
        f"round_{round+1}": {
            "mean_accuracy": float(np.mean(accuracies[round])),
            "standard_error": float(
                np.std(accuracies[round]) / (len(accuracies[round]) ** 0.5)
            ),
        }
        for round in range(args.rounds)
    }

    with open(f"gsm_performance_{args.num_agents}_{args.rounds}.json", "w") as f:
        json.dump(results, f, indent=4)

    print(f"Results saved to gsm_performance_{args.num_agents}_{args.rounds}.json")
