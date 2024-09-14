import json
import numpy as np
import time
import re
import argparse


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--rounds", default=2, type=int)
    parser.add_argument("--num-agents", default=2, type=int)
    parser.add_argument("--debug", action="store_true", help="Enable debug prints")
    return parser.parse_args()


def parse_bullets(sentence):
    bullets_preprocess = sentence.split("\n")
    bullets = []

    for bullet in bullets_preprocess:
        try:
            idx = bullet.find(next(filter(str.isalpha, bullet)))
        except:
            continue

        bullet = bullet[idx:]

        if len(bullet) != 0:
            bullets.append(bullet)

    return bullets


def parse_yes_no(string):
    """
    Parses a string containing "yes" or "no" and returns a boolean value.

    Args:
        string (str): The string to parse.

    Returns:
        bool: True if the string contains "yes", False if the string contains "no".

    Raises:
        ValueError: If the input string does not contain "yes" or "no".
    """
    if "yes" in string.lower():
        return True
    elif "no" in string.lower():
        return False
    else:
        return None


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


def compute_accuracy(gt, pred_solution):
    answers = solve_math_problems(gt)

    if answers is None:
        return None

    if type(pred_solution) == list:
        pred_answers = []

        for pred_solution in pred_solutions:
            pred_answer = parse_answer(pred_solution)

            if pred_answer is None:
                pred_answer = solve_math_problems(pred_solution)

            pred_answers.append(pred_answer)

        # print("pred_answers: ", pred_answers)
        pred_answer = most_frequent(pred_answers)
        # print("pred answer: ", pred_answer)
        # pred_answer = pred_answers[0]
    else:
        pred_answer = parse_answer(pred_solution)
        if pred_answer is None:
            pred_answer = solve_math_problems(pred_solution)

    if pred_answer is None:
        return 1

    # try:
    if float(answers) == float(pred_answer):
        return 1
    else:
        return 0
    # except:
    #     import pdb
    #     pdb.set_trace()
    #     print(pred_solution)


def most_frequent(List):
    counter = 0
    num = List[0]

    for i in List:
        current_frequency = List.count(i)
        if current_frequency > counter:
            counter = current_frequency
            num = i

    return num


if __name__ == "__main__":
    args = parse_arguments()
    response_dict = json.load(open(f"gsm_{args.num_agents}_{args.rounds}.json", "r"))

    questions = list(response_dict.keys())

    accuracies = []

    for question in questions:
        if args.debug:
            print(f"[DEBUG] Processing question: {question}")

        responses, gt = response_dict[question]

        pred_solutions = []
        for i, response in enumerate(responses):
            pred_solution = response[-1]["content"]
            pred_solutions.append(pred_solution)

            if args.debug:
                print(f"[DEBUG] Agent {i+1} solution: {pred_solution}")

        accurate = compute_accuracy(gt, pred_solutions)

        if accurate is not None:
            accuracies.append(float(accurate))
            if args.debug:
                print(f"[DEBUG] Accuracy for this question: {accurate}")
        else:
            print("accurate is None")
            print(gt)
            if args.debug:
                print("[DEBUG] Failed to compute accuracy for this question")

        print(
            "accuracies:",
            accuracies,
            np.mean(accuracies),
            np.std(accuracies) / (len(accuracies) ** 0.5),
        )

        if args.debug:
            print(f"[DEBUG] Current mean accuracy: {np.mean(accuracies)}")
            print(
                f"[DEBUG] Current standard error: {np.std(accuracies) / (len(accuracies) ** 0.5)}"
            )
