import json
import numpy as np
import random
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
import argparse


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--rounds", default=2, type=int)
    parser.add_argument("--num-agents", default=2, type=int)
    parser.add_argument("--evaluation", default=100, type=int)
    parser.add_argument("--seed", default=0, type=int)
    parser.add_argument("--temperature", default=0.7, type=float)
    parser.add_argument("--debug", action="store_true", help="Enable debug prints")
    return parser.parse_args()


def construct_message(agents, question, idx):
    if len(agents) == 0:
        return {
            "role": "user",
            "content": "Can you double check that your answer is correct. Please reiterate your answer, with your final answer a single numerical number, in the form \\boxed{{answer}}.",
        }

    prefix_string = "These are the solutions to the problem from other agents: "

    for agent in agents:
        agent_response = agent[idx]["content"]
        response = "\n\n One agent solution: ```{}```".format(agent_response)

        prefix_string = prefix_string + response

    prefix_string = (
        prefix_string
        + """\n\n Using the solutions from other agents as additional information, can you provide your answer to the math problem? \n The original math problem is {}. Your final answer should be a single numerical number, in the form \\boxed{{answer}}, at the end of your response.""".format(
            question
        )
    )
    return {"role": "user", "content": prefix_string}


def construct_assistant_message(completion):
    return {"role": "assistant", "content": completion}


def read_jsonl(path: str):
    with open(path) as fh:
        return [json.loads(line) for line in fh.readlines() if line]


if __name__ == "__main__":
    args = parse_arguments()
    torch.random.manual_seed(args.seed)
    random.seed(args.seed)
    agents = args.num_agents
    rounds = args.rounds

    if args.debug:
        print(f"[DEBUG] Number of agents: {agents}, Number of rounds: {rounds}")

    model = AutoModelForCausalLM.from_pretrained(
        "microsoft/Phi-3-mini-4k-instruct",
        device_map="cuda",
        torch_dtype="auto",
        trust_remote_code=True,
    )
    tokenizer = AutoTokenizer.from_pretrained("microsoft/Phi-3-mini-4k-instruct")
    pipe = pipeline("text-generation", model=model, tokenizer=tokenizer)
    generation_args = {
        "max_new_tokens": 500,
        "return_full_text": False,
        "do_sample": True,
        "top_p": 0.95,
        "temperature": args.temperature,
    }

    generated_description = {}

    questions = read_jsonl("gsm8k_test.jsonl")
    random.shuffle(questions)

    for data in questions[: args.evaluation]:
        question = data["question"]
        answer = data["answer"]

        if args.debug:
            print(f"[QUESTION] Processing question: {question}")

        agent_contexts = [
            [
                {
                    "role": "user",
                    "content": """Can you solve the following math problem? {} Explain your reasoning. Your final answer should be a single numerical number, in the form \\boxed{{answer}}, at the end of your response. """.format(
                        question
                    ),
                }
            ]
            for agent in range(agents)
        ]

        for round in range(rounds):
            if args.debug:
                print(f"[ROUND] Starting round {round + 1}")

            for i, agent_context in enumerate(agent_contexts):
                if args.debug:
                    print(f"[DEBUG] Processing agent {i + 1}")

                if round != 0:
                    agent_contexts_other = agent_contexts[:i] + agent_contexts[i + 1 :]
                    message = construct_message(
                        agent_contexts_other, question, 2 * round - 1
                    )
                    agent_context.append(message)

                print(f"[CONTEXT] Agent {i + 1} context: {agent_context}")
                completion = pipe(agent_context, **generation_args)[0]["generated_text"]

                if args.debug:
                    print(f"[ANSWER] Agent {i + 1} completion: {completion}")

                assistant_message = construct_assistant_message(completion)
                agent_context.append(assistant_message)

        generated_description[question] = (agent_contexts, answer)

    json.dump(generated_description, open("gsm_{}_{}.json".format(agents, rounds), "w"))

    if args.debug:
        print(f"[DEBUG] Results saved to gsm_{agents}_{rounds}.json")
