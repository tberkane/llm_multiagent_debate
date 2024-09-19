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

    other_answers = "These are the solutions to the problem from other agents: "

    for agent in agents:
        agent_response = agent[idx]["content"]
        response = f"\n\n One agent solution: ```{agent_response}```"
        other_answers += response

    prefix_string = f"""
You've already provided an answer to the math problem: {question}. Now, consider the following solutions from other LLMs:
{other_answers}
With this additional information:

Critically analyze your original answer and the solutions from other LLMs.
If you identify any errors in your initial solution or find merit in other approaches, revise your answer and explain your reasoning.
If you still believe your original answer is correct, explain why, addressing any discrepancies with other solutions.

Conclude with your final answer, either revised or reaffirmed, in the form \\boxed{{answer}}.
"""
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
    for idx, data in enumerate(questions[: args.evaluation], start=1):
        question = data["question"]
        answer = data["answer"]

        if args.debug:
            print(f"[QUESTION {idx}] Processing question: {question}")

        # Initialize contexts for each agent
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

        # Iterate through the specified number of rounds
        for round in range(rounds):
            if args.debug:
                print(f"[ROUND] Starting round {round + 1}")

            # Process each agent's context
            for i, agent_context in enumerate(agent_contexts):
                if args.debug:
                    print(f"[DEBUG] Processing agent {i + 1}")

                # For rounds after the first, add context from other agents
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

                # Construct and append the assistant's message to the context
                assistant_message = construct_assistant_message(completion)
                agent_context.append(assistant_message)

        generated_description[question] = (agent_contexts, answer)

    json.dump(generated_description, open("gsm_{}_{}.json".format(agents, rounds), "w"))

    if args.debug:
        print(f"[DEBUG] Results saved to gsm_{agents}_{rounds}.json")
