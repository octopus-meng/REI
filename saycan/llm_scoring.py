import os
import json
import re

from openai import OpenAI
import numpy as np
from heapq import nlargest
import matplotlib.pyplot as plt

from scene import PICK_TARGETS, PLACE_TARGETS

LLM_CACHE = {}

# ============ 配置区域 ============
MINIMAX_API_KEY = os.environ["MINIMAX_API_KEY"]  # 替换为你的 MiniMax API Key
MINIMAX_BASE_URL = "https://api.minimax.chat/v1"  # MiniMax API 端点
QWEN_API_KEY = os.environ["QWEN_API_KEY"]
QWEN_BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"
OLLAMA_API_KEY = "ollama"
OLLAMA_BASE_URL = "http://localhost:11434/v1"
LLM_DICT = {
    "minimax":[MINIMAX_API_KEY, MINIMAX_BASE_URL],
    "qwen":[QWEN_API_KEY, QWEN_BASE_URL],
    "ollama":[OLLAMA_API_KEY, OLLAMA_BASE_URL]
}
# ============PROMPTS==============
PLAN_PROMPT = """Role: You are a task planner responsible for decomposing natural language instructions into instructions executable by a robot.
**Task Description**
The "Objects" section contains interactive items; the "Targets" section contains locations where items can be placed; the "Abilities" section contains instructions executable by the robot; and "Task" is the current mission.
**Task Requirements**
1.Generate the result strictly in the following format, with no additional output. Format: \n 1. Instruction 1\n 2. Instruction 2\n .... \n n.done()
2.When the task planning is complete, output the instruction as done().
3.The items in the instructions must strictly adhere to the content specified in the task description and must not exceed the defined capabilities.
**examples**
***example1***
Objects:[red block, yellow block, blue block, green bowl]
Targets:[red block, yellow block, blue block, green bowl, top left corner, top right corner, middle, bottom left corner, bottom right corner]
Abilities:robot.pick_and_place(item1, item2)
Task:move all the blocks to the top left corner.
outputs:
1.robot.pick_and_place(blue block, top left corner)
2.robot.pick_and_place(red block, top left corner)
3.robot.pick_and_place(yellow block, top left corner)
4.done()
***example2***
Objects:[red block, yellow block, blue block, green bowl]
Targets:[red block, yellow block, blue block, green bowl, top left corner, top right corner, middle, bottom left corner, bottom right corner]
Abilities:robot.pick_and_place(item1, item2)
Task:put the yellow one the green thing.
outputs:
1.robot.pick_and_place(yellow block, green bowl)
2.done()
***example3***
Objects:[blue block, green bowl, red block, yellow bowl, green block]
Targets:[blue block, green bowl, red block, yellow bowl, green block, top left corner, top right corner, middle, bottom left corner, bottom right corner]
Abilities:robot.pick_and_place(item1, item2)
Task:stack the blocks.
1.robot.pick_and_place(green block, blue block)
2.robot.pick_and_place(red block, green block)
3.done()
"""

class LLM:
    def __init__(self, llm_mode="minimax", engine="MiniMax-M2.5"):
        self.llm_mode = llm_mode
        self.api_key = LLM_DICT[llm_mode][0]
        self.base_url = LLM_DICT[llm_mode][1]
        self.client = OpenAI(
            api_key=self.api_key,
            base_url=self.base_url,
        )
        self.engine = engine
        
    def llm_call(self, input="", max_tokens=8192, temperature=0):
        response = self.client.chat.completions.create(
            model=self.engine,
            messages=[{"role": "user", "content": input}],
            max_tokens=max_tokens,
            temperature=temperature,
            reasoning_effort="none"
        )
        return response

class LLM_Planner(LLM):
    def task_plan(self, query, objects, targets, abilities, prompt=PLAN_PROMPT):
        full_text = f"""{prompt}\n Objects:{objects}\n Targets:{targets}\n Abilities:{abilities}\n Task:{query}\n """
        response = self.llm_call(full_text)
        steps = self.plan_parse(response)
        return steps
    
    def plan_parse(self, response):
        content = response.choices[0].message.content.strip() if hasattr(response, 'choices') else str(response)
        content = re.sub(r'<think>.*?</think>', '', content, flags=re.DOTALL)
        lines = content.strip().split('\n')
        steps = []

        for line in lines:
            line = line.strip()
            if not line:
                continue
            if '.' in line:
                line = line.split('.', 1)[1].strip()
            if line.lower() == 'done()' or line.lower() == 'done':
                steps.append('done')
                break
            if 'robot.pick_and_place(' in line:
                step = line.replace("robot.pick_and_place(", "").replace(")", "")
                pick, place = step.split(", ")
                steps.append("Pick the " + pick + " and place it on the " + place + ".")
        return steps
    

class LLM_Scoring(LLM):
    def llm_call(self, input="", max_tokens=4096, temperature=0, score_mode="evaluation",
              logprobs=1, echo=True, reasoning=False):
        if score_mode == "evaluation":
            response = self.client.chat.completions.create(
                model=self.engine,
                messages=[{"role": "user", "content": input}],
                max_tokens=max_tokens,
                temperature=temperature,
                reasoning_effort="none"
            )
        elif score_mode == "likelihood":
            response = self.client.completions.create(
                model=self.engine,
                prompt=input,
                max_tokens=max_tokens,
                temperature=temperature,
                echo=echo,
                logprobs=logprobs
            )
        return response
    
    def llm_scoring(self, query, options, score_mode="evaluation", limit_num_options=None, option_start="\n", verbose=False):
        if limit_num_options:
            options = options[:limit_num_options]
        verbose and print("Scoring", len(options), "options", f"in {score_mode} mode")

        scores = {}
        for option in options:
            if score_mode == "evaluation":
                full_prompt = f"""Query:{query}
                                Option:{option}
                                Rate how well this option contributes to completing the query on a scale of 0-10. Output a JSON object: {{"score": number}}"""
                response = self.llm_call(
                    input=full_prompt,
                    max_tokens=4096,
                    temperature=0,
                    score_mode=score_mode
                )
                choice = response.choices[0]
                content = choice.message.content.strip() if choice.message.content else ""
                if "</think>" in content:
                    content = content.split("</think>")[-1].strip()
                try:
                    result = json.loads(content)
                    score = float(result.get("score", 0))
                    score = max(0, min(10, score))
                except (json.JSONDecodeError, ValueError):
                    score = 0
                scores[option] = score
                verbose and print(f"Option: {option[:50]}... Score: {score}")
            elif score_mode == "likelihood":
                full_prompt = f"{query} {option}\n"
                response = self.llm_call(
                    input=full_prompt,
                    max_tokens=0,
                    logprobs=1,
                    temperature=0,
                    score_mode=score_mode,
                    echo=True
                )
                choice = response.choices[0]
                logprobs_obj = choice.logprobs if hasattr(choice, 'logprobs') and choice.logprobs else None

                if logprobs_obj:
                    tokens = getattr(logprobs_obj, 'tokens', None)
                    token_logprobs = getattr(logprobs_obj, 'token_logprobs', None)

                    if tokens is None and hasattr(logprobs_obj, 'content'):
                        tokens = [getattr(c, 'token', '') for c in logprobs_obj.content]
                        token_logprobs = [getattr(c, 'logprob', None) for c in logprobs_obj.content]

                    if tokens is None:
                        raise AttributeError("No tokens field")
                else:
                    assert 0, "LLM doesn't has logprobs"


                scores[option] = token_logprobs[0]

        for i, option in enumerate(sorted(scores.items(), key=lambda x: -x[1])):
            verbose and print(option[1], "\t", option[0])
            if i >= 10:
                break

        return scores, response

    def batch_scoring(self, query, options, score_mode="evaluation", limit_num_options=None, verbose=False):
        if limit_num_options:
            options = options[:limit_num_options]
        verbose and print("Batch scoring", len(options), "options", f"in {score_mode} mode")

        scores = {}
        if score_mode == "evaluation":
            options_text = "\n".join([f"{i+1}. {opt}" for i, opt in enumerate(options)])
            full_prompt = f"""Query: {query}
                            Options:
                            {options_text}
                            Rate each option's contribution to completing the query on a scale of 0-10. Output a JSON object with keys as option numbers: {{"1": score1, "2": score2, ...}}"""
            response = self.llm_call(
                input=full_prompt,
                max_tokens=50000,
                temperature=0,
                score_mode=score_mode
            )
            choice = response.choices[0]
            content = choice.message.content.strip() if choice.message.content else ""
            if "</think>" in content:
                content = content.split("</think>")[-1].strip()

            try:
                result = json.loads(content)
                for i, option in enumerate(options):
                    key = str(i + 1)
                    score = float(result.get(key, 0))
                    score = max(0, min(10, score))
                    scores[option] = score
            except (json.JSONDecodeError, ValueError):
                print("[WARNING] json parse false!!")
                print(content)
                for option in options:
                    scores[option] = 0

            for opt, score in scores.items():
                verbose and print(f"Option: {opt[:50]}... Score: {score}")

        elif score_mode == "likelihood":
            raise NotImplementedError("likelihood method is not implemented")

        for i, option in enumerate(sorted(scores.items(), key=lambda x: -x[1])):
            verbose and print(option[1], "\t", option[0])
            if i >= 10:
                break

        return scores, response

    def normalize_scores(self, scores):
        max_score = max(scores.values())  
        normed_scores = {key: np.clip(scores[key] / max_score, 0, 1) for key in scores}
        return normed_scores
        
#@title Helper Functions
def build_scene_description(found_objects, block_name="box", bowl_name="circle"):
    scene_description = f"objects = {found_objects}"
    scene_description = scene_description.replace(block_name, "block")
    scene_description = scene_description.replace(bowl_name, "bowl")
    scene_description = scene_description.replace("'", "")
    return scene_description

def step_to_nlp(step):
    step = step.replace("robot.pick_and_place(", "")
    step = step.replace(")", "")
    pick, place = step.split(", ")
    return "Pick the " + pick + " and place it on the " + place + "."

# def plot_saycan(llm_scores, vfs, combined_scores, task, correct=True, show_top=None):
#     if show_top:
#         top_options = nlargest(show_top, combined_scores, key = combined_scores.get)
#         # add a few top llm options in if not already shown
#         top_llm_options = nlargest(show_top // 2, llm_scores, key = llm_scores.get)
#         for llm_option in top_llm_options:
#             if not llm_option in top_options:
#                 top_options.append(llm_option)
#         llm_scores = {option: llm_scores[option] for option in top_options}
#         vfs = {option: vfs[option] for option in top_options}
#         combined_scores = {option: combined_scores[option] for option in top_options}

#     sorted_keys = dict(sorted(combined_scores.items()))
#     keys = [key for key in sorted_keys]
#     positions = np.arange(len(combined_scores.items()))
#     width = 0.3

#     fig = plt.figure(figsize=(12, 6))
#     ax1 = fig.add_subplot(1,1,1)

#     plot_llm_scores = normalize_scores({key: np.exp(llm_scores[key]) for key in sorted_keys})
#     plot_llm_scores = np.asarray([plot_llm_scores[key] for key in sorted_keys])
#     plot_affordance_scores = np.asarray([vfs[key] for key in sorted_keys])
#     plot_combined_scores = np.asarray([combined_scores[key] for key in sorted_keys])
    
#     ax1.bar(positions, plot_combined_scores, 3 * width, alpha=0.6, color="#93CE8E", label="combined")
        
#     score_colors = ["#ea9999ff" for score in plot_affordance_scores]
#     ax1.bar(positions + width / 2, 0 * plot_combined_scores, width, color="#ea9999ff", label="vfs")
#     ax1.bar(positions + width / 2, 0 * plot_combined_scores, width, color="#a4c2f4ff", label="language")
#     ax1.bar(positions - width / 2, np.abs(plot_affordance_scores), width, color=score_colors)
    
#     plt.xticks(rotation="vertical")
#     ax1.set_ylim(0.0, 1.0)

#     ax1.grid(True, which="both")
#     ax1.axis("on")

#     ax1_llm = ax1.twinx()
#     ax1_llm.bar(positions + width / 2, plot_llm_scores, width, color="#a4c2f4ff", label="language")
#     ax1_llm.set_ylim(0.01, 1.0)
#     plt.yscale("log")
    
#     font = {"fontname":"Arial", "size":"16", "color":"k" if correct else "r"}
#     plt.title(task, **font)
#     key_strings = [key.replace("robot.pick_and_place", "").replace(", ", " to ").replace("(", "").replace(")","") for key in keys]
#     plt.xticks(positions, key_strings, **font)
#     ax1.legend()
#     plt.show()

def affordance_scoring(options, found_objects, verbose=False, block_name="box", bowl_name="circle", termination_string="done()"):
    affordance_scores = {}
    found_objects = [
                    found_object.replace(block_name, "block").replace(bowl_name, "bowl") 
                    for found_object in found_objects + list(PLACE_TARGETS.keys())[-5:]]
    verbose and print("found_objects", found_objects)
    for option in options:
        if option == termination_string:
            affordance_scores[option] = 0.2
            continue
        pick, place = option.replace("robot.pick_and_place(", "").replace(")", "").split(", ")
        affordance = 0
        found_objects_copy = found_objects.copy()
        if pick in found_objects_copy:
            found_objects_copy.remove(pick)
            if place in found_objects_copy:
                affordance = 1
        affordance_scores[option] = affordance
        verbose and print(affordance, '\t', option)
    return affordance_scores

def make_options(pick_targets=None, place_targets=None, options_in_api_form=True, termination_string="done()"):
    if not pick_targets:
        pick_targets = PICK_TARGETS
    if not place_targets:
        place_targets = PLACE_TARGETS
    options = []
    for pick in pick_targets:
        for place in place_targets:
            if options_in_api_form:
                option = "robot.pick_and_place({}, {})".format(pick, place)
            else:
                option = "Pick the {} and place it on the {}.".format(pick, place)
            options.append(option)

    options.append(termination_string)
    print("Considering", len(options), "options")
    return options

if __name__=="__main__":
    termination_string = "done()"
    # query = "To pick the blue block and put it on the red block, I should:\n"
    query = "put all the blocks in different corners." 

    # options = make_options(PICK_TARGETS, PLACE_TARGETS, termination_string=termination_string)
    # llm_scorer = LLM_Scoring()
    # llm_scores, _ = llm_scorer.batch_scoring(query, options, verbose=True, score_mode="evaluation")#"qwen3-30b-a3b-instruct-2507")
    # found_objects = ["blue block", "red block"]
    # affordance_scores = affordance_scoring(options, found_objects, block_name="box", bowl_name="circle", verbose=False, termination_string=termination_string)

    # combined_scores = {option: np.exp(llm_scores[option]) * affordance_scores[option] for option in options}
    # combined_scores = llm_scorer.normalize_scores(combined_scores)
    # selected_task = max(combined_scores, key=combined_scores.get)
    # print("Selecting: ", selected_task)
    objects = ["blue block", "red block"]
    places_targets = ["blue block", "red block", "top left corner", "top right corner", "middle", "bottom left corner", "bottom right corner"]
    planner = LLM_Planner()
    print(planner.task_plan(query, objects, places_targets, "robot.pick_and_place(item1, item2)"))