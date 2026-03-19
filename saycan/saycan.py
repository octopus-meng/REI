import cv2
import numpy as np
import imageio
import matplotlib.pyplot as plt
import clip

from scene import PickPlaceEnv, PICK_TARGETS, PLACE_TARGETS
from cliport import Cliport
from vild import ViLD
from llm_scoring import make_options, affordance_scoring, build_scene_description, step_to_nlp, LLM_Scoring, LLM_Planner
#@title Prompt

termination_string = "done()"
static_location = ["top left corner", "top right corner", "middle", "bottom left corner", "bottom right corner"]
category_names = ['blue block',
                  'red block',
                  'green block',
                  'orange block',
                  'yellow block',
                  'purple block',
                  'pink block',
                  'cyan block',
                  'brown block',
                  'gray block',

                  'blue bowl',
                  'red bowl',
                  'green bowl',
                  'orange bowl',
                  'yellow bowl',
                  'purple bowl',
                  'pink bowl',
                  'cyan bowl',
                  'brown bowl',
                  'gray bowl']
category_name_string = ";".join(category_names)

scorer_context = """
Examples:
Query: move all the blocks to the top left corner.
Options:
1. robot.pick_and_place(blue block, top left corner)
2. robot.pick_and_place(red block, top left corner)
3. robot.pick_and_place(yellow block, top left corner)
4. done()
Output: {"1": 8, "2": 8, "3": 8, "4": 10}

Query: put the yellow block in the green bowl.
Options:
1. robot.pick_and_place(yellow block, green bowl)
2. robot.pick_and_place(red block, green bowl)
3. robot.pick_and_place(blue block, green bowl)
4. done()
Output: {"1": 10, "2": 0, "3": 0, "4": 10}

Query: stack the blocks.
Options:
1. robot.pick_and_place(green block, blue block)
2. robot.pick_and_place(red block, green block)
3. robot.pick_and_place(blue block, red block)
4. done()
Output: {"1": 8, "2": 8, "3": 0, "4": 10}

Query: group the blue objects together.
Options:
1. robot.pick_and_place(blue block, blue bowl)
2. robot.pick_and_place(green bowl, blue bowl)
3. robot.pick_and_place(red block, blue bowl)
4. done()
Output: {"1": 10, "2": 0, "3": 0, "4": 10}

Query: sort all the blocks into their matching color bowls.
Options:
1. robot.pick_and_place(green block, green bowl)
2. robot.pick_and_place(red block, red bowl)
3. robot.pick_and_place(yellow block, yellow bowl)
4. done()
Output: {"1": 8, "2": 8, "3": 8, "4": 10}

Rate each option's contribution to completing the query on a scale of 0-10. Output a JSON object with keys as option numbers: {"1": score1, "2": score2, ...}. When the task is completed, give done() the highest score (10).
"""

gpt3_context = """
objects = [red block, yellow block, blue block, green bowl]
# move all the blocks to the top left corner.
robot.pick_and_place(blue block, top left corner)
robot.pick_and_place(red block, top left corner)
robot.pick_and_place(yellow block, top left corner)
done()

objects = [red block, yellow block, blue block, green bowl]
# put the yellow one the green thing.
robot.pick_and_place(yellow block, green bowl)
done()

objects = [yellow block, blue block, red block]
# move the light colored block to the middle.
robot.pick_and_place(yellow block, middle)
done()

objects = [blue block, green bowl, red block, yellow bowl, green block]
# stack the blocks.
robot.pick_and_place(green block, blue block)
robot.pick_and_place(red block, green block)
done()

objects = [red block, blue block, green bowl, blue bowl, yellow block, green block]
# group the blue objects together.
robot.pick_and_place(blue block, blue bowl)
done()

objects = [green bowl, red block, green block, red bowl, yellow bowl, yellow block]
# sort all the blocks into their matching color bowls.
robot.pick_and_place(green block, green bowl)
robot.pick_and_place(red block, red bowl)
robot.pick_and_place(yellow block, yellow bowl)
done()
"""

use_environment_description = False
gpt3_context_lines = gpt3_context.split("\n")
gpt3_context_lines_keep = []
for gpt3_context_line in gpt3_context_lines:
  if "objects =" in gpt3_context_line and not use_environment_description:
    continue
  gpt3_context_lines_keep.append(gpt3_context_line)

gpt3_context = "\n".join(gpt3_context_lines_keep)
# print(gpt3_context)

#@title Task and Config
only_plan = False

raw_input = "put all the blocks in different corners." 
config = {"pick":  ["red block", "yellow block", "green block", "blue block"],
          "place": ["red bowl"]}

#@title Setup Scene
image_path = "./2db.png"
np.random.seed(2)
if config is None:
  pick_items = list(PICK_TARGETS.keys())
  pick_items = np.random.choice(pick_items, size=np.random.randint(1, 5), replace=False)

  place_items = list(PLACE_TARGETS.keys())[:-9]
  place_items = np.random.choice(place_items, size=np.random.randint(1, 6 - len(pick_items)), replace=False)
  config = {"pick":  pick_items,
            "place": place_items}
  print(pick_items, place_items)

env = PickPlaceEnv()
obs = env.reset(config)

img_top = env.get_camera_image_top()
img_top_rgb = cv2.cvtColor(img_top, cv2.COLOR_BGR2RGB)
plt.imshow(img_top)

imageio.imsave(image_path, img_top)
#@title Runner
plot_on = True
max_tasks = 5

options = make_options(PICK_TARGETS, PLACE_TARGETS, termination_string=termination_string)
clip_model, _ = clip.load("ViT-B/32")

max_boxes_to_draw = 8 #@param {type:"integer"}
nms_threshold = 0.4 #@param {type:"slider", min:0, max:0.9, step:0.05}
min_rpn_score_thresh = 0.4  #@param {type:"slider", min:0, max:1, step:0.01}
min_box_area = 10 #@param {type:"slider", min:0, max:10000, step:1.0}
max_box_area = 3000  #@param {type:"slider", min:0, max:10000, step:1.0}
vild_params = max_boxes_to_draw, nms_threshold, min_rpn_score_thresh, min_box_area, max_box_area
vild = ViLD(clip_model=clip_model) 

found_objects = vild.inference(image_path, category_name_string, vild_params, plot_on=False)
scene_description = build_scene_description(found_objects)
env_description = scene_description

llm_planner = LLM_Planner()
steps_text = llm_planner.task_plan(raw_input, found_objects, found_objects.extend(static_location), "robot.pick_and_place(item1, item2)")

# all_llm_scores = []
# all_affordance_scores = []
# all_combined_scores = []
# affordance_scores = affordance_scoring(options, found_objects, block_name="box", bowl_name="circle", verbose=False)
# num_tasks = 0
# selected_task = ""
# steps_text = []
# llm_scorer = LLM_Scoring()
# while not selected_task == termination_string:
#   num_tasks += 1
#   if num_tasks > max_tasks:
#     break
#   llm_scores, _ = llm_scorer.batch_scoring(raw_input, options, verbose=True)
#   combined_scores = {option: np.exp(llm_scores[option]) * affordance_scores[option] for option in options}
#   combined_scores = llm_scorer.normalize_scores(combined_scores)
#   selected_task = max(combined_scores, key=combined_scores.get)
#   steps_text.append(selected_task)
#   print(num_tasks, "Selecting: ", selected_task)
#   gpt3_prompt += selected_task + "\n"

#   all_llm_scores.append(llm_scores)
#   all_affordance_scores.append(affordance_scores)
#   all_combined_scores.append(combined_scores)

# if plot_on:
#   for llm_scores, affordance_scores, combined_scores, step in zip(
#       all_llm_scores, all_affordance_scores, all_combined_scores, steps_text):
#     plot_saycan(llm_scores, affordance_scores, combined_scores, step, show_top=10)

print('**** Solution ****')
cliport = Cliport(clip_model=clip_model)
print(env_description)
print('# ' + raw_input)
# for i, step in enumerate(steps_text):
#   if step == '' or step == termination_string:
#     break
#   print('Step ' + str(i) + ': ' + step)
#   nlp_step = step_to_nlp(step)

if not only_plan:
  print('Initial state:')
  plt.imshow(env.get_camera_image())

  for i, step in enumerate(steps_text):
    if step == '' or step.lower() == termination_string or step.lower() == "done": 
      break
    #nlp_step = step_to_nlp(step)
    print('GPT-3 says next step:', step)

    obs = cliport.run_cliport(obs, step, env)

  # Show camera image after task.
  print('Final state:')
  plt.imshow(env.get_camera_image())
env.cache_video = []