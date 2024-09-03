

def get_prompt(goal_label_cxcy, bboxes, trial_num, sep_system=False, add_LCR_prompt=False):
    # 각 바운딩 박스에 대한 설명 구성
    # bbox_descriptions = [f"{label} at ({round(x_min, 2)}, {round(y_min, 2)}, {round(x_max, 2)}, {round(y_max, 2)})" for label, (x_min, y_min, x_max, y_max), _ in bboxes]
    bbox_descriptions = [f"{label} [{round(x_min, 4)}, {round(y_min, 4)}, {round(x_max, 4)}, {round(y_max, 4)}]" for
                         label, (x_min, y_min, x_max, y_max), _ in bboxes]
    bbox_list_str = ", ".join(bbox_descriptions)
    goal_label, goal_cxcy = goal_label_cxcy

    if len(goal_cxcy) == 2:  # point
        # dest_descriptions = f"{goal_label} at ({round(goal_cxcy[0], 2)}, {round(goal_cxcy[1], 2)})"
        dest_descriptions = f"{goal_label} [{round(goal_cxcy[0], 4)}, {round(goal_cxcy[1], 4)}]"
    elif len(goal_cxcy) == 4:  # bbox
        dest_descriptions = f"{goal_label} [{round(goal_cxcy[0], 4)}, {round(goal_cxcy[1], 4)}, {round(goal_cxcy[2], 4)}, {round(goal_cxcy[3], 4)}]"
    else:
        raise AssertionError('check ', goal_cxcy)

    if add_LCR_prompt:
        if goal_cxcy[0] < 0.33:
            dest_descriptions = dest_descriptions + ' on the left side of the image'
        elif goal_cxcy[0] > 0.66:
            dest_descriptions = dest_descriptions + ' on the right side of the image'
        else:
            dest_descriptions = dest_descriptions + ' on the center side of the image'

    # list_ids_sep = [45101, 45102, 45103, 45104, 45105, 45106, 45107]
    list_ids_sep = ['D', 'L', 'R', 'P', 'Not', 'Desc', 'Decs']

    list_prompt = []
    list_system = []
    # one turn
    if trial_num in list_ids_sep:
        if sep_system:
            list_system.append(
                'A chat between a human and an AI that understands visuals in English. '  # default
                'In images, [x, y] denotes points: top-left [0.0, 0.0], bottom-right [1.0, 1.0]. '  # LLaVA
                'Increasing x moves right; y moves down. Decreasing x moves left; y moves up. Bounding box: [x1, y1, x2, y2]. '  # LLaVA
                'Image size: 1.0x1.0.'  # LLaVA
                "The input image depicts the view from a pedestrian's position, taken at a point 80cm above the ground for pedestrian navigation purposes. "
                "The user is looking at the center [0.5, 0.5] of the image. "
                "Consider the starting point as the ground where the user is standing. "
                "Explain as if you were a navigation assistant explaining to a visually impaired person. "
                # "Don't talk about detailed image coordinates. Consider perspective view of the 2D image property. "
                "Rules: \n"
                "1. Do not talk about detailed image coordinates. Never use pixel positional information. "
                "2. Consider perspective view of the 2D image property. "
                # "3. Do not talk about orange point to represent the destination and yellow point to represent the starting point . "
            )

            if trial_num == list_ids_sep[6]:
                list_system.append(
                    '3. If the user is in front of a crosswalk, look for the pedestrian traffic light and tell me the color of the light. If there is a red pedestrian traffic light, say \'Stop and wait\' option. '
                    '4. If there is an obstacle in the path obstructing the user\'s walk, say \'Stop and wait\' option. '
                    '5. If the path leads the user into dangerous areas like roads or construction sites, say \'Stop and wait\' option. '
                    '6. If the path is clear of obstacles, say \'Follow the path\' option. '
                )
                list_system = [' '.join(list_system)]

            if len(bboxes) > 0:
                list_system.append((f"The image contains objects, {bbox_list_str}.\n"))
            list_system = [' '.join(list_system)]
        else:
            raise AssertionError('Unsupported')

        # str_dir=get_direction(goal_cxcy[0])
        # Summarize prompt
        list_prompt1 = []
        if trial_num == list_ids_sep[0]:
            # Destination,
            list_prompt1.append(
                f'Assume the user\'s destination is the {dest_descriptions} in the image. '  # only circle, same style of left and right
                'Provide a one-sentence description of the destination where the user\'s path is heading. '

                # Few Example
                'Example 1) The destination is ahead on the sidewalk. '
                'Example 2) The destination is ahead on the car road. '
                'Example 3) The destination is ahead but is obscured by a truck. '
                'Example 4) The destination is ahead at the entrance of the building. '
                )  # DESC1
        if trial_num == list_ids_sep[1]:
            # Left
            list_prompt1.append(
                f'Assume the user is moving along a 2-meter-wide path from the starting point [0.5, 1.0] to the destination {dest_descriptions} in the image. '
                'Analyze the input image and provide a one-sentence description that includes what is located closely to the left of the path. '
                # 'Analyze the visible area and provide a one-sentence description that includes what is located closely to the left of the path. '
                'If there are no special objects other than the floor, say \'nothing\'. '  # PathR
                'Example 1) There are cars on the left side. '
                'Example 2) There are nothing than the floor on the left side. '  # PathR
                )  # DESC1 or 2
        if trial_num == list_ids_sep[2]:
            # Right
            list_prompt1.append(
                f'Assume the user is moving along a 2-meter-wide path from the starting point [0.5, 1.0] to the destination {dest_descriptions} in the image. '
                'Analyze the input image and provide a one-sentence description that includes what is located closely to the right of the path. '
                # 'Analyze the visible area and provide a one-sentence description that includes what is located closely to the right of the path. '
                'If there are no special objects other than the floor, say \'nothing\'. '  # PathR
                'Example 1) There are people on the right side. '
                'Example 2) There are nothing than the floor on the right side. '  # PathR
                )  # DESC1 or 2
        if trial_num == list_ids_sep[3]:
            # Path
            list_prompt1.append(
                f'Assume the user is moving along a 2-meter-wide path from the starting point [0.5, 1.0] to the destination {dest_descriptions} in the image. '
                'Analyze the input image and provide a one-sentence description of what objects are on the path. '
                'Example 1) There are cars and people on the path. '
                'Example 2) There are nothing on the path. '
                )  # DESC1 or 2
        if trial_num == list_ids_sep[4]:
            raise AssertionError('Check the prompt id!')
        if trial_num == list_ids_sep[5]:
            # VLM, describe and find
            list_prompt1.append(f'Describe where the user is located based on the image. '
                                'If the user is in front of a crosswalk, look for the pedestrian traffic light and tell me the color of the light. '
                                'Example 1) The user is in front of a crosswalk. The pedestrian traffic light is red. '
                                'Example 2) The user is in front of a construction site. '
                                )
        if trial_num == list_ids_sep[6]:
            # VLM, find and decision
            list_prompt2 = []
            list_prompt2.append(
                'Based on the current description, evaluate the user is able to move along the path without collision or collide with other objects. '
                'Then, select the most appropriate action: \'Follow the path\' or \'Stop and wait\'. '
                'Say only the answer, explaining the selection and the reason in two sentences. '
                )  # DESC1,2
            list_prompt2.append(
                'Example 1) Stop and wait. A car is on the path, so walking to the destination is impossible. ')
            list_prompt2.append(
                'Example 2) Follow the path. The path is clear of obstacles, so walking to the destination is possible. ')
            list_prompt2.append(
                'Example 3) Stop and wait. It is in front of the crosswalk and the pedestrian traffic light is red, so walking to the destination is impossible. ')
            list_prompt1 = [' '.join(list_prompt2)]
        list_prompt.extend(list_prompt1)
    else:
        raise AssertionError(f'{trial_num} is not supported')

    return list_prompt, list_system


if __name__ == "__main__":

    list_prompt, list_system = get_prompt(['point', [0.1, 0.5]],
                                          [],
                                          trial_num='Decs',
                                          sep_system=True)
    print(list_system)
    print(list_prompt)

    # llm_system = list_system[0]
    # prefix_prompt = [' '.join(final_description)]
    # llm_prompt = f'The image description is following: {prefix_prompt[0]} {list_prompt[0]} Say only the answers. '
