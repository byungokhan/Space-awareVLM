from transformers import LlavaNextProcessor, LlavaNextForConditionalGeneration, AutoModelForCausalLM, AutoTokenizer
import torch
import os
from PIL import Image
import argparse
import glob
import json
import copy
from datetime import datetime
import xml.etree.ElementTree as ET

from eval_savlm import init_logging, eval_text
from eval_prompt import get_prompt
from gpt_wrapper import gpt_wrapper

from llava.mm_utils import get_model_name_from_path, process_images, tokenizer_image_token
from llava.model.builder import load_pretrained_model
from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN

# llava-next model names
# llava-hf/llava-v1.6-mistral-7b-hf
# llava-hf/llava-v1.6-vicuna-7b-hf
# llava-hf/llava-v1.6-vicuna-13b-hf
# llava-hf/llava-v1.6-34b-hf
# llava-hf/llama3-llava-next-8b-hf
# llava-hf/llava-next-72b-hf
# llava-hf/llava-next-110b-hf
# gpt-4o-2024-08-06
# gpt-4o-mini-2024-07-18

def llava_next_inference_latest(image, sys_prompt, user_prompt, model, processor, device):

    #########
    conversation = [
        # {
        #     "role": "system",
        #     "content": [
        #         {"type": "text", "text": sys_prompt},
        #     ],
        # },
        {
            "role": "user",
            "content": [
                {"type": "image"},
                {"type": "text", "text": sys_prompt + ' ' + user_prompt},
            ],
        },
    ]
    prompt = processor.apply_chat_template(conversation, add_generation_prompt=True)
    print(prompt)
    inputs = processor(prompt, image, return_tensors="pt").to(device)
    # autoregressively complete prompt
    output = model.generate(**inputs, max_new_tokens=2048)
    text_output = processor.decode(output[0], skip_special_tokens=True)

    return text_output

def llava_next_inference(image, sys_prompt, user_prompt, model, tokenizer, image_processor, device):

    image_tensor = process_images([image], image_processor, model.config)
    image_tensor = [_image.to(dtype=torch.float16, device=device) for _image in image_tensor]

    #conv_template = "qwen_1_5"  # Make sure you use correct chat template for different models
    # question = DEFAULT_IMAGE_TOKEN + "\nWhat is shown in this image?"
    # question = DEFAULT_IMAGE_TOKEN + "\nPlease provide a brief walking guide for a visually impaired person by analyzing the image based on the goal position (0.600, 0.750)."
    # conv = copy.deepcopy(conv_templates[conv_template])
    # conv.append_message(conv.roles[0], question)
    # conv.append_message(conv.roles[1], None)
    # prompt_question = conv.get_prompt()

    conversation = [
        {
            "role": "system",
            "content": [
                {"type": "text", "text": sys_prompt},
            ],
        },
        {
            "role": "user",
            "content": [
                {"type": "image"},
                {"type": "text", "text": user_prompt},
            ],
        },
    ]

    input_ids = tokenizer_image_token(conversation, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt").unsqueeze(
        0).to(device)
    image_sizes = [image.size]

    cont = model.generate(
        input_ids,
        images=image_tensor,
        image_sizes=image_sizes,
        do_sample=False,
        temperature=0,
        max_new_tokens=4096,
    )
    text_outputs = tokenizer.batch_decode(cont, skip_special_tokens=True)

    return text_outputs[0]

def evaluate_zero_shot_vlm(anno_list, model, tokenizer, processor, device, logger):

    eval_gt_degree = 'simple'  # or 'complex'
    eval_gt_tags = ['dest', 'left', 'right', 'path']
    for i, tag in enumerate(eval_gt_tags):
        eval_gt_tags[i] = './/' + tag + '/' + eval_gt_degree
    eval_gt_tags.append('.//recommend')
    list_ids_sep = ['D', 'L', 'R', 'P', 'Desc', 'Decs']

    tag_scores = {}
    avg_scores = {
        'm_scores': {tag: 0.0 for tag in eval_gt_tags},
        'b_scores': {tag: {'precision': 0.0, 'recall': 0.0, 'f1': 0.0} for tag in eval_gt_tags},
        'r_scores': {tag: {'rouge1': 0.0, 'rouge2': 0.0, 'rougeL': 0.0} for tag in eval_gt_tags},
        'llm_scores': {tag: 0.0 for tag in eval_gt_tags},
    }

    file_count = 1

    for xml_idx, xml_path in enumerate(anno_list):

        xml_filename = os.path.basename(xml_path)
        logger.info(f"[{file_count}] ********** [GT FILENAME]: {xml_path}")
        print(file_count, " : ", xml_path)
        tag_scores[xml_filename] = {
            'm_scores': {tag: [] for tag in eval_gt_tags},
            'b_scores': {tag: {'precision': [], 'recall': [], 'f1': []} for tag in eval_gt_tags},
            'r_scores': {tag: {'rouge1': [], 'rouge2': [], 'rougeL': []} for tag in eval_gt_tags},
            'llm_scores': {tag: [] for tag in eval_gt_tags},
        }

        gt_tree = ET.parse(xml_path)
        gt_root = gt_tree.getroot()
        gp_x = gt_root.find("goal_position/x").text.strip()
        gp_y = gt_root.find("goal_position/y").text.strip()
        # gp_xy = f"({float(gp_x):.3f}, {float(gp_y):.3f})"
        gp = ['point', [float(gp_x), float(gp_y)]]

        img_path = xml_path.replace('.xml', '.JPG')
        if not os.path.exists(img_path):
            img_path = xml_path.replace('.xml', '.jpeg')
        if not os.path.exists(img_path):
            img_path = xml_path.replace('.xml', '.jpg')
        if not os.path.exists(img_path):
            img_path = xml_path.replace('.xml', '.png')

        image = Image.open(img_path)

        responses = []
        for i, sep in enumerate(list_ids_sep):

            user_prompt, sys_prompt  = get_prompt(gp,
                                                  [],
                                                  trial_num=sep,
                                                  sep_system=True)
            user_prompt, sys_prompt = user_prompt[0], sys_prompt[0]

            if i<=3: # d, l, r, p,
                gt_tag = eval_gt_tags[i]
            elif i==5: # Decs
                gt_tag = eval_gt_tags[4]
                all_responses = ' '.join(responses)
                user_prompt = all_responses + ' ' + user_prompt[0]
            else:
                gt_tag = None

            if isinstance(model, gpt_wrapper):
                text_output = model.generate_llm_response(sys_prompt, user_prompt, img_path)
            else:
                #text_output = llava_next_inference(image, sys_prompt, user_prompt, model, tokenizer, image_processor, device)
                text_output = llava_next_inference_latest(image, sys_prompt, user_prompt, model, processor, device)

            responses.append(text_output)

            logger.info(f"***** gt_tag: {gt_tag}" )
            logger.info(f"[Query SYS]: {sys_prompt}")
            logger.info(f"[Query USER]: {user_prompt}")
            # 파싱 가능한 XML로 변경된 내용을 출력
            logger.info(f"[ANSWER]: {text_output}" )
            #print(ET.tostring(output_root, encoding='unicode'))

            if gt_tag != None:

                gt_dest_desc = gt_root.find(gt_tag)
                if gt_dest_desc == None or gt_dest_desc.text == None:
                    continue
                else:
                    gt_dest_desc = gt_dest_desc.text.strip()
                gt_dest_desc = gt_dest_desc.replace("India", "sidewalk")
                logger.info(f"   [gt]: {gt_dest_desc}")

                m_score, b_scores, r_scores, llm_score = eval_text(gt_dest_desc, text_output)

                tag_scores[xml_filename]['m_scores'][gt_tag].append(m_score)
                tag_scores[xml_filename]['b_scores'][gt_tag]['precision'].append(b_scores['precision'])
                tag_scores[xml_filename]['b_scores'][gt_tag]['recall'].append(b_scores['recall'])
                tag_scores[xml_filename]['b_scores'][gt_tag]['f1'].append(b_scores['f1'])
                tag_scores[xml_filename]['r_scores'][gt_tag]['rouge1'].append(r_scores['rouge1'].fmeasure)
                tag_scores[xml_filename]['r_scores'][gt_tag]['rouge2'].append(r_scores['rouge2'].fmeasure)
                tag_scores[xml_filename]['r_scores'][gt_tag]['rougeL'].append(r_scores['rougeL'].fmeasure)
                tag_scores[xml_filename]['llm_scores'][gt_tag].append(llm_score)

                avg_scores['m_scores'][gt_tag] += (m_score - avg_scores['m_scores'][gt_tag]) / file_count
                avg_scores['b_scores'][gt_tag]['precision'] += (b_scores['precision'] - avg_scores['b_scores'][gt_tag]['precision']) / file_count
                avg_scores['b_scores'][gt_tag]['recall'] += (b_scores['recall'] - avg_scores['b_scores'][gt_tag]['recall']) / file_count
                avg_scores['b_scores'][gt_tag]['f1'] += (b_scores['f1'] - avg_scores['b_scores'][gt_tag]['f1']) / file_count
                avg_scores['r_scores'][gt_tag]['rouge1'] += (r_scores['rouge1'].fmeasure - avg_scores['r_scores'][gt_tag]['rouge1']) / file_count
                avg_scores['r_scores'][gt_tag]['rouge2'] += (r_scores['rouge2'].fmeasure - avg_scores['r_scores'][gt_tag]['rouge2']) / file_count
                avg_scores['r_scores'][gt_tag]['rougeL'] += (r_scores['rougeL'].fmeasure - avg_scores['r_scores'][gt_tag]['rougeL']) / file_count
                avg_scores['llm_scores'][gt_tag] += (llm_score - avg_scores['llm_scores'][gt_tag]) / file_count

                logger.info(f"  METEOR Score: {m_score}")
                logger.info(f"  BERTScore Precision: {b_scores['precision']}")
                logger.info(f"  BERTScore Recall: {b_scores['recall']}")
                logger.info(f"  BERTScore F1: {b_scores['f1']}")
                logger.info(f"  ROUGE-1: {r_scores['rouge1'].fmeasure}")
                logger.info(f"  ROUGE-2: {r_scores['rouge2'].fmeasure}")
                logger.info(f"  ROUGE-L: {r_scores['rougeL'].fmeasure}")
                logger.info(f"  LLM Score: {llm_score}")

                logger.info(f"  [Avg] METEOR Score: {avg_scores['m_scores'][gt_tag]}")
                logger.info(f"  [Avg] BERTScore Precision: {avg_scores['b_scores'][gt_tag]['precision']}")
                logger.info(f"  [Avg] BERTScore Recall: {avg_scores['b_scores'][gt_tag]['recall']}")
                logger.info(f"  [Avg] BERTScore F1: {avg_scores['b_scores'][gt_tag]['f1']}")
                logger.info(f"  [Avg] ROUGE-1: {avg_scores['r_scores'][gt_tag]['rouge1']}")
                logger.info(f"  [Avg] ROUGE-2: {avg_scores['r_scores'][gt_tag]['rouge2']}")
                logger.info(f"  [Avg] ROUGE-L: {avg_scores['r_scores'][gt_tag]['rougeL']}")
                logger.info(f"  [Avg] LLM Score: {avg_scores['llm_scores'][gt_tag]}")

        # Update running averages
        file_count += 1

    return tag_scores, avg_scores


def main():

    parser = argparse.ArgumentParser(description='Evaluate the other VLMs')
    parser.add_argument('--model_ckpt_path', type=str, default="llava-hf/llava-v1.6-mistral-7b-hf", help='ckpt directory')
    parser.add_argument('--eval_db_dir', type=str, default='/mnt/data_disk/dbs/gd_space_aware/manual/240826en', help='dataset directory for evaluation')
    parser.add_argument('--output_dir', type=str, default='./eval_results', help='directory to save the output scores')
    args = parser.parse_args()

    # Create the output file name
    model_ckpt_name = os.path.basename(args.model_ckpt_path)
    output_dir = os.path.join(args.output_dir, model_ckpt_name)
    # Create output directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    if 'llava' in args.model_ckpt_path:
        processor = LlavaNextProcessor.from_pretrained(args.model_ckpt_path)
        model = LlavaNextForConditionalGeneration.from_pretrained(args.model_ckpt_path, torch_dtype=torch.float16, low_cpu_mem_usage=True)
        # tokenizer = AutoTokenizer.from_pretrained(args.model_ckpt_path)
        # model = AutoModelForCausalLM.from_pretrained(args.model_ckpt_path)
        device = "cuda"
        tokenizer = None
        model.to(device)

        # model_name = get_model_name_from_path(args.model_ckpt_path)
        # device = "cuda"
        # device_map = "auto"
        # tokenizer, model, image_processor, max_length = (
        #     load_pretrained_model(args.model_ckpt_path, None, model_name, device_map=device_map))  # Add any other thing you want to pass in llava_model_args
        # model.eval()

    elif 'gpt' in args.model_ckpt_path:
        model = gpt_wrapper('gpt-4o-2024-08-06', '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')
        tokenizer = None
        image_processor = None
        device = None

    # init logger
    logger = init_logging(output_dir, model_ckpt_name)
    # logging parameters
    for arg, value in vars(args).items():
        logger.info(f"{arg}: {value}")

    # data setting
    anno_list = glob.glob(f'{args.eval_db_dir}/**/*xml', recursive=True)
    # 특정 폴더명을 포함하는 파일만 필터링
    target_folders = ['1-500', 'm1-500']  # exclude 'demo' folder
    filtered_anno_list = [
        xml_file for xml_file in anno_list
        if any(folder in xml_file for folder in target_folders)
    ]

    tag_scores, avg_scores = evaluate_zero_shot_vlm(filtered_anno_list, model, tokenizer, processor, device, logger)

    current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file_name = f"{model_ckpt_name}_{current_time}.txt"
    output_file_path = os.path.join(output_dir, output_file_name)

    result = {
        "avg_scores": avg_scores,
        "tag_scores": tag_scores
    }

    # Write the result to the text file
    with open(output_file_path, 'w') as f:
        f.write(json.dumps(result, indent=4))

    print(f"Average scores and tag scores saved to {output_file_path}")


if __name__ == "__main__":
    main()
