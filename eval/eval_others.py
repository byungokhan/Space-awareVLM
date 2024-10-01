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
import time

from eval_savlm import init_logging, eval_text, update_and_logging_results
from eval_prompt import get_prompt
from gpt_wrapper import gpt_wrapper

from llava.mm_utils import get_model_name_from_path, process_images, tokenizer_image_token
from llava.model.builder import load_pretrained_model
from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN
from llava.conversation import conv_templates

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

def llava_next_inference_latest(image, sys_prompt, user_prompt, model, model_name, processor, device):

    if 'llama3' in model_name:
        custom_prompt = (
            f"<|start_header_id|>System:<|end_header_id|>\n\n"
            f"{sys_prompt}<|eot_id|>"
            f"<|start_header_id|>User:<|end_header_id|>\n\n"
            f"<image>\n{user_prompt}<|eot_id|>"
            f"<|start_header_id|>Assistant:<|end_header_id|>\n\n"
        )
    elif any(x in model_name for x in ['34b', '72b', '110b']):
        custom_prompt = (f"<|im_start|>System:\n{sys_prompt}<|im_end|>"
                         f"<|im_start|>User:\n<image>\n{user_prompt}<|im_end|>"
                         f"<|im_start|>Assistant:\n")
    else:
        custom_prompt = f"<image>\nSystem: {sys_prompt}\nUser: {user_prompt}\nAssistant:"

    inputs = processor(custom_prompt, image, return_tensors="pt").to(device)
    output = model.generate(**inputs,
                            temperature=0.0,
                            max_new_tokens=4096)
    text_output = processor.decode(output[0], skip_special_tokens=True)
    response_start = text_output.find("Assistant:") + len("Assistant:")
    response = text_output[response_start:].strip()

    return response

def llava_ov_inference(image, sys_prompt, user_prompt, model, tokenizer, image_processor, device):

    image_tensor = process_images([image], image_processor, model.config)
    image_tensor = [_image.to(dtype=torch.float16, device=device) for _image in image_tensor]

    conv_template = "qwen_1_5"  # Make sure you use correct chat template for different models
    question = DEFAULT_IMAGE_TOKEN + f"\nSystem: {sys_prompt}\nUser: {user_prompt}\nAssistant:"
    conv = copy.deepcopy(conv_templates[conv_template])
    conv.append_message(conv.roles[0], question)
    conv.append_message(conv.roles[1], None)
    prompt_question = conv.get_prompt()

    input_ids = tokenizer_image_token(prompt_question, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt").unsqueeze(0).to(device)
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

def evaluate_zero_shot_vlm(anno_list, model, model_name, tokenizer, processor, device, logger):

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
        'llm_scores_c': {tag: 0.0 for tag in eval_gt_tags},
        'num_words': {tag: 0.0 for tag in eval_gt_tags},
    }

    file_count = 1
    avg_infer_time=0.0

    for xml_idx, xml_path in enumerate(anno_list):

        xml_filename = os.path.basename(xml_path)
        logger.info(f"[{file_count}] ********** [GT FILENAME]: {xml_path}")
        print(file_count, " : ", xml_path)
        tag_scores[xml_filename] = {
            'm_scores': {tag: [] for tag in eval_gt_tags},
            'b_scores': {tag: {'precision': [], 'recall': [], 'f1': []} for tag in eval_gt_tags},
            'r_scores': {tag: {'rouge1': [], 'rouge2': [], 'rougeL': []} for tag in eval_gt_tags},
            'llm_scores': {tag: [] for tag in eval_gt_tags},
            'llm_scores_c': {tag: [] for tag in eval_gt_tags},
            'num_words': {tag: [] for tag in eval_gt_tags},
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
        inference_time = 0.0

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
                user_prompt = all_responses + ' ' + user_prompt
            else:
                gt_tag = None

            start_time = time.time()
            if 'gpt' in model_name:
                text_output = model.generate_llm_response(sys_prompt, user_prompt, img_path)
            else:
                if 'onevision' in model_name:
                    text_output = llava_ov_inference(image, sys_prompt, user_prompt, model, tokenizer, processor, device)
                else:
                    text_output = llava_next_inference_latest(image, sys_prompt, user_prompt, model, model_name, processor, device)
            end_time = time.time()
            elapsed_time = end_time - start_time
            inference_time += elapsed_time

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

                results = eval_text(gt_dest_desc, text_output)
                tag_scores, avg_scores = update_and_logging_results(tag_scores,
                                                                    avg_scores,
                                                                    xml_filename,
                                                                    gt_tag,
                                                                    file_count,
                                                                    results,
                                                                    logger)

        avg_infer_time = ((avg_infer_time * (file_count - 1)) + inference_time) / file_count
        logger.info(f"{file_count}: AVG. Inference Time: {avg_infer_time} secs")
        # Update running averages
        file_count += 1

    return tag_scores, avg_scores, avg_infer_time


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
        model_name = get_model_name_from_path(args.model_ckpt_path)
        if 'onevision' in args.model_ckpt_path:
            device = "cuda"
            device_map = "auto"
            tokenizer, model, processor, max_length = load_pretrained_model(args.model_ckpt_path, None, model_name, device_map=device_map)
            model.eval()

        else:
            processor = LlavaNextProcessor.from_pretrained(args.model_ckpt_path)
            model = LlavaNextForConditionalGeneration.from_pretrained(args.model_ckpt_path,
                                                                      torch_dtype=torch.float16,
                                                                      # low_cpu_mem_usage=True,
                                                                      device_map="auto")
            device = "cuda"
            tokenizer = None
            model.eval()
            #model.to(device)

    elif 'gpt' in args.model_ckpt_path:

        model_name = 'gpt'
        model = gpt_wrapper(args.model_ckpt_path, 'use-your-openai-key')
        tokenizer = None
        processor = None
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

    tag_scores, avg_scores, avg_infer_time = evaluate_zero_shot_vlm(filtered_anno_list, model, model_name, tokenizer, processor, device, logger)

    current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file_name = f"{model_ckpt_name}_{current_time}.txt"
    output_file_path = os.path.join(output_dir, output_file_name)

    result = {
        "avg_scores": avg_scores,
        "avg_inference_sec": avg_infer_time,
        "tag_scores": tag_scores
    }

    # Write the result to the text file
    with open(output_file_path, 'w') as f:
        f.write(json.dumps(result, indent=4))

    print(f"Average scores and tag scores saved to {output_file_path}")


if __name__ == "__main__":
    main()
