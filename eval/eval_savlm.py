from llava.model.builder import load_pretrained_model
from llava.mm_utils import get_model_name_from_path, process_images, tokenizer_image_token
from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN, IGNORE_INDEX
from llava.conversation import conv_templates, SeparatorStyle

from PIL import Image
import requests
import copy
import torch
import argparse
import glob
import xml.etree.ElementTree as ET
from nltk.translate.meteor_score import meteor_score
from bert_score import score
from rouge_score import rouge_scorer
import os
import numpy as np
from datetime import datetime
import json
import re

def eval_text(gt_text, infer_text):

    gt_tokens = gt_text.split()
    infer_tokens = infer_text.split()
    m_score = meteor_score([gt_tokens], infer_tokens)
    # print(f"METEOR Score: {m_score}")

    P, R, F1 = score([infer_text], [gt_text], lang="en", model_type="bert-base-uncased")
    # print(f"BERTScore Precision: {P.mean().item()}")
    # print(f"BERTScore Recall: {R.mean().item()}")
    # print(f"BERTScore F1: {F1.mean().item()}")
    b_scores = {
        'precision': P.mean().item(),
        'recall': R.mean().item(),
        'f1': F1.mean().item()
    }


    r_scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
    r_scores = r_scorer.score(gt_text, infer_text)
    # print("ROUGE-1: ", r_scores['rouge1'])
    # print("ROUGE-2: ", r_scores['rouge2'])
    # print("ROUGE-L: ", r_scores['rougeL'])

    return m_score, b_scores, r_scores


def _evaluate_vlm(anno_list, model, image_processor, tokenizer, device):


    eval_gt_degree = 'simple' # or 'complex'
    eval_gt_tags = ['dest', 'left', 'right', 'path', 'recommend']
    for i, tag in enumerate(eval_gt_tags):
        eval_gt_tags[i] = './/' + tag + '/' + eval_gt_degree
    eval_infer_tags = ['dest_desc', 'left_desc', 'right_desc', 'path_desc', 'recommend']

    tag_scores = {}

    for xml_idx, xml_path in enumerate(anno_list):

        xml_filename = os.path.basename(xml_path)
        gt_tree = ET.parse(xml_path)
        gt_root = gt_tree.getroot()
        gp_x = gt_root.find("goal_position/x")
        gp_y = gt_root.find("goal_position/y")

        tag_scores[xml_filename] = {
            'm_scores': {tag: [] for tag in eval_gt_tags},
            'b_scores': {tag: {'precision': [], 'recall': [], 'f1': []} for tag in eval_gt_tags},
            'r_scores': {tag: {'rouge1': [], 'rouge2': [], 'rougeL': []} for tag in eval_gt_tags},
        }

        img_path = xml_path.replace('xml', 'JPG')
        if os.path.exists(img_path) != True:
            img_path = xml_path.replace('xml', 'jpeg')

        image = Image.open(img_path)
        image_tensor = process_images([image], image_processor, model.config)
        image_tensor = [_image.to(dtype=torch.float16, device=device) for _image in image_tensor]

        conv_template = "llama_v3"  # Make sure you use correct chat template for different models
        question = (DEFAULT_IMAGE_TOKEN +
                    f"\nPlease provide a brief walking guide for a visually impaired person "
                    f"by analyzing the image based on the goal position ({gp_x:.3f}, {gp_y:.3f}). "
                    f"RULE: If a tag is open, it should be closed.")
        conv = copy.deepcopy(conv_templates[conv_template])
        conv.add_generation_prompt = True
        conv.append_message(conv.roles[0], question)
        conv.append_message(conv.roles[1], None)
        prompt_question = conv.get_prompt()
        input_ids = tokenizer_image_token(prompt_question,
                                          tokenizer,
                                          IMAGE_TOKEN_INDEX,
                                          return_tensors="pt").unsqueeze(0).to(device)
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
        print('<root>' + text_outputs[0] + '</root>')
        output_root = ET.fromstring('<root>' + text_outputs[0] + '</root>')

        for i, gt_tag in enumerate(eval_gt_tags):

            output_dest_desc = output_root.find(eval_infer_tags[i]).text.strip()
            gt_dest_desc = gt_root.find(gt_tag).text.strip()
            m_score, b_scores, r_scores = eval_text(gt_dest_desc, output_dest_desc)

            tag_scores[xml_filename]['m_scores'][gt_tag].append(m_score)
            tag_scores[xml_filename]['b_scores'][gt_tag]['precision'].append(b_scores['precision'])
            tag_scores[xml_filename]['b_scores'][gt_tag]['recall'].append(b_scores['recall'])
            tag_scores[xml_filename]['b_scores'][gt_tag]['f1'].append(b_scores['f1'])
            tag_scores[xml_filename]['r_scores'][gt_tag]['rouge1'].append(r_scores['rouge1'].fmeasure)
            tag_scores[xml_filename]['r_scores'][gt_tag]['rouge2'].append(r_scores['rouge2'].fmeasure)
            tag_scores[xml_filename]['r_scores'][gt_tag]['rougeL'].append(r_scores['rougeL'].fmeasure)


        # 중간 평균 계산 및 출력
        print(f"Intermediate results after {xml_idx + 1}/{len(anno_list)} files:")
        for gt_tag in eval_gt_tags:
            avg_m_score = np.mean([np.mean(tag_scores[xml_filename]['m_scores'][gt_tag]) for xml_filename in tag_scores])
            avg_b_scores = {
                key: np.mean([np.mean(tag_scores[xml_filename]['b_scores'][gt_tag][key]) for xml_filename in tag_scores])
                for key in ['precision', 'recall', 'f1']
            }
            avg_r_scores = {
                key: np.mean([np.mean(tag_scores[xml_filename]['r_scores'][gt_tag][key]) for xml_filename in tag_scores])
                for key in ['rouge1', 'rouge2', 'rougeL']
            }

            print(f"Tag: {gt_tag}")
            print(f"  Avg METEOR Score: {avg_m_score}")
            print(f"  Avg BERTScore Precision: {avg_b_scores['precision']}")
            print(f"  Avg BERTScore Recall: {avg_b_scores['recall']}")
            print(f"  Avg BERTScore F1: {avg_b_scores['f1']}")
            print(f"  Avg ROUGE-1: {avg_r_scores['rouge1']}")
            print(f"  Avg ROUGE-2: {avg_r_scores['rouge2']}")
            print(f"  Avg ROUGE-L: {avg_r_scores['rougeL']}")

    # 최종 평균 계산 및 반환
    avg_scores = {
        'm_scores': {tag: np.mean(tag_scores['m_scores'][tag]) for tag in eval_gt_tags},
        'b_scores': {
            tag: {key: np.mean(tag_scores['b_scores'][tag][key]) for key in ['precision', 'recall', 'f1']}
            for tag in eval_gt_tags
        },
        'r_scores': {
            tag: {key: np.mean(tag_scores['r_scores'][tag][key]) for key in ['rouge1', 'rouge2', 'rougeL']}
            for tag in eval_gt_tags
        }
    }

    return tag_scores, avg_scores


def evaluate_vlm(anno_list, model, image_processor, tokenizer, device):

    eval_gt_degree = 'simple'  # or 'complex'
    eval_gt_tags = ['dest', 'left', 'right', 'path']
    for i, tag in enumerate(eval_gt_tags):
        eval_gt_tags[i] = './/' + tag + '/' + eval_gt_degree
    eval_gt_tags.append('.//recommend')
    eval_infer_tags = ['dest_desc', 'left_desc', 'right_desc', 'path_desc', 'recommend']

    tag_scores = {}
    avg_scores = {
        'm_scores': {tag: 0.0 for tag in eval_gt_tags},
        'b_scores': {tag: {'precision': 0.0, 'recall': 0.0, 'f1': 0.0} for tag in eval_gt_tags},
        'r_scores': {tag: {'rouge1': 0.0, 'rouge2': 0.0, 'rougeL': 0.0} for tag in eval_gt_tags},
    }
    file_count = 0

    for xml_idx, xml_path in enumerate(anno_list):

        xml_filename = os.path.basename(xml_path)
        tag_scores[xml_filename] = {
            'm_scores': {tag: [] for tag in eval_gt_tags},
            'b_scores': {tag: {'precision': [], 'recall': [], 'f1': []} for tag in eval_gt_tags},
            'r_scores': {tag: {'rouge1': [], 'rouge2': [], 'rougeL': []} for tag in eval_gt_tags},
        }

        gt_tree = ET.parse(xml_path)
        gt_root = gt_tree.getroot()
        gp_x = gt_root.find("goal_position/x").text.strip()
        gp_y = gt_root.find("goal_position/y").text.strip()
        # from this line
        gp_x = f"{float(gp_x):.3f}"
        gp_y = f"{float(gp_y):.3f}"
        gp_xy = f"({gp_x}, {gp_y})"

        img_path = xml_path.replace('xml', 'JPG')
        if not os.path.exists(img_path):
            img_path = xml_path.replace('xml', 'jpeg')

        image = Image.open(img_path)
        image_tensor = process_images([image], image_processor, model.config)
        image_tensor = [_image.to(dtype=torch.float16, device=device) for _image in image_tensor]

        conv_template = "llama_v3"  # Make sure you use correct chat template for different models
        question = f"<image>\nPlease provide a brief walking guide for a visually impaired person by analyzing the image based on the goal position {gp_xy}."
        print("query: ", question)
        conv = copy.deepcopy(conv_templates[conv_template])
        conv.add_generation_prompt = True
        conv.append_message(conv.roles[0], question)
        conv.append_message(conv.roles[1], None)
        prompt_question = conv.get_prompt()
        input_ids = tokenizer_image_token(prompt_question,
                                          tokenizer,
                                          IMAGE_TOKEN_INDEX,
                                          return_tensors="pt").unsqueeze(0).to(device)
        image_sizes = [image.size]

        cont = model.generate(
            input_ids,
            images=image_tensor,
            image_sizes=image_sizes,
            do_sample=False,
            temperature=0,
            max_new_tokens=256,
        )

        text_outputs = tokenizer.batch_decode(cont, skip_special_tokens=True)

        # 태그를 자동으로 닫음
        #xml_output = fix_broken_xml('<root>' + text_outputs[0] + '</root>')

        # 파싱 가능한 XML로 변경된 내용을 출력
        print(text_outputs[0])
        output_root = ET.fromstring('<root>' + text_outputs[0] + '</root>')
        #print(ET.tostring(output_root, encoding='unicode'))

        # Update running averages
        file_count += 1

        for i, gt_tag in enumerate(eval_gt_tags):

            output_desc = output_root.find(eval_infer_tags[i])
            if output_desc == None:
                continue
            else:
                output_desc = output_desc.text.strip()

            gt_dest_desc = gt_root.find(gt_tag).text.strip()
            m_score, b_scores, r_scores = eval_text(gt_dest_desc, output_desc)

            tag_scores[xml_filename]['m_scores'][gt_tag].append(m_score)
            tag_scores[xml_filename]['b_scores'][gt_tag]['precision'].append(b_scores['precision'])
            tag_scores[xml_filename]['b_scores'][gt_tag]['recall'].append(b_scores['recall'])
            tag_scores[xml_filename]['b_scores'][gt_tag]['f1'].append(b_scores['f1'])
            tag_scores[xml_filename]['r_scores'][gt_tag]['rouge1'].append(r_scores['rouge1'].fmeasure)
            tag_scores[xml_filename]['r_scores'][gt_tag]['rouge2'].append(r_scores['rouge2'].fmeasure)
            tag_scores[xml_filename]['r_scores'][gt_tag]['rougeL'].append(r_scores['rougeL'].fmeasure)

            avg_scores['m_scores'][gt_tag] += (m_score - avg_scores['m_scores'][gt_tag]) / file_count
            avg_scores['b_scores'][gt_tag]['precision'] += (b_scores['precision'] - avg_scores['b_scores'][gt_tag]['precision']) / file_count
            avg_scores['b_scores'][gt_tag]['recall'] += (b_scores['recall'] - avg_scores['b_scores'][gt_tag]['recall']) / file_count
            avg_scores['b_scores'][gt_tag]['f1'] += (b_scores['f1'] - avg_scores['b_scores'][gt_tag]['f1']) / file_count
            avg_scores['r_scores'][gt_tag]['rouge1'] += (r_scores['rouge1'].fmeasure - avg_scores['r_scores'][gt_tag]['rouge1']) / file_count
            avg_scores['r_scores'][gt_tag]['rouge2'] += (r_scores['rouge2'].fmeasure - avg_scores['r_scores'][gt_tag]['rouge2']) / file_count
            avg_scores['r_scores'][gt_tag]['rougeL'] += (r_scores['rougeL'].fmeasure - avg_scores['r_scores'][gt_tag]['rougeL']) / file_count

        # 중간 평균 출력
        print(f"Intermediate results after {xml_idx + 1}/{len(anno_list)} files:")
        for gt_tag in eval_gt_tags:
            print(f"Tag: {gt_tag}")
            print(f"  Avg METEOR Score: {avg_scores['m_scores'][gt_tag]}")
            print(f"  Avg BERTScore Precision: {avg_scores['b_scores'][gt_tag]['precision']}")
            print(f"  Avg BERTScore Recall: {avg_scores['b_scores'][gt_tag]['recall']}")
            print(f"  Avg BERTScore F1: {avg_scores['b_scores'][gt_tag]['f1']}")
            print(f"  Avg ROUGE-1: {avg_scores['r_scores'][gt_tag]['rouge1']}")
            print(f"  Avg ROUGE-2: {avg_scores['r_scores'][gt_tag]['rouge2']}")
            print(f"  Avg ROUGE-L: {avg_scores['r_scores'][gt_tag]['rougeL']}")

    return tag_scores, avg_scores

def main():

    parser = argparse.ArgumentParser(description='Evaluate Space-aware Vision-Language Model')
    parser.add_argument('--model_ckpt_path', type=str, default="./checkpoints/llama3-llava-next-8b-lora-20240716_14_36_55", help='ckpt directory')
#    parser.add_argument('--model_ckpt_path', type=str,
#                        default="./checkpoints/llama3-llava-next-8b-lora-20240822_13_55_13", help='ckpt directory')
    parser.add_argument('--model_base_path', type=str, default="lmms-lab/llama3-llava-next-8b", help='base model directory')
    parser.add_argument('--eval_db_dir', type=str, default='/mnt/data_disk/dbs/gd_space_aware/manual/240819en', help='dataset directory for evaluation')
    parser.add_argument('--output_dir', type=str, default='./eval_results', help='directory to save the output scores')
    args = parser.parse_args()


    # model setting
    model_name = get_model_name_from_path(args.model_ckpt_path)
    print(model_name)
    device = 'cuda'
    device_map = 'auto'

    # Add any other thing you want to pass in llava_model_args
    tokenizer, model, image_processor, max_length = load_pretrained_model(args.model_ckpt_path,
                                                                          args.model_base_path,
                                                                          model_name,
                                                                          device_map=device_map)
    print('max_length=', max_length)
    model.eval()
    model.tie_weights()
    model.config.image_aspect_ratio = 'pad'  # or "anyres" or None

    # data setting
    anno_list = glob.glob(f'{args.eval_db_dir}/**/*xml', recursive=True)

    tag_scores, avg_scores = evaluate_vlm(anno_list, model, image_processor, tokenizer, device)

    # Create output directory if it doesn't exist
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    # Create the output file name
    model_ckpt_name = os.path.basename(args.model_ckpt_path)
    current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file_name = f"{model_ckpt_name}_{current_time}.txt"
    output_file_path = os.path.join(args.output_dir, output_file_name)

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