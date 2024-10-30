import os
import json
import argparse
import re
from translate_xmls import get_decision

def parse_args():
    parser = argparse.ArgumentParser(
        description='Generate json file for the llava training framework')

    parser.add_argument(
        '--gt_dir', default='/mnt/data_disk/dbs/gd_space_aware/auto_gen/d2024.09.08_20k/',
        metavar='DIRECTORY for ground truth',
        help='directory which contains images and object properties')

    parser.add_argument(
        '--output_filename', default='llava_gd_space_aware.json',
        metavar='FILEPATH',
        help='name for an output json file')

    return parser.parse_args()


def generate_query(base_sentence, xy):
    # 입력된 x, y 값을 소수점 셋째 자리까지 형식화
    formatted_x = f"{xy[0]:.3f}"
    formatted_y = f"{xy[1]:.3f}"

    # 새로운 문장을 생성
    new_sentence = f"<image>\n{base_sentence} ({formatted_x}, {formatted_y})."

    return new_sentence

def json_to_text(data):
    # JSON 문자열을 파이썬 딕셔너리로 변환

    # 소수점 셋째 자리까지만 표현하는 함수
    def format_float(value):
        return f"{value:.3f}"

    # 결과 텍스트 포맷 작성
#     result = f"""
#         <dest_desc>{data['dest_desc']}</dest_desc>
#         <left_desc>{data['left_desc']}</left_desc>
#         <right_desc>{data['right_desc']}</right_desc>
#         <path_desc>{data['path_desc']}</path_desc>
#         <recommend>{data['recommend']}</recommend>
#         <summary_answer>{data['summary_answer']}</summary_answer>
#         <bboxes>"""
# #         <goal_object_label>{data['goal_object_label']}</goal_object_label>
#
#     for bbox in data['bboxes']:
#         result += f"""
#                     <bbox>
#                         <label>{bbox[0]}</label>
#                         <coordinates>
#                             <x1>{format_float(bbox[1][0])}</x1>
#                             <y1>{format_float(bbox[1][1])}</y1>
#                             <x2>{format_float(bbox[1][2])}</x2>
#                             <y2>{format_float(bbox[1][3])}</y2>
#                         </coordinates>
#                         <confidence>{format_float(bbox[2])}</confidence>
#                     </bbox>"""
#
#     result += """
#                 </bboxes>
#                 <path_array>"""
#
#
#     for point in data['path_array'][:10]:
#         result += f"""
#                     <point>
#                         <x>{format_float(point[0])}</x>
#                         <y>{format_float(point[1])}</y>
#                     </point>"""
#
#     result += """
#                 </path_array>
#                     """
        #

        #     result = f"""
        #         <dest_desc>{data['dest_desc']}</dest_desc>
        #         <left_desc>{data['left_desc']}</left_desc>
        #         <right_desc>{data['right_desc']}</right_desc>
        #         <path_desc>{data['path_desc']}</path_desc>
        #         <recommend>{data['recommend']}</recommend>
        #         <summary_answer>{data['summary_answer']}</summary_answer>
        #         <bboxes>"""

    result=f"""
        <dest_desc>{data['dest_desc']}</dest_desc>
        <left_desc>{data['left_desc']}</left_desc>
        <right_desc>{data['right_desc']}</right_desc>
        <path_desc>{data['path_desc']}</path_desc>
        <recommend>{data['recommend']}</recommend>
        <decision>{get_decision(data['recommend'])}</decision>
        <summary_answer>{data['summary_answer']}</summary_answer>
        """
    result += "<path_array>"
    points = [data['path_array'][5]]
    #for point in data['path_array'][:10]:
    for point in points:
        result += f"""
                    <point>
                        <x>{format_float(point[0])}</x>
                        <y>{format_float(point[1])}</y>
                    </point>"""

    result += "</path_array>"

    # 공백, 줄바꿈 및 탭 제거, 데이터 내용의 공백은 유지
    result = re.sub(r'>\s+<', '><', result)
    result = result.replace('\n', '').replace('\t', '')

    return result.strip()


def main():
    args = parse_args()

    qa_json_dir = os.path.join(args.gt_dir, 'qa_json')
    files = os.listdir(qa_json_dir)
    json_files = [f for f in files if f.endswith(('.json'))]

    gt_list = []
    for idx, json_file in enumerate(json_files):
        json_path = os.path.join(qa_json_dir, json_file)
        print(idx+1, ' : ', json_path)
        with open(json_path, 'r', encoding='utf-8') as fp:
            data = json.load(fp)

        element = {
            "id": data['filename'].split('.')[0],
            "image": 'original_images/'+data['filename'],
            "conversations": [
                {
                    "from": "human",
                    "value": generate_query('Please provide a brief walking guide for a visually impaired person by analyzing the image based on the goal position', data["goal_position_xy"])

                },
                {
                    "from": "gpt",
                    "value": json_to_text(data)
                },
            ]

        }

        gt_list.append(element)

    with open(os.path.join(args.gt_dir, args.output_filename), 'w', encoding='utf-8') as llava_json:
        print('saving ', os.path.join(args.gt_dir, args.output_filename), '...')
        json.dump(gt_list, llava_json, indent="\t", ensure_ascii=False)

    return


if __name__ == '__main__':
    main()
