from openai import OpenAI
import xml.etree.ElementTree as ET
import glob
import os
import argparse
from eval.private import GPT_API_KEY

# OpenAI API 키 설정
client = OpenAI(api_key=GPT_API_KEY)

def load_and_parse_xml(file_path):
    tree = ET.parse(file_path)
    root = tree.getroot()
    return tree, root

def save_xml(tree, output_file_path):
    tree.write(output_file_path, encoding='utf-8', xml_declaration=True)

def translate_xml_elements(root, tags_to_translate):
    for elem in root.iter():
        if elem.tag in tags_to_translate and elem.text:
            elem.text = translate_text(elem.text)

def translate_text(text, dest_lang='en'):
    if not text:
        return text

    print("orgin: ", text)

    response = (client.chat.completions.create
                (model="gpt-4o",
                messages=[
                        {"role": "system", "content": "You are a helpful assistant that translates text to English."},
                        {"role": "user", "content": f"Translate the following text to an English sentence considering walking guide context.: '{text}'. Only answer please."}
                         ],
                max_tokens=1024,
                temperature=0.0))
    translated_text = response.choices[0].message.content.strip()
    # remove ", '
    translated_text = translated_text.replace('"', '')
    translated_text = translated_text.replace("'", '')
    print("translated: ", translated_text)

    return translated_text

def main():

    parser = argparse.ArgumentParser(description='Evaluate Space-aware Vision-Language Model')
    parser.add_argument('--xml_root_dir', type=str, default='/mnt/data_disk/dbs/gd_space_aware/manual/m240829', help='root dir of xml files')
    parser.add_argument('--dest_dir', type=str, default='/mnt/data_disk/dbs/gd_space_aware/manual/m240829en', help='destination root dir of xml files')
    args = parser.parse_args()

    xml_list = glob.glob(f'{args.xml_root_dir}/**/*xml', recursive=True)

    for idx, xml_path in enumerate(xml_list):

        # 사용 예시
        input_xml_file = xml_path  # 입력 XML 파일 경로
        output_xml_file =  xml_path.replace(args.xml_root_dir, args.dest_dir) # 출력 XML 파일 경로

        out_dir = os.path.dirname(output_xml_file)
        # 디렉토리가 없으면 생성
        if not os.path.exists(out_dir):
            os.makedirs(out_dir)

        # 번역할 태그 목록 240722 version
        # tags_to_translate = [
        #     'folder', 'dest', 'left', 'front_left', 'front',
        #     'front_right', 'right', 'path', 'recommend',  'direction',
        #     'location', 'path_shape', 'action', 'reason',
        #     'position', 'type'
        # ]

        # 240819 version
        tags_to_translate = [
            'folder', 'simple', 'complex', 'recommend',
        ]

        # XML 파일 로드 및 파싱
        tree, root = load_and_parse_xml(input_xml_file)

        # XML 요소 번역e
        translate_xml_elements(root, tags_to_translate)

        # 번역된 XML 파일 저장
        save_xml(tree, output_xml_file)

        print(f"{idx+1}: Translated XML saved to: {output_xml_file}")

if __name__ == "__main__":
    main()
