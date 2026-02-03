def calculate_average_sentence_length(file_path='/home/ubuntu/Baichuan_Harmony/APSEC-OUTPUT/ourapproach_codex_5shot.txt'):
    total_words = 0
    total_sentences = 0

    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            for line in file:
                line = line.strip()  # 去除行首尾的空白和换行符
                if not line:
                    continue  # 跳过空行

                words = line.split()  # 按空格分割单词
                total_words += len(words)
                total_sentences += 1

        if total_sentences == 0:
            print("文件中没有可统计的句子。")
        else:
            average_length = total_words / total_sentences
            print(f"所有句子的单词总数: {total_words}")
            print(f"句子的平均长度: {average_length:.2f} 个单词")

    except FileNotFoundError:
        print(f"文件 {file_path} 未找到，请确认路径是否正确。")

if __name__ == "__main__":
    calculate_average_sentence_length()