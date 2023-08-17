from multiprocessing import Pool
import pandas as pd
import os
import glob
import numpy as np
import pandas as pd
import requests
from bs4 import BeautifulSoup
import nltk
import string
import enchant
import sys
import re

if sys.version_info[0] > 2:
    unicode = str


df = pd.read_excel('Input.xlsx')


def scraper(list):
    start = list[0]
    end = list[1]
    print('From {} to {}:'.format(list[0], list[1]))
    headers = {
        'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10.12; rv:55.0) Gecko/20100101 Firefox/55.0',
    }
    for i in range(start-1, end):
        print(f'Page: {i}')
        blog_data = pd.DataFrame(
            columns=['URL_ID', 'URL', 'blog_title', 'blog_body'])
        link = df['URL'][i]
        response = requests.get(f'{link}', headers=headers).text
        soup = BeautifulSoup(response, features="lxml")
        if soup.find('div', class_="td-404-title"):
            continue
        blog_title = soup.find('header').h1.text
        blog_body = ""
        for div in soup.find_all('div', class_='td-post-content'):
            for p in div.find_all('p'):
                blog_body = blog_body + p.text
        name = int(df['URL_ID'][i])
        temp = pd.DataFrame({'URL_ID': [name], 'URL': [link], 'blog_title': [
                            blog_title], 'blog_body': [blog_body]})
        blog_data = pd.concat([blog_data, temp], ignore_index=True)
        blog_data.to_csv(f'Scraped_Data/{name}.csv')


def start_end(n, parts):
    lists = []
    ct = 0
    i = 1
    while (ct < parts):
        lists.append([i, i+int(n/parts)-1])
        i += int(n/parts)
        ct = ct+1
    return lists


def run_parallel(n, parts):
    # list of ranges to execute for each parallel process
    list_ranges = start_end(n, parts)  # start_end function defined above
    # pool object with number of elements in the list
    pool = Pool(processes=len(list_ranges))
    # map the function to the list and pass
    # function and list_ranges as arguments
    pool.map(scraper, list_ranges)


def get_stop_words():
    path = r'StopWords'  # use your path
    all_files = glob.glob(path + "/*.txt")
    stop_words = []
    for file in all_files:
        if file == 'StopWords\StopWords_Currencies.txt':
            with open(file, 'r') as f:
                stop_words.append(word.upper() for word in f.read().split())
        else:
            with open(file, 'r') as f:
                stop_words.append(f.read().splitlines())
    stop_words.append([*string.punctuation])
    stop_words = [item for sublist in stop_words for item in sublist]
    return stop_words


def stop_words_clean():
    path = r'Scraped_Data'  # use your path
    all_files = glob.glob(path + "/*.csv")
    scraped_df = pd.DataFrame(
        columns=['URL_ID', 'URL', 'blog_title', 'blog_body', 'blog_body_without_sw'])
    global cleaned_body_length
    cleaned_body_length = []
    for filename in all_files:
        df = pd.read_csv(filename)
        URL_ID = df['URL_ID'].values[0]
        URL = df['URL'].values[0]
        blog_title = df['blog_title'].values[0]
        blog_body = df['blog_body'].values[0]
        temp = pd.DataFrame({'URL_ID': [URL_ID], 'URL': [URL], 'blog_title': [
                            blog_title], 'blog_body': [blog_body]})
        scraped_df = pd.concat([scraped_df, temp], ignore_index=True)
    # scraped_df.to_csv('Scraped_Data/scraped_df.csv')
    # scraped_df.drop('Unnamed: 0',axis=1,inplace=True)
    stop_words = get_stop_words()
    for i in scraped_df.index:
        text = scraped_df['blog_body'][i]
        words = [word for word in nltk.word_tokenize(
            text) if word.upper() not in stop_words]
        new_text = " ".join(words)
        cleaned_body_length.append(len(words)+0.000001)
        scraped_df['blog_body_without_sw'][i] = new_text
    scraped_df.to_csv('Scraped_Data/cleaned_data.csv')


def get_negative_words():
    with open('MasterDictionary/negative-words.txt', 'r') as f:
        negative_words = (f.read().splitlines())
    return negative_words


def get_positive_words():
    with open('MasterDictionary/positive-words.txt', 'r') as f:
        positive_words = (f.read().splitlines())
    return positive_words


def derived_variables():
    df = pd.read_csv('Scraped_Data/cleaned_data.csv')
    df['POSITIVE SCORE'] = ''
    df['NEGATIVE SCORE'] = ''
    positive_words = get_positive_words()
    negative_words = get_negative_words()
    for i in df.index:
        text = df['blog_body'][i]
        pos_words_list = [word for word in nltk.word_tokenize(
            text) if word.lower() in positive_words]
        neg_words_list = [word for word in nltk.word_tokenize(
            text) if word.lower() in negative_words]
        df.loc[i, 'POSITIVE SCORE'] = len(pos_words_list)
        df.loc[i, 'NEGATIVE SCORE'] = len(neg_words_list)
        df['POSITIVE SCORE'] = pd.to_numeric(
            df['POSITIVE SCORE'], errors='coerce').fillna(0).astype(float)
        df['NEGATIVE SCORE'] = pd.to_numeric(
            df['NEGATIVE SCORE'], errors='coerce').fillna(0).astype(float)
        total_score = df['POSITIVE SCORE']+df['NEGATIVE SCORE']
        df['POLARITY SCORE'] = (df['POSITIVE SCORE'] -
                                df['NEGATIVE SCORE'])/(total_score+0.000001)
        df['SUBJECTIVITY SCORE'] = (
            df['POSITIVE SCORE']+df['NEGATIVE SCORE'])/(cleaned_body_length)
        df['SUBJECTIVITY SCORE'] = pd.to_numeric(
            df['SUBJECTIVITY SCORE'], errors='coerce').fillna(0).astype(float)
    df.drop('Unnamed: 0', axis=1, inplace=True)
    df['AVG SENTENCE LENGTH'] = ''
    df['PERCENTAGE OF COMPLEX WORDS'] = ''
    df['FOG INDEX'] = ''
    df['AVG NUMBER OF WORDS PER SENTENCE'] = ''
    df['COMPLEX WORD COUNT'] = ''
    df['WORD COUNT'] = ''
    df['SYLLABLE PER WORD'] = ''
    df['PERSONAL PRONOUNS'] = ''
    df['AVG WORD LENGTH'] = ''
    df.to_csv('Scraped_Data/cleaned_data.csv')


def syllables(word):
    word = word.lower()
    word = word + " "  # word extended
    length = len(word)
    ending = ["ing ", "ed ", "es ", "ous ", "tion ",
              "nce ", "ness "]  # not included in complex words
    vowels = "aeiouy"

    for end in ending:
        x = word.find(end)
        if x > -1:
            x = length - x
            word = word[:-x]
    syllable_count = 0
    if word[-1] == " ":
        word = word[:-1]
    # removing the extra " " at the end if failed and dropping last letter if e
    if word[-1] == "e":
        try:
            if word[-3:] == "nce" and word[-3:] == "rce":
                syllable_count = 0

            elif word[-3] not in vowels and word[-2] not in vowels and word[-3:] != "nce" and word[-3:] != "rce":
                if word[-3] != "'":
                    syllable_count += 1  # e cannot be dropped as it contributes to a syllable
            word = word[:-1]
        except IndexError:
            syllable_count += 0

    one_syllable_beg = ["ya", "ae", "oe", "ea", "yo", "yu", "ye"]
    two_syllables = ["ao", "uo", "ia", "eo", "ea", "uu",
                     "eous", "uou", "ii", "io", "ua", "ya", "yo", "yu", "ye"]
    last_letter = str()  # last letter is null for the first alphabet
    for index, alphabet in enumerate(word):
        if alphabet in vowels:
            current_combo = last_letter + alphabet
            if len(current_combo) == 1:  # if it's the first alphabet
                try:
                    if word[1] not in vowels:  # followed by a consonant, then one syllable
                        syllable_count += 1
                        last_letter = word[1]
                    else:
                        syllable_count += 1  # followed by a vowel
                        last_letter = alphabet
                except IndexError:
                    syllable_count += 1  # followed by a vowel
                    last_letter = alphabet

            else:
                if current_combo in two_syllables:
                    try:
                        # if they're only 1 syllable at the beginning of a word, don't increment
                        if current_combo == word[:2] and current_combo in one_syllable_beg:
                            syllable_count += 0
                        elif word[index - 2] + current_combo + word[index + 1] == "tion" or word[index - 2] + current_combo + \
                                word[index + 1] == "sion":  # here io is one syllable :
                            syllable_count += 0
                        else:
                            syllable_count += 1  # vowel combination forming 2 syllables
                        last_letter = alphabet
                    except IndexError:
                        syllable_count += 0
                else:  # two vowels as well as non vowel combination
                    if last_letter not in vowels:
                        syllable_count += 1
                        last_letter = alphabet
                    else:
                        last_letter = alphabet
        else:
            last_letter = alphabet
    if word[-3:] == "ier":  # word ending with ier has 2 syllables
        syllable_count += 1
    return syllable_count


def __concat(object1, object2):
    if isinstance(object1, str) or isinstance(object1, unicode):
        object1 = [object1]
    if isinstance(object2, str) or isinstance(object2, unicode):
        object2 = [object2]
    return object1 + object2


def __capitalize_first_char(word):
    return word[0].upper() + word[1:]


def __split(word, language='en_US'):
    dictionary = enchant.Dict(language)
    max_index = len(word)
    if max_index < 3:
        return word
    for index, char in enumerate(word, 2):
        left_word = word[0:index]
        right_word = word[index:]
        if index == max_index - 1:
            break
        if dictionary.check(left_word) and dictionary.check(right_word):
            return [compound for compound in __concat(left_word, right_word)]
    return word


def split(compound_word, language='en_US'):
    words = compound_word.split('-')
    word = ""
    for x in words:
        word += x
    result = __split(word, language)
    if result == compound_word:
        return [result]
    return result


def length_all():
    words = file_contents.split()
    length_of_all_words = 0
    for word in words:
        length_of_all_words += len(word)
    return length_of_all_words


def syllables_per_words():
    words = file_contents.split()
    syllables_per_word = []
    for word in words:
        syllables_per_word.append(syllables(word))
    return syllables_per_word


def complexwords():
    syllablecount = 0
    beg_each_Sentence = re.findall(r"\.\s*(\w+)", file_contents)
    capital_words = re.findall(r"\b[A-Z][a-z]+\b", file_contents)
    words = file_contents.split()
    for word in words:
        # all lower case words
        if word not in capital_words and len(word) >= 3:
            if syllables(word) >= 3 and len(split(word)) == 1:
                syllablecount += 1
        if word in capital_words and word in beg_each_Sentence:  # beginning of each sentence is uppercase
            if syllables(word) >= 3:
                syllablecount += 1
    return syllablecount


def wordcount():
    # Regex to match all words, hyphenated words count as a compound words
    return len(re.findall("[a-zA-Z-]+", file_contents))


def sentencecount():
    # regex to count sentences, can end with a period, "?" or "!"
    return (len(re.split("[.!?]+", file_contents))-1)


def pp(text):
    pronounRegex = re.compile(r'\b(I|we|my|ours|(?-i:us))\b', re.I)
    pronouns = pronounRegex.findall(text)
    return pronouns


def analysis(list):
    data = pd.read_csv('Scraped_Data/cleaned_data.csv')
    start = list[0]
    end = list[1]
    print('From {} to {}:'.format(list[0], list[1]))
    global file_contents
    for i in range(start-1, end):
        print(f"Page: {i}")
        file_contents = data['blog_body'][i]
        try:
            word_count = wordcount()
            sentence_count = sentencecount()
            complex_words = complexwords()
            personal_pronouns = pp(file_contents)
            syllables_per_word = syllables_per_words()
            length_of_all_words = length_all()
            fog_index_calculated = (
                (word_count/sentence_count) + complex_words)*0.4
            gunning_fog_index = (
                (word_count/sentence_count) + 100*(complex_words/word_count))*0.4
        except ZeroDivisionError:
            fog_index_calculated = gunning_fog_index = 0
        data.loc[i, 'AVG SENTENCE LENGTH'] = word_count/sentence_count
        data.loc[i, 'PERCENTAGE OF COMPLEX WORDS'] = complex_words/word_count
        data.loc[i, 'FOG INDEX'] = gunning_fog_index
        data.loc[i, 'AVG NUMBER OF WORDS PER SENTENCE'] = word_count / \
            sentence_count
        data.loc[i, 'COMPLEX WORD COUNT'] = complex_words
        data.loc[i, 'SYLLABLE PER WORD'] = sum(
            syllables_per_word) / len(syllables_per_word)
        data.loc[i, 'PERSONAL PRONOUNS'] = len(personal_pronouns)
        data.loc[i, 'AVG WORD LENGTH'] = length_of_all_words/word_count
    data.to_csv('Scraped_Data/cleaned_data.csv')


def cleaned_word_ops(list):
    global cleaned_file_contents
    data = pd.read_csv('Scraped_Data/cleaned_data.csv')
    start = list[0]
    end = list[1]
    print('From {} to {}:'.format(list[0], list[1]))
    for i in range(start-1, end):
        print(f"Page: {i}")
        cleaned_file_contents = data['blog_body_without_sw'][i]
        try:
            word_count = wordcount()
        except ZeroDivisionError:
            word_count = 0
        data.loc[i, 'WORD COUNT'] = word_count
    data.to_csv('Scraped_Data/cleaned_data.csv')


def run_parallel_2(n, parts):
    # list of ranges to execute for each parallel process
    list_ranges = start_end(n, parts)  # start_end function defined above
    # pool object with number of elements in the list
    pool = Pool(processes=len(list_ranges))
    # map the function to the list and pass
    # function and list_ranges as arguments
    pool.map(analysis, list_ranges)


def save():
    d = pd.read_csv('Scraped_Data/cleaned_data.csv', usecols=['URL_ID', 'URL', 'POSITIVE SCORE', 'NEGATIVE SCORE', 'POLARITY SCORE', 'SUBJECTIVITY SCORE', 'AVG SENTENCE LENGTH',
                    'PERCENTAGE OF COMPLEX WORDS', 'FOG INDEX', 'AVG NUMBER OF WORDS PER SENTENCE', 'COMPLEX WORD COUNT', 'WORD COUNT', 'SYLLABLE PER WORD', 'PERSONAL PRONOUNS', 'AVG WORD LENGTH'])
    d['URL_ID'].astype('int')
    d.sort_values(by=['URL_ID'], ascending=True, inplace=True)
    d.set_index("URL_ID", inplace=True)
    d.to_csv('Output.csv')


if __name__ == '__main__':
    final_dir = os.path.join(os.getcwd(), r'Scraped_Data')
    os.mkdir(final_dir)
    run_parallel(114, 19)
    stop_words_clean()
    print('Scraping Completed')
    derived_variables()
    analysis([1, 111])
    cleaned_word_ops([1, 111])
    save()
    print('Analysis Completed')
    print('Output stored in Output.csv')
