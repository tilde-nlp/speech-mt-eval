import re


def remove_nonalpha(text: str):
    return ''.join([c for c in text if c.isalpha() or c == ' '])


def remove_nonalphanumeric(text: str):
    return ''.join([c for c in text if c.isalpha() or c.isdigit() or c == ' '])


def clean_text(text: str):
    text = text.replace("<unk>", "")
    text = re.sub(r'\([^()]*\)', '', text)  # remove parenthesis and their contents
    text = re.sub(r'\[[^()]*\]', '', text)  # remove parenthesis and their contents
    text = " ".join(text.split())  # remove repeating whitespaces
    return text


def fix_overlap(segs):
    """
    Fixes overlapping segments for SER calculation
    :param segs:
    :return:
    """
    for i in range(len(segs)-1):
        curr = segs[i+1]
        prev = segs[i]
        if curr[0] < prev[1]:
            segs[i] = (prev[0], curr[0])

