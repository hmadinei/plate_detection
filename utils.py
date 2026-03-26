from __future__ import annotations
import numpy as np

from typing import List
from bidi.algorithm import get_display
import arabic_reshaper

# DIGITS = [str(i) for i in range(10)]
# LETTERS = [chr(c) for c in range(ord('A'), ord('Z') + 1)]
DIGITS = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
LETTERS = ['D', 'S', 'T', 'ا', 'ب', 'ت', 'ث', 'ج', 'د', 'ز', 'س', 'ش', 'ص', 'ط', 'ظ', 'ع', 'ف', 'ق', 'ل', 'م', 'ن', 'ه',
           'و', 'پ', 'ژ', 'ک', 'گ', 'ی']
ALL_CHARS = DIGITS + LETTERS
PATTERN = ["digit", "digit", "letter", "digit", "digit", "digit", "digit", "digit"]
ADV_FACTORS = [0, 1.0, 1.25, 1.25, 1.0, 1.0, 1.15, 0.95]
LETTER_INDEX = 2
PERSIAN_DIGITS = {
    "0": "۰",
    "1": "۱",
    "2": "۲",
    "3": "۳",
    "4": "۴",
    "5": "۵",
    "6": "۶",
    "7": "۷",
    "8": "۸",
    "9": "۹",
    "?": "؟"
}

# ----------------------------- Utility helpers ------------------------------

def is_digit(c: str) -> bool:
    return c in DIGITS


def is_letter(c: str) -> bool:
    return c in LETTERS

def is_unknown(ch: str) -> bool:
    return ch in ["?"] 


def softmax(xs: List[float], temp: float = 1.0) -> List[float]:
    if not xs:
        return []
    x = np.array(xs, dtype=float) / max(1e-6, temp)
    x = x - np.max(x)
    ex = np.exp(x)
    p = ex / np.sum(ex)
    return p.tolist()


def fix_persian_text(text):   
    reshaped_text = arabic_reshaper.reshape(text)
    bidi_text = get_display(reshaped_text)
    return bidi_text


def to_persian_numbers(text):
    return "".join(PERSIAN_DIGITS.get(ch, ch) for ch in text)


def make_rtl(data):
    # Unicode constants
    RLE = '\u202B'  # Start RTL
    PDF = '\u202C'  # End RTL

    return f"{RLE}{data}{PDF}"