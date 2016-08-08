#!/usr/bin/env python

import sys
import string
from collections import defaultdict
words = []
wordcounts = defaultdict(int)

def clean_text(line):
    words = []
    keep = "abcdefghijklmnopqrstuvwxyz0123456789"

    for word in line.split(" "):
        word=word.lower()
        word = ''.join(ch for ch in word if ch in keep)
        words.append(word)

    return words


for line in sys.stdin:
    # line = line.encode('ascii')
    line = line.strip()
    line = line.translate(None, string.punctuation)
    line = " ".join(clean_text(line))
    words.extend(word.lower() for word in line.split())

    # for word in words:
    #     wordcounts[word] = 1

# for w, n in sorted(wordcounts.items()):
#     print '%s\t%s' %(w,n)


# for w in words:
#     print w

worddict= {}
for word in words:
	if word in worddict:
		worddict[word] +=1
	else:
		worddict.update({word:1})

for w, n in sorted(wordcounts.items()):
    print '%s\t%s' %(w,n)