#!/usr/bin/env python

import sys
import string
from collections import defaultdict

import my_cleaner

wordcounts = defaultdict(int)

for line in sys.stdin:
    # line = line.encode('ascii')
    # line = line.strip()
    # line = line.translate(None, string.punctuation)
    # words = [word.lower() for word in line.split()]
    line = my_cleaner.clean_text(line)
    words = [word for word in line.split()]

    for word in words:
        wordcounts[word] += 1

for w, n in sorted(wordcounts.items()):
    print '%s\t%s' %(w,n)