import string

def clean_text(x):
	x = x.encode('ascii')
	x = x.strip()
	x = x.lower()
	x = x.translate(None, string.punctuation)
	return x
	