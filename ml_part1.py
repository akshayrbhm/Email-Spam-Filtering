# GROUP NUMBER : 27
# GROUP MEMBERS:
#	Akshay Ramesh Bhivagade	(22CL60R16)
#	Anubhav Dhar			(20CS30004)
#	Rahul Arvind Mool		(22CS60R72)
#
# EMAIL SPAM FILTER USING NAIVE BAYES CLASSIFIER LEARNING MODEL (SFNB)
#





import pandas as pd
import numpy as np
import seaborn as sb
import matplotlib.pyplot as plt

# Done
import re
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords,wordnet
from nltk.stem import WordNetLemmatizer
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('omw-1.4')
def clean_text(text,boolean):
    # Lowercase the text
    text = text.lower()
    #Clean the text
    cleaned_text = re.sub("[\r\n]", "", text)
    # Tokenize the text
    tokens = word_tokenize(cleaned_text)
    if boolean == False:
        return tokens
    # Remove stop words
    stop_words = set(stopwords.words('english'))
    tokens = [token for token in tokens if ((token not in stop_words)  and (not token.isnumeric()))]
    # Perform lemmatization
    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(token) for token in tokens]
    return tokens



# problem 1 
def problem_1(): 
	data = pd.read_csv("email_spam_dataset.csv")

	is_spam = data['label_num'].tolist()		# This contains the label for spam
	mail_text = data['text'].tolist()			# This contains the text

	N = len(is_spam)

	word_freq_spam = {}							# will contain the most freqent words in spam
	word_freq_ham = {}							# will contain the most freqent words in ham

	spam_count = 0
	ham_count = 0

	word_lens_spam = []
	word_lens_ham = []

	print("\n\n--------------------------------------------------")
	print("\nPROBLEM 1")


	# problem 1(a)
	for i in range(0, N):

		# determine spam_count and ham_count
		if is_spam[i] == 1:
			spam_count += 1
		else:
			ham_count += 1

		# do the text cleanup
		word_list = clean_text(mail_text[i], True) 		# cleanup the text
		for word in word_list:							# extract words from text
			if word.isalpha():							# only consider words which are alphabetic
				if is_spam[i] == 1:
					word_lens_spam.append(len(word))
					if word in word_freq_spam:
						word_freq_spam[word] += 1
					else:
						word_freq_spam.update({word: 1})
				else:
					word_lens_ham.append(len(word))
					if word in word_freq_ham:
						word_freq_ham[word] += 1
					else:
						word_freq_ham.update({word: 1})


	# problem 1(b)
	ITERATIONS = 10;
	print("\n\nTop 10 words in SPAM are: ")
	for iteration in range(0, ITERATIONS): 				# finding top occuring word everytime
		mx = 0
		mx_str = ""
		for word in word_freq_spam:
			if mx < word_freq_spam[word]:
				mx = word_freq_spam[word]
				mx_str =  word
		print(str(iteration + 1) + ".\t" + mx_str, end = "")
		for i in range(0, 20 - len(mx_str)):
			print(" ", end = "")
		print("[Count : " + str(word_freq_spam[mx_str]) + "]")
		del word_freq_spam[mx_str]						# delete the top occuring word

	print("\n\nTop 10 words in HAM are: ")
	for iteration in range(0, ITERATIONS): 				# finding top occuring word everytime
		mx = 0
		mx_str = ""
		for word in word_freq_ham:
			if mx < word_freq_ham[word]:
				mx = word_freq_ham[word]
				mx_str =  word
		print(str(iteration + 1) + ".\t" + mx_str, end = "")
		for i in range(0, 20 - len(mx_str)):
			print(" ", end = "")
		print("[Count : " + str(word_freq_ham[mx_str]) + "]")
		del word_freq_ham[mx_str]						# delete the top occuring word
	print("");
	

	# problem 1(c)
	data = {'spam':spam_count, 'ham':ham_count}
	fig = plt.figure(figsize = (6, 6))
	plt.bar(data.keys(), data.values(), color = 'blue', width = 0.3)
	plt.xlabel("Type of mail")
	plt.ylabel("Number of mails")
	plt.title("1(c): Class Distribution of the given dataset")
	plt.show()


	# problem 1(d)

	# box plot:
	fig = plt.figure(figsize = (10, 5))
	axes = fig.add_subplot(111)

	box_plot = axes.boxplot([word_lens_spam, word_lens_ham], patch_artist = True, notch = 'True', vert = 0)

	colors = ['red', 'olive']
	for patch, color in zip(box_plot['boxes'], colors):
		patch.set_facecolor(color)

	for attr in box_plot['whiskers']:
		attr.set(color = 'brown', linewidth = 1.5, linestyle = ":")

	for attr in box_plot['caps']:
		attr.set(color = 'black', linewidth = 2)

	for attr in box_plot['medians']:
		attr.set(color = 'cyan', linewidth = 2)

	for attr in box_plot['fliers']:
		attr.set(marker = 'D', color = 'cyan', alpha = 0.5)

	axes.set_yticklabels(['spam', 'ham'])
	axes.get_xaxis().tick_bottom()
	axes.get_yaxis().tick_left()
	plt.xlabel("word length")
	plt.title("1(d): Box plot of word length for both the classes")
	plt.show()

	#kernel density estimation (KDE) plot
	fig = plt.figure(figsize = (12, 5))
	plt.subplot(1, 2, 1)
	sb.kdeplot(word_lens_spam, color = 'red', label = 'Spam')
	plt.xlabel("word length")
	plt.title("KDE for SPAM")
	plt.subplot(1, 2, 2)
	sb.kdeplot(word_lens_ham, color = 'green', label = 'Not Spam')
	plt.xlabel("word length")
	plt.title("KDE for HAM")
	plt.show()

	print("--------------------------------------------------\n\n")

problem_1()