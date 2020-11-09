
# import libraries 

import nltk
from newspaper import Article
import random
import string
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import warnings
 
 # filter on warnings
warnings.filterwarnings('ignore')

# download the punkt package
nltk.download('punkt' , quiet = True)

# get the article
'''
Steps :
Fetch
Download
Parse
Apply NLP
Define a corpus
'''

article = Article(input(r'Enter Article URl: '))  # change url as per need
article.download()
article.parse()
article.nlp()

corpus = article.text


# show corpus : Article texts
#print(corpus)

# Tokenizing
text = corpus
sentence_list = nltk.sent_tokenize(text) # list of all sentneces

# print list of sentences
#print(sentence_list)

# function to return a random greeting response from bot to user

def greeting(text):
    text = text.lower()

    # bot greeting
    bot_response = ['hello' , 'hi' , 'hi there' , 'howdy' , 'welcome' ]
    # users greeting
    user_response = ['hi' , 'hello' , 'hey' , 'anyone here' , 'whatsup' , 'greetings']

    # for user greet , return random bot greet
    for word in text.split():
        if word in user_response:
            return random.choice(bot_response)

def idx_sort(list_var):
    length = len(list_var)  # len of list
    list_idx = list(range(0 ,length)) # takes index of list ranging from values 0 to lenth of that list_var(whole list length)
    
    # loop for sorting index of list and appending highest value to new list below
    x = list_var # call list_var as x


# if index of list at pos x is > index of list at pos j then swap
    for i in range(length):
        for j in range(length):
            if x[list_idx[i]] > x[list_idx[j]]:
            # Swap
                temp = list_idx[i]
                list_idx[i] = list_idx[j]
                list_idx[j] = temp

            '''
            or
            list_idx[i],list_idx[j] = list_idx[j] , list_idx[i]
            
            '''

    return list_idx

# create a func for bots response
def bot_greet(user_input):
    user_input = user_input.lower() # lowers the input
    sentence_list.append(user_input) # append it to tokenized list - sentence list so it gets tokenized

    # create empty response
    bot_greet = ''
    count_mtx = CountVectorizer().fit_transform(sentence_list)  # create count matrix
    sim_score = cosine_similarity(count_mtx[-1] , count_mtx)  # get similarity scores between last_sentnece_user_input and everthyhing in count_matx
    sim_score_list = sim_score.flatten()  # reduce the dimension of similarity scores


    # finding index of highest score in sim_score_list
    idx = idx_sort(sim_score_list)
  
  # knowing where highest values are in sim_score_list by geeting highest idx


#print(sim_score_list)
#idx
  
    idx = idx[1:]    # only value from 1 onwards index should contain only values that are not itself
    response_flag = 0  # get response back if user id similarity scores(query or text) is similar to what the users query or input is


    j = 0  # a variable which keeps account of score that are above 0

# returns only top 2 similarity sentences
    for i in range(len(idx)):
        if sim_score_list[idx[i]] > 0.0:   # similarity
            bot_greet = bot_greet+''+ sentence_list[idx[i]]
            response_flag = 1
            j = j+1
        
        if j > 2 :  # if more than 2 sentences
            break
        
    if response_flag == 0:     # no similarity to data bot has
        bot_greet = bot_greet + ' ' + 'Unable to Understand'

    sentence_list.remove(user_input)    
    return  bot_greet



'''
     use tokenizer class for tensorflow and keras
'''

# start the chat
print('CHAT BOT : I Am Ready To Help You , Ask Subject Queries , Else Kindly Close Me By Typing Exit')

exit_list = ['exit' , 'bye' , 'quit' , 'break' , 'later']
while(True):
    user_input = input('Your Input : ')
    if user_input.lower() in exit_list:
        print ('CHAT BOT : Signing Off!!!')
        break
    else:
        if greeting(user_input) != None:
            print('CHAT BOT : ' + greeting(user_input))
        else:
            print('CHAT BOT :' + bot_greet(user_input))
