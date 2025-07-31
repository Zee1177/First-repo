#!/usr/bin/env python
# coding: utf-8

# In[14]:


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# In[15]:


Amazonrev=pd.read_csv('Amazon_review.csv')


# In[16]:


Amazonrev.head()


# In[17]:


Amazonrev.columns


# In[18]:


Amazonrev.dtypes


# In[19]:


pip install neattext


# In[20]:


import neattext.functions as nfx


# In[22]:


Amazonrev['reviewText']


# In[24]:


Amazonrev["reviewText"].apply(nfx.extract_hashtags)


# In[25]:


Amazonrev["reviewText"].apply(nfx.remove_hashtags)


# In[26]:


T=Amazonrev['reviewText'].apply(nfx.remove_hashtags)


# In[27]:


T.apply(lambda x:nfx.remove_userhandles(x))


# In[28]:


R=T.apply(lambda x:nfx.remove_userhandles(x))


# In[29]:


R.apply(nfx.remove_urls)


# In[30]:


Q=R.apply(nfx.remove_urls)


# In[31]:


Q.apply(nfx.remove_punctuations)


# In[32]:


O=Q.apply(nfx.remove_punctuations)


# In[33]:


O.apply(nfx.remove_multiple_spaces)


# In[34]:


Z=O.apply(nfx.remove_multiple_spaces)


# In[35]:


A= Z.apply(nfx.remove_non_ascii)


# In[36]:


B= A.apply(nfx.remove_numbers)


# In[37]:


B


# In[38]:


pip install textblob


# In[39]:


from textblob import TextBlob


# In[40]:


def get_sentiment(text):
    blob=TextBlob(text)
    sentiment_polarity=blob.sentiment.polarity
    sentiment_subjectivity=blob.sentiment.subjectivity
    if sentiment_polarity>0:
        sentiment_label='Positive'
    elif sentiment_polarity<0:
        sentiment_label='Negative'
    else:
        sentiment_label='Neutral'
    result={'polarity':sentiment_polarity,'subjectivity':sentiment_subjectivity,'sentiment':sentiment_label}
    return result


# In[41]:


A.apply(get_sentiment)


# In[42]:


pd.json_normalize(A.apply(get_sentiment))


# In[43]:


CLEAN_TEXT=pd.json_normalize(A.apply(get_sentiment))


# In[44]:


CLEAN_TEXT.value_counts('sentiment')


# In[45]:


CLEAN_TEXT.value_counts('sentiment').plot(kind='bar')


# In[46]:


Amazonrev['Summary']=B


# In[47]:


Amazonrev.head()


# In[48]:


Amazonrev['sentiments']=CLEAN_TEXT['sentiment']


# In[49]:


Amazonrev.head()


# In[50]:


positive_tweets=Amazonrev[Amazonrev['sentiments']=='Positive']['Summary']


# In[51]:


negative_tweets=Amazonrev[Amazonrev['sentiments']=='Negative']['Summary']
neutral_tweets=Amazonrev[Amazonrev['sentiments']=='Neutral']['Summary']


# In[52]:


positive_tweet_list=positive_tweets.apply(nfx.remove_stopwords).tolist()


# In[53]:


negative_tweet_list=negative_tweets.apply(nfx.remove_stopwords).tolist()
neutral_tweet_list=neutral_tweets.apply(nfx.remove_stopwords).tolist()


# In[54]:


for line in positive_tweet_list:
    for token in line.split():
        print(token)


# In[55]:


pos_token=[token for line in positive_tweet_list for token in line.split()]


# In[56]:


neg_token=[token for line in negative_tweet_list for token in line.split()]
neu_token=[token for line in neutral_tweet_list for token in line.split()]


# In[57]:


from collections import Counter


# In[58]:


def get_tokens(docx,num=30):
    word_tokens=Counter(docx)
    most_common=word_tokens.most_common(num)
    result=dict(most_common)
    return result


# In[59]:


most_common_pos_words=get_tokens(pos_token)
most_common_neg_words=get_tokens(neg_token)
most_common_neu_words=get_tokens(neu_token)


# In[60]:


pip install wordcloud


# In[61]:


from wordcloud import WordCloud


# In[62]:


pos_docx=' '.join(pos_token)
neg_docx=' '.join(neg_token)
neu_docx=' '.join(neu_token)


# In[63]:


positive = WordCloud(width = 800, height = 800,
                      background_color ='black',
                      min_font_size = 10).generate(pos_docx)
negative = WordCloud(width = 800, height = 800,
                      background_color ='black',
                      min_font_size = 10).generate(neg_docx)
neutral = WordCloud(width = 800, height = 800,
                      background_color ='black',
                      min_font_size = 10).generate(neu_docx)


# In[64]:


plt.imshow(positive)
plt.axis("off")
plt.tight_layout(pad = 0)
plt.show()


# In[65]:


plt.imshow(negative)
plt.axis("off")
plt.tight_layout(pad = 0)
plt.show()


# In[66]:


plt.imshow(neutral)
plt.axis("off")
plt.tight_layout(pad = 0)
plt.show()


# In[ ]:




