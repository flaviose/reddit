# PRAW to interact with reddit
import numpy as np
import praw
#install textblob if not already installed using "pip install -U textblob"
from textblob import TextBlob
import nltk
# Download VADER, if not downloaded
# nltk.download('vader_lexicon')
from nltk.sentiment.vader import SentimentIntensityAnalyzer

# create object for VADER sentiment function interaction
sia = SentimentIntensityAnalyzer()


my_client_id = 'aRKBB5Iob671nA'
my_client_secret = 'o8rkAnqC62qOa6tuQhNZ6TaELV6ARQ'
my_user_agent = 'reddit_scrapping'

reddit = praw.Reddit(client_id=my_client_id,
                     client_secret=my_client_secret,
                     user_agent=my_user_agent)

# get 10 hot posts from the showerthoughts subreddit



# Sentiment analysis function for TextBlob tools
def text_blob_sentiment(review, sub_entries_textblob,score_text_blob):
    analysis = TextBlob(review)
    #score_text_blob=[0,1,2]#negative_positive_neutral
    if analysis.sentiment.polarity >= 0.0001:
        if analysis.sentiment.polarity > 0:
            sub_entries_textblob['positive'] = sub_entries_textblob['positive'] + 1
            
            score_text_blob[1]=score_text_blob[1]+1
            
            return 'Positive'

    elif analysis.sentiment.polarity <= -0.0001:
        if analysis.sentiment.polarity <= 0:
            sub_entries_textblob['negative'] = sub_entries_textblob['negative'] + 1
            
            score_text_blob[0]=score_text_blob[0]+1
            
            return 'Negative'
    else:
        sub_entries_textblob['neutral'] = sub_entries_textblob['neutral'] + 1
        
        score_text_blob[2]=score_text_blob[2]+1
        
        return 'Neutral'
    

# sentiment analysis function for VADER tool
def nltk_sentiment(review, sub_entries_nltk,score_nltk):
    vs = sia.polarity_scores(review)
    #score_nltk = [0,1,2]# negative_positive_neutral
    if not vs['neg'] > 0.05:
        if vs['pos'] - vs['neg'] > 0:
            sub_entries_nltk['positive'] = sub_entries_nltk['positive'] + 1
            
            score_nltk[1]=score_nltk[1]+1
            
            return 'Positive'
        else:
            sub_entries_nltk['neutral'] = sub_entries_nltk['neutral'] + 1
            
            score_nltk[2]=score_nltk[2]+1
            
            return 'Neutral'

    elif not vs['pos'] > 0.05:
        if vs['pos'] - vs['neg'] <= 0:
            sub_entries_nltk['negative'] = sub_entries_nltk['negative'] + 1
            
            score_nltk[0] = score_nltk[0]+1
            
            return 'Negative'
        else:
            sub_entries_nltk['neutral'] = sub_entries_nltk['neutral'] + 1
            
            score_nltk[2]=score_nltk[2]+1
            
            return 'Neutral'
    else:
        sub_entries_nltk['neutral'] = sub_entries_nltk['neutral'] + 1
        
        score_nltk[2]=score_nltk[2]+1
        
        return 'Neutral'


# replication of comment section of reddit post
def replies_of(top_level_comment, count_comment, sub_entries_textblob, sub_entries_nltk):
    if len(top_level_comment.replies) == 0:
        count_comment = 0
        return
    else:
        for num, comment in enumerate(top_level_comment.replies):
            try:
                count_comment += 1
                #print('-' * count_comment, comment.body)
                text_blob_sentiment(comment.body, sub_entries_textblob)
                nltk_sentiment(comment.body, sub_entries_nltk)
            except:
                continue
            replies_of(comment, count_comment, sub_entries_textblob,sub_entries_nltk)


def stats(data_text):
    max_element_array=np.empty(20,dtype=float)
    max_element_idx_array=np.empty(20,dtype=int)
    sentiment = np.empty(20, dtype = "U10")## dtype = "S10" stores the strings as bytes, dtype="U10" stores the data as char
    
    stats_val=np.empty(shape=(21,6), dtype="U10")
    stats_val[0,0]="index_el"
    stats_val[0,1]="negative"
    stats_val[0,2]="positive"
    stats_val[0,3]="neutral"
    stats_val[0,4]="max_element"
    stats_val[0,5]="#"
    #stats_val[0,6]="leading_sentiment"
    
    for idx in range(len(data_text[:,0])):
  
        negative = (data_text[idx,0])/(np.sum(data_text[idx,:]))
        positive = (data_text[idx,1])/(np.sum(data_text[idx,:]))
        neutral  = (data_text[idx,2])/(np.sum(data_text[idx,:]))
    
        max_element =  data_text[idx,0]
        max_element_idx = 0
        for k in range(len(data_text[idx,:])):
        
        
            if (k+1<len(data_text[idx,:]))and(data_text[idx,k+1]>max_element):#we need to put the index inside bounds condition first 
                max_element = data_text[idx,k+1]#otherwise the second expression raises and error
                max_element_idx = k+1
        
        max_element_array[idx] = max_element/(np.sum(data_text[idx,:]))
        max_element_idx_array[idx] = max_element_idx
    
    
        if max_element_idx ==0:
            sentiment[idx]='negative'
        
        elif max_element_idx == 1:
            sentiment[idx] = 'positive'
        
        else:
            sentiment[idx] = 'neutral'
            
        stats_val[idx+1,0]=  idx+1
        stats_val[idx+1,1] = negative
        stats_val[idx+1,2] = positive
        stats_val[idx+1,3] = neutral
        stats_val[idx+1,4] = max_element_array[idx]
        stats_val[idx+1,5] = max_element_idx_array[idx]+1
        #stats_val[idx+1,6] = sentiment[idx]
        
        
        
        print(idx+1,' ',negative,' ',positive,' ',neutral,' ',max_element_array[idx],max_element_idx_array[idx]+1,' ',sentiment[idx])
        
    
    return stats_val

def plot(r_corona_virus_text_blob,name_of_image):
    index_val=(r_corona_virus_text_blob[1:,0]).astype(np.int)
    print(type(index_val[0]))

    negative_sentiment = (r_corona_virus_text_blob[1:,1]).astype(np.float)
    positive_sentiment=  (r_corona_virus_text_blob[1:,2]).astype(np.float)
    neutral_sentiment=   (r_corona_virus_text_blob[1:,3]).astype(np.float)
    print(negative_sentiment)

    fig = plt.figure()
    plt.plot(index_val,negative_sentiment,'o',color='red')
    plt.plot(index_val, positive_sentiment,'o',color='blue')
    plt.plot(index_val, neutral_sentiment,'o',color='green')
    
    plt.savefig(name_of_image+'.png',dpi = 600)

    avg_dist = 0

    print(r_corona_virus_text_blob[20,3])

    for i in range(1,len(r_corona_virus_text_blob[:,1])): 

        avg_dist = avg_dist + np.absolute(r_corona_virus_text_blob[i,1].astype(np.float)-r_corona_virus_text_blob[i,2].astype(np.float))


    avg_dist = avg_dist/len(r_corona_virus_text_blob[:,1])
    
    return avg_dist

## time can only be 'month' or 'week', subreddit for our project is Coronavirus or China_Flu
def main(subreddit,data_text_blob,data_nltk,time,Limit):
    
    #data_text_blob = np.ndarray(shape=(20,3), dtype = int)
    #data_nltk      = np.ndarray(shape=(20,3), dtype = int)
    
    top_posts = reddit.subreddit(subreddit).top(time, limit=Limit)
    i = 0
    for submission in top_posts:
        sub_entries_textblob = {'negative': 0, 'positive' : 0, 'neutral' : 0}
        sub_entries_nltk = {'negative': 0, 'positive' : 0, 'neutral' : 0}
        
        score_text_blob=[0,0,0]
        score_nltk     =[0,0,0]
        
        
        print('Title of the post :', submission.title)
        text_blob_sentiment(submission.title, sub_entries_textblob,score_text_blob)
        nltk_sentiment(submission.title, sub_entries_nltk,score_nltk)
        print("\n")
        submission_comm = reddit.submission(id=submission.id)
        
        
    
        for count, top_level_comment in enumerate(submission_comm.comments):
            print(f"-------------{count} top level comment start--------------")
            count_comm = 0
            try :
                print(top_level_comment.body)
                text_blob_sentiment(top_level_comment.body, sub_entries_textblob,score_text_blob)
                nltk_sentiment(top_level_comment.body, sub_entries_nltk,score_nltk)
                replies_of(top_level_comment,
                           count_comm,
                           sub_entries_textblob,
                           sub_entries_nltk)
            except:
                continue
                
        for j in range(3):
            data_text_blob[i,j]=score_text_blob[j]
            data_nltk[i,j]     =score_nltk[j]
        
        i = i+1
        print('Over all Sentiment of Topic by TextBlob :', sub_entries_textblob,' ',score_text_blob)
        print('Over all Sentiment of Topic by VADER :',sub_entries_nltk,' ',score_nltk)
        print("\n\n\n")

    print(data_text_blob)
    



if __name__ == '__main__':
    
    data_text_blob = np.ndarray(shape=(20,3), dtype = int)
    data_nltk      = np.ndarray(shape=(20,3), dtype = int)
    
    main('Coronavirus',data_text_blob,data_nltk,'month',20)
    
    r_coronavirus_text_blob=stats(data_text_blob)
    #print(r_coronavirus_text_blob)
    #print(r_coronavirus_text_blob)
    np.savetxt("r_coronavirus_text_blob.csv", r_coronavirus_text_blob, delimiter=",",fmt='%s')

    r_coronavirus_nltk=stats(data_nltk)
    #print(r_corona_virus_nltk)
    np.savetxt("r_coronavirus_nltk.csv", r_coronavirus_nltk, delimiter=",",fmt='%s')
    #print(r_coronavirus_nltk)

    ## the second argument of plot can be changed to save the plot for another set of parameters
    plot(r_coronavirus_text_blob,'r_Coronavirus_text_blob_month')