import streamlit as st
from pathlib import Path
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer as SIA
import time as tm
import pandas as pd
from datetime import datetime, timedelta
import plotly_express as px
import re
import requests
from bs4 import BeautifulSoup
import plotly.graph_objects as go
from wordcloud import WordCloud,ImageColorGenerator, STOPWORDS
import numpy as np 
import PIL.Image

nltk.download('vader_lexicon')

def scraper(topic):
    load_bar = st.progress(29, text="Scraping!")
    page = 0
    max_page = 29
    total_articles = 0
    
    total_info = []

    while page < max_page:
        load_bar.progress(int(100 * page / max_page))
        tm.sleep(1)
        page += 1
        url = f"https://www.bbc.co.uk/search?q={topic}&d=SEARCH_PS&page={page}"
        webpage = requests.get(url).text
        soup = BeautifulSoup(webpage, "html.parser")
        print(f'Printing {page}')   
        if soup.find('div', class_='ssrcss-1qik2p5-Stack e1y4nx260'):
            st.error('No Articles Found')
            break
        for article in soup.find_all('div',class_='ssrcss-tq7xfh-PromoContent exn3ah99'):
            total_articles += 1
            title = article.find('span').text #get article titles
            try:
                description = article.find('p',class_='ssrcss-1q0x1qg-Paragraph e1jhz7w10').text #get article descs
            except:
                description = ""
            misc_list = article.find('ul',class_='ssrcss-1xpwu3-MetadataStripContainer eh44mf03') # get miscellanous info such as date published, section and area
            
            if misc_list is None:
                continue

            time, site, section = None, None, None

            for list_item in misc_list.findAll('li'):
                print(list_item.text)

                if list_item.text.startswith('Published'):
                    time = list_item.text.split('Published')[1]
            
                if list_item.text.startswith('Site'):
                    site = list_item.text.split('Site')[1]
            
                if list_item.text.startswith('Section'):
                    section = list_item.text.split('Section')[1]
                
            



            row_info = [title,description,time,site,section]   
            total_info.append(row_info)  

    load_bar.empty()
    return total_info

def get_sentiment(polarity_score):
    neutrality_threshold = 0.2
    sentiment = polarity_score['compound']
    if abs(sentiment) >= neutrality_threshold:
        if sentiment > 0:
            return 'Positive'
        else:
            return 'Negative'
    else:
        return 'Neutral'
    
def df_sentiment_scorer(df):
    df = df.groupby(by=['publisheddate','sentiment'])['sentiment'].count().to_frame('count').reset_index()
    df['score'] = 0
    for index in range(1,len(df)):
        if df.loc[index]['sentiment'] == 'Positive':
            df.loc[index, 'score'] = df.loc[index - 1, 'score'] + ( 1 * df.loc[index, 'count'])
        elif df.loc[index]['sentiment'] == 'Neutral':
            df.loc[index, 'score'] = df.loc[index - 1, 'score'] + 0 
        elif df.loc[index]['sentiment'] == 'Negative':
            df.loc[index, 'score'] = df.loc[index - 1, 'score'] - ( 1 * df.loc[index, 'count'])

    return df



def time_processor(date):
    try:
        today = datetime.today().date()
        
        if date is None:
            return None

        if date.startswith('Today'):
            date = today.strftime('%d %B %Y')
        
        if date.startswith('in'):
            date = today.strftime('%d %B %Y')
        
        if date.startswith('Tomorrow'):
            date = (today + timedelta(days=1)).strftime('%d %B %Y')

        if date.endswith('ago'):
            if 'hour' in date:
                date = today.strftime('%d %B %Y')
            if 'day' in date:
                date = (today - timedelta(days=int(date.split(' ')[0]))).strftime('%d %B %Y')
        elif date[len(date)-4:].isnumeric() == False:
            date = date + ' 2023'
        return date
    except Exception as e:
        return None


def data_pipeline(info):
    #Make DF
    df = pd.DataFrame(info,columns=['title','description','publisheddate','section','site'])

    #Convert Written Dates to Datetime OBJ
    df['publisheddate'] = df['publisheddate'].apply(lambda x : time_processor(x))
    df['publisheddate'] = pd.to_datetime(df['publisheddate'])

    #process text
    # Lower Case all text
    df['title'] = df['title'].apply(lambda x:x.lower())
    df['title'] = df['title'].apply(lambda x: re.sub('[\W]+',' ',x))
    df['description'] = df['description'].apply(lambda x:x.lower())
    df['description'] = df['description'].apply(lambda x: re.sub('[\W]+',' ',x))

    df = df.sort_values(by='publisheddate',ascending=False).reset_index(drop=True)

    return df

def sentimentise(df):
    sia = SIA()
    for index in range(len(df)):
        pol_score_title = sia.polarity_scores(df['title'][index])
        pol_score_description = sia.polarity_scores(df['description'][index])
        combined_score = {
            'compound': (pol_score_title['compound'] + pol_score_description['compound']) / 2
        }
        df.loc[index,'sentiment'] = get_sentiment(combined_score)

    return df


def topic_searcher():
    st.title('Topic Sentiment Tracker')
    st.write("""Simply input your topic of interest, 
             and we'll scour the BBC portal, providing you with a clear 
             and concise sentiment score over time. Track the highs and lows as 
             we also compare this sentiment data against Malaysia's sentiment graph""")

    topic = st.text_input('Enter Topic :')
    if len(topic) > 0:
        info = scraper(topic)
        if len(info) > 0:
            df = data_pipeline(info)
            df = sentimentise(df)
            df = df_sentiment_scorer(df)
            sentiment_score_gen_search(df2=df,df2name=topic)

    st.write("Caveats")
    st.write("""Please be aware of certain limitations with our "Topic Sentiment Tracker." 
            The sentiment analysis and comparison against Malaysia's graph 
            are based solely on BBC portal news articles, excluding sports and other
            categories in Malaysia's graph. Changes to the BBC portal's data structure
            may impact accuracy. Sentiment analysis is inherently subjective, offering 
            estimations rather than absolute measures, and temporal variations may be influenced 
            by external factors. Users should consider these factors
            """)
        

def word_cloud_gen(df):
    with st.spinner("Loading..."):
        stop_words = set(STOPWORDS)
        stop_words.update(['malaysia','malaysian','ex','correspondent','s'])

        malaysianFlag_mask =np.array(PIL.Image.open(f'{Path(__file__).parent}/assets/Flag_of_Malaysia.svg.png'))
        malaysianFlag_colormap =ImageColorGenerator(malaysianFlag_mask)

        sentiment_option = st.selectbox('Select sentiment category', ['Positive', 'Neutral', 'Negative'])

        text = " ".join(df[df['sentiment'] == sentiment_option]['title'])

        wordcloud= WordCloud(stopwords=stop_words, 
                                    mask=malaysianFlag_mask,
                                    background_color='white').generate(text)
        wordcloud.recolor(color_func=malaysianFlag_colormap)


        fig = px.imshow(wordcloud.to_array(), binary_string=True)
        fig.update_layout(title=f'{sentiment_option} Sentiment', xaxis=dict(visible=False), yaxis=dict(visible=False))
        
        st.plotly_chart(fig)

        if sentiment_option == 'Positive':
            st.write("""
                    The positive sentiment wordcloud simply displays words which have positive associations from
                    BBC articles about Malaysia. It's a straightforward visual 
                    summary of the 'good' language used in the titles and descriptions 
                    of those articles.
                    """)
        if sentiment_option == 'Negative':
            st.write("""
                    The negative sentiment wordcloud simply displays words which have negative associations from
                    BBC articles about Malaysia. It's a straightforward visual 
                    summary of the 'bad' language used in the titles and descriptions 
                    of those articles.
                    """)
        if sentiment_option == 'Neutral':
            st.write("""
                    The negative sentiment wordcloud simply displays words which have neutral associations from
                    BBC articles about Malaysia. It's a straightforward visual 
                    summary of the 'normal' language used in the titles and descriptions 
                    of those articles.
                    """)

def sentiment_score_gen():
    with st.spinner("Loading..."):
        df = pd.read_csv(f'{Path(__file__).parent}/data/SentimentScoreBBC.csv')
        fig = go.Figure()
        fig.update_layout(
        title="BBC News Report's Sentiment Score on Malaysia",
        xaxis_title="Time",
        yaxis_title="Sentiment Score",
        template = 'plotly_dark'
        )
        fig.add_trace(go.Line(y=df['score'],x=df['publisheddate'], connectgaps=False))
        fig.add_vrect(x0='2013-05-01',x1='2017-05-31',annotation_text="No News Reports", annotation_position="bottom left",
                    fillcolor="grey", opacity=0.25, line_width=0)
        fig.add_hline(y=0,annotation_text='Neutral Sentiment',line_dash="dash")
        fig.add_annotation(x='2018-05-09',y=-18, 
                    text="GE15",
                    showarrow=True,
                    arrowhead=1)
        fig.add_annotation(x='2018-07-03',y=-18, 
                    text="DS Najib's Arrest",
                    font=dict(color = 'lightblue'),
                    showarrow=True,
                    arrowhead=7,
                    ax=50,
                    ay=-30,)
        
        st.plotly_chart(fig)

        st.write("""
                We graphed the cumulative sentiment score over time for the BBC. 
                Dates with positive sentiment scores were incremented by 1, 
                negative scores by -1, and neutral scores remained at 0. 
                This provides an overview of how sentiment changes over time in the BBC content.
                """)


def sentiment_score_gen_search(df2,df2name):
    with st.spinner("Loading..."):
        df = pd.read_csv(f'{Path(__file__).parent}/data/SentimentScoreBBC.csv')
        fig = go.Figure()
        fig.update_layout(
        title=f"BBC News Report's Sentiment Score on Malaysia Vs {df2name}",
        xaxis_title="Time",
        yaxis_title="Sentiment Score",
        template = 'plotly_dark'
        )
        fig.add_trace(go.Line(name="Sentiments of Malaysia",y=df['score'],x=df['publisheddate'], connectgaps=False))
        fig.add_hline(y=0,annotation_text='Neutral Sentiment',line_dash="dash")
        fig.add_trace(go.Line(name=f'Sentiments of {df2name}',y=df2['score'],x=df2['publisheddate'], connectgaps=False))

        st.plotly_chart(fig)


def gen_pie_chart(df,site):
    with st.spinner("Loading..."):
        st.write(""" Each site represents the category under which the BBC classifies the article. 
                Each pie chart illustrates the ratio of positive to negative sentiments, 
                helping us identify which categories depict Malaysia in a positive light.
                """)
        sentiment_value_counts = df[df['site'] == site]['sentiment'].value_counts().to_list()
        fig = px.pie(df,values=sentiment_value_counts,names=df[df['site'] == site]['sentiment'].unique(),template='plotly_dark',title=site)

        st.plotly_chart(fig)

        

def visualisation_playground():
    df_sentiment = pd.read_csv(f'{Path(__file__).parent}/data/BBCSentimentProcessed.csv',index_col=0)
    st.header('Visualisation Playground')

    option = st.selectbox('Choose an option',['None','WordClouds','Sentiment Score','Pie Charts'])

    if option == 'WordClouds':
        word_cloud_gen(df_sentiment)

    if option == 'Sentiment Score':
        sentiment_score_gen()

    if option == 'Pie Charts':
        option = st.selectbox('Choose an option',df_sentiment['site'].unique())
        gen_pie_chart(df_sentiment,option)

def about():
    st.title('About Page.')
    st.header('Caveats')
    st.write("""
            - Our project relies on data from the BBC portal, and any changes to their setup may affect the accuracy of sentiment analysis.

- Malaysia's sentiment graph excludes sports and other categories, providing a focused view on specific news topics.

- Sentiment analysis provides estimations rather than precise measures, capturing general emotions expressed in articles.

- External events and global sentiment shifts may influence sentiment trends, so results should be interpreted with this in mind.
             """)
    st.divider()
    st.header('Disclaimers')
    st.write("""As a student, I want to make it clear that the data presented in this project has been collected from the BBC website solely for educational and research purposes. I do not have explicit permission from the BBC to scrape their website, and all efforts have been made to respect intellectual property and copyright laws.
Furthermore, the data representation in this project is intended to provide an unbiased and objective analysis. It is not intended to alter views, serve as propaganda, or misrepresent any information. The aim is to present insights and findings in a fair and transparent manner. If there are any concerns regarding the use of the data or its representation, please contact me, a
and I will promptly address any issues. At : marcuso1710@gmail.com
""")
    st.divider() 
    st.header('Process Flow')
    st.image(PIL.Image.open(f'{Path(__file__).parent}/assets/ProcessFlow.png'))
    st.divider()
 
def app():
    st.sidebar.header("Hello!")
    st.sidebar.text("""Explore more here! 
There's more to come!""")
    
    sidebar_option = st.sidebar.selectbox("Select an option", ["Home", "Visualisation Playground", "About",'Topic Searcher'])

   
    if sidebar_option == "Home":
        main_df = pd.read_csv(f'{Path(__file__).parent}/data/BBCNews.csv')

        st.title("Narratives in Focus: Exploring Malaysia Through BBC's Lens")

        st.write("""Take a dive into Malaysia's stories through the BBC's perspective—a global journalism hub. 
                In a world where media molds opinions, understanding the biases in news coverage is key. 
                We'll dig into BBC articles, examining language, tone, and framing to uncover the underlying perspectives 
                that shape the global view of Malaysia. It's a laid-back exploration into the nuances of media representation.""")

        st.divider()

        st.subheader("Data Acquisition")
        st.write("""
    We used a script in Python to collect information from the BBC website. As of December 13, 2023, 
                we searched for words like "Malaysia" and found about 290 articles written by the BBC about Malaysia.""")
        
        st.table(data=main_df.head(3))

        st.divider()

        st.subheader('Sentiment Analysis')
        st.write("""We initially exclude non-news articles for this experiment. 
                Using nltk's built-in SentimentIntensityAnalysis (SIA), which operates on VADER, we assign a compounded score to each word within the title and description of the article. 
                We apply a neutrality threshold of 0.2 for neutral sentiments.
                """)
        st.write("Explained more at : [Medium Article](https://medium.com/@piocalderon/vader-sentiment-analysis-explained-f1c4f9101cd9)")
        
        df_sentiment = pd.read_csv(f'{Path(__file__).parent}/data/BBCSentimentProcessed.csv',index_col=0)
        st.table(data=df_sentiment.head(3))

        st.divider()

        st.info('Check out more at the sidebar!')


    if sidebar_option == 'Visualisation Playground':
        visualisation_playground()
        
    if sidebar_option == 'About':
        about()
    
    if sidebar_option == 'Topic Searcher':
        topic_searcher()
if __name__ == "__main__":
    app()
