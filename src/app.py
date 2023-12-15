import streamlit as st
import pandas as pd
import plotly_express as px
import plotly.graph_objects as go
from wordcloud import WordCloud,ImageColorGenerator
import numpy as np 
import PIL.Image
import matplotlib.pyplot as plt
from nltk.corpus import stopwords


def word_cloud_gen(df):
    stop_words = set(stopwords.words('english'))
    stop_words.update(['malaysia','malaysian','ex','correspondent'])

    malaysianFlag_mask =np.array(PIL.Image.open('assets/Flag_of_Malaysia.svg.png'))
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

def sentiment_score_gen():
    df = pd.read_csv('data/SentimentScoreBBC.csv')
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

def gen_pie_chart(df,site):
    sentiment_value_counts = df[df['site'] == site]['sentiment'].value_counts().to_list()
    fig = px.pie(df,values=sentiment_value_counts,names=df[df['site'] == site]['sentiment'].unique(),template='plotly_dark',title=site)

    st.plotly_chart(fig)

 
def app():
    main_df = pd.read_csv('data/BBCNews.csv')

    st.title("Narratives in Focus: Exploring Malaysia Through BBC's Lens")

    st.write("""Embark on a journey through Malaysia as we examine its stories through the lens of the BBC, a beacon of global journalism. 
             In a world where media shapes perceptions, understanding the biases within news coverage is crucial.
             By closely examining the language, tone, and framing of BBC articles, 
             we strive to decode the underlying perspectives that influence how Malaysia, is viewed on the global stage.""")

    st.subheader("Data Acquisition")
    st.write("""All data is scraped from the BBC website using Python. (13/12/2023) 
             By searching for keywords like "Malaysia," we acquire up to 290 articles 
             that the BBC has written about Malaysia.""")
    
    st.table(data=main_df.head(3))

    st.subheader('Sentiment Analysis')
    st.write("""We initially exclude non-news articles for this experiment. 
             Using nltk's built-in SentimentIntensityAnalysis (SIA), which operates on VADER, we assign a compounded score to each word within the title and description of the article. 
             We apply a neutrality threshold of 0.2 for neutral sentiments.
             """)
    st.write("Explained more at : [Medium Article](https://medium.com/@piocalderon/vader-sentiment-analysis-explained-f1c4f9101cd9)")
    
    df_sentiment = pd.read_csv('data/BBCSentimentProcessed.csv',index_col=0)
    st.table(data=df_sentiment.head(3))

    st.header('Visualisation Playground')

    option = st.selectbox('Choose an option',['None','WordClouds','Sentiment Score','Pie Charts'])

    if option == 'WordClouds':
        word_cloud_gen(df_sentiment)

    if option == 'Sentiment Score':
        sentiment_score_gen()

    if option == 'Pie Charts':
        option = st.selectbox('Choose an option',df_sentiment['site'].unique())
        gen_pie_chart(df_sentiment,option)


if __name__ == "__main__":
    app()
