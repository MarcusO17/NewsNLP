import streamlit as st
from pathlib import Path
import pandas as pd
import plotly_express as px
import plotly.graph_objects as go
from wordcloud import WordCloud,ImageColorGenerator, STOPWORDS
import numpy as np 
import PIL.Image




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
    st.header('Roadmap')
    st.write('Coming Soon')
    st.write("""This is meant to be a short term project 
             however I wanted to give a heads up that
              there are some new features in the works for the 
             platform such as live updates, a prediction system and more! Stay tuned for updates. Thank you.
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
    
    sidebar_option = st.sidebar.selectbox("Select an option", ["Home", "Visualisation Playground", "About"])

   
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
if __name__ == "__main__":
    app()
