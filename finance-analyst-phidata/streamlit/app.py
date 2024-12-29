import streamlit as st
# Must be the first Streamlit command
st.set_page_config(page_title="Financial and Web Search Assistant", layout="wide")

import yfinance as yf
from duckduckgo_search import DDGS
import pandas as pd
from datetime import datetime, timedelta
import os
from dotenv import load_dotenv
from groq import Groq

# Load environment variables
load_dotenv()

# Initialize Groq client
client = Groq(api_key=os.getenv("GROQ_API_KEY"))

class YFinanceTools:
    @staticmethod
    def get_stock_price(symbol):
        stock = yf.Ticker(symbol)
        return stock.history(period="1d")['Close'].iloc[-1]
    
    @staticmethod
    def get_analyst_recommendations(symbol):
        stock = yf.Ticker(symbol)
        # Get only the last 5 recommendations
        recommendations = stock.recommendations
        if isinstance(recommendations, pd.DataFrame):
            return recommendations.tail(5)
        return None
    
    @staticmethod
    def get_stock_fundamentals(symbol):
        stock = yf.Ticker(symbol)
        info = stock.info
        
        # Select only the most important metrics
        key_metrics = {
            'marketCap': info.get('marketCap'),
            'forwardPE': info.get('forwardPE'),
            'dividendYield': info.get('dividendYield'),
            'fiftyTwoWeekHigh': info.get('fiftyTwoWeekHigh'),
            'fiftyTwoWeekLow': info.get('fiftyTwoWeekLow'),
            'trailingEps': info.get('trailingEps'),
            'sector': info.get('sector'),
            'industry': info.get('industry')
        }
        
        # Get only the most recent financial statements
        balance_sheet = stock.balance_sheet.iloc[:, 0] if not stock.balance_sheet.empty else None
        cash_flow = stock.cashflow.iloc[:, 0] if not stock.cashflow.empty else None
        earnings = stock.earnings.iloc[-1] if not stock.earnings.empty else None
        
        return {
            'key_metrics': key_metrics,
            'balance_sheet': balance_sheet,
            'cash_flow': cash_flow,
            'earnings': earnings
        }
    
    @staticmethod
    def get_company_news(symbol):
        stock = yf.Ticker(symbol)
        # Get only the last 3 news items
        news = stock.news[:3] if stock.news else []
        return [{'title': item['title'], 'publisher': item['publisher']} for item in news]

class WebSearchTools:
    @staticmethod
    def search(query, num_results=5):
        with DDGS() as ddgs:
            results = list(ddgs.text(query, max_results=num_results))
        return results

def process_financial_query(query, symbol):
    # Create system message for financial analysis
    system_message = """You are a financial analysis AI. Analyze the provided financial data and respond with insights.
    Focus on the most important metrics and recent trends. Use markdown formatting for tables and structure your response clearly."""
    
    # Get financial data
    yf_tools = YFinanceTools()
    
    try:
        # Collect and format data more concisely
        current_price = yf_tools.get_stock_price(symbol)
        fundamentals = yf_tools.get_stock_fundamentals(symbol)
        recommendations = yf_tools.get_analyst_recommendations(symbol)
        news = yf_tools.get_company_news(symbol)
        
        # Create a more condensed data summary
        data_summary = {
            'symbol': symbol,
            'current_price': current_price,
            'key_metrics': fundamentals['key_metrics'],
            'recent_news': news
        }
        
        if recommendations is not None:
            data_summary['recent_recommendations'] = recommendations.to_dict('records')
        
        # Format the data for the model
        messages = [
            {"role": "system", "content": system_message},
            {"role": "user", "content": f"Analyze this data for {symbol}: {str(data_summary)}. Query: {query}"}
        ]
        
        # Get response from Groq
        response = client.chat.completions.create(
            messages=messages,
            model="llama3-groq-70b-8192-tool-use-preview",
            temperature=0.7,
        )
        
        return response.choices[0].message.content
    
    except Exception as e:
        st.error(f"Error processing data: {str(e)}")
        return "Unable to process financial data. Please try a different stock symbol or query."

def process_web_search_query(query):
    # Perform web search
    search_results = WebSearchTools.search(query)
    
    # Create system message for web search analysis
    system_message = """You are a web search analysis AI. Analyze the search results and provide a comprehensive response.
    Always include sources and use markdown formatting."""
    
    # Format the message for the model
    messages = [
        {"role": "system", "content": system_message},
        {"role": "user", "content": f"Analyze these search results: {str(search_results)}. Query: {query}"}
    ]
    
    # Get response from Groq
    response = client.chat.completions.create(
        messages=messages,
        model="llama3-groq-70b-8192-tool-use-preview",
        temperature=0.7,
    )
    
    return response.choices[0].message.content

# Streamlit UI
st.title("Financial and Web Search Assistant")

# Create tabs for different functionalities
tab1, tab2 = st.tabs(["Financial Analysis", "Web Search"])

with tab1:
    st.header("Financial Analysis")
    symbol = st.text_input("Enter Stock Symbol (e.g., AAPL):")
    financial_query = st.text_area("Enter your financial analysis query:")
    
    if st.button("Analyze Financial Data") and symbol and financial_query:
        with st.spinner("Analyzing financial data..."):
            try:
                response = process_financial_query(financial_query, symbol)
                st.markdown(response)
            except Exception as e:
                st.error(f"Error processing financial query: {str(e)}")

with tab2:
    st.header("Web Search")
    web_query = st.text_area("Enter your web search query:")
    
    if st.button("Search Web") and web_query:
        with st.spinner("Searching and analyzing..."):
            try:
                response = process_web_search_query(web_query)
                st.markdown(response)
            except Exception as e:
                st.error(f"Error processing web search: {str(e)}")

if __name__ == "__main__":
    pass