import yfinance as yf
from langchain_community.tools.yahoo_finance_news import YahooFinanceNewsTool
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser
import mlflow

# function to get the stock code
def get_stock_code(company: str) -> str:
    if not company:
        return {"error": "No company name provided"}
    try:
        search_res = yf.Search(company, max_results=1)
        if search_res.quotes and len(search_res.quotes) > 0:
            top_symbol = search_res.quotes[0]['symbol']
            return {"ticker": top_symbol}
        else:
            return {"error": f"No ticker found for '{company}'"}
    except Exception as e:
        return {"error": f"Ticker lookup failed: {e}"}
    

# function to fetch the company news
def fetch_company_news(ticker: str) -> str:
    if not ticker:
        return {}
    try:
        tool = YahooFinanceNewsTool()
        try:
            news = tool.run(ticker)
        except:
            news = None
        if (not news) or ("No news found" in news):
            ticker_obj = yf.Ticker(ticker)
            news_items = ticker_obj.news[:3]  # Limit to 3
            print(news_items)
            news_list = [
                f"{item['content'].get('title', '')}: {item['content'].get('provider', {}).get('displayName', '')}"
                for item in news_items if item.get('content', {}).get('title')
            ]
            news = "\n".join(news_list) if news_list else "No news found."
        news_summary = news[:500] + "..." if len(news) > 500 else news
        return {"news_summary": news_summary}
    except Exception as e:
        return {"error": f"YahooFinanceNewsTool failed: {e}"}
    
# function to analyse sentiment
def analyze_sentiment(model, company_name, ticker, news):
    """Analyze news sentiment using LLM."""
    with mlflow.start_run(nested=True, run_name="raj_sentiment_analysis"):
        try:
            prompt = mlflow.genai.load_prompt("prompts:/arun_prakash-sentiment_analysis-prompt@latest")
            prompt = ChatPromptTemplate.from_template(prompt.template)
            chain = prompt | model | JsonOutputParser()
            result = chain.invoke({"company_name": company_name, "ticker": ticker, "news": news})
            return result
        except Exception as e:
            return {"error": f"Sentiment analysis failed: {e}"}