import asyncio, os
import pandas as pd
from src.preprocess import fetch_and_preprocess_data

async def main():
    tickers = ['AAPL', 'GOOGL', 'MSFT', 'AMZN']  # Add more tickers as needed
    today = pd.Timestamp('today').date()
    result = await fetch_and_preprocess_data(tickers)
    
    # Print the result to console
    print(result)
    
    # Save the result to an Excel file
    if not os.path.exists(f"analysis/{today}"):
        os.mkdir(f"analysis/{today}")
    output_file = f'analysis/{today}/stock_analysis_results.xlsx'
    result.to_excel(output_file, index=False)
    print(f"Results saved to {output_file}")

if __name__ == '__main__':
    asyncio.run(main())