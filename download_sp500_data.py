"""
Data Download Script for S&P 500 (^GSPC) OHLCV Data
Downloads daily data from 2015-01-01 to 2025-01-01 and saves as CSV.
"""

import yfinance as yf
import pandas as pd
import os
import argparse
from datetime import datetime


def main(out_path=None):
    """
    Download S&P 500 data and save to CSV.
    
    Args:
        out_path (str, optional): Output file path. Defaults to 'data/sp500_2015_2025.csv'
    """
    if out_path is None:
        out_path = "data/sp500_2015_2025.csv"
    
    # Create data directory if it doesn't exist
    os.makedirs(os.path.dirname(out_path) if os.path.dirname(out_path) else ".", exist_ok=True)
    
    print(f"Downloading S&P 500 (^GSPC) data from 2015-01-01 to 2025-01-01...")
    print(f"Output path: {out_path}")
    
    try:
        # Download data using yfinance
        ticker = "^GSPC"
        start_date = "2015-01-01"
        end_date = "2025-01-01"
        
        data = yf.download(ticker, start=start_date, end=end_date, progress=True)
        
        if data.empty:
            raise ValueError("Downloaded data is empty. Please check your internet connection and date range.")
        
        # Reset index to make Date a column
        data.reset_index(inplace=True)
        
        # Ensure we have the required columns
        required_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
        missing_cols = [col for col in required_cols if col not in data.columns]
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")
        
        # Save to CSV
        data.to_csv(out_path, index=False)
        
        print(f"\n✓ Successfully downloaded {len(data)} rows of data")
        print(f"✓ Data saved to: {out_path}")
        print(f"\nData columns: {list(data.columns)}")
        print(f"Date range: {data['Date'].min()} to {data['Date'].max()}")
        
    except Exception as e:
        print(f"✗ Error downloading data: {str(e)}")
        raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download S&P 500 data")
    parser.add_argument(
        "--out_path",
        type=str,
        default=None,
        help="Output CSV file path (default: data/sp500_2015_2025.csv)"
    )
    args = parser.parse_args()
    
    main(out_path=args.out_path)

