#! python3
#
# pip install yfinance pandas
#
# python stock_collector.py MSFT AAPL --ofile stock.csv
# python stock_collector.py VOW3.DE BMW.DE --start-date 2023-10-01 --end-date 2023-10-20 --ofile stock.csv
#
# C2025 George Biro
#
import sys
import yfinance as yf
import pandas as pd
import csv
import argparse
from datetime import date, timedelta, datetime


def fetch_stock_data(start_date_obj, end_date_obj, ofile="stock_data.csv"):
    """
    Collects asset information for given tickers and date range, then writes to a CSV.

    Args:
        tickers (list): List of stock ticker symbols (e.g., ['MSFT', 'AAPL']).
        start_date_obj (datetime.date): The start date for data collection.
        end_date_obj (datetime.date): The end date for data collection (inclusive).
        ofile (str): Name of the CSV file to save data.
    """

    # yfinance history's end_date is exclusive for daily data,
    # so add one day to ensure the user's specified end_date is included.
    yf_query_end_date = end_date_obj + timedelta(days=1)

    fieldnames = [
        "Asset Ticker",
        "Date",
        "Asset Name",
        "Asset Type",
        "Industry",
        "Opening Price",
        "Closing Price",
        "High Price",
        "Low Price",
        "Dividend",
        "Currency",
        "Volume",
        "Last Dividend Date",
        "Last Dividend Value",
        "Total Revenue",
        "Market Cap",
        "Total Debt",
        "Total Cash",
    ]

    all_rows_data = []  # To store data for all tickers before writing to CSV

    for ticker_symbol in start_date_obj:
        print(
            f"\nProcessing {ticker_symbol} from {start_date_obj[ticker_symbol].strftime('%Y-%m-%d')} to {end_date_obj.strftime('%Y-%m-%d')}"
        )
        try:
            ticker = yf.Ticker(ticker_symbol)

            # Get general info (mostly static for the asset)
            info = ticker.info

            currency = info.get("currency", "N/A")
            asset_type = info.get("quoteType", "N/A")
            prevClose = info.get("previousClose")
            if (prevClose < 2) and (asset_type == "EQUITY"):
                print(f"Bowli {ticker_symbol} as the price is {prevClose}{currency}")
                continue

            # Use .get() with a default value to avoid KeyErrors if a field is missing
            # Fallback to ticker if name not found
            asset_name = info.get("shortName", ticker_symbol)
            echange = info.get("fullExchangeName", "N/A")
            industry = info.get("industry", "N/A")
            lastDividendDate = info.get("lastDividendDate", "N/A")
            lastDividendValue = info.get("lastDividendValue", "N/A")
            dividendRate = info.get("dividendRate")
            profitMargins = info.get("profitMargins", "N/A")
            totalCash = info.get("totalCash", "N/A")
            totalDebt = info.get("totalDebt", "N/A")
            totalRevenue = info.get("totalRevenue", "N/A")
            heldPercentInsiders = info.get("heldPercentInsiders", "N/A")
            heldPercentInstitutions = info.get("heldPercentInstitutions", "N/A")
            currentPrice = info.get("currentPrice", "N/A")
            companyOfficers = info.get("companyOfficers", "N/A")
            marketCap = info.get("marketCap", "N/A")

            start_date = start_date_obj[ticker_symbol].strftime("%Y-%m-%d")
            end_date = yf_query_end_date.strftime("%Y-%m-%d")
            if start_date > end_date:
                continue
            # Get historical market data for the specified range
            # interval="1d" is the default for daily data
            hist_df = ticker.history(start=start_date, end=end_date)

            if hist_df.empty:
                print(
                    f"No historical data found for {ticker_symbol} in the given date range."
                )
                continue

            # Iterate through each day in the historical data
            for index_date, row_data in hist_df.iterrows():
                # index_date is a pandas Timestamp object
                formatted_date = index_date.strftime("%Y-%m-%d")

                # Extract data, providing 'N/A' or 0.0 for missing values
                opening_price = row_data.get("Open", "N/A")
                closing_price = row_data.get("Close", "N/A")
                high_price = row_data.get("High", "N/A")
                low_price = row_data.get("Low", "N/A")
                # Dividends can be 0.0, which is valid
                dividend = row_data.get("Dividends", 0.0)
                volume = row_data.get("Volume", "N/A")

                data_row = {
                    "Asset Ticker": ticker_symbol,
                    "Date": formatted_date,
                    "Asset Name": asset_name,
                    "Asset Type": asset_type,
                    "Industry": industry,
                    "Opening Price": (
                        f"{opening_price:.2f}"
                        if isinstance(opening_price, (int, float))
                        else opening_price
                    ),
                    "Closing Price": (
                        f"{closing_price:.2f}"
                        if isinstance(closing_price, (int, float))
                        else closing_price
                    ),
                    "High Price": (
                        f"{high_price:.2f}"
                        if isinstance(high_price, (int, float))
                        else high_price
                    ),
                    "Low Price": (
                        f"{low_price:.2f}"
                        if isinstance(low_price, (int, float))
                        else low_price
                    ),
                    # Dividends can be small
                    "Dividend": (
                        f"{dividend:.4f}"
                        if isinstance(dividend, (int, float))
                        else dividend
                    ),
                    "Volume": (
                        int(volume)
                        if isinstance(volume, (int, float)) and not pd.isna(volume)
                        else volume
                    ),
                    "Currency": currency,
                    "Last Dividend Date": (
                        datetime.utcfromtimestamp(lastDividendDate)
                        if isinstance(lastDividendDate, (int))
                        else lastDividendDate
                    ),
                    "Last Dividend Value": (
                        f"{lastDividendValue:.2f}"
                        if isinstance(lastDividendValue, (int, float))
                        else lastDividendValue
                    ),
                    "Total Revenue": (
                        f"{totalRevenue:.0g}"
                        if isinstance(totalRevenue, (int, float))
                        else totalRevenue
                    ),
                    "Market Cap": (
                        f"{marketCap:.0g}"
                        if isinstance(marketCap, (int, float))
                        else marketCap
                    ),
                    "Total Debt": (
                        f"{totalDebt:.0g}"
                        if isinstance(totalDebt, (int, float))
                        else totalDebt
                    ),
                    "Total Cash": (
                        f"{totalCash:.0g}"
                        if isinstance(totalCash, (int, float))
                        else totalCash
                    ),
                }
                all_rows_data.append(data_row)

            print(
                f"Successfully fetched {len(hist_df)} day(s) of data for {ticker_symbol}."
            )

        except Exception as e:
            # This can catch various errors: invalid ticker, network issues, missing data in .info
            print(f"Could not retrieve or process data for {ticker_symbol}: {e}")
            # For example, yfinance might raise yfinance.exceptions.YFinanceException for some tickers
            # or if .info doesn't contain expected keys (e.g., for indices or certain asset types).

    if not all_rows_data:
        print("\nNo data collected. CSV file will not be created.")
        return

    # Write all collected data to the CSV file
    try:
        with open(ofile, "a+", newline="", encoding="utf-8") as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            if csvfile.tell() == 0:
                #                csvfile.write("#")
                writer.writeheader()

            writer.writerows(all_rows_data)
        print(f"\nData successfully written to {ofile}")
    except IOError:
        print(f"Error: Could not write to CSV file {ofile}.")


def read_asset_list(file_path):
    try:
        with open(file_path, "r") as f:
            tlist2 = [stripped for line in f if (stripped := line.strip())]
    except FileNotFoundError:
        print(f"Error: The file '{file_path}' was not found.")
    except Execption as e:
        print(f"An Error occured: {e}")
    return list(set(tlist2))


def retreive_start_date(file_path, tlist, start_date, end_date):
    assets = {}
    try:
        df_orig_raw = pd.read_csv(
            file_path, parse_dates=["Date"], na_values=["N/A"]
        ).sort_values(by=["Date", "Asset Ticker"])
        df_raw_pivoted = df_orig_raw.pivot_table(
            index="Date", columns="Asset Ticker", values=["Closing Price"]
        )

        if isinstance(df_raw_pivoted.columns, pd.MultiIndex):
            df_raw_pivoted.columns = [
                f"{ticker}_{feat.replace(' ', '_')}"
                for feat, ticker in df_raw_pivoted.columns
            ]
        else:
            df_raw_pivoted.columns = [
                f"{str(col).replace(' ', '_')}" for col in df_raw_pivoted.columns
            ]

        for i in tlist:
            close_col = f"{i}_Closing_Price"
            if close_col not in df_raw_pivoted.columns:
                assets[i] = start_date
            else:
                current_price_series = df_raw_pivoted[close_col]
                assets[i] = max(
                    (current_price_series.index.max() + timedelta(days=1)).date(),
                    end_date,
                )

    except Exception as e:
        for i in tlist:
            assets[i] = start_date

    return assets


def main():
    parser = argparse.ArgumentParser(
        description="Collects stock asset information from Yahoo Finance and saves it to a CSV file.",
        formatter_class=argparse.RawTextHelpFormatter,  # For better help text formatting
    )
    parser.add_argument(
        "--start-date", help="Start date for data collection (YYYY-MM-DD format)."
    )
    parser.add_argument(
        "--end-date",
        help="End date for data collection (YYYY-MM-DD format). If not provided, defaults to yesterday.",
    )
    parser.add_argument(
        "--ofile",
        default="stock_data.csv",
        help="Output CSV file name (default: stock_data.csv).",
    )
    parser.add_argument(
        "--asset-list", default="asset_list.txt", help="Asset list file"
    )

    try:
        args = parser.parse_args()
    except SystemExit as e:
        # argparse raises SystemExit on -h/--help or error, which is fine
        # If you need to catch it specifically for other reasons, you can
        if e.code == 0:  # Help was requested
            sys.exit(0)
        else:  # An error occurred
            sys.exit(e.code)  # e.code is usually 2 for errors

    # --- Date Logic ---
    # Default: if no dates provided, get data for the "last day"
    # (which is typically considered yesterday's closing data).
    if args.end_date:
        try:
            end_date_obj = datetime.strptime(args.end_date, "%Y-%m-%d").date()
        except ValueError:
            print("Error: Invalid end_date format. Please use YYYY-MM-DD.")
            return
    else:
        # Default end_date to yesterday (as typically the last fully closed trading day's data available)
        end_date_obj = date.today() - timedelta(days=1)

    if end_date_obj >= date.today():
        print(
            "Warning: End date is today or in the future. Data for today might be incomplete or unavailable until market close."
        )
        print(
            "To get yesterday's data (last full trading day), do not specify dates or set end_date to yesterday."
        )

    tlist = read_asset_list(args.asset_list)

    if args.start_date:
        try:
            start_date_obj = datetime.strptime(args.start_date, "%Y-%m-%d").date()
        except ValueError:
            print("Error: Invalid start_date format. Please use YYYY-MM-DD.")
            return
    else:
        start_date_obj = date.today() - timedelta(days=3000)

    if start_date_obj > end_date_obj:
        print("Error: Start date is later then end date!")
        return

    assets = retreive_start_date(args.ofile, tlist, start_date_obj, end_date_obj)
    fetch_stock_data(assets, end_date_obj, args.ofile)


if __name__ == "__main__":
    main()
