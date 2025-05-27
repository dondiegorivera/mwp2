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


def fetch_stock_data(tickers, start_date_obj, end_date_obj, ofile="stock_data.csv"):
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

    print(f"Fetching data for tickers: {', '.join(tickers)}")
    print(
        f"Period: {start_date_obj.strftime('%Y-%m-%d')} to {end_date_obj.strftime('%Y-%m-%d')}"
    )

    for ticker_symbol in tickers:
        print(f"\nProcessing {ticker_symbol}...")
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
            asset_name = info.get(
                "shortName", ticker_symbol
            )  # Fallback to ticker if name not found
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

            # Get historical market data for the specified range
            # interval="1d" is the default for daily data
            hist_df = ticker.history(
                start=start_date_obj.strftime("%Y-%m-%d"),
                end=yf_query_end_date.strftime("%Y-%m-%d"),
            )

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
                dividend = row_data.get(
                    "Dividends", 0.0
                )  # Dividends can be 0.0, which is valid
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
                    "Dividend": (
                        f"{dividend:.4f}"
                        if isinstance(dividend, (int, float))
                        else dividend
                    ),  # Dividends can be small
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

    if args.start_date:
        try:
            start_date_obj = datetime.strptime(args.start_date, "%Y-%m-%d").date()
        except ValueError:
            print("Error: Invalid start_date format. Please use YYYY-MM-DD.")
            return
    else:
        # If no start_date is provided, default start_date to be the same as end_date.
        # This means we'll fetch data for that single day (the end_date).
        start_date_obj = end_date_obj

    # Validate date range
    if start_date_obj > end_date_obj:
        print("Error: Start date cannot be after end date.")
        return
    if end_date_obj >= date.today():
        print(
            "Warning: End date is today or in the future. Data for today might be incomplete or unavailable until market close."
        )
        print(
            "To get yesterday's data (last full trading day), do not specify dates or set end_date to yesterday."
        )

    tlist = [
        "BTC-EUR",
        "ETH-EUR",
        "EURUSD=X",
        "EURJPY=X",
        "CL=F",
        "GC=F",
        "SI=F",
        "EOAN.DE",
        "EBK.DE",
        "ENI.DE",
        "UN0.DE",
        "WWG.F",
        "LEC.F",
        "RIO1.DE",
        "CRC.F",
        "IBE.MC",
        "CLP.F",
        "ALV.DE",
        "DB1.DE",
        "MUV2.DE",
        "HNR1.DE",
        "CBK.DE",
        "TLX.DE",
        "3V64.DE",
        "FRE.DE",
        "FME.DE",
        "MRK.DE",
        "AFX.DE",
        "PFE.F",
        "UNH.F",
        "2M6.DE",
        "SNW.F",
        "GIS.DE",
        "BRM.DE",
        "LLY.F",
        "VNA.DE",
        "LHA.DE",
        "UPAB.F",
        "FDX.DE",
        "CY2.F",
        "CTM.F",
        "NTT.F",
        "DIP.F",
        "IFX.DE",
        "SRT3.DE",
        "RAA.DE",
        "LEG.DE",
        "INL.F",
        "IBM.F",
        "CIS.F",
        "TII.F",
        "VOW3.DE",
        "BMW.DE",
        "HDM.F",
        "FMC1.DE",
        "NISA.F",
        "SUK.F",
        "VOW.DE",
        "ADS.DE",
        "BEI.DE",
        "PSM.DE",
        "BOSS.DE",
        "PUM.DE",
        "JNJ.F",
        "PRG.F",
        "CCC3.DE",
        "WDP.F",
        "PEP.DE",
        "MMM.F",
        "LOR.F",
        "MOH.F",
        "CPA.F",
        "HEI.DE",
        "HOT.DE",
        "LWE.F",
        "ILT.F",
        "KMY.DE",
        "SQU.F",
        "DCO.DE",
        "UTDI.DE",
        "TUI1.DE",
        "ALD.F",
        "TN8.F",
        "ENL.F",
        "H4W.F",
        "WF3.F",
        "FIE.DE",
        "TTK.DE",
        "HBH.DE",
        "WMT.F",
        "HDI.DE",
        "SRB.DE",
        "CTO.F",
        "DYH.F",
        "CAR.F",
        "BAS.DE",
        "BAYN.DE",
        "SY1.DE",
        "LXS.DE",
        "FPE3.DE",
        "WCH.DE",
        "SDF.DE",
        "AIL.F",
        "DLY.F",
        "LIN",
        "BHP",
        "RIO",
        "SHW",
        "SCCO",
        "APD",
        "FCX",
        "ECL",
        "CRH",
        "NUE",
        "CTVA",
        "DOW",
        "DD",
        "PPG",
        "NTR",
        "NEM",
        "LYB",
        "VMC",
        "MLM",
        "CMCSA",
        "DIS",
        "ATVI",
        "NTES",
        "AMX",
        "EA",
        "CHT",
        "TTWO",
        "HD",
        "TM",
        "NKE",
        "LOW",
        "SBUX",
        "TJX",
        "MELI",
        "JD",
        "STLA",
        "RACE",
        "F",
        "HMC",
        "GM",
        "WMT",
        "PG",
        "KO",
        "PEP",
        "COST",
        "FMX",
        "UL",
        "MDLZ",
        "CL",
        "TGT",
        "EL",
        "MNST",
        "KDP",
        "HSY",
        "ADM",
        "KHC",
        "KMB",
        "GIS",
        "XOM",
        "CVX",
        "SHEL",
        "TTE",
        "COP",
        "BP",
        "EQNR",
        "PBR",
        "PBR-A",
        "SLB",
        "EOG",
        "CNQ",
        "MPC",
        "OXY",
        "PXD",
        "V",
        "MA",
        "PSX",
        "E",
        "WDS",
        "HES",
        "VLO",
        "SU",
        "ET",
        "UNH",
        "JNJ",
        "LLY",
        "MRK",
        "NVS",
        "AZN",
        "TMO",
        "PFE",
        "DHR",
        "ABT",
        "SNY",
        "BMY",
        "MDT",
        "ELV",
        "SYK",
        "GILD",
        "CVS",
        "CI",
        "ZTS",
        "UPS",
        "BA",
        "HON",
        "DE",
        "RTX",
        "GE",
        "LMT",
        "ADP",
        "ETN",
        "CNI",
        "ITW",
        "NOC",
        "FDX",
        "WM",
        "RELX",
        "GD",
        "TRI",
        "MMM",
        "EMR",
        "WY",
        "AAPL",
        "MSFT",
        "NVDA",
        "ASML",
        "ADBE",
        "CSCO",
        "ACN",
        "SAP",
        "TXN",
        "INTC",
        "INTU",
        "QCOM",
        "IBM",
        "AMAT",
        "SONY",
        "ADI",
        "CEG",
        "ELP",
        "ALI=F",
        "MGC=F",
        "SB=F",
        "SPAX.PVT",
        "NOK",
        "GLD",
        "SPY",
        "QBTS",
        "RGTI",
        "LCID",
        "IONQ",
        "TSLA",
        "MARA",
        "PLTR",
        "JWN",
        "BTG",
        "CLSK",
        "SOFI",
        "ITUB",
        "NU",
        "AGNC",
        "BBD",
        "HIMS",
        "AAL",
        "RIOT",
        "DFS",
        "GRAB",
        "1211.HK",
        "P911.DE",
        "DPE5.DE",
        "EJ7.F",
        "SAP.DE",
        "ZAL.DE",
        "SIE.DE",
        "8TP0.F",
        "50V.F",
        "SHA0.DE",
        "DTG.DE",
        "EVD.DE",
        "OB7.F",
        "SHL.DE",
        "HTG.DE",
        "KO32.F",
        "TMV.DE",
        "DHER.DE",
        "AIXA.DE",
        "MAV.F",
        "DEZ.DE",
        "HAG.DE",
        "PLTS.DE",
        "9OC.F",
        "9OC.SG",
        "NDX1.DE",
        "PAH3.DE",
        "BSP.DE",
        "ULF1.F",
        "HEN3.DE",
        "4JH.F",
        "TEG.DE",
        "E3T.F",
        "S4D.BE",
        "SGDE.DE",
        "FPQ1.F",
        "AG1.DE",
        "CEC.DE",
        "MJ4.F",
        "TLT5.DE",
        "3FR0.F",
        "YBB.F",
        "BT81.F",
        "EBM.F",
        "BPE5.DE",
        "9DO.F",
        "BR01.F",
        "BJ4.F",
        "CHP.SG",
        "B4X.F",
        "12J0.F",
        "HFG.DE",
        "PNY.F",
        "SZU.DE",
        "GLY.F",
        "HDD.DE",
        "PW5.BE",
        "S92.DE",
        "DBK.SG",
        "PBB.DE",
        "CON.DE",
        "RRU.DE",
        "QQ3S.DE",
        "BEI.DE",
        "BVB.DE",
        "LEG.DE",
        "ETHA.DE",
        "BC8.DE",
        "SZG.DE",
        "EJT1.DE",
        "RHM.DE",
        "KGX.DE",
        "NOV.DE",
        "AIR.DE",
        "JEN.DE",
        "MLP.DE",
        "RQ0.F",
        "VBTC.DE",
        "NVD.DE",
        "DOU.DE",
        "FTK.DE",
        "TL0.DE",
        "IOS.DE",
        "4GLD.DE",
        "ZETH.DE",
        "3CP.F",
        "KTN.DE",
        "BTG4.SG",
        "HEN.DE",
        "NOA3.DE",
        "HBC1.DE",
        "ABEA.DE",
        "KCO.DE",
        "SGL.DE",
        "PFE.DE",
        "TLX.DE",
        "DCHB.MU",
        "PFSE.DE",
        "VS0L.DE",
        "E3B.DE",
        "FNM.SG",
        "JUN3.DE",
        "DWNI.DE",
        "1CO.DE",
        "NDA.DE",
        "DWS.DE",
        "BMT.DE",
        "NEM.DE",
        "WAF.DE",
        "MTX.DE",
        "RRTL.DE",
        "GBF.DE",
        "F3C.DE",
        "ORC.DE",
        "MICR",
    ]

    fetch_stock_data(tlist, start_date_obj, end_date_obj, args.ofile)


if __name__ == "__main__":
    main()
