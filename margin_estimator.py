# margin_estimator.py
# This script calculates the initial margin for a given portfolio using the Deutsche Börse Prisma Margin Estimator API.
# It fetches the margin estimation and drills down details, then saves the results into an Excel file.
# Dependencies: requests, pandas

import requests
import pandas as pd
from datetime import datetime

url_base = "https://api.developer.deutsche-boerse.com/prod/prisma-margin-estimator-2-0/2.0.0/"
api_header = {"X-DBP-APIKEY": "fa6ba0f7-dbcc-4ba5-bbde-17eb8f517763"}

proxies = {
    'http': 'http://webproxy.deutsche-boerse.de:8080/',
    'https': 'http://webproxy.deutsche-boerse.de:8080/',
}

# Create the session and set the proxies.
s = requests.Session()
s.proxies = proxies

# User input
business_date = "20240429"  # Ensure this is in the correct format (YYYYMMDD)
maturity = "202405"         # Ensure this is in the correct format (YYYYMM)
Productid = "OESX"
call_put_flag = "C"
Exercise_price = 4925
Net_ls_balance = 1  # -1 means short 1 position, +1 means long 1 position
xm = False  # Set cross margin - xm to False
Currency = "EUR"
#################################################

# Format the business_date to ensure correct format (if needed)
business_date = datetime.strptime(business_date, "%Y%m%d").strftime("%Y%m%d")
maturity = datetime.strptime(maturity, "%Y%m").strftime("%Y%m")

request_payload = {
    "snapshot": {"business_date": int(business_date)},
    'portfolio_components': [
        {
            'type': 'etd_portfolio',
            'etd_portfolio': [
                {
                    "line_no": 1,
                    "product_id": Productid,
                    "maturity": maturity,
                    "call_put_flag": call_put_flag,
                    "exercise_price": Exercise_price,
                    "net_ls_balance": Net_ls_balance
                }
            ]
        }
    ],
    'clearing_currency': Currency,
    'is_cross_margined': xm
}

# Sending POST request
response = s.post(url_base + 'estimator', headers=api_header, json=request_payload)

# Parse the response and extract initial margin value
response_json = response.json()
initial_margin = response_json['portfolio_margin'][0]['initial_margin']
print(f"Initial Margin: {initial_margin}")

# Process the response data and save as Excel file
df_portfolio_margin = pd.DataFrame(response_json['portfolio_margin'])
df_drilldowns = pd.DataFrame(response_json['drilldowns'])

output_filename = f"margin_info_xm_{xm}.xlsx"
with pd.ExcelWriter(output_filename) as writer:
    df_portfolio_margin.to_excel(writer, sheet_name='Portfolio Margin', index=False)
    df_drilldowns.to_excel(writer, sheet_name='Drilldowns', index=False)

print(f"Excel file saved successfully for Cross Margined: {xm}")
