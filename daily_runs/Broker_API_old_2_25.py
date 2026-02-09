import requests
import json
import time
import os
import requests

class BrokerAPI:
    def __init__(self, config_path, config_file="config.json"):
        with open(config_path, "r") as file:
            self.config = json.load(file)

        self.client_id = self.config["client_id"]
        self.client_secret = self.config["client_secret"]
        self.redirect_uri = self.config["redirect_uri"]
        self.access_token = self.config["access_token"]
        self.refresh_token = self.config["refresh_token"]
        self.base_url = "https://api.schwab.com"

    def fetch_auth_code_from_local():
        """
        Fetch the authorization code from the home laptop via ngrok.
        """
        local_server_url = "https://platypus-outgoing-annually.ngrok-free.app/get_auth_code"
        try:
            response = requests.get(local_server_url)
            if response.status_code == 200:
                return response.json().get("auth_code")
            else:
                print("[ERROR] Failed to fetch authorization code from local machine.")
                return None
        except requests.RequestException as e:
            print(f"[ERROR] Could not connect to local machine: {e}")
            return None

    def request_access_token(self):
        """
        Exchanges the authorization code for an access token.
        """
        authorization_code = fetch_auth_code_from_local()
        if not authorization_code:
            print("[ERROR] No valid authorization code retrieved.")
            return

        url = f"{self.base_url}/oauth2/token"
        payload = {
            "grant_type": "authorization_code",
            "client_id": self.client_id,
            "client_secret": self.client_secret,
            "redirect_uri": self.redirect_uri,
            "code": authorization_code
        }
        headers = {"Content-Type": "application/x-www-form-urlencoded"}

        response = requests.post(url, data=payload, headers=headers)

        if response.status_code == 200:
            data = response.json()
            self.access_token = data["access_token"]
            self.refresh_token = data["refresh_token"]
            self.config["access_token"] = self.access_token
            self.config["refresh_token"] = self.refresh_token
            with open("config.json", "w") as file:
                json.dump(self.config, file, indent=4)

            print("[INFO] Access token obtained successfully.")
        else:
            print("[ERROR] Failed to obtain access token:", response.text)

    def ensure_valid_token(self):
        """
        Refreshes the access token only if expired.
        """
        if "expires_at" in self.config and time.time() < self.config["expires_at"]:
            return  # Token is still valid, no need to refresh

        print("[INFO] Refreshing access token...")

        url = f"{self.base_url}/oauth2/token"
        payload = {
            "grant_type": "refresh_token",
            "client_id": self.client_id,
            "client_secret": self.client_secret,
            "refresh_token": self.refresh_token
        }
        headers = {"Content-Type": "application/x-www-form-urlencoded"}

        response = requests.post(url, data=payload, headers=headers)

        if response.status_code == 200:
            new_tokens = response.json()
            self.access_token = new_tokens["access_token"]
            self.refresh_token = new_tokens["refresh_token"]
            expires_in = new_tokens.get("expires_in", 1800)  # Default to 30 minutes if not provided
            self.config["access_token"] = self.access_token
            self.config["refresh_token"] = self.refresh_token
            self.config["expires_at"] = time.time() + expires_in

            with open("config.json", "w") as file:
                json.dump(self.config, file, indent=4)

            print("[INFO] Access token refreshed successfully.")
        else:
            print("[ERROR] Failed to refresh token:", response.text)

    
    def _send_request(self, endpoint, method="GET", payload=None):
        """
        Internal helper method to send API requests.
        """
        self.ensure_valid_token()  # Refresh token if expired

        url = f"{self.base_url}{endpoint}"
        headers = {
            "Authorization": f"Bearer {self.access_token}",
            "Content-Type": "application/json"
        }

        try:
            if method == "GET":
                response = requests.get(url, headers=headers)
            elif method == "POST":
                response = requests.post(url, headers=headers, json=payload)
            else:
                raise ValueError("Unsupported HTTP method")

            response.raise_for_status()
            return response.json()

        except requests.exceptions.RequestException as e:
            print(f"[ERROR] API request failed: {e}")
            return None

    def get_account_details(self):
        """
        Fetches the account details, including settled cash.

        Returns:
            dict: Account details response from the API.
        """
        if "account_id" not in self.config:
            print("[WARNING] `account_id` missing in config.json. Attempting to fetch from API...")
            account_data = self._send_request("/v1/accounts")

            if account_data and "accounts" in account_data:
                self.config["account_id"] = account_data["accounts"][0]["accountId"]
                with open("config.json", "w") as file:
                    json.dump(self.config, file, indent=4)
                print(f"[INFO] Retrieved and saved account ID: {self.config['account_id']}")
            else:
                print("[ERROR] Unable to fetch account ID. Please update config.json manually.")
                return None

        return self._send_request(f"/v1/accounts/{self.config['account_id']}")

    def get_settled_cash(self):
        """
        Fetches the available settled cash for trading.

        Returns:
            float: Amount of settled cash available.
        """
        account_details = self.get_account_details()
        if account_details and "settled_cash" in account_details:
            return float(account_details["settled_cash"])
        
        print("[ERROR] Failed to fetch settled cash.")
        return 0.0

    def get_stock_prices(self, tickers):
        """
        Fetches real-time stock prices for the given tickers.

        Args:
            tickers (list): List of stock ticker symbols.

        Returns:
            dict: Dictionary mapping tickers to current prices.
        """
        tickers_str = ",".join(tickers)
        endpoint = f"/v1/marketdata/quotes?symbols={tickers_str}"
        response = self._send_request(endpoint)

        if response and "quotes" in response:
            return {quote["symbol"]: float(quote["lastPrice"]) for quote in response["quotes"]}
        
        print("[ERROR] Failed to fetch stock prices.")
        return {}

    def get_previous_closes(self, tickers):
        """
        Fetches previous day's closing prices for the given tickers.

        Args:
            tickers (list): List of stock ticker symbols.

        Returns:
            dict: Dictionary mapping tickers to their previous close prices.
        """
        tickers_str = ",".join(tickers)
        endpoint = f"/v1/marketdata/quotes?symbols={tickers_str}"
        response = self._send_request(endpoint)

        if response and "quotes" in response:
            return {quote["symbol"]: float(quote["previousClose"]) for quote in response["quotes"]}

        print("[ERROR] Failed to fetch previous close prices.")
        return {}

    def place_1st_triggers_sequential_order(self, ticker, quantity, limit_price, cancel_price, trigger_price, trailing_percent):
        """
        Place a 1st Triggers Sequential Order with Charles Schwab's API.

        Args:
            ticker (str): Stock ticker symbol.
            quantity (int): Number of shares to buy.
            limit_price (float): Limit price for the buy order.
            cancel_price (float): Price above which the buy order will be canceled.
            trigger_price (float): Price to trigger the trailing stop order.
            trailing_percent (float): Percentage for the trailing stop.

        Returns:
            dict: API response.
        """
        # Define the 1st order (limit buy order)
        first_order = {
            "orderType": "LIMIT",
            "session": "NORMAL",
            "duration": "DAY",
            "orderStrategyType": "SINGLE",
            "price": limit_price,
            "orderLegCollection": [
                {
                    "instruction": "BUY",
                    "quantity": quantity,
                    "instrument": {
                        "symbol": ticker,
                        "assetType": "EQUITY"
                    }
                }
            ],
            "cancelTime": {
                "price": cancel_price  # Cancel if price exceeds this value
            }
        }

        # Define the 2nd order (trailing stop order)
        second_order = {
            "orderType": "TRAILING_STOP",
            "session": "NORMAL",
            "duration": "GTC",  # Good Till Canceled
            "trailingStopAmountType": "PERCENT",
            "trailingStopAmount": trailing_percent,
            "triggerPrice": trigger_price,
            "orderLegCollection": [
                {
                    "instruction": "SELL",
                    "quantity": quantity,
                    "instrument": {
                        "symbol": ticker,
                        "assetType": "EQUITY"
                    }
                }
            ]
        }

        # Combine into a 1st triggers sequential order
        sequential_order = {
            "orderStrategyType": "TRIGGER",
            "childOrderStrategies": [first_order, second_order]
        }

        # Send the request to Schwab's API
        url = f"{self.base_url}/v1/accounts/{self.config['account_id']}/orders"
        headers = {"Authorization": f"Bearer {self.access_token}", "Content-Type": "application/json"}
        response = requests.post(url, json=sequential_order, headers=headers)

        if response.status_code == 201:
            return response.json()
        else:
            response.raise_for_status()