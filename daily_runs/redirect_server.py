from flask import Flask, request

app = Flask(__name__)

@app.route("/callback")
def callback():
    """
    Handles the OAuth 2.0 callback and extracts the authorization code.
    """
    auth_code = request.args.get("code")
    
    if auth_code:
        print(f"[INFO] Authorization Code: {auth_code}")
        with open("auth_code.txt", "w") as f:
            f.write(auth_code)
        return "Authorization successful. You may close this window.", 200
    else:
        return "Error: No authorization code found.", 400

if __name__ == "__main__":
    app.run(port=8080)  # Run the server on port 8080
