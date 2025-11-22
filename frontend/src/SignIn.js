import React from "react";
import { GoogleLogin } from "@react-oauth/google";
import { getGreeting } from "./Greeting";
import { useNavigate } from "react-router-dom";
import "./Landing.css";

function SignIn() {
  const navigate = useNavigate();
  return (
    <div className="landing">
      <header className="landing-header">
        <div
          className="logo"
          style={{ cursor: "pointer" }}
          onClick={() => navigate("/")}
        >
          Clean My Data
        </div>
        <div className="greeting">{getGreeting()}</div>
      </header>
      <main className="landing-main">
        <h1 className="headline">Sign in to continue</h1>
        <div
          style={{
            display: "flex",
            flexDirection: "column",
            alignItems: "center",
            margin: "32px 0",
          }}
        >
          <GoogleLogin
            size="large"
            shape="pill"
            text="signin_with"
            width="280"
            onSuccess={(credentialResponse) => {
              if (credentialResponse.credential) {
                localStorage.setItem(
                  "google_jwt",
                  credentialResponse.credential
                );
              }
              console.log("Google login success", credentialResponse);
              navigate("/app");
            }}
            onError={() => {
              console.log("Google login failed");
            }}
          />
        </div>
        <div
          style={{
            color: "#888",
            fontSize: "1rem",
            marginTop: 12,
            textAlign: "center",
          }}
        >
          We use Google to keep your data safe and track your monthly cleanings.
        </div>
      </main>
    </div>
  );
}

export default SignIn;
