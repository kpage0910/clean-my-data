import React from "react";
import { useNavigate } from "react-router-dom";
import { getGreeting } from "./Greeting";
import "./Landing.css";

function Landing() {
  const navigate = useNavigate();
  return (
    <div className="landing">
      {/* Simple header */}
      <header className="landing-header">
        <div className="logo">Clean My Data</div>
        <div className="greeting">{getGreeting()}</div>
      </header>

      {/* Hero section */}
      <main className="landing-main">
        <h1 className="headline">Clean data. Simple.</h1>

        <p className="subheadline">
          Upload messy data. Get it back clean. No scripts, no hassle.
        </p>

        <button className="cta-button" onClick={() => navigate("/signin")}>
          Get Started
        </button>

        {/* Demo visual placeholder */}
        <div className="demo-container">
          <div className="demo-placeholder">
            <div className="demo-step">
              <div className="demo-icon">📁</div>
              <div className="demo-label">Upload</div>
            </div>
            <div className="demo-arrow">→</div>
            <div className="demo-step">
              <div className="demo-icon">🤖</div>
              <div className="demo-label">AI Clean</div>
            </div>
            <div className="demo-arrow">→</div>
            <div className="demo-step">
              <div className="demo-icon">✅</div>
              <div className="demo-label">Ready</div>
            </div>
          </div>
        </div>

        {/* Three bullets */}
        <div className="features">
          <div className="feature">
            <strong>Automatic cleaning.</strong> No scripts needed.
          </div>
          <div className="feature">
            <strong>Instant insights.</strong> Get summaries and anomalies.
          </div>
          <div className="feature">
            <strong>Model-ready output.</strong> Download clean CSV or code.
          </div>
        </div>
      </main>
    </div>
  );
}

export default Landing;
