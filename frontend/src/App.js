// src/App.js
import React, { useState, useEffect } from "react";
import {
  BrowserRouter as Router,
  Routes,
  Route,
  Link,
  useNavigate,
} from "react-router-dom";
import "./App.css";
import Landing from "./Landing";
import SignIn from "./SignIn";
import UsageBanner from "./UsageBanner";
const USAGE_LIMITS = {
  maxFileSizeMB: 5,
};
import { getGreeting } from "./Greeting";
import { jwtDecode } from "jwt-decode";

function DataCleaner() {
  const [file, setFile] = useState(null);
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [usageStats, setUsageStats] = useState(null);
  const [showBanner, setShowBanner] = useState(true);
  const [usageWarning, setUsageWarning] = useState(null);
  const [userAvatar, setUserAvatar] = useState(null);
  const [showAvatarMenu, setShowAvatarMenu] = useState(false);
  const navigate = useNavigate();

  // Check for Google JWT and decode avatar
  useEffect(() => {
    const jwt = localStorage.getItem("google_jwt");
    if (jwt) {
      try {
        const decoded = jwtDecode(jwt);
        if (decoded && decoded.picture) {
          setUserAvatar(decoded.picture);
        }
      } catch (e) {
        setUserAvatar(null);
      }
    } else {
      setUserAvatar(null);
    }
  }, []);

  // Fetch usage stats from backend
  useEffect(() => {
    const fetchUsage = async () => {
      const googleJwt = localStorage.getItem("google_jwt");
      if (!googleJwt) return;
      try {
        const response = await fetch(`${process.env.REACT_APP_API_URL}/usage`, {
          headers: { Authorization: `Bearer ${googleJwt}` },
        });
        const stats = await response.json();
        setUsageStats(stats);
        if (stats.requestsRemaining === 1) {
          setUsageWarning(
            "⚠️ WARNING: This is your LAST cleaning for this month!"
          );
        } else if (stats.requestsRemaining === 2) {
          setUsageWarning(
            `⚠️ Only ${stats.requestsRemaining} cleanings remaining this month.`
          );
        } else {
          setUsageWarning(null);
        }
      } catch (e) {
        setUsageStats(null);
      }
    };
    fetchUsage();
  }, []);

  const handleFileUpload = (event) => {
    const uploadedFile = event.target.files[0];

    if (!uploadedFile) return;

    // Check file type
    if (uploadedFile.type !== "text/csv") {
      setError("Please upload a valid CSV file");
      return;
    }

    // Check file size
    const fileSizeMB = uploadedFile.size / (1024 * 1024);
    if (fileSizeMB > USAGE_LIMITS.maxFileSizeMB) {
      setError(
        `File too large! Maximum size: ${USAGE_LIMITS.maxFileSizeMB} MB`
      );
      return;
    }

    setFile(uploadedFile);
    setResult(null);
    setError(null);
  };

  const cleanData = async () => {
    if (!file) return;

    // Usage limit is now enforced by backend, so no need to check here

    setLoading(true);
    setError(null);

    const formData = new FormData();
    formData.append("file", file);

    // Get Google JWT from localStorage
    const googleJwt = localStorage.getItem("google_jwt");

    try {
      const response = await fetch(`${process.env.REACT_APP_API_URL}/clean`, {
        method: "POST",
        body: formData,
        headers: googleJwt ? { Authorization: `Bearer ${googleJwt}` } : {},
      });
      const data = await response.json();

      if (data.success) {
        // Refetch usage stats from backend
        const googleJwt = localStorage.getItem("google_jwt");
        if (googleJwt) {
          try {
            const response = await fetch(
              `${process.env.REACT_APP_API_URL}/usage`,
              {
                headers: { Authorization: `Bearer ${googleJwt}` },
              }
            );
            const stats = await response.json();
            setUsageStats(stats);
            if (stats.requestsRemaining === 1) {
              setUsageWarning(
                "⚠️ WARNING: This was your LAST cleaning for this month!"
              );
            } else if (stats.requestsRemaining === 2) {
              setUsageWarning(
                `⚠️ Only ${stats.requestsRemaining} cleanings remaining this month.`
              );
            } else {
              setUsageWarning(null);
            }
          } catch (e) {
            setUsageStats(null);
          }
        }
        setResult(data);
      } else {
        setError(data.error || "Something went wrong");
      }
    } catch (err) {
      console.error("Fetch error:", err);
      setError(
        "Cannot connect to backend. Make sure it's running on port 8000."
      );
    } finally {
      setLoading(false);
    }
  };

  const downloadCleanedCSV = () => {
    if (!result || !result.csv_string) return;

    const blob = new Blob([result.csv_string], { type: "text/csv" });
    const url = window.URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = `cleaned_${file.name}`;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    window.URL.revokeObjectURL(url);
  };

  const startOver = () => {
    setFile(null);
    setResult(null);
    setError(null);
  };

  return (
    <div className="App">
      {/* Consistent header */}

      <header className="landing-header">
        <div
          className="logo"
          style={{ cursor: "pointer" }}
          onClick={() => navigate("/")}
        >
          Clean My Data
        </div>
        <div className="greeting">{getGreeting()}</div>
        {userAvatar && (
          <div style={{ position: "relative", display: "inline-block" }}>
            <img
              src={userAvatar}
              alt="Google user avatar"
              style={{
                width: 36,
                height: 36,
                borderRadius: "50%",
                marginLeft: 16,
                border: "1.5px solid #eee",
                cursor: "pointer",
              }}
              title="Account menu"
              onClick={() => setShowAvatarMenu((v) => !v)}
            />
            {showAvatarMenu && (
              <div
                style={{
                  position: "absolute",
                  right: 0,
                  top: 44,
                  background: "#fff",
                  border: "1px solid #eee",
                  borderRadius: 6,
                  boxShadow: "0 2px 8px rgba(0,0,0,0.08)",
                  minWidth: 120,
                  zIndex: 100,
                }}
              >
                <button
                  style={{
                    width: "100%",
                    padding: "10px 18px",
                    background: "none",
                    border: "none",
                    textAlign: "left",
                    cursor: "pointer",
                    fontSize: 15,
                  }}
                  onClick={() => {
                    localStorage.removeItem("google_jwt");
                    setUserAvatar(null);
                    setShowAvatarMenu(false);
                    navigate("/");
                  }}
                >
                  Sign out
                </button>
              </div>
            )}
          </div>
        )}
      </header>

      <main className="cleaner-main">
        {/* Usage Banner */}
        {usageStats && showBanner && !result && (
          <UsageBanner
            usageStats={usageStats}
            onClose={() => setShowBanner(false)}
          />
        )}

        {/* Upload Section */}
        {!result && (
          <div className="upload-container">
            <h1 className="upload-title">Upload your messy data.</h1>
            <p className="upload-subtitle">
              We'll clean it automatically. No questions asked.
            </p>

            {/* Disclaimer */}
            <div
              className="disclaimer"
              style={{
                margin: "10px 0 18px 0",
                color: "#b77c00",
                background: "#fffbe6",
                padding: "12px 18px",
                borderRadius: "6px",
                fontSize: "1rem",
                border: "1px solid #ffe58f",
                textAlign: "center",
              }}
            >
              <strong>Disclaimer:</strong> Data cleaning is powered by AI and
              may not work perfectly on all files, especially very large or
              highly inconsistent CSVs. Please review cleaned results.
            </div>

            <label className="file-upload-area">
              <input
                type="file"
                accept=".csv"
                onChange={handleFileUpload}
                className="file-input-hidden"
              />
              <div className="upload-content">
                <div className="upload-icon">📊</div>
                <p className="upload-text">
                  {file ? (
                    <>
                      <strong>{file.name}</strong>
                      <br />
                      <span style={{ fontSize: "14px", color: "#86868b" }}>
                        Ready to clean
                      </span>
                    </>
                  ) : (
                    <>Click to browse or drag CSV here</>
                  )}
                </p>
              </div>
            </label>

            {error && <div className="error-message">{error}</div>}

            {file && (
              <button
                onClick={cleanData}
                disabled={loading}
                className="primary-button"
              >
                {loading ? "Cleaning..." : "Clean My Data"}
              </button>
            )}
          </div>
        )}

        {/* Results Section */}
        {result && (
          <div className="preview-container">
            <h2 className="preview-title">✨ All clean!</h2>

            {/* Show updated usage after cleaning */}
            {usageStats && (
              <div className="usage-status">
                📊 Usage: {usageStats.requestsUsed}/{usageStats.requestsLimit}{" "}
                cleanings used this month
              </div>
            )}

            <div className="summary-box">
              <p className="ai-summary">{result.summary}</p>
            </div>

            <div className="stats-row">
              <div className="stat-item">
                <div className="stat-number">
                  {result.before.rows.toLocaleString()}
                </div>
                <div className="stat-label">Rows Before</div>
              </div>
              <div className="stat-arrow">→</div>
              <div className="stat-item">
                <div className="stat-number">
                  {result.after.rows.toLocaleString()}
                </div>
                <div className="stat-label">Rows After</div>
              </div>
            </div>

            {/* What was fixed */}
            {result.operations && result.operations.length > 0 && (
              <div className="operations-section">
                <h3>What we fixed:</h3>
                <ul className="operations-list">
                  {result.operations.map((op, i) => (
                    <li key={i}>{op}</li>
                  ))}
                </ul>
              </div>
            )}

            {/* Preview table */}
            {result.preview && result.before_preview && (
              <div className="preview-section">
                <h3>Preview (first 10 rows):</h3>
                <div className="preview-tables-row">
                  <div style={{ flex: 1, minWidth: 320, maxWidth: "100%" }}>
                    <h4
                      style={{
                        marginBottom: 8,
                        textAlign: "center",
                      }}
                    >
                      Before Cleaning
                    </h4>
                    <div className="table-scroll">
                      <table className="preview-table">
                        <thead>
                          <tr>
                            {result.before_preview.columns.map((col, i) => (
                              <th key={i}>{col}</th>
                            ))}
                          </tr>
                        </thead>
                        <tbody>
                          {result.before_preview.data.map((row, i) => (
                            <tr key={i}>
                              {row.map((cell, j) => (
                                <td key={j}>{cell !== null ? cell : "—"}</td>
                              ))}
                            </tr>
                          ))}
                        </tbody>
                      </table>
                    </div>
                  </div>
                  <div style={{ flex: 1, minWidth: 320, maxWidth: "100%" }}>
                    <h4
                      style={{
                        marginBottom: 8,
                        textAlign: "center",
                      }}
                    >
                      After Cleaning
                    </h4>
                    <div className="table-scroll">
                      <table className="preview-table">
                        <thead>
                          <tr>
                            {result.preview.columns.map((col, i) => (
                              <th key={i}>{col}</th>
                            ))}
                          </tr>
                        </thead>
                        <tbody>
                          {result.preview.data.map((row, i) => (
                            <tr key={i}>
                              {row.map((cell, j) => (
                                <td key={j}>{cell !== null ? cell : "—"}</td>
                              ))}
                            </tr>
                          ))}
                        </tbody>
                      </table>
                    </div>
                  </div>
                </div>
              </div>
            )}

            <div className="action-buttons">
              <button onClick={downloadCleanedCSV} className="primary-button">
                📥 Download Cleaned CSV
              </button>
              <button onClick={startOver} className="secondary-button">
                Clean Another File
              </button>
            </div>
          </div>
        )}
      </main>
    </div>
  );
}

function App() {
  return (
    <Router>
      <Routes>
        <Route path="/" element={<Landing />} />
        <Route path="/signin" element={<SignIn />} />
        <Route path="/app" element={<DataCleaner />} />
      </Routes>
    </Router>
  );
}

export default App;
