// src/UsageBanner.js
import React from "react";
import "./UsageBanner.css";

function UsageBanner({ usageStats }) {
  return (
    <div className="usage-banner">
      <div className="usage-banner-content">
        <div className="banner-header">
          <h3>Usage This Month</h3>
        </div>

        <div className="usage-stats">
          <div className="stat-item">
            <div className="stat-label">Cleanings Used</div>
            <div className="stat-value">
              {usageStats.requestsUsed} / {usageStats.requestsLimit}
            </div>
          </div>

          <div className="stat-item">
            <div className="stat-label">Remaining</div>
            <div className="stat-value highlight">
              {usageStats.requestsRemaining}
            </div>
          </div>

          <div className="stat-item">
            <div className="stat-label">Resets</div>
            <div className="stat-value small">{usageStats.nextResetDate}</div>
          </div>
        </div>

        <div className="usage-limits-info">
          <p>
            <strong>Limits:</strong> 1000 rows, 5MB per file
          </p>
        </div>
      </div>
    </div>
  );
}

export default UsageBanner;
