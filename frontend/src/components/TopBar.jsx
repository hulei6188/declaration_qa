import React from 'react';
import { Activity } from 'lucide-react';

export default function TopBar({ healthStatus, healthLabel }) {
  return (
    <header className="topbar">
      <div>
        <p className="eyebrow">Declaration QA Workflow</p>
        <h1>申报智能问答工作台</h1>
      </div>
      <div className="status-strip">
        <span className={`status-dot ${healthStatus}`} />
        <span>{healthLabel}</span>
      </div>
    </header>
  );
}
