import React, { useState, useEffect } from 'react';
import { Activity, ShieldAlert, ShieldCheck, Zap, Server, ActivitySquare } from 'lucide-react';
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer } from 'recharts';
import './index.css';

const mockChartData = [
  { time: '10:00', score: 12 },
  { time: '10:05', score: 25 },
  { time: '10:10', score: 15 },
  { time: '10:15', score: 45 },
  { time: '10:20', score: 85 },
  { time: '10:25', score: 20 },
  { time: '10:30', score: 35 },
];

const initialTransactions = [
  { id: 'txn-1', score: 85, status: 'Flagged', time: '10:20:15', amount: '$1,200.00' },
  { id: 'txn-2', score: 12, status: 'Approved', time: '10:20:12', amount: '$45.00' },
  { id: 'txn-3', score: 25, status: 'Approved', time: '10:19:45', amount: '$89.99' },
  { id: 'txn-4', score: 45, status: 'Review', time: '10:18:22', amount: '$350.00' },
];

function App() {
  const [health, setHealth] = useState(null);
  const [modelInfo, setModelInfo] = useState(null);
  const [transactions, setTransactions] = useState(initialTransactions);
  const [txnId, setTxnId] = useState('');
  const [scoring, setScoring] = useState(false);
  const [scoreResult, setScoreResult] = useState(null);

  // Fetch Health & Model Info
  useEffect(() => {
    const fetchStatus = async () => {
      try {
        const [hRes, mRes] = await Promise.all([
          fetch('http://127.0.0.1:8000/api/v1/health').catch(() => ({ ok: false })),
          fetch('http://127.0.0.1:8000/api/v1/model-info').catch(() => ({ ok: false }))
        ]);
        
        if (hRes.ok) setHealth(await hRes.json());
        if (mRes.ok) setModelInfo(await mRes.json());
      } catch (err) {
        console.error("Failed to fetch system status", err);
      }
    };
    fetchStatus();
    // Refresh health every 30s
    const intv = setInterval(fetchStatus, 30000);
    return () => clearInterval(intv);
  }, []);

  const handleScore = async (e) => {
    e.preventDefault();
    if (!txnId) return;
    
    setScoring(true);
    setScoreResult(null);
    try {
      const res = await fetch('http://127.0.0.1:8000/api/v1/predict', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ transaction_id: txnId, include_features: false })
      });
      if (res.ok) {
        const data = await res.json();
        setScoreResult(data);
        // Add to feed
        setTransactions(prev => [
          {
            id: txnId.substring(0, 8) + '...',
            score: data.risk_score || 0,
            status: data.is_fraud ? 'Flagged' : 'Approved',
            time: new Date().toLocaleTimeString(),
            amount: '---'
          },
          ...prev.slice(0, 4)
        ]);
      }
    } catch (err) {
      console.error(err);
    }
    setScoring(false);
  };

  return (
    <div className="app-container">
      {/* Sidebar */}
      <aside className="sidebar">
        <div className="sidebar-header">
          <ShieldAlert className="pulse" color="var(--accent-color)" />
          <span>Risk Platform</span>
        </div>
        <nav style={{ display: 'flex', flexDirection: 'column', gap: '0.5rem' }}>
          <a className="nav-item active"><ActivitySquare size={18} /> Dashboard</a>
          <a className="nav-item"><Server size={18} /> Models Registry</a>
          <a className="nav-item"><ShieldAlert size={18} /> Review Queue</a>
        </nav>
      </aside>

      {/* Main Content */}
      <main className="main-content">
        <header className="header">
          <div>
            <h1>Real-Time Fraud Dashboard</h1>
            <p>Monitor transactions and model health metrics.</p>
          </div>
          <div style={{ display: 'flex', gap: '1rem' }}>
            <span className={`badge ${health?.status === 'ok' ? 'success' : 'warning'}`}>
              System: {health?.status || 'Unknown'}
            </span>
            <span className="badge info">Model: {modelInfo?.model_version || 'Loading...'}</span>
          </div>
        </header>

        {/* Top Widgets */}
        <div className="grid grid-cols-3" style={{ marginBottom: '2rem' }}>
          <div className="card glass">
            <h3 className="card-title"><Activity size={18} /> System Uptime</h3>
            <div className="stat-value">{health ? `${Number(health.uptime_seconds).toFixed(1)}s` : '---'}</div>
          </div>
          <div className="card glass">
            <h3 className="card-title"><ShieldCheck size={18} /> Active Model F1</h3>
            <div className="stat-value">{modelInfo?.metrics?.f1 ? Number(modelInfo.metrics.f1).toFixed(3) : '---'}</div>
          </div>
          <div className="card glass">
            <h3 className="card-title"><ShieldAlert size={18} /> AUC ROC</h3>
            <div className="stat-value">{modelInfo?.metrics?.roc_auc ? Number(modelInfo.metrics.roc_auc).toFixed(3) : '---'}</div>
          </div>
        </div>

        {/* Main Grid */}
        <div className="grid" style={{ gridTemplateColumns: '2fr 1fr', gap: '2rem' }}>
          
          {/* Left Column */}
          <div style={{ display: 'flex', flexDirection: 'column', gap: '2rem' }}>
            
            {/* Chart Card */}
            <div className="card">
              <h3 className="card-title"><Zap size={18} /> Risk Score Trends (Last 30 Mins)</h3>
              <div style={{ height: '300px', width: '100%' }}>
                <ResponsiveContainer>
                  <LineChart data={mockChartData} margin={{ top: 10, right: 10, left: -20, bottom: 0 }}>
                    <CartesianGrid strokeDasharray="3 3" stroke="#2e3440" />
                    <XAxis dataKey="time" stroke="#a0a5b1" tick={{ fill: '#a0a5b1', fontSize: 12 }} />
                    <YAxis stroke="#a0a5b1" tick={{ fill: '#a0a5b1', fontSize: 12 }} domain={[0, 100]} />
                    <Tooltip 
                      contentStyle={{ backgroundColor: '#232730', borderColor: '#2e3440', color: '#f0f2f5' }}
                      itemStyle={{ color: '#3b82f6' }}
                    />
                    <Line type="monotone" dataKey="score" stroke="#3b82f6" strokeWidth={3} dot={{ r: 4, fill: '#3b82f6' }} activeDot={{ r: 6 }} />
                  </LineChart>
                </ResponsiveContainer>
              </div>
            </div>

            {/* Live Feed Table */}
            <div className="card">
              <h3 className="card-title">Recent Transactions</h3>
              <div className="table-container">
                <table>
                  <thead>
                    <tr>
                      <th>Time</th>
                      <th>Txn ID</th>
                      <th>Amount</th>
                      <th>Score</th>
                      <th>Status</th>
                    </tr>
                  </thead>
                  <tbody>
                    {transactions.map((t, i) => (
                      <tr key={i}>
                        <td>{t.time}</td>
                        <td style={{ fontFamily: 'monospace' }}>{t.id}</td>
                        <td>{t.amount}</td>
                        <td>
                          <span style={{ fontWeight: 'bold', color: t.score > 80 ? 'var(--danger)' : t.score > 40 ? 'var(--warning)' : 'var(--success)'}}>
                            {t.score.toFixed(0)}
                          </span>
                        </td>
                        <td>
                          <span className={`badge ${t.status === 'Flagged' ? 'danger' : t.status === 'Review' ? 'warning' : 'success'}`}>
                            {t.status}
                          </span>
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </div>

          </div>

          {/* Right Column */}
          <div style={{ display: 'flex', flexDirection: 'column', gap: '2rem' }}>
            
            {/* Manual Scoring Form */}
            <div className="card">
              <h3 className="card-title">Manual Scoring</h3>
              <form onSubmit={handleScore} style={{ display: 'flex', flexDirection: 'column', gap: '1rem' }}>
                <div>
                  <label style={{ display: 'block', marginBottom: '0.5rem', fontSize: '0.875rem', color: 'var(--text-secondary)' }}>
                    Transaction UUID
                  </label>
                  <input 
                    type="text" 
                    className="input-field" 
                    placeholder="Enter UUID..."
                    value={txnId}
                    onChange={(e) => setTxnId(e.target.value)}
                  />
                </div>
                <button type="submit" className="btn btn-primary" disabled={scoring}>
                  {scoring ? 'Scoring...' : 'Score Transaction'}
                </button>
              </form>

              {scoreResult && (
                <div style={{ marginTop: '1.5rem', padding: '1rem', backgroundColor: 'var(--bg-color)', borderRadius: '0.5rem', border: '1px solid var(--border-color)' }}>
                  <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: '1rem' }}>
                    <span style={{ color: 'var(--text-secondary)' }}>Risk Score</span>
                    <span style={{ fontWeight: 'bold', color: scoreResult.is_fraud ? 'var(--danger)' : 'var(--success)' }}>
                      {Number(scoreResult.risk_score).toFixed(2)} / 100
                    </span>
                  </div>
                  <div style={{ display: 'flex', justifyContent: 'space-between' }}>
                    <span style={{ color: 'var(--text-secondary)' }}>Decision</span>
                    <span className={`badge ${scoreResult.is_fraud ? 'danger' : 'success'}`}>
                      {scoreResult.is_fraud ? 'Flagged' : 'Approved'}
                    </span>
                  </div>
                </div>
              )}
            </div>

          </div>
        </div>
      </main>
    </div>
  );
}

export default App;
