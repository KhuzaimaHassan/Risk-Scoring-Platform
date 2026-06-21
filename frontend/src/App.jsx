import React, { useState, useEffect } from 'react';
import { Activity, ShieldAlert, ShieldCheck, Zap, Server, ActivitySquare, Lock } from 'lucide-react';
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer } from 'recharts';
import './index.css';

function App() {
  const [token, setToken] = useState(localStorage.getItem('token') || null);
  const [username, setUsername] = useState('');
  const [password, setPassword] = useState('');
  const [loginError, setLoginError] = useState('');
  const [isLoggingIn, setIsLoggingIn] = useState(false);

  const [health, setHealth] = useState(null);
  const [modelInfo, setModelInfo] = useState(null);
  const [transactions, setTransactions] = useState([]);
  const [txnId, setTxnId] = useState('');
  const [scoring, setScoring] = useState(false);
  const [scoreResult, setScoreResult] = useState(null);
  const [dashboardStats, setDashboardStats] = useState(null);
  const [chartData, setChartData] = useState([]);

  const logout = () => {
    setToken(null);
    localStorage.removeItem('token');
  };

  // Fetch Dashboard Data
  useEffect(() => {
    if (!token) return;

    const fetchDashboard = async () => {
      try {
        const headers = { 'Authorization': `Bearer ${token}` };
        
        const [hRes, mRes, statsRes, chartRes, txnRes] = await Promise.all([
          fetch('/api/v1/health', { headers }).catch(() => ({ ok: false })),
          fetch('/api/v1/model-info', { headers }).catch(() => ({ ok: false })),
          fetch('/api/v1/dashboard/stats', { headers }).catch(() => ({ ok: false })),
          fetch('/api/v1/dashboard/chart-data', { headers }).catch(() => ({ ok: false })),
          fetch('/api/v1/dashboard/recent-transactions', { headers }).catch(() => ({ ok: false }))
        ]);
        
        if (hRes.status === 401) return logout();

        if (hRes.ok) setHealth(await hRes.json());
        if (mRes.ok) setModelInfo(await mRes.json());
        if (statsRes.ok) setDashboardStats(await statsRes.json());
        if (chartRes.ok) setChartData(await chartRes.json());
        if (txnRes.ok) setTransactions(await txnRes.json());
      } catch (err) {
        console.error("Failed to fetch dashboard data", err);
      }
    };
    
    fetchDashboard();
    // Auto-refresh every 30s
    const intv = setInterval(fetchDashboard, 30000);
    return () => clearInterval(intv);
  }, [token]);

  const handleLogin = async (e) => {
    e.preventDefault();
    setLoginError('');
    setIsLoggingIn(true);
    try {
      const formData = new URLSearchParams();
      formData.append('username', username);
      formData.append('password', password);

      const res = await fetch('/api/v1/token', {
        method: 'POST',
        headers: { 'Content-Type': 'application/x-www-form-urlencoded' },
        body: formData
      });

      if (res.ok) {
        const data = await res.json();
        setToken(data.access_token);
        localStorage.setItem('token', data.access_token);
      } else {
        setLoginError('Invalid username or password');
      }
    } catch (err) {
      setLoginError('Network error');
    }
    setIsLoggingIn(false);
  };

  const handleScore = async (e) => {
    e.preventDefault();
    if (!txnId) return;
    
    setScoring(true);
    setScoreResult(null);
    try {
      const res = await fetch('/api/v1/predict', {
        method: 'POST',
        headers: { 
          'Content-Type': 'application/json',
          'Authorization': `Bearer ${token}`
        },
        body: JSON.stringify({ transaction_id: txnId, include_features: false })
      });
      if (res.status === 401) {
        setScoring(false);
        return logout();
      }
      if (res.ok) {
        const data = await res.json();
        setScoreResult(data);
        // Add to feed
        setTransactions(prev => [
          {
            id: txnId.substring(0, 8) + '...',
            score: data.risk_score * 100 || 0,
            status: data.is_fraud ? 'Flagged' : 'Approved',
            time: new Date().toLocaleTimeString(),
            amount: '---'
          },
          ...prev.slice(0, 9)
        ]);
      }
    } catch (err) {
      console.error(err);
    }
    setScoring(false);
  };

  if (!token) {
    return (
      <div style={{ display: 'flex', justifyContent: 'center', alignItems: 'center', minHeight: '100vh', backgroundColor: 'var(--bg-color)' }}>
        <div className="card glass" style={{ width: '400px', padding: '2rem' }}>
          <div style={{ display: 'flex', flexDirection: 'column', alignItems: 'center', marginBottom: '2rem' }}>
            <div style={{ padding: '1rem', backgroundColor: 'rgba(59, 130, 246, 0.1)', borderRadius: '50%', marginBottom: '1rem' }}>
              <Lock size={32} color="var(--accent-color)" />
            </div>
            <h2 style={{ fontSize: '1.5rem', fontWeight: 'bold' }}>Admin Login</h2>
            <p style={{ color: 'var(--text-secondary)' }}>Risk Scoring Platform</p>
          </div>

          {loginError && (
            <div style={{ padding: '0.75rem', marginBottom: '1rem', backgroundColor: 'rgba(239, 68, 68, 0.1)', color: 'var(--danger)', borderRadius: '0.5rem', fontSize: '0.875rem', textAlign: 'center' }}>
              {loginError}
            </div>
          )}

          <form onSubmit={handleLogin} style={{ display: 'flex', flexDirection: 'column', gap: '1rem' }}>
            <div>
              <label style={{ display: 'block', marginBottom: '0.5rem', fontSize: '0.875rem', color: 'var(--text-secondary)' }}>Username</label>
              <input 
                type="text" 
                className="input-field" 
                value={username} 
                onChange={e => setUsername(e.target.value)} 
                required 
              />
            </div>
            <div>
              <label style={{ display: 'block', marginBottom: '0.5rem', fontSize: '0.875rem', color: 'var(--text-secondary)' }}>Password</label>
              <input 
                type="password" 
                className="input-field" 
                value={password} 
                onChange={e => setPassword(e.target.value)} 
                required 
              />
            </div>
            <button type="submit" className="btn btn-primary" style={{ marginTop: '1rem' }} disabled={isLoggingIn}>
              {isLoggingIn ? 'Authenticating...' : 'Sign In'}
            </button>
          </form>
        </div>
      </div>
    );
  }

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
        <div style={{ marginTop: 'auto', paddingTop: '2rem' }}>
          <button className="btn" style={{ width: '100%', backgroundColor: 'transparent', border: '1px solid var(--border-color)', color: 'var(--text-secondary)' }} onClick={logout}>
            Sign Out
          </button>
        </div>
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
            <h3 className="card-title"><Activity size={18} /> Total Volume</h3>
            <div className="stat-value">{dashboardStats ? `$${(dashboardStats.total_volume_usd/1000).toFixed(1)}k` : '---'}</div>
          </div>
          <div className="card glass">
            <h3 className="card-title"><ShieldCheck size={18} /> Processed Txns</h3>
            <div className="stat-value">{dashboardStats ? dashboardStats.total_transactions.toLocaleString() : '---'}</div>
          </div>
          <div className="card glass">
            <h3 className="card-title"><ShieldAlert size={18} /> Total Fraud Caught</h3>
            <div className="stat-value" style={{color: 'var(--danger)'}}>{dashboardStats ? dashboardStats.total_fraud_caught.toLocaleString() : '---'}</div>
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
                  <LineChart data={chartData} margin={{ top: 10, right: 10, left: -20, bottom: 0 }}>
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
