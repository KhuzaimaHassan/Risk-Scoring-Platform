import { useState, useEffect, useCallback, useRef } from 'react';
import { Activity, ShieldAlert, ShieldCheck, Zap, Server, ActivitySquare, Lock } from 'lucide-react';
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer } from 'recharts';
import './index.css';

const money = value => new Intl.NumberFormat(undefined, { style: 'currency', currency: 'USD', maximumFractionDigits: 4 }).format(value);
const timestamp = value => new Date(value).toLocaleString();
const message = data => Array.isArray(data.detail) ? data.detail.map(e => e.msg).join('; ') : data.detail || 'Request failed. Please try again.';

function App() {
  const [token, setToken] = useState(() => sessionStorage.getItem('token'));
  const activeToken = useRef(token);
  const [username, setUsername] = useState('');
  const [password, setPassword] = useState('');
  const [loginError, setLoginError] = useState('');
  const [isLoggingIn, setIsLoggingIn] = useState(false);
  const [health, setHealth] = useState(null);
  const [modelInfo, setModelInfo] = useState(null);
  const [transactions, setTransactions] = useState([]);
  const [available, setAvailable] = useState([]);
  const [models, setModels] = useState([]);
  const [reviews, setReviews] = useState([]);
  const [page, setPage] = useState('dashboard');
  const [txnId, setTxnId] = useState('');
  const [scoring, setScoring] = useState(false);
  const [scoreResult, setScoreResult] = useState(null);
  const [scoreError, setScoreError] = useState('');
  const [dashboardStats, setDashboardStats] = useState(null);
  const [chartData, setChartData] = useState([]);
  const [dashboardError, setDashboardError] = useState('');
  const [updatedAt, setUpdatedAt] = useState(null);
  const [loading, setLoading] = useState(false);
  const [reviewing, setReviewing] = useState(null);

  const logout = useCallback(() => {
    activeToken.current = null;
    setToken(null);
    sessionStorage.removeItem('token');
    localStorage.removeItem('token');
    setPassword(''); setUsername(''); setScoreResult(null); setScoreError('');
    setDashboardStats(null); setTransactions([]); setChartData([]); setModels([]); setReviews([]);
    setAvailable([]); setHealth(null); setModelInfo(null); setUpdatedAt(null); setPage('dashboard');
    setDashboardError(''); setTxnId('');
  }, []);

  const request = useCallback(async (path, options = {}) => {
    const res = await fetch(`/api/v1${path}`, { ...options, signal: AbortSignal.timeout(15000),
      headers: { Authorization: `Bearer ${token}`, 'Content-Type': 'application/json', ...options.headers } });
    const data = await res.json().catch(() => ({}));
    if (res.status === 401) { logout(); setLoginError('Your session expired. Please sign in again.'); throw new Error('Your session expired. Please sign in again.'); }
    if (!res.ok && !(path === '/health' && res.status === 503)) throw new Error(message(data));
    return data;
  }, [token, logout]);

  const loadDashboard = useCallback(async () => {
    if (!token) return;
    setLoading(true);
    try {
      const [h, m, stats, chart, txns, choices, registry, queue] = await Promise.all([
        request('/health'), request('/model-info'), request('/dashboard/stats'), request('/dashboard/chart-data'),
        request('/dashboard/recent-transactions'), request('/dashboard/transactions'), request('/models'), request('/dashboard/review-queue')
      ]);
      if (activeToken.current !== token) return;
      setHealth(h); setModelInfo(m); setDashboardStats(stats); setChartData(chart); setTransactions(txns);
      setAvailable(choices); setModels(registry.versions); setReviews(queue);
      setUpdatedAt(new Date()); setDashboardError('');
    } catch (err) {
      if (activeToken.current === token) setDashboardError(`Unable to refresh: ${err.message} Displayed data may be stale.`);
    } finally { setLoading(false); }
  }, [token, request]);

  useEffect(() => {
    if (!token) return;
    const initial = setTimeout(loadDashboard, 0);
    const interval = setInterval(loadDashboard, 30000);
    return () => { clearTimeout(initial); clearInterval(interval); };
  }, [token, loadDashboard]);

  const handleLogin = async e => {
    e.preventDefault(); setLoginError(''); setIsLoggingIn(true);
    try {
      const res = await fetch('/api/v1/token', { method: 'POST', signal: AbortSignal.timeout(15000),
        headers: { 'Content-Type': 'application/x-www-form-urlencoded' }, body: new URLSearchParams({ username, password }) });
      const data = await res.json().catch(() => ({}));
      if (!res.ok) throw new Error(message(data));
      activeToken.current = data.access_token;
      sessionStorage.setItem('token', data.access_token); setToken(data.access_token); setPassword('');
    } catch (err) { setLoginError(err.message); }
    finally { setIsLoggingIn(false); }
  };

  const handleScore = async e => {
    e.preventDefault(); setScoring(true); setScoreError(''); setScoreResult(null);
    try {
      const data = await request('/predict', { method: 'POST', body: JSON.stringify({ transaction_id: txnId.trim(), include_features: false }) });
      if (activeToken.current !== token) return;
      setScoreResult(data);
      // The server owns the feed, score scale, amounts, chart, and KPI calculations.
      await loadDashboard();
    } catch (err) { if (activeToken.current === token) setScoreError(err.message); }
    finally { setScoring(false); }
  };

  const review = async id => {
    setReviewing(id);
    try { await request(`/dashboard/reviews/${id}`, { method: 'POST', body: JSON.stringify({ notes: 'Reviewed in dashboard' }) }); await loadDashboard(); }
    catch (err) { setDashboardError(err.message); }
    finally { setReviewing(null); }
  };

  if (!token) return <div className="login-container"><div className="card glass login-card">
    <Lock size={32} color="var(--accent-color)" /><h1>Admin Login</h1><p>Risk Scoring Platform</p>
    {loginError && <p role="alert" className="error-message">{loginError}</p>}
    <form onSubmit={handleLogin} className="form-stack">
      <label htmlFor="username">Username</label><input id="username" autoComplete="username" className="input-field" value={username} onChange={e => setUsername(e.target.value)} required />
      <label htmlFor="password">Password</label><input id="password" autoComplete="current-password" type="password" className="input-field" value={password} onChange={e => setPassword(e.target.value)} required />
      <button className="btn btn-primary" disabled={isLoggingIn}>{isLoggingIn ? 'Authenticating...' : 'Sign In'}</button>
    </form></div></div>;

  return <div className="app-container">
    <aside className="sidebar"><div className="sidebar-header"><ShieldAlert color="var(--accent-color)" />Risk Platform</div>
      <nav aria-label="Main navigation" className="form-stack">
        {[['dashboard', 'Dashboard', ActivitySquare], ['models', 'Models Registry', Server], ['reviews', 'Review Queue', ShieldAlert]].map(([key, label, Icon]) =>
          <button key={key} className={`nav-item ${page === key ? 'active' : ''}`} aria-current={page === key ? 'page' : undefined} onClick={() => setPage(key)}><Icon size={18} />{label}</button>)}
      </nav><button className="btn btn-outline sign-out" onClick={logout}>Sign Out</button>
    </aside>
    <main className="main-content"><header className="header"><div><h1>{page === 'dashboard' ? 'Real-Time Fraud Dashboard' : page === 'models' ? 'Models Registry' : 'Review Queue'}</h1>
      <p>{page === 'dashboard' ? 'Monitor scored transactions and model health.' : page === 'models' ? 'Registered model versions and evaluation metrics.' : 'Flagged predictions awaiting analyst review.'}</p></div>
      <div className="status-bar"><span className={`badge ${health?.status === 'ok' ? 'success' : 'warning'}`}>System: {health?.status || 'Unknown'}</span>
        <span className="badge info">Model: {modelInfo?.model_version || 'Loading...'}</span></div></header>
      <div className="refresh-bar"><p>{updatedAt ? `Last updated ${updatedAt.toLocaleTimeString()} · refreshes every 30 seconds` : 'Loading live data...'}</p>
        <button className="btn btn-outline" disabled={loading} onClick={loadDashboard}>{loading ? 'Refreshing...' : 'Refresh'}</button></div>
      {dashboardError && <p role="alert" className="error-message">{dashboardError}</p>}
      {page === 'dashboard' && <>
        <div className="grid grid-cols-3 widgets">
          <div className="card glass"><h2 className="card-title"><Activity size={18} />Scored Volume (USD)</h2><div className="stat-value">{dashboardStats ? money(dashboardStats.total_volume_usd) : '—'}</div></div>
          <div className="card glass"><h2 className="card-title"><ShieldCheck size={18} />Scored Transactions</h2><div className="stat-value">{dashboardStats?.total_transactions.toLocaleString() ?? '—'}</div><p>{dashboardStats?.stored_transactions.toLocaleString() ?? '—'} stored transactions</p></div>
          <div className="card glass"><h2 className="card-title"><ShieldAlert size={18} />Confirmed Fraud Detected</h2><div className="stat-value">{dashboardStats?.total_fraud_caught.toLocaleString() ?? '—'}</div><p>{dashboardStats?.flagged_transactions.toLocaleString() ?? '—'} currently flagged transactions</p></div>
        </div>
        <div className="grid dashboard-grid"><div className="form-stack">
          <section className="card"><h2 className="card-title"><Zap size={18} />Risk Score Trends (Last 30 Minutes)</h2>
            {chartData.length === 0 ? <p>No predictions in the last 30 minutes.</p> : <div className="chart-container"><ResponsiveContainer><LineChart data={chartData}>
              <CartesianGrid strokeDasharray="3 3" stroke="#2e3440" /><XAxis dataKey="time" tickFormatter={v => new Date(v).toLocaleTimeString()} stroke="#a0a5b1" />
              <YAxis domain={[0, 100]} stroke="#a0a5b1" /><Tooltip labelFormatter={timestamp} /><Line type="monotone" dataKey="score" stroke="#3b82f6" strokeWidth={3} />
            </LineChart></ResponsiveContainer></div>}
          </section><section className="card"><h2 className="card-title">Recent Predictions</h2>
            {transactions.length === 0 ? <p>No predictions yet. Select a stored transaction to score.</p> : <div className="table-container"><table><thead><tr><th>Scored at</th><th>Transaction</th><th>Amount (USD)</th><th>Score / 100</th><th>Decision</th></tr></thead>
              <tbody>{transactions.map(t => <tr key={t.log_id}><td>{timestamp(t.time)}</td><td><button className="id-button" title={t.id} onClick={() => setTxnId(t.id)}>{t.id.slice(0, 8)}…</button></td><td>{money(t.amount_usd)}</td><td data-testid="feed-score">{t.score.toFixed(2)}</td><td><span className={`badge ${t.status === 'Flagged' ? 'danger' : 'success'}`}>{t.status}</span></td></tr>)}</tbody></table></div>}
          </section></div>
          <section className="card scoring-card"><h2 className="card-title">Manual Scoring</h2><form onSubmit={handleScore} className="form-stack">
            <label htmlFor="transaction-id">Transaction UUID</label><input id="transaction-id" className="input-field" placeholder="Enter UUID..." value={txnId} onChange={e => setTxnId(e.target.value)} required />
            <label htmlFor="stored-transaction">Or select a stored transaction</label><select id="stored-transaction" className="input-field" value={available.some(t => t.id === txnId) ? txnId : ''} onChange={e => setTxnId(e.target.value)}>
              <option value="">Select transaction...</option>{available.map(t => <option key={t.id} value={t.id}>{t.id.slice(0, 8)}… · {money(t.amount_usd)}</option>)}</select>
            <button className="btn btn-primary" disabled={scoring}>{scoring ? 'Scoring...' : 'Score Transaction'}</button></form>
            {scoreError && <p role="alert" className="error-message">{scoreError}</p>}
            {scoreResult && <div className="score-result" aria-live="polite"><p>Risk Score <strong data-testid="result-score">{scoreResult.risk_score.toFixed(2)} / 100</strong></p>
              <p>Decision <span className={`badge ${scoreResult.is_fraud ? 'danger' : 'success'}`}>{scoreResult.is_fraud ? 'Flagged' : 'Approved'}</span></p>
              <p>Recommended action: {scoreResult.risk_context.recommended_action}</p><p>Completed in {scoreResult.latency_ms} ms</p></div>}
          </section></div>
      </>}
      {page === 'models' && <section className="card"><div className="table-container"><table><thead><tr><th>Version</th><th>Stage</th><th>Trained</th><th>ROC AUC</th><th>F1</th></tr></thead>
        <tbody>{models.map(m => <tr key={m.version}><td>{m.version}</td><td>{m.is_production ? 'Serving' : m.stage}</td><td>{timestamp(m.trained_at)}</td><td>{m.roc_auc?.toFixed(3) ?? '—'}</td><td>{m.f1?.toFixed(3) ?? '—'}</td></tr>)}</tbody></table></div>{models.length === 0 && <p>No registered models.</p>}
        {modelInfo && <p className="model-details">Serving threshold: {modelInfo.metrics.threshold_used} · Recall: {((modelInfo.metrics.recall ?? 0) * 100).toFixed(1)}% · Precision: {((modelInfo.metrics.precision ?? 0) * 100).toFixed(1)}%</p>}</section>}
      {page === 'reviews' && <section className="card">{reviews.length === 0 ? <p>No flagged predictions awaiting review.</p> : <div className="table-container"><table><thead><tr><th>Transaction</th><th>Amount (USD)</th><th>Score / 100</th><th>Scored at</th><th>Review</th></tr></thead><tbody>{reviews.map(r => <tr key={r.log_id}><td title={r.id}>{r.id.slice(0, 8)}…</td><td>{money(r.amount_usd)}</td><td>{r.score.toFixed(2)}</td><td>{timestamp(r.time)}</td><td><button className="btn btn-outline" disabled={reviewing === r.log_id} onClick={() => review(r.log_id)}>Mark reviewed</button></td></tr>)}</tbody></table></div>}<p className="model-details">Review acknowledges inspection; it does not confirm a fraud label or execute a payment decision.</p></section>}
    </main></div>;
}

export default App;
