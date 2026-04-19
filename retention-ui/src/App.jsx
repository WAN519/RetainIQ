import { useState, useEffect, useCallback } from "react";

const API_BASE = import.meta.env.VITE_API_BASE ?? "";  // prod: Render URL, dev: Vite proxy

const riskColor = (v) => v >= 80 ? "#FF3B30" : v >= 65 ? "#FF9500" : "#34C759";
const riskLabel = (v) => v >= 80 ? "Critical" : v >= 65 ? "High" : "Moderate";

const GaugeRing = ({ value, size = 64 }) => {
  const r = (size - 10) / 2, circ = 2 * Math.PI * r, fill = (value / 100) * circ;
  return (
    <svg width={size} height={size} style={{ flexShrink: 0 }}>
      <circle cx={size/2} cy={size/2} r={r} fill="none" stroke="#F2F2F7" strokeWidth={6}/>
      <circle cx={size/2} cy={size/2} r={r} fill="none" stroke={riskColor(value)} strokeWidth={6}
        strokeDasharray={`${fill} ${circ}`} strokeLinecap="round" transform={`rotate(-90 ${size/2} ${size/2})`}/>
      <text x="50%" y="50%" textAnchor="middle" dominantBaseline="middle" fontSize="12" fontWeight="700" fill={riskColor(value)}>{value}%</text>
    </svg>
  );
};

const Badge = ({ text }) => {
  const map = {
    "Salary + Career": "#007AFF", "Salary + Equity": "#5856D6",
    "Workload + Salary": "#FF9500", "Management": "#FF3B30",
    "Career": "#34C759", "Salary": "#FF9500", "Internal Equity": "#5856D6",
  };
  const c = map[text] || "#8E8E93";
  return <span style={{ fontSize: 12, fontWeight: 600, color: c, background: c+"18", borderRadius: 6, padding: "3px 10px", whiteSpace: "nowrap" }}>{text}</span>;
};

const Avatar = ({ name, id }) => {
  const hue = (id || "E0").charCodeAt(1) * 37 % 360;
  const initials = name ? name.split(/[\s_]/).filter(Boolean).slice(0,2).map(n=>n[0].toUpperCase()).join("") : "?";
  return <div style={{ width: 46, height: 46, borderRadius: "50%", background: `hsl(${hue},55%,88%)`, display: "flex", alignItems: "center", justifyContent: "center", fontSize: 15, fontWeight: 700, color: `hsl(${hue},55%,38%)`, flexShrink: 0 }}>{initials}</div>;
};

const Spinner = () => (
  <div style={{ display: "flex", justifyContent: "center", alignItems: "center", padding: 80 }}>
    <div style={{ width: 36, height: 36, border: "3px solid #E5E5EA", borderTopColor: "#007AFF", borderRadius: "50%", animation: "spin 0.7s linear infinite" }}/>
    <style>{`@keyframes spin { to { transform: rotate(360deg); } }`}</style>
  </div>
);

const globalStyle = `
  *, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }
  html, body, #root { width: 100%; min-height: 100vh; }
  body { font-family: -apple-system, BlinkMacSystemFont, 'SF Pro Display', 'Helvetica Neue', sans-serif; background: #F2F2F7; color: #1C1C1E; }
`;

export default function App() {
  const [screen, setScreen]       = useState("home");
  const [loginForm, setLoginForm] = useState({ user: "", pass: "" });
  const [loginError, setLoginError] = useState("");
  const [selected, setSelected]   = useState(null);

  // API state
  const [session, setSession]     = useState(null);   // {role, dept, username}
  const [employees, setEmployees] = useState([]);
  const [loading, setLoading]     = useState(false);
  const [loginLoading, setLoginLoading] = useState(false);
  const [dataError, setDataError] = useState("");

  // ---------- helpers ----------
  const signOut = () => {
    setScreen("home");
    setSelected(null);
    setSession(null);
    setEmployees([]);
    setDataError("");
  };

  const fetchEmployees = useCallback(async (role, dept) => {
    setLoading(true);
    setDataError("");
    try {
      const params = new URLSearchParams({ role });
      if (dept) params.append("dept", dept);
      const res = await fetch(`${API_BASE}/api/employees?${params}`);
      if (!res.ok) throw new Error(`Server error ${res.status}`);
      const data = await res.json();
      setEmployees(data.employees || []);
    } catch (err) {
      setDataError(err.message || "Failed to load employee data.");
      setEmployees([]);
    } finally {
      setLoading(false);
    }
  }, []);

  const enterView = (view) => {
    setSelected(null);
    setScreen(view);
    fetchEmployees(view, view === "manager" ? session?.dept : null);
  };

  const login = async () => {
    setLoginError("");
    setLoginLoading(true);
    try {
      const res = await fetch(`${API_BASE}/api/login`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ username: loginForm.user, password: loginForm.pass }),
      });
      if (!res.ok) {
        const err = await res.json().catch(() => ({}));
        setLoginError(err.detail || "Incorrect username or password.");
        return;
      }
      const sess = await res.json();  // {role, dept, username}
      setSession(sess);
      setScreen("portal");
    } catch {
      setLoginError("Cannot reach server. Please wait — the server may be waking up (first load can take ~30s).");
    } finally {
      setLoginLoading(false);
    }
  };

  const toggle = (emp) => setSelected(s => s?.id === emp.id ? null : emp);

  // ── Shared subcomponents ────────────────────────────────────────

  const NavBar = ({ title }) => (
    <div style={{ width: "100%", background: "rgba(255,255,255,0.88)", backdropFilter: "blur(24px)", borderBottom: "1px solid #E5E5EA", height: 56, display: "flex", justifyContent: "space-between", alignItems: "center", padding: "0 5%", position: "sticky", top: 0, zIndex: 100 }}>
      <div style={{ display: "flex", alignItems: "center", gap: 10 }}>
        <button onClick={() => setScreen("portal")} style={{ background: "none", border: "none", color: "#007AFF", fontSize: 13, cursor: "pointer", fontFamily: "inherit", padding: 0, fontWeight: 500 }}>← Portal</button>
        <span style={{ color: "#D1D1D6" }}>/</span>
        <span style={{ fontSize: 14, fontWeight: 700 }}>{title}</span>
        {session?.dept && <span style={{ fontSize: 12, background: "#F2F2F7", borderRadius: 6, padding: "2px 8px", color: "#636366" }}>{session.dept}</span>}
      </div>
      <button onClick={signOut} style={{ background: "none", border: "1.5px solid #E5E5EA", borderRadius: 10, padding: "6px 18px", fontSize: 13, color: "#636366", cursor: "pointer", fontFamily: "inherit" }}>Sign Out</button>
    </div>
  );

  const Card = ({ children, style = {}, onClick }) => (
    <div onClick={onClick}
      onMouseEnter={e => e.currentTarget.style.boxShadow = "0 6px 24px rgba(0,0,0,0.11)"}
      onMouseLeave={e => e.currentTarget.style.boxShadow = "0 1px 4px rgba(0,0,0,0.07)"}
      style={{ background: "#fff", borderRadius: 18, padding: "22px 26px", boxShadow: "0 1px 4px rgba(0,0,0,0.07)", transition: "box-shadow 0.15s", ...style }}>
      {children}
    </div>
  );

  const ErrorBanner = ({ msg }) => (
    <div style={{ background: "#FFF0EE", border: "1px solid #FFD0CA", borderRadius: 12, padding: "14px 18px", color: "#C0392B", fontSize: 14, marginBottom: 20 }}>
      ⚠️ {msg}
    </div>
  );

  // ── HOME ─────────────────────────────────────────────────────────
  if (screen === "home") return (
    <>
      <style>{globalStyle}</style>
      <div style={{ width: "100%", minHeight: "100vh", display: "flex", flexDirection: "column", alignItems: "center", justifyContent: "center", padding: "48px 24px", background: "linear-gradient(160deg,#F2F2F7 0%,#FAFAFA 100%)" }}>
        <div style={{ textAlign: "center", maxWidth: 520 }}>
          <div style={{ display: "inline-block", background: "#007AFF", borderRadius: 12, padding: "8px 18px", fontSize: 11, fontWeight: 700, letterSpacing: 2, color: "#fff", textTransform: "uppercase", marginBottom: 24 }}>RetentionAgent</div>
          <div style={{ fontSize: "clamp(28px, 4vw, 48px)", fontWeight: 700, letterSpacing: -1, color: "#1C1C1E", lineHeight: 1.15 }}>People Intelligence<br/>Platform</div>
          <div style={{ fontSize: "clamp(14px, 1.5vw, 17px)", color: "#8E8E93", marginTop: 14, marginBottom: 40, lineHeight: 1.6 }}>AI-powered retention risk analysis for HR & Management</div>
          <button onClick={() => { setLoginForm({ user:"", pass:"" }); setLoginError(""); setScreen("login"); }}
            style={{ padding: "16px 48px", background: "#007AFF", color: "#fff", border: "none", borderRadius: 14, fontSize: 16, fontWeight: 600, cursor: "pointer", fontFamily: "inherit", boxShadow: "0 4px 18px rgba(0,122,255,0.35)" }}>
            Sign In →
          </button>
        </div>
      </div>
    </>
  );

  // ── PORTAL ───────────────────────────────────────────────────────
  if (screen === "portal") {
    const isHR = session?.role === "hr";
    const views = [
      {
        id: "hr", icon: "👥", label: "HR Dashboard",
        sub: "Full workforce overview",
        desc: ["All employees — every risk tier", "Market & internal salary gaps", "AI compensation adjustment plans"],
        allowed: isHR,
      },
      {
        id: "manager", icon: "📊", label: "Manager View",
        sub: session?.dept ? `${session.dept} team` : "Your team",
        desc: ["Your team's flight risk scores", "AI retention recommendations", "Action timeline guidance"],
        allowed: true,
      },
    ];
    return (
      <>
        <style>{globalStyle}</style>
        <div style={{ width: "100%", minHeight: "100vh" }}>
          <div style={{ width: "100%", background: "rgba(255,255,255,0.88)", backdropFilter: "blur(24px)", borderBottom: "1px solid #E5E5EA", height: 56, display: "flex", justifyContent: "space-between", alignItems: "center", padding: "0 5%", position: "sticky", top: 0, zIndex: 100 }}>
            <span style={{ fontSize: 14, fontWeight: 700 }}>RetentionAgent</span>
            <div style={{ display: "flex", alignItems: "center", gap: 14 }}>
              <span style={{ fontSize: 13, color: "#8E8E93" }}>{session?.username}</span>
              <button onClick={signOut} style={{ background: "none", border: "1.5px solid #E5E5EA", borderRadius: 10, padding: "6px 18px", fontSize: 13, color: "#636366", cursor: "pointer", fontFamily: "inherit" }}>Sign Out</button>
            </div>
          </div>

          <div style={{ width: "100%", maxWidth: 760, margin: "0 auto", padding: "64px 24px", display: "flex", flexDirection: "column", alignItems: "center" }}>
            <div style={{ textAlign: "center", marginBottom: 48 }}>
              <div style={{ fontSize: "clamp(22px, 3vw, 32px)", fontWeight: 700, letterSpacing: -0.5 }}>Where would you like to go?</div>
              <div style={{ fontSize: 15, color: "#8E8E93", marginTop: 8 }}>Select a view to continue</div>
            </div>

            <div style={{ display: "flex", gap: 24, flexWrap: "wrap", justifyContent: "center", width: "100%" }}>
              {views.map(({ id, icon, label, sub, desc, allowed }) => (
                <div key={id}
                  onClick={() => allowed && enterView(id)}
                  onMouseEnter={e => { if (allowed) { e.currentTarget.style.transform = "translateY(-4px)"; e.currentTarget.style.boxShadow = "0 12px 36px rgba(0,0,0,0.13)"; }}}
                  onMouseLeave={e => { e.currentTarget.style.transform = ""; e.currentTarget.style.boxShadow = allowed ? "0 2px 20px rgba(0,0,0,0.08)" : "0 1px 8px rgba(0,0,0,0.05)"; }}
                  style={{ background: "#fff", borderRadius: 24, padding: "36px 32px", flex: "1 1 260px", maxWidth: 320, boxShadow: "0 2px 20px rgba(0,0,0,0.08)", transition: "transform 0.18s, box-shadow 0.18s", cursor: allowed ? "pointer" : "default", opacity: allowed ? 1 : 0.42, position: "relative" }}>
                  {!allowed && (
                    <div style={{ position: "absolute", top: 14, right: 16, fontSize: 11, fontWeight: 700, color: "#8E8E93", background: "#F2F2F7", borderRadius: 6, padding: "3px 10px", letterSpacing: 0.5 }}>No Access</div>
                  )}
                  <div style={{ fontSize: 38, marginBottom: 16 }}>{icon}</div>
                  <div style={{ fontSize: 22, fontWeight: 700 }}>{label}</div>
                  <div style={{ fontSize: 14, color: "#8E8E93", marginTop: 6, marginBottom: 20, lineHeight: 1.55 }}>{sub}</div>
                  <div style={{ borderTop: "1px solid #F2F2F7", paddingTop: 16, marginBottom: allowed ? 24 : 0 }}>
                    {desc.map(d => <div key={d} style={{ fontSize: 13, color: "#636366", marginBottom: 8, display: "flex", gap: 8 }}><span style={{ color: allowed ? "#34C759" : "#C7C7CC" }}>✓</span>{d}</div>)}
                  </div>
                  {allowed && (
                    <button style={{ width: "100%", padding: "13px 0", background: "#007AFF", color: "#fff", border: "none", borderRadius: 12, fontSize: 15, fontWeight: 600, cursor: "pointer", fontFamily: "inherit", pointerEvents: "none" }}>
                      Enter →
                    </button>
                  )}
                </div>
              ))}
            </div>
          </div>
        </div>
      </>
    );
  }

  // ── LOGIN ────────────────────────────────────────────────────────
  if (screen === "login") return (
    <>
      <style>{globalStyle}</style>
      <div style={{ width: "100%", minHeight: "100vh", display: "flex", alignItems: "center", justifyContent: "center", background: "linear-gradient(160deg,#F2F2F7 0%,#FAFAFA 100%)", padding: 24 }}>
        <div style={{ background: "#fff", borderRadius: 28, padding: "44px 48px", width: "100%", maxWidth: 400, boxShadow: "0 4px 40px rgba(0,0,0,0.11)" }}>
          <button onClick={() => setScreen("home")} style={{ background: "none", border: "none", color: "#007AFF", fontSize: 14, cursor: "pointer", padding: 0, marginBottom: 28, fontFamily: "inherit" }}>← Back</button>
          <div style={{ fontSize: 11, fontWeight: 700, letterSpacing: 1.8, color: "#8E8E93", textTransform: "uppercase", marginBottom: 8 }}>RetentionAgent</div>
          <div style={{ fontSize: 30, fontWeight: 700, marginBottom: 32 }}>Sign In</div>
          {[{ f: "user", label: "Username", type: "text" }, { f: "pass", label: "Password", type: "password" }].map(({ f, label, type }) => (
            <div key={f} style={{ marginBottom: 16 }}>
              <div style={{ fontSize: 13, fontWeight: 600, color: "#636366", marginBottom: 7 }}>{label}</div>
              <input type={type} value={loginForm[f]} onChange={e => setLoginForm(p => ({ ...p, [f]: e.target.value }))} onKeyDown={e => e.key === "Enter" && !loginLoading && login()}
                onFocus={e => e.target.style.borderColor = "#007AFF"} onBlur={e => e.target.style.borderColor = "#E5E5EA"}
                style={{ width: "100%", padding: "13px 16px", borderRadius: 12, border: "1.5px solid #E5E5EA", fontSize: 15, outline: "none", fontFamily: "inherit", color: "#1C1C1E", transition: "border-color 0.15s" }}/>
            </div>
          ))}
          {loginError && <div style={{ fontSize: 13, color: "#FF3B30", marginBottom: 12, padding: "10px 14px", background: "#FFF0EE", borderRadius: 10 }}>{loginError}</div>}
          <button onClick={login} disabled={loginLoading}
            style={{ width: "100%", padding: "14px 0", background: loginLoading ? "#63AEFF" : "#007AFF", color: "#fff", border: "none", borderRadius: 13, fontSize: 16, fontWeight: 600, cursor: loginLoading ? "not-allowed" : "pointer", marginTop: 8, fontFamily: "inherit", display: "flex", alignItems: "center", justifyContent: "center", gap: 10, transition: "background 0.2s" }}>
            {loginLoading && (
              <div style={{ width: 16, height: 16, border: "2px solid rgba(255,255,255,0.4)", borderTopColor: "#fff", borderRadius: "50%", animation: "spin 0.7s linear infinite", flexShrink: 0 }}/>
            )}
            {loginLoading ? "Signing in..." : "Continue"}
          </button>
        </div>
      </div>
    </>
  );

  // ── HR DASHBOARD ─────────────────────────────────────────────────
  if (screen === "hr") {
    const criticalCount = employees.filter(e => e.attrition >= 80).length;
    const gaps = employees.map(e => e.salaryGap).filter(v => v !== 0);
    const avgGap = gaps.length ? (gaps.reduce((s, v) => s + v, 0) / gaps.length).toFixed(1) : "—";

    return (
      <>
        <style>{globalStyle}</style>
        <div style={{ width: "100%", minHeight: "100vh" }}>
          <NavBar title="HR Dashboard" />
          <div style={{ width: "100%", maxWidth: 1200, margin: "0 auto", padding: "40px 5%" }}>
            <div style={{ marginBottom: 32 }}>
              <div style={{ fontSize: "clamp(24px, 2.5vw, 32px)", fontWeight: 700, letterSpacing: -0.5 }}>At-Risk Employees</div>
              <div style={{ fontSize: 15, color: "#8E8E93", marginTop: 6 }}>Sorted by attrition probability · Powered by RetentionAgent ML pipeline</div>
            </div>

            {dataError && <ErrorBanner msg={dataError} />}

            {!loading && (
              <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(200px, 1fr))", gap: 16, marginBottom: 28 }}>
                {[
                  { label: "Monitored",      value: employees.length, unit: "employees",             accent: "#007AFF" },
                  { label: "Critical Risk",  value: criticalCount,    unit: "require immediate action", accent: "#FF3B30" },
                  { label: "Avg Market Gap", value: avgGap + "%",     unit: "below market median",   accent: "#FF9500" },
                ].map(({ label, value, unit, accent }) => (
                  <div key={label} style={{ background: "#fff", borderRadius: 18, padding: "22px 24px", boxShadow: "0 1px 4px rgba(0,0,0,0.07)" }}>
                    <div style={{ fontSize: 11, fontWeight: 700, color: "#8E8E93", textTransform: "uppercase", letterSpacing: 1 }}>{label}</div>
                    <div style={{ fontSize: "clamp(28px, 3vw, 40px)", fontWeight: 700, color: accent, marginTop: 8, letterSpacing: -1 }}>{value}</div>
                    <div style={{ fontSize: 13, color: "#C7C7CC", marginTop: 4 }}>{unit}</div>
                  </div>
                ))}
              </div>
            )}

            {loading ? <Spinner /> : (
              <div style={{ display: "flex", flexDirection: "column", gap: 12 }}>
                {employees.map(emp => (
                  <Card key={emp.id} onClick={() => toggle(emp)} style={{ borderLeft: `5px solid ${riskColor(emp.attrition)}`, cursor: "pointer" }}>
                    <div style={{ display: "flex", alignItems: "center", gap: 18, flexWrap: "wrap" }}>
                      <GaugeRing value={emp.attrition} size={64}/>
                      <div style={{ flex: 1, minWidth: 200 }}>
                        <div style={{ display: "flex", alignItems: "center", gap: 10, flexWrap: "wrap" }}>
                          <span style={{ fontWeight: 700, fontSize: 16 }}>{emp.name}</span>
                          <span style={{ fontSize: 13, color: "#8E8E93" }}>{emp.role} · {emp.dept}</span>
                          <Badge text={emp.flag}/>
                          <span style={{ fontSize: 12, fontWeight: 700, color: riskColor(emp.attrition), background: riskColor(emp.attrition)+"15", padding: "2px 10px", borderRadius: 6 }}>{riskLabel(emp.attrition)}</span>
                        </div>
                        <div style={{ display: "flex", gap: 24, marginTop: 8, flexWrap: "wrap" }}>
                          {[
                            ["Market Gap",  emp.salaryGap  + "%", "#FF3B30"],
                            ["Internal Gap",emp.internalGap + "%", "#FF9500"],
                            ["Urgency",     emp.urgency,          "#1C1C1E"],
                          ].map(([l,v,c]) => (
                            <div key={l} style={{ fontSize: 13, color: "#8E8E93" }}>{l}: <span style={{ color: c, fontWeight: 600 }}>{v}</span></div>
                          ))}
                        </div>
                      </div>
                      <span style={{ fontSize: 20, color: "#C7C7CC", transition: "transform 0.2s", transform: selected?.id === emp.id ? "rotate(90deg)" : "" }}>›</span>
                    </div>

                    {selected?.id === emp.id && (
                      <div style={{ marginTop: 22, paddingTop: 22, borderTop: "1px solid #F2F2F7", display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(240px, 1fr))", gap: 14 }}>
                        <div style={{ background: "#F9F9FB", borderRadius: 14, padding: "18px 20px" }}>
                          <div style={{ fontSize: 11, fontWeight: 700, color: "#8E8E93", textTransform: "uppercase", letterSpacing: 1, marginBottom: 10 }}>💰 Compensation Plan</div>
                          <div style={{ fontSize: 14, lineHeight: 1.65 }}>{emp.plan}</div>
                          {emp.currentSalary && (
                            <div style={{ marginTop: 10, display: "flex", gap: 16, flexWrap: "wrap" }}>
                              <div style={{ fontSize: 13, color: "#8E8E93" }}>Current: <strong>${emp.currentSalary?.toLocaleString()}</strong></div>
                              <div style={{ fontSize: 13, color: "#8E8E93" }}>Market: <strong>${emp.marketMedian?.toLocaleString()}</strong></div>
                            </div>
                          )}
                        </div>
                        <div style={{ background: "#F9F9FB", borderRadius: 14, padding: "18px 20px" }}>
                          <div style={{ fontSize: 11, fontWeight: 700, color: "#8E8E93", textTransform: "uppercase", letterSpacing: 1, marginBottom: 10 }}>📋 Risk Details</div>
                          {[
                            ["Department",     emp.dept],
                            ["Employee ID",    emp.id],
                            ["Risk Bucket",    emp.riskBucket],
                            ["Attrition Score",emp.attrition + "%"],
                          ].map(([k, v]) => (
                            <div key={k} style={{ fontSize: 14, marginBottom: 4 }}>{k}: <strong style={k === "Attrition Score" ? { color: riskColor(emp.attrition) } : {}}>{v}</strong></div>
                          ))}
                        </div>
                        {(emp.keyConcerns?.length > 0 || emp.rootCauses?.length > 0) && (
                          <div style={{ background: "#F9F9FB", borderRadius: 14, padding: "18px 20px" }}>
                            <div style={{ fontSize: 11, fontWeight: 700, color: "#8E8E93", textTransform: "uppercase", letterSpacing: 1, marginBottom: 10 }}>🔍 Key Concerns</div>
                            {(emp.keyConcerns?.length > 0 ? emp.keyConcerns : emp.rootCauses).map((c, i) => (
                              <div key={i} style={{ fontSize: 13, color: "#636366", marginBottom: 6, display: "flex", gap: 8 }}>
                                <span style={{ color: "#FF9500" }}>•</span>{c}
                              </div>
                            ))}
                          </div>
                        )}
                      </div>
                    )}
                  </Card>
                ))}
              </div>
            )}
          </div>
        </div>
      </>
    );
  }

  // ── MANAGER DASHBOARD ─────────────────────────────────────────────
  if (screen === "manager") return (
    <>
      <style>{globalStyle}</style>
      <div style={{ width: "100%", minHeight: "100vh" }}>
        <NavBar title="Manager View" />
        <div style={{ width: "100%", maxWidth: 1200, margin: "0 auto", padding: "40px 5%" }}>
          <div style={{ marginBottom: 32 }}>
            <div style={{ fontSize: "clamp(24px, 2.5vw, 32px)", fontWeight: 700, letterSpacing: -0.5 }}>Team Retention Board</div>
            <div style={{ fontSize: 15, color: "#8E8E93", marginTop: 6 }}>High &amp; Mid risk employees only · AI-generated retention recommendations</div>
          </div>

          {dataError && <ErrorBanner msg={dataError} />}

          {loading ? <Spinner /> : (
            <div style={{ display: "flex", flexDirection: "column", gap: 14 }}>
              {employees.map(emp => (
                <Card key={emp.id} onClick={() => toggle(emp)} style={{ cursor: "pointer" }}>
                  <div style={{ display: "flex", alignItems: "center", gap: 16, flexWrap: "wrap" }}>
                    <Avatar name={emp.name} id={emp.id}/>
                    <div style={{ flex: 1, minWidth: 200 }}>
                      <div style={{ display: "flex", alignItems: "center", gap: 10, flexWrap: "wrap" }}>
                        <span style={{ fontWeight: 700, fontSize: 16 }}>{emp.name}</span>
                        <span style={{ fontSize: 13, color: "#8E8E93" }}>{emp.role} · {emp.city || emp.dept}</span>
                        <Badge text={emp.flag}/>
                      </div>
                      <div style={{ display: "flex", alignItems: "center", gap: 10, marginTop: 10, flexWrap: "wrap" }}>
                        <div style={{ height: 7, width: 160, background: "#F2F2F7", borderRadius: 4, overflow: "hidden" }}>
                          <div style={{ height: "100%", width: `${emp.attrition}%`, background: riskColor(emp.attrition), borderRadius: 4 }}/>
                        </div>
                        <span style={{ fontSize: 14, fontWeight: 700, color: riskColor(emp.attrition) }}>{emp.attrition}%</span>
                        <span style={{ fontSize: 13, color: "#8E8E93" }}>flight risk</span>
                        <span style={{ fontSize: 12, fontWeight: 600, color: riskColor(emp.attrition), background: riskColor(emp.attrition)+"15", padding: "2px 10px", borderRadius: 6 }}>{riskLabel(emp.attrition)}</span>
                      </div>
                    </div>
                    <span style={{ fontSize: 20, color: "#C7C7CC", transition: "transform 0.2s", transform: selected?.id === emp.id ? "rotate(90deg)" : "" }}>›</span>
                  </div>

                  {selected?.id === emp.id && (
                    <div style={{ marginTop: 22, paddingTop: 22, borderTop: "1px solid #F2F2F7", display: "flex", flexDirection: "column", gap: 12 }}>
                      <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(160px, 1fr))", gap: 12 }}>
                        {[
                          ["Urgency",        emp.urgency,         null],
                          ["Attrition Risk", emp.attrition + "%", riskColor(emp.attrition)],
                          ["Priority",       emp.priority,        emp.priority === "Urgent" ? "#FF3B30" : emp.priority === "High" ? "#FF9500" : "#34C759"],
                        ].map(([l, v, c]) => (
                          <div key={l} style={{ background: "#F9F9FB", borderRadius: 12, padding: "14px 16px" }}>
                            <div style={{ fontSize: 11, color: "#8E8E93", fontWeight: 700, textTransform: "uppercase", letterSpacing: 1 }}>{l}</div>
                            <div style={{ fontSize: 15, fontWeight: 700, marginTop: 6, color: c || "#1C1C1E" }}>{v}</div>
                          </div>
                        ))}
                      </div>

                      {emp.plan && (
                        <div style={{ background: "linear-gradient(135deg,#EEF5FF,#E5EFFF)", borderRadius: 14, padding: "18px 20px", border: "1px solid #C8DEFF" }}>
                          <div style={{ fontSize: 11, fontWeight: 700, color: "#007AFF", textTransform: "uppercase", letterSpacing: 1.2, marginBottom: 10 }}>🤖 AI Recommendation — Action for Manager</div>
                          <div style={{ fontSize: 14, lineHeight: 1.7 }}>{emp.plan}</div>
                        </div>
                      )}

                      {emp.keyConcerns?.length > 0 && (
                        <div style={{ background: "#F9F9FB", borderRadius: 14, padding: "16px 20px" }}>
                          <div style={{ fontSize: 11, color: "#8E8E93", fontWeight: 700, textTransform: "uppercase", letterSpacing: 1, marginBottom: 8 }}>🔍 Key Concerns</div>
                          {emp.keyConcerns.map((c, i) => (
                            <div key={i} style={{ fontSize: 13, color: "#636366", marginBottom: 5, display: "flex", gap: 8 }}>
                              <span style={{ color: "#FF9500" }}>•</span>{c}
                            </div>
                          ))}
                        </div>
                      )}

                      {emp.managerNote && (
                        <div style={{ background: "#F9F9FB", borderRadius: 14, padding: "14px 18px" }}>
                          <div style={{ fontSize: 11, color: "#8E8E93", fontWeight: 700, textTransform: "uppercase", letterSpacing: 1, marginBottom: 6 }}>📋 Risk Context</div>
                          <div style={{ fontSize: 13, color: "#636366", lineHeight: 1.65 }}>{emp.managerNote}</div>
                        </div>
                      )}

                      <div style={{ display: "flex", alignItems: "center", gap: 10 }}>
                        <div style={{ width: 9, height: 9, borderRadius: "50%", background: riskColor(emp.attrition), flexShrink: 0 }}/>
                        <span style={{ fontSize: 14, color: "#636366" }}>
                          {emp.urgency === "Immediate"
                            ? "⚠️ Action required within 1 week"
                            : emp.urgency === "Near-term"
                            ? "Follow up within 2 weeks"
                            : "Monitor in next review cycle"}
                        </span>
                      </div>
                    </div>
                  )}
                </Card>
              ))}
              {employees.length === 0 && !dataError && (
                <div style={{ textAlign: "center", padding: 60, color: "#8E8E93" }}>No at-risk employees found for your team.</div>
              )}
            </div>
          )}
        </div>
      </div>
    </>
  );
}