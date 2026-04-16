import { useState, useRef, useEffect, useCallback } from "react";

const API = "http://localhost:8002";

const ALG = {
  svm: { name: "SVM", full: "Support Vector Machine", color: "#A78BFA", icon: "⚡" },
  rf: { name: "RF", full: "Random Forest", color: "#34D399", icon: "🌲" },
  xgb: { name: "XGB", full: "Extreme Gradient Boosting", color: "#FFE66D", icon: "🚀" },
  cnn: { name: "CNN", full: "Convolutional Neural Network", color: "#FF6B6B", icon: "🧠" },
  lstm: { name: "LSTM", full: "BiLSTM + Attention", color: "#4ECDC4", icon: "🔄" },
};

const CLS = {
  gunshot: { icon: "🔫", color: "#FF3B30", label: "Gunshot" },
  rifle_fire: { icon: "🎯", color: "#FF9500", label: "Rifle Fire" },
  vehicle: { icon: "🚛", color: "#FFCC00", label: "Vehicle" },
  aircraft: { icon: "✈️", color: "#34C759", label: "Aircraft" },
  comms_signal: { icon: "📡", color: "#007AFF", label: "Comms Signal" },
  explosion: { icon: "💥", color: "#FF2D55", label: "Explosion" },
  ambient: { icon: "🔇", color: "#8E8E93", label: "Ambient" },
};
const getC = n => CLS[n?.toLowerCase()] || { icon: "🎵", color: "#8E8E93", label: n || "Unknown" };

// ─── THREAT LEVEL ─────────────────────────────────────────────────────────────
function getThreatLevel(conf1, conf2, primaryClass, secondaryClass) {
  const highThreatClasses = ["gunshot", "rifle_fire", "aircraft", "explosion"];
  const hasHighThreatClass = highThreatClasses.includes(primaryClass) || (secondaryClass && highThreatClasses.includes(secondaryClass) && conf2 > 0.25);

  if (conf1 > 0.75) return { level: "CRITICAL", color: "#FF3B30", bg: "rgba(255,59,48,0.15)" };
  if (hasHighThreatClass) return { level: "HIGH", color: "#FF9500", bg: "rgba(255,149,0,0.15)" };
  if (conf1 > 0.55) return { level: "HIGH", color: "#FF9500", bg: "rgba(255,149,0,0.15)" };
  if (conf1 > 0.35) return { level: "MODERATE", color: "#FFE66D", bg: "rgba(255,230,109,0.15)" };
  return { level: "LOW", color: "#34C759", bg: "rgba(52,199,89,0.15)" };
}

// ─── GAUGE ────────────────────────────────────────────────────────────────────
function Gauge({ value, color, size = 80 }) {
  const r = 32, cv = 2 * Math.PI * r, p = Math.min(Math.max(value, 0), 1);
  return (
    <svg width={size} height={size} viewBox="0 0 72 72">
      <circle cx="36" cy="36" r={r} fill="none" stroke="rgba(255,255,255,0.06)"
        strokeWidth="7" strokeDasharray={`${cv * .75} ${cv * .25}`} strokeLinecap="round"
        transform="rotate(135 36 36)" />
      <circle cx="36" cy="36" r={r} fill="none" stroke={color} strokeWidth="7"
        strokeDasharray={`${p * cv * .75} ${cv - p * cv * .75 + cv * .25}`} strokeLinecap="round"
        transform="rotate(135 36 36)" style={{ transition: "stroke-dasharray 1s ease" }} />
      <text x="36" y="40" textAnchor="middle" fill="#fff" fontSize="12"
        fontWeight="700" fontFamily="monospace">{Math.round(p * 100)}%</text>
    </svg>
  );
}

// ─── MATRIX ───────────────────────────────────────────────────────────────────
function Matrix({ matrix, classNames }) {
  if (!matrix?.length || !classNames?.length) return null;
  const mx = Math.max(...matrix.flat(), 1);
  return (
    <div style={{ overflowX: "auto" }}>
      <div style={{ display: "inline-block", fontSize: 8, fontFamily: "monospace" }}>
        <div style={{ display: "flex", marginBottom: 2, marginLeft: 60 }}>
          {classNames.map(n => <div key={n} style={{ width: 34, textAlign: "center", color: "#444" }}>{getC(n).icon}</div>)}
        </div>
        {matrix.map((row, i) => (
          <div key={i} style={{ display: "flex", alignItems: "center", marginBottom: 1 }}>
            <div style={{ width: 58, textAlign: "right", paddingRight: 5, color: "#444", fontSize: 8 }}>
              {getC(classNames[i]).icon} {classNames[i]?.slice(0, 6)}
            </div>
            {row.map((v, j) => {
              const p = v / mx;
              return <div key={j} style={{
                width: 34, height: 24, display: "flex", alignItems: "center",
                justifyContent: "center",
                background: i === j ? `rgba(0,255,136,${.08 + p * .7})` : `rgba(255,59,48,${p * .55})`,
                border: "1px solid rgba(255,255,255,0.03)", color: p > .2 ? "#fff" : "#333", fontSize: 8
              }}>{v}</div>;
            })}
          </div>
        ))}
        <div style={{ color: "#222", marginTop: 3, marginLeft: 60, fontSize: 7 }}>← Predicted | Actual ↓</div>
      </div>
    </div>
  );
}

// ─── EPOCH CHART ──────────────────────────────────────────────────────────────
function EpochChart({ history, color, label }) {
  if (!history?.val_acc?.length) return null;
  const vals = history.val_acc, mx = Math.max(...vals, 0.01);
  const W = 280, H = 80, pad = 8;
  const pts = vals.map((v, i) => {
    const x = pad + (i / (vals.length - 1 || 1)) * (W - pad * 2);
    const y = H - pad - (v / mx) * (H - pad * 2);
    return `${x},${y}`;
  }).join(" ");
  return (
    <div>
      <div style={{ fontSize: 9, color: "#555", marginBottom: 4 }}>{label} VALIDATION ACCURACY</div>
      <svg width={W} height={H} style={{ background: "rgba(255,255,255,0.02)", borderRadius: 6 }}>
        <polyline points={pts} fill="none" stroke={color} strokeWidth="1.5" strokeLinecap="round" />
        {vals.map((v, i) => {
          const x = pad + (i / (vals.length - 1 || 1)) * (W - pad * 2);
          const y = H - pad - (v / mx) * (H - pad * 2);
          return <circle key={i} cx={x} cy={y} r="2" fill={color} />;
        })}
      </svg>
      <div style={{ display: "flex", justifyContent: "space-between", fontSize: 8, color: "#444", marginTop: 2 }}>
        <span>Epoch 1</span>
        <span style={{ color }}>{(vals[vals.length - 1] * 100).toFixed(1)}% final</span>
        <span>Epoch {vals.length}</span>
      </div>
    </div>
  );
}

// ─── WAVEFORM ─────────────────────────────────────────────────────────────────
function Waveform({ audioURL }) {
  const canvasRef = useRef(null);
  const [decoded, setDecoded] = useState(null);
  useEffect(() => {
    if (!audioURL) return;
    const ctx = new (window.AudioContext || window.webkitAudioContext)();
    fetch(audioURL).then(r => r.arrayBuffer()).then(b => ctx.decodeAudioData(b))
      .then(ab => setDecoded(ab.getChannelData(0))).catch(() => { });
  }, [audioURL]);
  useEffect(() => {
    if (!decoded || !canvasRef.current) return;
    const canvas = canvasRef.current, ctx = canvas.getContext("2d");
    const W = canvas.width, H = canvas.height;
    ctx.clearRect(0, 0, W, H);
    const step = Math.ceil(decoded.length / W);
    ctx.beginPath(); ctx.strokeStyle = "#00FF88"; ctx.lineWidth = 1;
    for (let i = 0; i < W; i++) {
      let mn = 1, mx = -1;
      for (let j = 0; j < step; j++) { const v = decoded[i * step + j] || 0; if (v < mn) mn = v; if (v > mx) mx = v; }
      ctx.moveTo(i, (1 + mn) / 2 * H); ctx.lineTo(i, (1 + mx) / 2 * H);
    }
    ctx.stroke();
  }, [decoded]);
  return <canvas ref={canvasRef} width={520} height={70}
    style={{
      width: "100%", height: 70, borderRadius: 8,
      background: "rgba(0,255,136,0.03)", border: "1px solid rgba(0,255,136,0.1)"
    }} />;
}

// ─── LOG ──────────────────────────────────────────────────────────────────────
function TLog({ lines }) {
  const ref = useRef(null);
  useEffect(() => { if (ref.current) ref.current.scrollTop = ref.current.scrollHeight; }, [lines]);
  return (
    <div ref={ref} style={{
      background: "#030608", border: "1px solid rgba(0,255,136,0.1)",
      borderRadius: 10, padding: "10px 14px", fontFamily: "monospace", fontSize: 10,
      height: 160, overflowY: "auto", lineHeight: 1.7
    }}>
      {lines.length === 0 ? <span style={{ color: "#1a1a1a" }}>Waiting…</span>
        : lines.map((l, i) => (
          <div key={i} style={{
            color: l.includes("✅") ? "#00FF88" : l.includes("❌") ? "#FF3B30"
              : l.includes("⚙️") ? "#FFE66D" : l.includes("🏆") ? "#FF9500" : "#4ade80"
          }}>{l}</div>
        ))}
    </div>
  );
}

// ─── TOP-2 DETECTION CARD ─────────────────────────────────────────────────────
function DualThreatCard({ names, preds }) {
  // Aggregate probabilities across all models
  const n = names.length;
  const avgProba = new Array(n).fill(0);
  const algoList = Object.values(preds);
  algoList.forEach(p => { p.proba.forEach((v, i) => { avgProba[i] += v; }); });
  avgProba.forEach((_, i) => { avgProba[i] /= algoList.length; });

  // Get top 2
  const sorted = avgProba.map((v, i) => ({ idx: i, conf: v })).sort((a, b) => b.conf - a.conf);
  const top1 = sorted[0], top2 = sorted[1];
  const c1 = getC(names[top1.idx]), c2 = getC(names[top2.idx]);
  const isDual = top2.conf > 0.20; // show dual only if 2nd class > 20%
  const threat = getThreatLevel(
    top1.conf,
    top2.conf,
    names[top1.idx]?.toLowerCase(),
    names[top2.idx]?.toLowerCase()
  );

  // Consensus from individual model votes
  const votes = {};
  algoList.forEach(p => { votes[p.pred] = (votes[p.pred] || 0) + 1; });
  const topVoteIdx = parseInt(Object.entries(votes).sort((a, b) => b[1] - a[1])[0][0]);

  return (
    <div>
      {/* Threat level banner */}
      <div style={{
        background: threat.bg, border: `1.5px solid ${threat.color}`,
        borderRadius: 12, padding: "10px 18px", marginBottom: 16,
        display: "flex", alignItems: "center", justifyContent: "space-between"
      }}>
        <div style={{ fontSize: 10, color: threat.color, fontWeight: 700, letterSpacing: 2 }}>
          ⚠️ THREAT LEVEL: {threat.level}
        </div>
        <div style={{ fontSize: 9, color: "#555" }}>
          {Object.keys(preds).length}/5 models analyzed
        </div>
      </div>

      {/* Primary + Secondary detection */}
      <div style={{ display: "grid", gridTemplateColumns: isDual ? "1fr 1fr" : "1fr", gap: 14, marginBottom: 16 }}>

        {/* PRIMARY */}
        <div style={{
          background: `linear-gradient(135deg,${c1.color}12,rgba(0,0,0,0.6))`,
          border: `2px solid ${c1.color}`, borderRadius: 14, padding: 20, textAlign: "center"
        }}>
          <div style={{ fontSize: 8, color: "#555", letterSpacing: 3, marginBottom: 8 }}>
            PRIMARY DETECTION
          </div>
          <div style={{ fontSize: 46, marginBottom: 6 }}>{c1.icon}</div>
          <div style={{ fontSize: 20, fontWeight: 700, color: c1.color, letterSpacing: 1 }}>
            {(c1.label || names[top1.idx]).toUpperCase()}
          </div>
          <div style={{ display: "flex", justifyContent: "center", marginTop: 10 }}>
            <Gauge value={top1.conf} color={c1.color} size={90} />
          </div>
          <div style={{ fontSize: 9, color: "#555", marginTop: 4 }}>ENSEMBLE CONFIDENCE</div>
        </div>

        {/* SECONDARY — only if prominent */}
        {isDual && (
          <div style={{
            background: `linear-gradient(135deg,${c2.color}08,rgba(0,0,0,0.6))`,
            border: `1.5px solid ${c2.color}88`, borderRadius: 14, padding: 20, textAlign: "center",
            position: "relative"
          }}>
            <div style={{
              position: "absolute", top: 10, right: 10,
              background: "rgba(255,149,0,0.2)", border: "1px solid #FF9500",
              borderRadius: 99, padding: "2px 8px", fontSize: 7, color: "#FF9500"
            }}>
              ALSO DETECTED
            </div>
            <div style={{ fontSize: 8, color: "#555", letterSpacing: 3, marginBottom: 8 }}>
              SECONDARY DETECTION
            </div>
            <div style={{ fontSize: 46, marginBottom: 6 }}>{c2.icon}</div>
            <div style={{ fontSize: 20, fontWeight: 700, color: c2.color, letterSpacing: 1 }}>
              {(c2.label || names[top2.idx]).toUpperCase()}
            </div>
            <div style={{ display: "flex", justifyContent: "center", marginTop: 10 }}>
              <Gauge value={top2.conf} color={c2.color} size={90} />
            </div>
            <div style={{ fontSize: 9, color: "#555", marginTop: 4 }}>ENSEMBLE CONFIDENCE</div>
          </div>
        )}
      </div>

      {/* All class probability bars */}
      <div style={{
        background: "rgba(255,255,255,0.02)", border: "1px solid rgba(255,255,255,0.06)",
        borderRadius: 12, padding: "14px 16px", marginBottom: 16
      }}>
        <div style={{ fontSize: 9, color: "#555", letterSpacing: 2, marginBottom: 12 }}>
          ALL CLASS PROBABILITIES (ENSEMBLE AVERAGE)
        </div>
        {names.map((n, i) => {
          const c = getC(n), v = avgProba[i];
          const isTop = i === top1.idx || (isDual && i === top2.idx);
          return (
            <div key={i} style={{ marginBottom: 8 }}>
              <div style={{ display: "flex", justifyContent: "space-between", fontSize: 10, marginBottom: 3 }}>
                <span style={{ color: isTop ? c.color : "#666" }}>{c.icon} {c.label || n}</span>
                <span style={{ color: isTop ? c.color : "#444", fontFamily: "monospace" }}>
                  {(v * 100).toFixed(1)}%
                  {i === top1.idx && " ← PRIMARY"}
                  {isDual && i === top2.idx && " ← SECONDARY"}
                </span>
              </div>
              <div style={{ height: isTop ? 7 : 4, background: "rgba(255,255,255,0.04)", borderRadius: 99 }}>
                <div style={{
                  height: "100%", width: `${v * 100}%`,
                  background: isTop ? c.color : `${c.color}44`, borderRadius: 99,
                  transition: "width 1s ease"
                }} />
              </div>
            </div>
          );
        })}
      </div>

      {/* Per model breakdown */}
      <div style={{ fontSize: 9, color: "#555", letterSpacing: 2, marginBottom: 10 }}>
        PER-MODEL INDIVIDUAL RESULTS
      </div>
      <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 10 }}>
        {Object.entries(preds).map(([id, res]) => {
          const am = ALG[id], cls = names[res.pred] || "Unknown", cm = getC(cls);
          const conf = Math.max(...res.proba);
          // Get this model's top 2
          const modelSorted = res.proba.map((v, i) => ({ idx: i, v })).sort((a, b) => b.v - a.v);
          const m2 = modelSorted[1];
          const showM2 = m2.v > 0.18;
          return (
            <div key={id} style={{
              background: "rgba(255,255,255,0.02)",
              border: `1px solid ${res.pred === topVoteIdx ? am.color + "55" : "rgba(255,255,255,0.06)"}`,
              borderRadius: 10, padding: "12px 14px"
            }}>
              <div style={{ display: "flex", justifyContent: "space-between", alignItems: "flex-start", marginBottom: 8 }}>
                <div>
                  <div style={{ fontSize: 11, fontWeight: 700, color: am.color }}>{am.icon} {am.name}</div>
                  <div style={{ fontSize: 8, color: "#333", marginTop: 1 }}>{am.full}</div>
                </div>
                <Gauge value={conf} color={am.color} size={55} />
              </div>

              {/* Model's top detection */}
              <div style={{ display: "flex", gap: 6, marginBottom: 8 }}>
                <div style={{
                  flex: 1, display: "flex", alignItems: "center", gap: 6,
                  background: `${cm.color}10`, borderRadius: 7, padding: "5px 8px",
                  border: `1px solid ${cm.color}33`
                }}>
                  <span style={{ fontSize: 14 }}>{cm.icon}</span>
                  <div>
                    <div style={{ fontSize: 9, color: cm.color, fontWeight: 700 }}>{cm.label || cls}</div>
                    <div style={{ fontSize: 7, color: "#444" }}>{(conf * 100).toFixed(1)}%</div>
                  </div>
                </div>
                {showM2 && (() => {
                  const c2m = getC(names[m2.idx]);
                  return (
                    <div style={{
                      flex: 1, display: "flex", alignItems: "center", gap: 6,
                      background: `${c2m.color}08`, borderRadius: 7, padding: "5px 8px",
                      border: `1px solid ${c2m.color}22`
                    }}>
                      <span style={{ fontSize: 14 }}>{c2m.icon}</span>
                      <div>
                        <div style={{ fontSize: 9, color: `${c2m.color}bb`, fontWeight: 700 }}>{c2m.label || names[m2.idx]}</div>
                        <div style={{ fontSize: 7, color: "#444" }}>{(m2.v * 100).toFixed(1)}%</div>
                      </div>
                    </div>
                  );
                })()}
              </div>

              {/* Mini prob bars */}
              {names.map((n, i) => (
                <div key={i} style={{ display: "flex", alignItems: "center", gap: 5, marginBottom: 2 }}>
                  <div style={{ fontSize: 8, width: 12 }}>{getC(n).icon}</div>
                  <div style={{ flex: 1, height: 3, background: "rgba(255,255,255,0.04)", borderRadius: 99 }}>
                    <div style={{
                      height: "100%", width: `${(res.proba[i] || 0) * 100}%`,
                      background: i === res.pred ? am.color : `${am.color}33`, borderRadius: 99
                    }} />
                  </div>
                  <div style={{ width: 26, fontSize: 7, color: "#333", fontFamily: "monospace", textAlign: "right" }}>
                    {((res.proba[i] || 0) * 100).toFixed(0)}%
                  </div>
                </div>
              ))}
            </div>
          );
        })}
      </div>
    </div>
  );
}

// ─── MAIN APP ─────────────────────────────────────────────────────────────────
export default function App() {
  const [tab, setTab] = useState("train");
  const [training, setTraining] = useState(false);
  const [log, setLog] = useState([]);
  const [progress, setProgress] = useState(0);
  const [curModel, setCurModel] = useState("");
  const [results, setResults] = useState(null);
  const [classNames, setClassNames] = useState([]);
  const [modelsReady, setModelsReady] = useState(false);
  const [trainError, setTrainError] = useState("");
  const [selectedAlgo, setSelectedAlgo] = useState("xgb");
  const [datasetStats, setDatasetStats] = useState(null);
  const [epochHistory, setEpochHistory] = useState({});
  const [folderPath, setFolderPath] = useState(
    "C:\\Projects\\ALGORITHM\\threat_detection_system\\data\\audio"
  );
  const [audioFile, setAudioFile] = useState(null);
  const [audioURL, setAudioURL] = useState(null);
  const [predicting, setPredicting] = useState(false);
  const [predictions, setPredictions] = useState(null);
  const audioRef = useRef(null);
  const pollRef = useRef(null);

  const fetchResults = useCallback(async () => {
    try {
      const r = await fetch(`${API}/results`);
      if (r.ok) {
        const d = await r.json();
        setResults(d.results); setClassNames(d.class_names || []);
        setDatasetStats(d.dataset_stats || null);
        setEpochHistory(d.epoch_history || {});
        setProgress(100); setModelsReady(true); return true;
      }
    } catch (_) { }
    return false;
  }, []);

  useEffect(() => {
    const init = async () => {
      try {
        const sr = await fetch(`${API}/status`); const sd = await sr.json();
        if (sd.status === "done") { await fetchResults(); return; }
        const lr = await fetch(`${API}/load`);
        if (lr.ok) { const ld = await lr.json(); if (ld.loaded?.length > 0) await fetchResults(); }
      } catch (_) { }
    };
    init();
  }, [fetchResults]);

  const pollStatus = useCallback(async () => {
    try {
      const r = await fetch(`${API}/status`); const d = await r.json();
      setLog(d.log || []); setProgress(d.progress || 0); setCurModel(d.current_model || "");
      if (d.class_names?.length) setClassNames(d.class_names);
      if (d.dataset_stats) setDatasetStats(d.dataset_stats);
      if (d.epoch_history && Object.keys(d.epoch_history).length > 0) setEpochHistory(d.epoch_history);
      if (d.status === "done") {
        clearInterval(pollRef.current); setTraining(false);
        await fetchResults(); setTab("compare");
      } else if (d.status === "error") {
        clearInterval(pollRef.current); setTraining(false); setTrainError(d.error || "Error");
      }
    } catch (_) { }
  }, [fetchResults]);

  useEffect(() => () => clearInterval(pollRef.current), []);

  const startTraining = async () => {
    if (!folderPath.trim()) return;
    setTraining(true); setTrainError(""); setResults(null);
    setModelsReady(false); setProgress(0); setLog([]);
    try {
      const r = await fetch(`${API}/train/folder?dataset_path=${encodeURIComponent(folderPath)}`, { method: "POST" });
      if (!r.ok) { const e = await r.json(); throw new Error(e.detail); }
      pollRef.current = setInterval(pollStatus, 1500);
    } catch (e) { setTraining(false); setTrainError(e.message); }
  };

  const handleAudio = f => {
    if (!f) return;
    setAudioFile(f); setAudioURL(URL.createObjectURL(f)); setPredictions(null);
  };

  const runPredict = async () => {
    if (!audioFile || !modelsReady) return;
    setPredicting(true);
    const fd = new FormData(); fd.append("file", audioFile);
    try {
      const r = await fetch(`${API}/predict`, { method: "POST", body: fd });
      const d = await r.json(); setPredictions(d); setTab("compare");
    } catch (e) { alert("Failed: " + e.message); }
    setPredicting(false);
  };

  const sortedAlgos = results
    ? Object.entries(results).filter(([, m]) => m?.accuracy != null).sort((a, b) => b[1].accuracy - a[1].accuracy)
    : [];

  const tabs = [
    { id: "train", label: "TRAIN", icon: "⚡" },
    { id: "predict", label: "PREDICT", icon: "🎯" },
    { id: "compare", label: "COMPARE", icon: "📊" },
  ];

  const BG = `radial-gradient(ellipse at 10% 10%,rgba(0,255,136,0.03) 0%,transparent 50%),
    radial-gradient(ellipse at 90% 90%,rgba(0,122,255,0.03) 0%,transparent 50%),
    repeating-linear-gradient(0deg,transparent,transparent 39px,rgba(255,255,255,0.01) 40px),
    repeating-linear-gradient(90deg,transparent,transparent 39px,rgba(255,255,255,0.01) 40px)`;

  return (
    <div style={{
      minHeight: "100vh", background: "#07090E", color: "#E8E8E8",
      fontFamily: "'Space Mono','Courier New',monospace", backgroundImage: BG
    }}>

      {/* HEADER */}
      <div style={{
        borderBottom: "1px solid rgba(0,255,136,0.12)", background: "rgba(0,0,0,0.7)",
        backdropFilter: "blur(20px)", padding: "0 28px", position: "sticky", top: 0, zIndex: 100
      }}>
        <div style={{
          maxWidth: 1280, margin: "0 auto", display: "flex", alignItems: "center",
          justifyContent: "space-between", height: 58
        }}>
          <div style={{ display: "flex", alignItems: "center", gap: 12 }}>
            <div style={{
              width: 30, height: 30, borderRadius: 7,
              background: "linear-gradient(135deg,#00FF88,#007AFF)",
              display: "flex", alignItems: "center", justifyContent: "center", fontSize: 15
            }}>⚡</div>
            <div>
              <div style={{ fontSize: 12, fontWeight: 700, color: "#fff", letterSpacing: 1 }}>
                ACOUSTIC THREAT DETECTION
              </div>
              <div style={{ fontSize: 8, color: "#00FF88", letterSpacing: 3 }}>
                DUAL-CLASS DETECTION · 5 ALGORITHMS · v3.0
              </div>
            </div>
          </div>
          <div style={{ display: "flex", alignItems: "center", gap: 16 }}>
            {modelsReady && (
              <div style={{ fontSize: 9, color: "#00FF88" }}>
                ✅ {Object.keys(results || {}).length} models · {classNames.length} classes
                {datasetStats && ` · ${datasetStats.total} samples`}
              </div>
            )}
            <div style={{ display: "flex", gap: 6, alignItems: "center" }}>
              <div style={{
                width: 6, height: 6, borderRadius: "50%", background: "#00FF88",
                boxShadow: "0 0 6px #00FF88", animation: "pulse 2s infinite"
              }} />
              <span style={{ fontSize: 8, color: "#00FF88", letterSpacing: 2 }}>ONLINE</span>
            </div>
          </div>
        </div>
      </div>

      {/* TABS */}
      <div style={{ borderBottom: "1px solid rgba(255,255,255,0.05)", background: "rgba(0,0,0,0.4)" }}>
        <div style={{ maxWidth: 1280, margin: "0 auto", display: "flex", padding: "0 28px" }}>
          {tabs.map(t => (
            <button key={t.id} onClick={() => setTab(t.id)} style={{
              background: "none", border: "none", cursor: "pointer",
              padding: "14px 22px", fontSize: 9, letterSpacing: 2,
              color: tab === t.id ? "#00FF88" : "#444",
              borderBottom: tab === t.id ? "2px solid #00FF88" : "2px solid transparent",
              fontFamily: "inherit", transition: "all 0.2s"
            }}>{t.icon} {t.label}</button>
          ))}
        </div>
      </div>

      <div style={{ maxWidth: 1280, margin: "0 auto", padding: 22 }}>

        {/* ══ TRAIN ══ */}
        {tab === "train" && (
          <div style={{ maxWidth: 700, margin: "0 auto" }}>
            {modelsReady && (
              <div style={{
                background: "rgba(0,255,136,0.05)", border: "1px solid rgba(0,255,136,0.2)",
                borderRadius: 12, padding: "14px 18px", marginBottom: 16,
                display: "flex", alignItems: "center", justifyContent: "space-between"
              }}>
                <div>
                  <div style={{ fontSize: 11, color: "#00FF88", fontWeight: 700 }}>✅ Models Ready!</div>
                  <div style={{ fontSize: 9, color: "#444", marginTop: 3 }}>
                    {Object.keys(results || {}).length} models · {classNames.length} classes
                    {datasetStats && ` · ${datasetStats.total} samples`}
                  </div>
                </div>
                <div style={{ display: "flex", gap: 8 }}>
                  <button onClick={() => setTab("compare")} style={{
                    background: "rgba(255,255,255,0.06)",
                    border: "1px solid rgba(255,255,255,0.1)", borderRadius: 7, padding: "7px 14px",
                    color: "#fff", fontFamily: "inherit", fontSize: 10, fontWeight: 700, cursor: "pointer"
                  }}>
                    COMPARE →
                  </button>
                  <button onClick={() => setTab("predict")} style={{
                    background: "linear-gradient(135deg,#00FF88,#007AFF)",
                    border: "none", borderRadius: 7, padding: "7px 14px", color: "#000",
                    fontFamily: "inherit", fontSize: 10, fontWeight: 700, cursor: "pointer"
                  }}>
                    PREDICT →
                  </button>
                </div>
              </div>
            )}

            <div style={{
              background: "rgba(255,255,255,0.02)", border: "1px solid rgba(0,122,255,0.2)",
              borderRadius: 12, padding: "16px 18px", marginBottom: 14
            }}>
              <div style={{ fontSize: 9, color: "#007AFF", letterSpacing: 2, marginBottom: 8 }}>📁 DATASET PATH</div>
              <input value={folderPath} onChange={e => setFolderPath(e.target.value)}
                placeholder="C:\path\to\audio\folder"
                style={{
                  width: "100%", background: "rgba(0,0,0,0.4)",
                  border: "1px solid rgba(255,255,255,0.1)", borderRadius: 8,
                  padding: "10px 14px", color: "#00FF88", fontFamily: "monospace",
                  fontSize: 11, outline: "none", marginBottom: 8, boxSizing: "border-box"
                }} />
              <div style={{ fontSize: 8, color: "#333", lineHeight: 1.8, fontFamily: "monospace" }}>
                Needs: train.csv · test.csv (optional) · files/training/*/wav
              </div>
            </div>

            <button onClick={startTraining} disabled={training || !folderPath.trim()} style={{
              width: "100%", padding: 13, border: "none", borderRadius: 10,
              background: training ? "rgba(255,149,0,0.2)" : "linear-gradient(135deg,#007AFF,#00FF88)",
              color: training ? "#FF9500" : "#000", fontFamily: "inherit", fontSize: 11,
              fontWeight: 700, letterSpacing: 2, cursor: training ? "wait" : "pointer", marginBottom: 14
            }}>
              {training ? `⚙️ TRAINING — ${curModel || "preparing"}…` : "⚡ TRAIN ALL 5 MODELS"}
            </button>

            {(training || progress > 0) && (
              <div style={{ marginBottom: 14 }}>
                <div style={{ display: "flex", justifyContent: "space-between", marginBottom: 5, fontSize: 9 }}>
                  <span style={{ color: "#FF9500" }}>{curModel || "Processing"}</span>
                  <span style={{ fontFamily: "monospace" }}>{progress}%</span>
                </div>
                <div style={{ height: 5, background: "rgba(255,255,255,0.05)", borderRadius: 99 }}>
                  <div style={{
                    height: "100%", borderRadius: 99, width: `${progress}%`,
                    background: "linear-gradient(90deg,#00FF88,#007AFF)", transition: "width 0.6s"
                  }} />
                </div>
                <div style={{ display: "flex", gap: 6, marginTop: 9, flexWrap: "wrap" }}>
                  {Object.entries(ALG).map(([id, m]) => {
                    const done = log.some(l => l.includes(m.name) && l.includes("✅"));
                    const active = curModel?.toLowerCase().includes(m.name.toLowerCase());
                    return (
                      <div key={id} style={{
                        padding: "3px 9px", borderRadius: 99, fontSize: 9,
                        background: done ? `${m.color}18` : active ? "rgba(255,149,0,0.12)" : "rgba(255,255,255,0.03)",
                        border: `1px solid ${done ? m.color : active ? "#FF9500" : "rgba(255,255,255,0.06)"}`,
                        color: done ? m.color : active ? "#FF9500" : "#333"
                      }}>
                        {done ? "✅" : active ? "⚙️" : "⏳"} {m.name}
                      </div>
                    );
                  })}
                </div>
              </div>
            )}

            {(training || log.length > 0) && <TLog lines={log} />}

            {datasetStats && (
              <div style={{
                background: "rgba(255,255,255,0.02)", border: "1px solid rgba(255,255,255,0.06)",
                borderRadius: 10, padding: "12px 14px", marginTop: 12
              }}>
                <div style={{ fontSize: 9, color: "#555", letterSpacing: 2, marginBottom: 10 }}>DATASET</div>
                <div style={{ display: "flex", gap: 18, marginBottom: 10 }}>
                  {[[datasetStats.total, "SAMPLES", "#00FF88"], [classNames.length, "CLASSES", "#007AFF"],
                  [datasetStats.feature_dim, "FEATURES", "#FFE66D"]].map(([v, l, c]) => (
                    <div key={l} style={{ textAlign: "center" }}>
                      <div style={{ fontSize: 16, fontWeight: 700, color: c, fontFamily: "monospace" }}>{v}</div>
                      <div style={{ fontSize: 7, color: "#444" }}>{l}</div>
                    </div>
                  ))}
                </div>
                {datasetStats.classes && Object.entries(datasetStats.classes).map(([cls, count]) => {
                  const c = getC(cls), mx = Math.max(...Object.values(datasetStats.classes));
                  return (
                    <div key={cls} style={{ display: "flex", alignItems: "center", gap: 8, marginBottom: 5 }}>
                      <div style={{ width: 78, fontSize: 9, color: "#555" }}>{c.icon} {c.label || cls}</div>
                      <div style={{ flex: 1, height: 4, background: "rgba(255,255,255,0.04)", borderRadius: 99 }}>
                        <div style={{ height: "100%", width: `${(count / mx) * 100}%`, background: c.color, borderRadius: 99, opacity: .7 }} />
                      </div>
                      <div style={{ width: 28, fontSize: 8, color: "#444", fontFamily: "monospace", textAlign: "right" }}>{count}</div>
                    </div>
                  );
                })}
              </div>
            )}

            {trainError && (
              <div style={{
                background: "rgba(255,59,48,0.07)", border: "1px solid rgba(255,59,48,0.2)",
                borderRadius: 10, padding: 12, marginTop: 12, fontSize: 10, color: "#FF3B30"
              }}>❌ {trainError}</div>
            )}

            {Object.keys(epochHistory).length > 0 && (
              <div style={{
                background: "rgba(255,255,255,0.02)", border: "1px solid rgba(255,255,255,0.06)",
                borderRadius: 12, padding: "14px 16px", marginTop: 12
              }}>
                <div style={{ fontSize: 9, color: "#555", letterSpacing: 2, marginBottom: 12 }}>TRAINING CURVES</div>
                <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 14 }}>
                  {Object.entries(epochHistory).map(([id, h]) => (
                    <EpochChart key={id} history={h} color={ALG[id]?.color || "#00FF88"} label={ALG[id]?.name || id} />
                  ))}
                </div>
              </div>
            )}
          </div>
        )}

        {/* ══ PREDICT ══ */}
        {tab === "predict" && (
          <div style={{ display: "grid", gridTemplateColumns: "340px 1fr", gap: 20 }}>
            <div>
              {!modelsReady && (
                <div style={{
                  background: "rgba(255,149,0,0.07)", border: "1px solid rgba(255,149,0,0.2)",
                  borderRadius: 10, padding: "10px 13px", marginBottom: 12, fontSize: 10, color: "#FF9500"
                }}>
                  ⚠️ Train models first
                </div>
              )}

              <div style={{ fontSize: 9, color: "#444", letterSpacing: 2, marginBottom: 9 }}>UPLOAD AUDIO</div>
              <div onClick={() => document.getElementById("ai").click()}
                onDrop={e => { e.preventDefault(); handleAudio(e.dataTransfer.files[0]); }}
                onDragOver={e => e.preventDefault()}
                style={{
                  border: `2px dashed ${audioFile ? "rgba(0,255,136,0.5)" : "rgba(255,255,255,0.1)"}`,
                  borderRadius: 12, padding: "26px 16px", textAlign: "center", cursor: "pointer",
                  background: audioFile ? "rgba(0,255,136,0.03)" : "rgba(255,255,255,0.02)",
                  marginBottom: 11, transition: "all 0.2s"
                }}>
                <input id="ai" type="file" accept="audio/*" style={{ display: "none" }}
                  onChange={e => handleAudio(e.target.files[0])} />
                <div style={{ fontSize: 26, marginBottom: 7 }}>{audioFile ? "🎵" : "🎤"}</div>
                <div style={{ fontSize: 11, color: audioFile ? "#00FF88" : "#444" }}>
                  {audioFile ? audioFile.name : "Drop WAV / MP3 or click"}
                </div>
                {audioFile && <div style={{ fontSize: 9, color: "#333", marginTop: 3 }}>{(audioFile.size / 1024).toFixed(1)} KB</div>}
              </div>

              {audioURL && (
                <div style={{ marginBottom: 11 }}>
                  <div style={{ fontSize: 9, color: "#555", letterSpacing: 2, marginBottom: 5 }}>WAVEFORM</div>
                  <Waveform audioURL={audioURL} />
                  <div style={{ marginTop: 7 }}>
                    <audio ref={audioRef} src={audioURL} controls
                      style={{ width: "100%", borderRadius: 8, filter: "invert(0.8) hue-rotate(180deg)" }} />
                  </div>
                </div>
              )}

              {/* Info box */}
              <div style={{
                background: "rgba(0,122,255,0.05)", border: "1px solid rgba(0,122,255,0.15)",
                borderRadius: 9, padding: "10px 13px", marginBottom: 11, fontSize: 9, color: "#555", lineHeight: 1.9
              }}>
                <div style={{ color: "#007AFF", fontWeight: 700, marginBottom: 4, fontSize: 10 }}>
                  🆕 DUAL-CLASS DETECTION
                </div>
                <div>Detects the TWO most prominent sounds in the audio</div>
                <div>e.g. Aircraft comms + Rifle fire simultaneously</div>
                <div>Secondary shown only if confidence &gt; 20%</div>
              </div>

              <button onClick={runPredict} disabled={!audioFile || !modelsReady || predicting} style={{
                width: "100%", padding: 12, border: "none", borderRadius: 10,
                background: !audioFile || !modelsReady ? "rgba(255,255,255,0.04)"
                  : predicting ? "rgba(255,149,0,0.2)" : "linear-gradient(135deg,#FF3B30,#FF9500)",
                color: !audioFile || !modelsReady ? "#333" : "#fff",
                fontFamily: "inherit", fontSize: 11, fontWeight: 700, letterSpacing: 2,
                cursor: audioFile && modelsReady ? "pointer" : "not-allowed"
              }}>
                {predicting ? "⚡ ANALYZING ALL 5 MODELS…" : "🔍 DETECT THREAT CLASSES"}
              </button>

              <div style={{ marginTop: 14 }}>
                <div style={{ fontSize: 8, color: "#2a2a2a", letterSpacing: 2, marginBottom: 7 }}>DETECTABLE</div>
                <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 4 }}>
                  {(classNames.length > 0 ? classNames : Object.keys(CLS)).map(cls => {
                    const c = getC(cls);
                    return (
                      <div key={cls} style={{
                        display: "flex", alignItems: "center", gap: 6,
                        background: "rgba(255,255,255,0.02)", borderRadius: 6, padding: "4px 8px",
                        border: "1px solid rgba(255,255,255,0.04)"
                      }}>
                        <span>{c.icon}</span>
                        <span style={{ fontSize: 8, color: "#444" }}>{c.label || cls}</span>
                      </div>
                    );
                  })}
                </div>
              </div>
            </div>

            {/* RIGHT — dual detection results */}
            <div>
              {predictions ? (
                <DualThreatCard names={predictions.class_names || classNames} preds={predictions.predictions} />
              ) : (
                <div style={{
                  height: "100%", minHeight: 460, display: "flex", flexDirection: "column",
                  alignItems: "center", justifyContent: "center",
                  background: "rgba(255,255,255,0.01)", border: "1px dashed rgba(255,255,255,0.05)", borderRadius: 14
                }}>
                  <div style={{ fontSize: 44, marginBottom: 14, opacity: .2 }}>🎧</div>
                  <div style={{ fontSize: 12, color: "#333" }}>Upload audio to detect threats</div>
                  <div style={{ fontSize: 9, color: "#222", marginTop: 6 }}>Detects up to 2 simultaneous threat classes</div>
                  <div style={{ display: "flex", gap: 8, marginTop: 16, flexWrap: "wrap", justifyContent: "center" }}>
                    {Object.values(CLS).map(c => (
                      <div key={c.label} style={{
                        fontSize: 9, color: "#2a2a2a", padding: "3px 9px",
                        border: "1px solid #1a1a1a", borderRadius: 99
                      }}>{c.icon} {c.label}</div>
                    ))}
                  </div>
                </div>
              )}
            </div>
          </div>
        )}

        {/* ══ COMPARE ══ */}
        {tab === "compare" && (
          <div>
            {!results ? (
              <div style={{ textAlign: "center", padding: "80px 20px", color: "#333" }}>
                <div style={{ fontSize: 36, marginBottom: 14, opacity: .2 }}>📊</div>
                <div style={{ fontSize: 12 }}>Train models first</div>
                <button onClick={() => setTab("train")} style={{
                  marginTop: 14,
                  background: "rgba(0,255,136,0.07)", border: "1px solid rgba(0,255,136,0.2)",
                  borderRadius: 7, padding: "9px 22px", color: "#00FF88",
                  fontFamily: "inherit", fontSize: 10, cursor: "pointer"
                }}>→ Go to Training</button>
              </div>
            ) : (
              <>
                {sortedAlgos.length > 0 && (() => {
                  const [bestId, bestM] = sortedAlgos[0], am = ALG[bestId];
                  return (
                    <div style={{
                      background: `linear-gradient(135deg,${am.color}10,rgba(0,0,0,0.5))`,
                      border: `1.5px solid ${am.color}`, borderRadius: 14, padding: "16px 22px", marginBottom: 18,
                      display: "flex", alignItems: "center", justifyContent: "space-between", flexWrap: "wrap", gap: 12
                    }}>
                      <div>
                        <div style={{ fontSize: 8, color: am.color, letterSpacing: 3 }}>🏆 BEST MODEL</div>
                        <div style={{ fontSize: 20, fontWeight: 700, color: "#fff", marginTop: 4 }}>
                          {am.icon} {am.full} — {bestM.accuracy}%
                        </div>
                        <div style={{ fontSize: 10, color: "#444", marginTop: 3 }}>
                          {classNames.length} classes · {classNames.join(", ")}
                        </div>
                      </div>
                      <div style={{ display: "flex", gap: 18 }}>
                        {[["F1", bestM.f1], ["Prec", bestM.precision], ["Rec", bestM.recall]].map(([l, v]) => (
                          <div key={l} style={{ textAlign: "center" }}>
                            <div style={{ fontSize: 18, fontWeight: 700, color: am.color, fontFamily: "monospace" }}>{v}%</div>
                            <div style={{ fontSize: 8, color: "#444" }}>{l}</div>
                          </div>
                        ))}
                      </div>
                    </div>
                  );
                })()}

                <div style={{ display: "grid", gridTemplateColumns: "repeat(5,1fr)", gap: 10, marginBottom: 18 }}>
                  {sortedAlgos.map(([id, m], rank) => {
                    const am = ALG[id], rc = ["#FFD700", "#C0C0C0", "#CD7F32"];
                    return (
                      <div key={id} onClick={() => setSelectedAlgo(id)} style={{
                        background: selectedAlgo === id ? `${am.color}10` : "rgba(255,255,255,0.02)",
                        border: `1.5px solid ${selectedAlgo === id ? am.color : "rgba(255,255,255,0.06)"}`,
                        borderRadius: 10, padding: "11px 12px", cursor: "pointer",
                        transition: "all 0.2s", position: "relative"
                      }}>
                        {rank < 3 && (
                          <div style={{
                            position: "absolute", top: 6, right: 6, background: rc[rank],
                            color: "#000", borderRadius: 99, width: 15, height: 15, display: "flex",
                            alignItems: "center", justifyContent: "center", fontSize: 7, fontWeight: 800
                          }}>
                            #{rank + 1}
                          </div>
                        )}
                        <div style={{ fontSize: 11, fontWeight: 700, color: am.color, marginBottom: 1 }}>{am.icon} {am.name}</div>
                        <div style={{ fontSize: 7, color: "#333", marginBottom: 7 }}>{am.full}</div>
                        <div style={{ fontSize: 18, fontWeight: 700, color: "#fff", fontFamily: "monospace" }}>{m.accuracy}%</div>
                        <div style={{ fontSize: 7, color: "#333", marginBottom: 5 }}>ACCURACY</div>
                        <div style={{ height: 3, background: "rgba(255,255,255,0.05)", borderRadius: 99 }}>
                          <div style={{ height: "100%", width: `${m.accuracy}%`, background: am.color, borderRadius: 99 }} />
                        </div>
                        <div style={{ display: "flex", justifyContent: "space-between", marginTop: 4, fontSize: 7, color: "#333" }}>
                          <span>F1:{m.f1}%</span><span>P:{m.precision}%</span>
                        </div>
                        {predictions?.predictions?.[id] && (() => {
                          const pr = predictions.predictions[id];
                          const cls = (predictions.class_names || classNames)[pr.pred];
                          const c = getC(cls);
                          return (
                            <div style={{
                              marginTop: 6, padding: "3px 6px", background: `${c.color}15`,
                              borderRadius: 5, border: `1px solid ${c.color}33`,
                              fontSize: 7, color: c.color, textAlign: "center"
                            }}>
                              {c.icon} {c.label || cls} · {(Math.max(...pr.proba) * 100).toFixed(0)}%
                            </div>
                          );
                        })()}
                      </div>
                    );
                  })}
                </div>

                {results[selectedAlgo] && (() => {
                  const m = results[selectedAlgo], am = ALG[selectedAlgo];
                  return (
                    <div style={{
                      background: "rgba(255,255,255,0.02)", border: "1px solid rgba(255,255,255,0.06)",
                      borderRadius: 14, padding: 20
                    }}>
                      <div style={{ display: "flex", alignItems: "center", gap: 8, marginBottom: 16 }}>
                        <div style={{ width: 3, height: 20, background: am.color, borderRadius: 2 }} />
                        <div style={{ fontSize: 13, fontWeight: 700, color: am.color }}>{am.icon} {am.full}</div>
                      </div>
                      <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 22, marginBottom: 20 }}>
                        <div>
                          <div style={{ fontSize: 8, color: "#444", letterSpacing: 2, marginBottom: 10 }}>PER-CLASS ACCURACY</div>
                          {classNames.map(cls => {
                            const pc = m.per_class?.[cls] ?? 0, c = getC(cls);
                            return (
                              <div key={cls} style={{ marginBottom: 8 }}>
                                <div style={{ display: "flex", justifyContent: "space-between", fontSize: 10, marginBottom: 3 }}>
                                  <span>{c.icon} {c.label || cls}</span>
                                  <span style={{ color: c.color, fontFamily: "monospace" }}>{pc}%</span>
                                </div>
                                <div style={{ height: 5, background: "rgba(255,255,255,0.04)", borderRadius: 99 }}>
                                  <div style={{ height: "100%", width: `${pc}%`, background: c.color, borderRadius: 99 }} />
                                </div>
                              </div>
                            );
                          })}
                        </div>
                        <div>
                          <div style={{ fontSize: 8, color: "#444", letterSpacing: 2, marginBottom: 10 }}>CONFUSION MATRIX</div>
                          <Matrix matrix={m.conf_matrix} classNames={classNames} />
                        </div>
                      </div>
                      {epochHistory[selectedAlgo] && (
                        <div style={{ borderTop: "1px solid rgba(255,255,255,0.05)", paddingTop: 16, marginBottom: 16 }}>
                          <EpochChart history={epochHistory[selectedAlgo]} color={am.color} label={am.name} />
                        </div>
                      )}
                      <div style={{ borderTop: "1px solid rgba(255,255,255,0.05)", paddingTop: 16 }}>
                        <div style={{ fontSize: 8, color: "#444", letterSpacing: 2, marginBottom: 10 }}>ALL MODELS</div>
                        {sortedAlgos.map(([id, mt]) => {
                          const a = ALG[id];
                          return (
                            <div key={id} style={{ display: "flex", alignItems: "center", gap: 9, marginBottom: 7 }}>
                              <div style={{ width: 78, fontSize: 10, color: id === selectedAlgo ? a.color : "#444" }}>{a.icon} {a.name}</div>
                              <div style={{ flex: 1, height: 6, background: "rgba(255,255,255,0.04)", borderRadius: 99 }}>
                                <div style={{
                                  height: "100%", borderRadius: 99, width: `${mt.accuracy}%`,
                                  background: id === selectedAlgo ? a.color : "rgba(255,255,255,0.1)", transition: "width 0.8s"
                                }} />
                              </div>
                              <div style={{ width: 42, textAlign: "right", fontSize: 10, fontFamily: "monospace" }}>{mt.accuracy}%</div>
                            </div>
                          );
                        })}
                      </div>
                      <div style={{
                        display: "grid", gridTemplateColumns: "repeat(4,1fr)", gap: 9, marginTop: 16,
                        borderTop: "1px solid rgba(255,255,255,0.05)", paddingTop: 16
                      }}>
                        {[["Accuracy", m.accuracy], ["F1 Score", m.f1], ["Precision", m.precision], ["Recall", m.recall]].map(([l, v]) => (
                          <div key={l} style={{
                            background: "rgba(255,255,255,0.02)", borderRadius: 8,
                            padding: 13, textAlign: "center", border: "1px solid rgba(255,255,255,0.04)"
                          }}>
                            <div style={{ fontSize: 18, fontWeight: 700, color: am.color, fontFamily: "monospace" }}>{v}%</div>
                            <div style={{ fontSize: 9, color: "#777", marginTop: 3 }}>{l}</div>
                          </div>
                        ))}
                      </div>
                    </div>
                  );
                })()}
              </>
            )}
          </div>
        )}
      </div>
      <style>{`
        @import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&display=swap');
        *{box-sizing:border-box;margin:0;padding:0;}
        @keyframes pulse{0%,100%{opacity:1}50%{opacity:0.4}}
        ::-webkit-scrollbar{width:3px;}
        ::-webkit-scrollbar-thumb{background:#1a1a1a;border-radius:2px;}
        input:focus{border-color:rgba(0,255,136,0.4)!important;}
      `}</style>
    </div>
  );
}