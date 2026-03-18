import { useState, useRef, useEffect, useCallback } from "react";

const API = "http://localhost:8001";

const ALG_META = {
  svm:  { name: "SVM",           full: "Support Vector Machine",         color: "#A78BFA", icon: "⚡" },
  rf:   { name: "Random Forest",  full: "Random Forest Classifier",       color: "#34D399", icon: "🌲" },
  xgb:  { name: "XGBoost",        full: "Extreme Gradient Boosting",      color: "#FFE66D", icon: "🚀" },
  cnn:  { name: "CNN",            full: "Convolutional Neural Network",   color: "#FF6B6B", icon: "🧠" },
  lstm: { name: "LSTM",           full: "Bidirectional LSTM + Attention", color: "#4ECDC4", icon: "🔄" },
};

const CLASS_META = {
  gunshot:      { icon: "🔫", color: "#FF3B30", label: "Gunshot"      },
  rifle_fire:   { icon: "🎯", color: "#FF9500", label: "Rifle Fire"   },
  vehicle:      { icon: "🚛", color: "#FFCC00", label: "Vehicle"      },
  aircraft:     { icon: "✈️", color: "#34C759", label: "Aircraft"     },
  comms_signal: { icon: "📡", color: "#007AFF", label: "Comms Signal" },
  explosion:    { icon: "💥", color: "#FF2D55", label: "Explosion"    },
  ambient:      { icon: "🔇", color: "#8E8E93", label: "Ambient/Safe" },
  class_0: { icon: "🔫", color: "#FF3B30", label: "Class 0" },
  class_1: { icon: "🎯", color: "#FF9500", label: "Class 1" },
  class_2: { icon: "🚛", color: "#FFCC00", label: "Class 2" },
  class_3: { icon: "✈️", color: "#34C759", label: "Class 3" },
  class_4: { icon: "📡", color: "#007AFF", label: "Class 4" },
  class_5: { icon: "💥", color: "#FF2D55", label: "Class 5" },
  class_6: { icon: "🔇", color: "#8E8E93", label: "Class 6" },
};

function getC(name) {
  return CLASS_META[name?.toLowerCase()] || CLASS_META[name] || { icon: "🎵", color: "#8E8E93", label: name || "Unknown" };
}

function Gauge({ value, color, size = 80 }) {
  const r = 32, circ = 2 * Math.PI * r;
  const pct = Math.min(Math.max(value, 0), 1);
  const dash = pct * circ * 0.75;
  return (
    <svg width={size} height={size} viewBox="0 0 72 72">
      <circle cx="36" cy="36" r={r} fill="none" stroke="rgba(255,255,255,0.06)"
        strokeWidth="7" strokeDasharray={`${circ*0.75} ${circ*0.25}`}
        strokeLinecap="round" transform="rotate(135 36 36)" />
      <circle cx="36" cy="36" r={r} fill="none" stroke={color}
        strokeWidth="7" strokeDasharray={`${dash} ${circ-dash+circ*0.25}`}
        strokeLinecap="round" transform="rotate(135 36 36)"
        style={{ transition: "stroke-dasharray 1s ease" }} />
      <text x="36" y="40" textAnchor="middle" fill="#fff"
        fontSize="12" fontWeight="700" fontFamily="monospace">
        {Math.round(pct * 100)}%
      </text>
    </svg>
  );
}

function Matrix({ matrix, classNames }) {
  if (!matrix?.length || !classNames?.length) return null;
  const maxV = Math.max(...matrix.flat(), 1);
  return (
    <div style={{ overflowX: "auto" }}>
      <div style={{ display: "inline-block", fontSize: 9, fontFamily: "monospace" }}>
        <div style={{ display: "flex", marginBottom: 2, marginLeft: 64 }}>
          {classNames.map(n => (
            <div key={n} style={{ width: 36, textAlign: "center", color: "#444", fontSize: 8 }}>{getC(n).icon}</div>
          ))}
        </div>
        {matrix.map((row, i) => (
          <div key={i} style={{ display: "flex", alignItems: "center", marginBottom: 1 }}>
            <div style={{ width: 62, textAlign: "right", paddingRight: 6, color: "#444", fontSize: 8 }}>
              {getC(classNames[i]).icon} {classNames[i]?.slice(0, 7)}
            </div>
            {row.map((v, j) => {
              const p = v / maxV;
              return (
                <div key={j} style={{
                  width: 36, height: 26, display: "flex", alignItems: "center", justifyContent: "center",
                  background: i===j ? `rgba(0,255,136,${0.08+p*0.7})` : `rgba(255,59,48,${p*0.55})`,
                  border: "1px solid rgba(255,255,255,0.03)",
                  color: p > 0.2 ? "#fff" : "#333", fontSize: 8
                }}>{v}</div>
              );
            })}
          </div>
        ))}
        <div style={{ color: "#222", marginTop: 4, marginLeft: 64, fontSize: 8 }}>← Predicted | Actual ↓</div>
      </div>
    </div>
  );
}

function TrainingLog({ lines }) {
  const ref = useRef(null);
  useEffect(() => { if (ref.current) ref.current.scrollTop = ref.current.scrollHeight; }, [lines]);
  return (
    <div ref={ref} style={{
      background: "#030608", border: "1px solid rgba(0,255,136,0.1)", borderRadius: 10,
      padding: "12px 14px", fontFamily: "monospace", fontSize: 11,
      height: 180, overflowY: "auto", lineHeight: 1.7
    }}>
      {lines.length === 0
        ? <span style={{ color: "#1a1a1a" }}>Waiting for logs…</span>
        : lines.map((l, i) => (
          <div key={i} style={{
            color: l.includes("✅") ? "#00FF88" : l.includes("❌") ? "#FF3B30"
                 : l.includes("⚙️") ? "#FFE66D" : l.includes("🏆") ? "#FF9500"
                 : l.includes("📂") ? "#007AFF" : "#4ade80"
          }}>{l}</div>
        ))}
    </div>
  );
}

export default function App() {
  const [tab, setTab]               = useState("train");
  const [training, setTraining]     = useState(false);
  const [trainLog, setTrainLog]     = useState([]);
  const [progress, setProgress]     = useState(0);
  const [curModel, setCurModel]     = useState("");
  const [results, setResults]       = useState(null);
  const [classNames, setClassNames] = useState([]);
  const [trainError, setTrainError] = useState("");
  const [selectedAlgo, setSelectedAlgo] = useState("xgb");
  const [modelsReady, setModelsReady]   = useState(false);
  const [audioFile, setAudioFile]   = useState(null);
  const [audioURL, setAudioURL]     = useState(null);
  const [predicting, setPredicting] = useState(false);
  const [predictions, setPredictions] = useState(null);
  const pollRef = useRef(null);

  const fetchResults = useCallback(async () => {
    try {
      const rr = await fetch(`${API}/results`);
      if (rr.ok) {
        const rd = await rr.json();
        setResults(rd.results);
        setClassNames(rd.class_names || []);
        setProgress(100);
        setModelsReady(true);
        return true;
      }
    } catch (_) {}
    return false;
  }, []);

  useEffect(() => {
    const init = async () => {
      try {
        const sr = await fetch(`${API}/status`);
        const sd = await sr.json();
        if (sd.status === "done") { await fetchResults(); return; }
        const lr = await fetch(`${API}/load`);
        if (lr.ok) {
          const ld = await lr.json();
          if (ld.loaded?.length > 0) await fetchResults();
        }
      } catch (_) {}
    };
    init();
  }, [fetchResults]);

  const pollStatus = useCallback(async () => {
    try {
      const r = await fetch(`${API}/status`);
      const d = await r.json();
      setTrainLog(d.log || []);
      setProgress(d.progress || 0);
      setCurModel(d.current_model || "");
      if (d.class_names?.length) setClassNames(d.class_names);
      if (d.status === "done") {
        clearInterval(pollRef.current);
        setTraining(false);
        const ok = await fetchResults();
        if (ok) setTab("predict");
      } else if (d.status === "error") {
        clearInterval(pollRef.current);
        setTraining(false);
        setTrainError(d.error || "Unknown error");
      }
    } catch (_) {}
  }, [fetchResults]);

  useEffect(() => () => clearInterval(pollRef.current), []);

  const handleDatasetUpload = async (file) => {
    if (!file) return;
    setTraining(true); setTrainError(""); setResults(null);
    setModelsReady(false); setProgress(0); setTrainLog([]);
    const fd = new FormData();
    fd.append("file", file);
    try {
      const r = await fetch(`${API}/train`, { method: "POST", body: fd });
      if (!r.ok) { const e = await r.json(); throw new Error(e.detail); }
      pollRef.current = setInterval(pollStatus, 1500);
    } catch (e) {
      setTraining(false);
      setTrainError(e.message);
    }
  };

  const handleAudioDrop = (file) => {
    if (!file) return;
    setAudioFile(file);
    setAudioURL(URL.createObjectURL(file));
    setPredictions(null);
  };

  const runPredict = async () => {
    if (!audioFile || !modelsReady) return;
    setPredicting(true);
    const fd = new FormData();
    fd.append("file", audioFile);
    try {
      const r = await fetch(`${API}/predict`, { method: "POST", body: fd });
      const d = await r.json();
      setPredictions(d);
      setTab("compare");
    } catch (e) { alert("Prediction failed: " + e.message); }
    setPredicting(false);
  };

  const sortedAlgos = (results && typeof results === "object")
    ? Object.entries(results).filter(([,m]) => m && m.accuracy != null).sort((a, b) => b[1].accuracy - a[1].accuracy)
    : [];

  const tabs = [
    { id: "train",   label: "TRAIN",      icon: "⚡" },
    { id: "predict", label: "PREDICT",    icon: "🎯" },
    { id: "compare", label: "COMPARISON", icon: "📊" },
  ];

  return (
    <div style={{
      minHeight: "100vh", background: "#07090E", color: "#E8E8E8",
      fontFamily: "'Space Mono','Courier New',monospace",
      backgroundImage: `radial-gradient(ellipse at 10% 10%,rgba(0,255,136,0.03) 0%,transparent 50%),
        radial-gradient(ellipse at 90% 90%,rgba(0,122,255,0.03) 0%,transparent 50%),
        repeating-linear-gradient(0deg,transparent,transparent 39px,rgba(255,255,255,0.01) 40px),
        repeating-linear-gradient(90deg,transparent,transparent 39px,rgba(255,255,255,0.01) 40px)`
    }}>
      {/* HEADER */}
      <div style={{ borderBottom:"1px solid rgba(0,255,136,0.12)", background:"rgba(0,0,0,0.7)",
        backdropFilter:"blur(20px)", padding:"0 28px", position:"sticky", top:0, zIndex:100 }}>
        <div style={{ maxWidth:1200, margin:"0 auto", display:"flex", alignItems:"center",
          justifyContent:"space-between", height:60 }}>
          <div style={{ display:"flex", alignItems:"center", gap:12 }}>
            <div style={{ width:32, height:32, borderRadius:8,
              background:"linear-gradient(135deg,#00FF88,#007AFF)",
              display:"flex", alignItems:"center", justifyContent:"center", fontSize:16 }}>⚡</div>
            <div>
              <div style={{ fontSize:13, fontWeight:700, color:"#fff", letterSpacing:1 }}>ACOUSTIC THREAT DETECTION</div>
              <div style={{ fontSize:9, color:"#00FF88", letterSpacing:3 }}>5-ALGORITHM ML COMPARISON SYSTEM</div>
            </div>
          </div>
          <div style={{ display:"flex", alignItems:"center", gap:16 }}>
            {modelsReady && (
              <div style={{ fontSize:10, color:"#00FF88" }}>
                ✅ {Object.keys(results||{}).length} models ready · {classNames.length} classes
              </div>
            )}
            <div style={{ display:"flex", gap:6, alignItems:"center" }}>
              <div style={{ width:7, height:7, borderRadius:"50%", background:"#00FF88",
                boxShadow:"0 0 8px #00FF88", animation:"pulse 2s infinite" }} />
              <span style={{ fontSize:9, color:"#00FF88", letterSpacing:2 }}>ONLINE</span>
            </div>
          </div>
        </div>
      </div>

      {/* TABS */}
      <div style={{ borderBottom:"1px solid rgba(255,255,255,0.05)", background:"rgba(0,0,0,0.4)" }}>
        <div style={{ maxWidth:1200, margin:"0 auto", display:"flex", padding:"0 28px" }}>
          {tabs.map(t => (
            <button key={t.id} onClick={() => setTab(t.id)} style={{
              background:"none", border:"none", cursor:"pointer",
              padding:"15px 24px", fontSize:10, letterSpacing:2,
              color: tab===t.id ? "#00FF88" : "#444",
              borderBottom: tab===t.id ? "2px solid #00FF88" : "2px solid transparent",
              fontFamily:"inherit", transition:"all 0.2s"
            }}>{t.icon} {t.label}</button>
          ))}
        </div>
      </div>

      <div style={{ maxWidth:1200, margin:"0 auto", padding:24 }}>

        {/* ══════ TRAIN ══════ */}
        {tab === "train" && (
          <div style={{ maxWidth:680, margin:"0 auto" }}>
            {modelsReady ? (
              <div style={{ background:"rgba(0,255,136,0.05)", border:"1px solid rgba(0,255,136,0.2)",
                borderRadius:14, padding:"16px 20px", marginBottom:20,
                display:"flex", alignItems:"center", justifyContent:"space-between" }}>
                <div>
                  <div style={{ fontSize:12, color:"#00FF88", fontWeight:700 }}>✅ Models Already Trained & Loaded!</div>
                  <div style={{ fontSize:10, color:"#444", marginTop:4 }}>
                    {Object.keys(results||{}).length} models · {classNames.length} classes · Ready to predict
                  </div>
                </div>
                <div style={{ display:"flex", gap:10 }}>
                  <button onClick={() => setTab("compare")} style={{
                    background:"rgba(255,255,255,0.06)", border:"1px solid rgba(255,255,255,0.1)",
                    borderRadius:8, padding:"8px 18px", color:"#fff",
                    fontFamily:"inherit", fontSize:11, fontWeight:700, cursor:"pointer"
                  }}>VIEW COMPARISON →</button>
                  <button onClick={() => setTab("predict")} style={{
                    background:"linear-gradient(135deg,#00FF88,#007AFF)", border:"none",
                    borderRadius:8, padding:"8px 18px", color:"#000",
                    fontFamily:"inherit", fontSize:11, fontWeight:700, cursor:"pointer"
                  }}>PREDICT AUDIO →</button>
                </div>
              </div>
            ) : (
              <div style={{ background:"rgba(0,122,255,0.05)", border:"1px solid rgba(0,122,255,0.2)",
                borderRadius:14, padding:"16px 20px", marginBottom:20,
                display:"flex", alignItems:"center", justifyContent:"space-between" }}>
                <div>
                  <div style={{ fontSize:12, color:"#007AFF", fontWeight:700 }}>🔄 Saved Models Found</div>
                  <div style={{ fontSize:10, color:"#444", marginTop:4 }}>Click to load your previously trained models</div>
                </div>
                <button onClick={async () => {
                  await fetch(`${API}/load`);
                  await fetchResults();
                }} style={{
                  background:"linear-gradient(135deg,#007AFF,#00FF88)", border:"none",
                  borderRadius:8, padding:"8px 18px", color:"#000",
                  fontFamily:"inherit", fontSize:11, fontWeight:700, cursor:"pointer"
                }}>⚡ LOAD SAVED MODELS</button>
              </div>
            )}

            <div style={{ background:"rgba(0,122,255,0.05)", border:"1px solid rgba(0,122,255,0.15)",
              borderRadius:14, padding:"16px 20px", marginBottom:18 }}>
              <div style={{ fontSize:9, color:"#007AFF", letterSpacing:2, marginBottom:10 }}>📋 ZIP YOUR DATASET</div>
              <pre style={{ background:"rgba(0,0,0,0.5)", borderRadius:8, padding:"10px 14px",
                fontSize:11, color:"#4ade80", lineHeight:1.8 }}>{`cd C:\\Projects\\threat_detection_system\\data
Compress-Archive -Path audio -DestinationPath dataset.zip`}</pre>
              <div style={{ fontSize:10, color:"#333", marginTop:8 }}>
                ⏱ ~25–35 min on CPU (10 epochs) · Delete old models/ folder if retraining from scratch
              </div>
            </div>

            <div onClick={() => !training && document.getElementById("dsInput").click()}
              onDrop={e => { e.preventDefault(); if (!training) handleDatasetUpload(e.dataTransfer.files[0]); }}
              onDragOver={e => e.preventDefault()}
              style={{
                border:`2px dashed ${training ? "rgba(255,149,0,0.6)" : "rgba(0,255,136,0.3)"}`,
                borderRadius:14, padding:"44px 24px", textAlign:"center",
                cursor: training ? "wait" : "pointer",
                background: training ? "rgba(255,149,0,0.02)" : "rgba(255,255,255,0.02)",
                marginBottom:16, transition:"all 0.3s"
              }}>
              <input id="dsInput" type="file" accept=".zip" style={{ display:"none" }}
                onChange={e => handleDatasetUpload(e.target.files[0])} />
              <div style={{ fontSize:36, marginBottom:10 }}>{training ? "⏳" : "📦"}</div>
              <div style={{ fontSize:13, color: training ? "#FF9500" : "#00FF88" }}>
                {training ? `Training — ${curModel || "preparing"}…` : "Drop dataset.zip here or click to browse"}
              </div>
              <div style={{ fontSize:10, color:"#333", marginTop:8 }}>
                Trains all 5 models: SVM · Random Forest · XGBoost · CNN · LSTM
              </div>
            </div>

            {(training || progress > 0) && (
              <div style={{ marginBottom:16 }}>
                <div style={{ display:"flex", justifyContent:"space-between", marginBottom:6, fontSize:10 }}>
                  <span style={{ color:"#FF9500" }}>⚙️ {curModel || "Processing"}</span>
                  <span style={{ fontFamily:"monospace" }}>{progress}%</span>
                </div>
                <div style={{ height:5, background:"rgba(255,255,255,0.05)", borderRadius:99 }}>
                  <div style={{ height:"100%", borderRadius:99, width:`${progress}%`,
                    background:"linear-gradient(90deg,#00FF88,#007AFF)", transition:"width 0.6s ease" }} />
                </div>
                <div style={{ display:"flex", gap:8, marginTop:10, flexWrap:"wrap" }}>
                  {Object.entries(ALG_META).map(([id, m]) => {
                    const done   = trainLog.some(l => l.includes(m.name) && l.includes("✅"));
                    const active = curModel?.toLowerCase().includes(m.name.toLowerCase());
                    return (
                      <div key={id} style={{
                        padding:"4px 10px", borderRadius:99, fontSize:9,
                        background: done ? `${m.color}18` : active ? "rgba(255,149,0,0.12)" : "rgba(255,255,255,0.03)",
                        border:`1px solid ${done ? m.color : active ? "#FF9500" : "rgba(255,255,255,0.06)"}`,
                        color: done ? m.color : active ? "#FF9500" : "#333"
                      }}>
                        {done ? "✅" : active ? "⚙️" : "⏳"} {m.name}
                      </div>
                    );
                  })}
                </div>
              </div>
            )}

            {(training || trainLog.length > 0) && <TrainingLog lines={trainLog} />}
            {trainError && (
              <div style={{ background:"rgba(255,59,48,0.07)", border:"1px solid rgba(255,59,48,0.2)",
                borderRadius:10, padding:14, marginTop:14, fontSize:11, color:"#FF3B30" }}>
                ❌ {trainError}
              </div>
            )}
          </div>
        )}

        {/* ══════ PREDICT ══════ */}
        {tab === "predict" && (
          <div style={{ display:"grid", gridTemplateColumns:"360px 1fr", gap:20 }}>
            <div>
              {!modelsReady && (
                <div style={{ background:"rgba(255,149,0,0.07)", border:"1px solid rgba(255,149,0,0.2)",
                  borderRadius:10, padding:"12px 16px", marginBottom:14, fontSize:11, color:"#FF9500" }}>
                  ⚠️ Train models first before predicting
                </div>
              )}
              <div style={{ fontSize:9, color:"#444", letterSpacing:2, marginBottom:12 }}>UPLOAD AUDIO FILE</div>
              <div onClick={() => document.getElementById("wavInput").click()}
                onDrop={e => { e.preventDefault(); handleAudioDrop(e.dataTransfer.files[0]); }}
                onDragOver={e => e.preventDefault()}
                style={{
                  border:`2px dashed ${audioFile ? "rgba(0,255,136,0.5)" : "rgba(255,255,255,0.1)"}`,
                  borderRadius:14, padding:"32px 20px", textAlign:"center", cursor:"pointer",
                  background: audioFile ? "rgba(0,255,136,0.03)" : "rgba(255,255,255,0.02)",
                  marginBottom:14, transition:"all 0.2s"
                }}>
                <input id="wavInput" type="file" accept="audio/*" style={{ display:"none" }}
                  onChange={e => handleAudioDrop(e.target.files[0])} />
                <div style={{ fontSize:32, marginBottom:10 }}>{audioFile ? "🎵" : "🎤"}</div>
                <div style={{ fontSize:12, color: audioFile ? "#00FF88" : "#444" }}>
                  {audioFile ? audioFile.name : "Drop WAV / MP3 or click to browse"}
                </div>
                {audioFile && <div style={{ fontSize:10, color:"#333", marginTop:4 }}>{(audioFile.size/1024).toFixed(1)} KB</div>}
              </div>

              {audioURL && (
                <div style={{ marginBottom:14 }}>
                  <audio src={audioURL} controls style={{ width:"100%", borderRadius:8 }} />
                </div>
              )}

              <div style={{ background:"rgba(255,255,255,0.02)", border:"1px solid rgba(255,255,255,0.05)",
                borderRadius:10, padding:"12px 14px", marginBottom:14, fontSize:10, color:"#444", lineHeight:1.9 }}>
                <div style={{ color:"#666", marginBottom:6, fontWeight:700 }}>HOW IT WORKS</div>
                <div>1. Upload any audio file (.wav, .mp3)</div>
                <div>2. All 5 models analyze simultaneously</div>
                <div>3. Each model identifies the sound class</div>
                <div>4. Comparison tab shows which model is most confident</div>
              </div>

              <button onClick={runPredict} disabled={!audioFile || !modelsReady || predicting} style={{
                width:"100%", padding:14, border:"none", borderRadius:12,
                background: !audioFile || !modelsReady ? "rgba(255,255,255,0.04)"
                  : predicting ? "rgba(255,149,0,0.2)"
                  : "linear-gradient(135deg,#FF3B30,#FF9500)",
                color: !audioFile || !modelsReady ? "#333" : "#fff",
                fontFamily:"inherit", fontSize:12, fontWeight:700,
                letterSpacing:2, cursor: audioFile && modelsReady ? "pointer" : "not-allowed"
              }}>
                {predicting ? "⚡ ANALYZING WITH 5 MODELS…" : "🔍 IDENTIFY SOUND CLASS"}
              </button>

              <div style={{ marginTop:18 }}>
                <div style={{ fontSize:9, color:"#2a2a2a", letterSpacing:2, marginBottom:10 }}>DETECTABLE CLASSES</div>
                <div style={{ display:"grid", gridTemplateColumns:"1fr 1fr", gap:6 }}>
                  {(classNames.length > 0 ? classNames : ["gunshot","rifle_fire","vehicle","aircraft","comms_signal","explosion","ambient"]).map(cls => {
                    const c = getC(cls);
                    return (
                      <div key={cls} style={{ display:"flex", alignItems:"center", gap:8,
                        background:"rgba(255,255,255,0.02)", borderRadius:8, padding:"6px 10px",
                        border:"1px solid rgba(255,255,255,0.04)" }}>
                        <span>{c.icon}</span>
                        <span style={{ fontSize:10, color:"#555" }}>{c.label || cls}</span>
                      </div>
                    );
                  })}
                </div>
              </div>
            </div>

            {/* RIGHT — results */}
            <div>
              {predictions ? (() => {
                const preds    = predictions.predictions;
                const names    = predictions.class_names || classNames;
                const algoList = Object.entries(preds);
                const votes = {};
                algoList.forEach(([,p]) => { votes[p.pred] = (votes[p.pred]||0)+1; });
                const topIdx   = parseInt(Object.entries(votes).sort((a,b)=>b[1]-a[1])[0][0]);
                const topName  = names[topIdx] || "Unknown";
                const topC     = getC(topName);
                const voteCount = votes[topIdx];
                return (
                  <div>
                    <div style={{
                      background:`linear-gradient(135deg,${topC.color}10,rgba(0,0,0,0.6))`,
                      border:`2px solid ${topC.color}`,
                      borderRadius:16, padding:24, textAlign:"center", marginBottom:20
                    }}>
                      <div style={{ fontSize:9, color:"#444", letterSpacing:3, marginBottom:8 }}>
                        ENSEMBLE VERDICT — {voteCount}/{algoList.length} MODELS AGREE
                      </div>
                      <div style={{ fontSize:52, marginBottom:8 }}>{topC.icon}</div>
                      <div style={{ fontSize:26, fontWeight:700, color:topC.color, letterSpacing:2 }}>
                        {(topC.label||topName).toUpperCase()}
                      </div>
                      <div style={{
                        marginTop:14, display:"inline-block", padding:"6px 18px", borderRadius:99,
                        background: voteCount>=4?"rgba(255,59,48,0.15)":voteCount>=3?"rgba(255,149,0,0.15)":"rgba(52,199,89,0.15)",
                        border:`1px solid ${voteCount>=4?"#FF3B30":voteCount>=3?"#FF9500":"#34C759"}`,
                        fontSize:10, fontWeight:700, letterSpacing:1,
                        color: voteCount>=4?"#FF3B30":voteCount>=3?"#FF9500":"#34C759"
                      }}>
                        {voteCount>=4?"⚠️ HIGH CONFIDENCE":voteCount>=3?"⚡ MODERATE CONFIDENCE":"✅ LOW CONFIDENCE — REVIEW"}
                      </div>
                    </div>
                    <div style={{ fontSize:9, color:"#444", letterSpacing:2, marginBottom:12 }}>ALL 5 MODELS — INDIVIDUAL RESULTS</div>
                    <div style={{ display:"grid", gridTemplateColumns:"1fr 1fr", gap:12 }}>
                      {algoList.map(([id, res]) => {
                        const am  = ALG_META[id];
                        const cls = names[res.pred]||"Unknown";
                        const cm2 = getC(cls);
                        const conf = Math.max(...res.proba);
                        return (
                          <div key={id} style={{
                            background:"rgba(255,255,255,0.02)",
                            border:`1px solid ${res.pred===topIdx?am.color+"55":"rgba(255,255,255,0.06)"}`,
                            borderRadius:12, padding:"14px 16px"
                          }}>
                            <div style={{ display:"flex", justifyContent:"space-between", alignItems:"flex-start", marginBottom:10 }}>
                              <div>
                                <div style={{ fontSize:12, fontWeight:700, color:am.color }}>{am.icon} {am.name}</div>
                                <div style={{ fontSize:9, color:"#333", marginTop:2 }}>{am.full}</div>
                              </div>
                              <Gauge value={conf} color={am.color} size={62} />
                            </div>
                            <div style={{
                              display:"flex", alignItems:"center", gap:8,
                              background:`${cm2.color}10`, borderRadius:8, padding:"6px 10px",
                              border:`1px solid ${cm2.color}33`, marginBottom:10
                            }}>
                              <span style={{ fontSize:18 }}>{cm2.icon}</span>
                              <div>
                                <div style={{ fontSize:11, color:cm2.color, fontWeight:700 }}>{cm2.label||cls}</div>
                                <div style={{ fontSize:9, color:"#444" }}>{(conf*100).toFixed(1)}% confidence</div>
                              </div>
                            </div>
                            {names.map((n, i) => (
                              <div key={i} style={{ display:"flex", alignItems:"center", gap:6, marginBottom:3 }}>
                                <div style={{ fontSize:9, width:14 }}>{getC(n).icon}</div>
                                <div style={{ flex:1, height:3, background:"rgba(255,255,255,0.04)", borderRadius:99 }}>
                                  <div style={{ height:"100%", width:`${(res.proba[i]||0)*100}%`,
                                    background: i===res.pred?am.color:`${am.color}33`, borderRadius:99 }} />
                                </div>
                                <div style={{ width:30, fontSize:8, color:"#333", textAlign:"right", fontFamily:"monospace" }}>
                                  {((res.proba[i]||0)*100).toFixed(0)}%
                                </div>
                              </div>
                            ))}
                          </div>
                        );
                      })}
                    </div>
                  </div>
                );
              })() : (
                <div style={{ height:"100%", minHeight:480, display:"flex", flexDirection:"column",
                  alignItems:"center", justifyContent:"center",
                  background:"rgba(255,255,255,0.01)", border:"1px dashed rgba(255,255,255,0.05)", borderRadius:16 }}>
                  <div style={{ fontSize:48, marginBottom:16, opacity:0.2 }}>🎧</div>
                  <div style={{ fontSize:13, color:"#333" }}>Upload an audio file and click Identify</div>
                  <div style={{ fontSize:10, color:"#222", marginTop:8 }}>Results from all 5 models appear here</div>
                </div>
              )}
            </div>
          </div>
        )}

        {/* ══════ COMPARE ══════ */}
        {tab === "compare" && (
          <div>
            {!results ? (
              <div style={{ textAlign:"center", padding:"80px 20px", color:"#333" }}>
                <div style={{ fontSize:40, marginBottom:16, opacity:0.2 }}>📊</div>
                <div style={{ fontSize:13 }}>Train models first</div>
                <button onClick={() => setTab("train")} style={{
                  marginTop:16, background:"rgba(0,255,136,0.07)", border:"1px solid rgba(0,255,136,0.2)",
                  borderRadius:8, padding:"10px 24px", color:"#00FF88",
                  fontFamily:"inherit", fontSize:11, cursor:"pointer"
                }}>→ Go to Training</button>
              </div>
            ) : (
              <>
                {sortedAlgos.length > 0 && (() => {
                  const [bestId, bestM] = sortedAlgos[0];
                  const am = ALG_META[bestId];
                  return (
                    <div style={{
                      background:`linear-gradient(135deg,${am.color}10,rgba(0,0,0,0.5))`,
                      border:`1.5px solid ${am.color}`, borderRadius:16,
                      padding:"18px 24px", marginBottom:24,
                      display:"flex", alignItems:"center", justifyContent:"space-between", flexWrap:"wrap", gap:16
                    }}>
                      <div>
                        <div style={{ fontSize:9, color:am.color, letterSpacing:3 }}>🏆 BEST PERFORMING MODEL</div>
                        <div style={{ fontSize:22, fontWeight:700, color:"#fff", marginTop:6 }}>
                          {am.icon} {am.full} — {bestM.accuracy}% Accuracy
                        </div>
                        <div style={{ fontSize:11, color:"#444", marginTop:4 }}>
                          {classNames.length} classes · {classNames.join(", ")}
                        </div>
                      </div>
                      <div style={{ display:"flex", gap:24 }}>
                        {[["F1",bestM.f1],["Precision",bestM.precision],["Recall",bestM.recall]].map(([l,v])=>(
                          <div key={l} style={{ textAlign:"center" }}>
                            <div style={{ fontSize:20, fontWeight:700, color:am.color, fontFamily:"monospace" }}>{v}%</div>
                            <div style={{ fontSize:9, color:"#444" }}>{l}</div>
                          </div>
                        ))}
                      </div>
                    </div>
                  );
                })()}

                <div style={{ display:"grid", gridTemplateColumns:"repeat(5,1fr)", gap:10, marginBottom:24 }}>
                  {sortedAlgos.map(([id, m], rank) => {
                    const am = ALG_META[id];
                    const rc = ["#FFD700","#C0C0C0","#CD7F32"];
                    return (
                      <div key={id} onClick={() => setSelectedAlgo(id)} style={{
                        background: selectedAlgo===id?`${am.color}10`:"rgba(255,255,255,0.02)",
                        border:`1.5px solid ${selectedAlgo===id?am.color:"rgba(255,255,255,0.06)"}`,
                        borderRadius:12, padding:"14px 14px", cursor:"pointer", transition:"all 0.2s", position:"relative"
                      }}>
                        {rank<3&&(
                          <div style={{ position:"absolute", top:8, right:8, background:rc[rank], color:"#000",
                            borderRadius:99, width:18, height:18, display:"flex", alignItems:"center",
                            justifyContent:"center", fontSize:9, fontWeight:800 }}>#{rank+1}</div>
                        )}
                        <div style={{ fontSize:13, fontWeight:700, color:am.color, marginBottom:2 }}>{am.icon} {am.name}</div>
                        <div style={{ fontSize:8, color:"#333", marginBottom:10 }}>{am.full}</div>
                        <div style={{ fontSize:20, fontWeight:700, color:"#fff", fontFamily:"monospace" }}>{m.accuracy}%</div>
                        <div style={{ fontSize:8, color:"#333", marginBottom:8 }}>ACCURACY</div>
                        <div style={{ height:4, background:"rgba(255,255,255,0.05)", borderRadius:99 }}>
                          <div style={{ height:"100%", width:`${m.accuracy}%`, background:am.color, borderRadius:99 }} />
                        </div>
                        <div style={{ display:"flex", justifyContent:"space-between", marginTop:6, fontSize:8, color:"#333" }}>
                          <span>F1: {m.f1}%</span><span>P: {m.precision}%</span>
                        </div>
                        {predictions?.predictions?.[id]&&(()=>{
                          const pr=predictions.predictions[id];
                          const cls=(predictions.class_names||classNames)[pr.pred];
                          const c=getC(cls);
                          return (
                            <div style={{ marginTop:8, padding:"4px 8px", background:`${c.color}15`,
                              borderRadius:6, border:`1px solid ${c.color}33`,
                              fontSize:9, color:c.color, textAlign:"center" }}>
                              {c.icon} {c.label||cls} · {(Math.max(...pr.proba)*100).toFixed(0)}%
                            </div>
                          );
                        })()}
                      </div>
                    );
                  })}
                </div>

                {results[selectedAlgo]&&(()=>{
                  const m=results[selectedAlgo], am=ALG_META[selectedAlgo];
                  return (
                    <div style={{ background:"rgba(255,255,255,0.02)", border:"1px solid rgba(255,255,255,0.06)", borderRadius:16, padding:24 }}>
                      <div style={{ display:"flex", alignItems:"center", gap:10, marginBottom:20 }}>
                        <div style={{ width:4, height:20, background:am.color, borderRadius:2 }} />
                        <div style={{ fontSize:14, fontWeight:700, color:am.color }}>{am.icon} {am.full}</div>
                      </div>
                      <div style={{ display:"grid", gridTemplateColumns:"1fr 1fr", gap:28, marginBottom:24 }}>
                        <div>
                          <div style={{ fontSize:9, color:"#444", letterSpacing:2, marginBottom:14 }}>PER-CLASS ACCURACY</div>
                          {classNames.map(cls=>{
                            const pc=m.per_class?.[cls]??0, c=getC(cls);
                            return (
                              <div key={cls} style={{ marginBottom:10 }}>
                                <div style={{ display:"flex", justifyContent:"space-between", fontSize:11, marginBottom:4 }}>
                                  <span>{c.icon} {c.label||cls}</span>
                                  <span style={{ color:c.color, fontFamily:"monospace" }}>{pc}%</span>
                                </div>
                                <div style={{ height:6, background:"rgba(255,255,255,0.04)", borderRadius:99 }}>
                                  <div style={{ height:"100%", width:`${pc}%`, background:c.color, borderRadius:99 }} />
                                </div>
                              </div>
                            );
                          })}
                        </div>
                        <div>
                          <div style={{ fontSize:9, color:"#444", letterSpacing:2, marginBottom:14 }}>CONFUSION MATRIX</div>
                          <Matrix matrix={m.conf_matrix} classNames={classNames} />
                        </div>
                      </div>
                      <div style={{ borderTop:"1px solid rgba(255,255,255,0.05)", paddingTop:20 }}>
                        <div style={{ fontSize:9, color:"#444", letterSpacing:2, marginBottom:14 }}>ALL MODELS ACCURACY</div>
                        {sortedAlgos.map(([id,mt])=>{
                          const a=ALG_META[id];
                          return (
                            <div key={id} style={{ display:"flex", alignItems:"center", gap:12, marginBottom:10 }}>
                              <div style={{ width:88, fontSize:11, color:id===selectedAlgo?a.color:"#444" }}>{a.icon} {a.name}</div>
                              <div style={{ flex:1, height:7, background:"rgba(255,255,255,0.04)", borderRadius:99 }}>
                                <div style={{ height:"100%", borderRadius:99, width:`${mt.accuracy}%`,
                                  background:id===selectedAlgo?a.color:"rgba(255,255,255,0.1)", transition:"width 0.8s ease" }} />
                              </div>
                              <div style={{ width:46, textAlign:"right", fontSize:11, fontFamily:"monospace" }}>{mt.accuracy}%</div>
                            </div>
                          );
                        })}
                      </div>
                      <div style={{ display:"grid", gridTemplateColumns:"repeat(4,1fr)", gap:12, marginTop:20,
                        borderTop:"1px solid rgba(255,255,255,0.05)", paddingTop:20 }}>
                        {[["Accuracy",m.accuracy],["F1 Score",m.f1],["Precision",m.precision],["Recall",m.recall]].map(([l,v])=>(
                          <div key={l} style={{ background:"rgba(255,255,255,0.02)", borderRadius:10, padding:16, textAlign:"center" }}>
                            <div style={{ fontSize:22, fontWeight:700, color:am.color, fontFamily:"monospace" }}>{v}%</div>
                            <div style={{ fontSize:11, color:"#777", marginTop:4 }}>{l}</div>
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
        * { box-sizing:border-box; margin:0; padding:0; }
        @keyframes pulse { 0%,100%{opacity:1} 50%{opacity:0.4} }
        ::-webkit-scrollbar { width:4px; }
        ::-webkit-scrollbar-thumb { background:#1a1a1a; border-radius:2px; }
      `}</style>
    </div>
  );
}
