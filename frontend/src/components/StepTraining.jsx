import { useState, useEffect, useRef } from 'react'

const API = 'http://localhost:8000'

export default function StepTraining({ uploadData, config, modelConfig, onDone, onBack }) {
  const [jobId,      setJobId]      = useState(null)
  const [status,     setStatus]     = useState('checking') // checking | cache_found | idle | running | complete | failed
  const [progress,   setProgress]   = useState(0)
  const [step,       setStep]       = useState('Checking for cached results...')
  const [logs,       setLogs]       = useState([])
  const [error,      setError]      = useState(null)
  const [cachedMeta, setCachedMeta] = useState(null)   // meta from cache hit
  const [sessionName,setSessionName]= useState('')
  const logRef  = useRef(null)
  const pollRef = useRef(null)

  // ── On mount: auto-check cache ──────────────────────────────────────────
  useEffect(() => {
    checkCache()
    return () => clearInterval(pollRef.current)
  }, [])

  useEffect(() => {
    if (logRef.current) logRef.current.scrollTop = logRef.current.scrollHeight
  }, [logs])

  // Auto-suggest session name from config if blank
  const suggestedName = () => {
    const ds = uploadData?.filename?.replace('.csv','') || 'Dataset'
    const pa = config?.protected_attribute || ''
    const steps = modelConfig?.mitigation_steps || []
    const stage = steps.includes('CEO') ? 'Full Pipeline' : steps.includes('ADB') ? 'Reweigh+ADB' : 'Reweighing'
    return `${ds} — ${pa} — ${stage}`
  }

  const buildPayload = () => ({
    dataset_id:          uploadData.dataset_id,
    target_column:       config.target_column,
    protected_attribute: config.protected_attribute,
    privileged_value:    config.privileged_value,
    favorable_value:     config.favorable_value    || null,
    protected_threshold: config.protected_threshold || null,
    feature_columns:     config.feature_columns,
    models:              modelConfig.models,
    mitigation_steps:    modelConfig.mitigation_steps,
    smote_variants:      modelConfig.smote_variants,
    test_size:           modelConfig.test_size,
    session_name:        sessionName || suggestedName(),
  })

  const checkCache = async () => {
    setStatus('checking')
    try {
      const res  = await fetch(`${API}/api/check-cache`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(buildPayload()),
      })
      const data = await res.json()
      if (data.cached) {
        setCachedMeta(data.meta)
        setStatus('cache_found')
      } else {
        setStatus('idle')
      }
    } catch {
      setStatus('idle')
    }
  }

  // ── Load cached results instantly ──────────────────────────────────────
  const loadFromCache = async () => {
    setStatus('running')
    setStep('Loading cached results...')
    setProgress(80)
    try {
      const res     = await fetch(`${API}/api/sessions/${cachedMeta.session_id}`)
      const results = await res.json()
      setProgress(100)
      setStep('Loaded from cache!')
      setStatus('complete')
      setTimeout(() => onDone(cachedMeta.session_id, results), 600)
    } catch (e) {
      setError(e.message)
      setStatus('idle')
    }
  }

  // ── Train fresh ────────────────────────────────────────────────────────
  const startTraining = async () => {
    setStatus('running')
    setError(null)
    try {
      const res  = await fetch(`${API}/api/train`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(buildPayload()),
      })
      const data = await res.json()
      if (!res.ok) throw new Error(data.detail || 'Failed to start training')
      setJobId(data.job_id)
      pollRef.current = setInterval(() => pollStatus(data.job_id), 1500)
    } catch (e) {
      setError(e.message)
      setStatus('failed')
    }
  }

  const pollStatus = async (jid) => {
    try {
      const res  = await fetch(`${API}/api/status/${jid}`)
      const data = await res.json()
      setProgress(data.progress)
      setStep(data.current_step)
      setLogs(data.logs || [])
      if (data.status === 'complete') {
        clearInterval(pollRef.current)
        setStatus('complete')
        const res2    = await fetch(`${API}/api/results/${jid}`)
        const results = await res2.json()
        setTimeout(() => onDone(jid, results), 800)
      } else if (data.status === 'failed') {
        clearInterval(pollRef.current)
        setStatus('failed')
        setError(data.current_step)
      }
    } catch { /* keep polling */ }
  }

  const stages = [
    { label: 'Dataset preparation',    threshold: 10 },
    { label: 'Bias baseline analysis', threshold: 20 },
    { label: 'SMOTE experiments',      threshold: 40 },
    { label: 'Model training',         threshold: 80 },
    { label: 'Saving to cache',        threshold: 100 },
  ]
  const getStageStatus = (t) => progress >= t ? 'done' : progress >= t - 20 ? 'active' : 'pending'

  // ── Render ─────────────────────────────────────────────────────────────
  return (
    <div className="fade-up max-w-2xl mx-auto space-y-6">
      <div className="text-center mb-8">
        <h2 className="font-display font-bold text-2xl text-white mb-2">
          {status === 'complete'    ? '✓ Ready'
           : status === 'cache_found' ? '⚡ Cached Results Found'
           : 'Model Training'}
        </h2>
        <p className="text-ink-400 font-body text-sm">
          {status === 'checking'    && 'Checking for previously saved results...'}
          {status === 'cache_found' && 'An identical run was saved. Load instantly or retrain fresh.'}
          {status === 'idle'        && 'No cached results found. Configure a name and launch training.'}
          {status === 'running'     && 'Training in progress — results will be saved automatically.'}
          {status === 'complete'    && 'Done! Loading results dashboard...'}
          {status === 'failed'      && 'Training failed. Check the error below.'}
        </p>
      </div>

      {/* Checking spinner */}
      {status === 'checking' && (
        <div className="card-glow flex flex-col items-center py-12 gap-4">
          <div className="w-10 h-10 rounded-full border-2 border-cyan/30 border-t-cyan animate-spin" />
          <span className="font-display text-sm text-ink-400">Scanning session cache...</span>
        </div>
      )}

      {/* ── CACHE HIT ── */}
      {status === 'cache_found' && cachedMeta && (
        <div className="space-y-4 fade-up">
          <div className="card-glow border-emerald/30 bg-emerald/5 space-y-4">
            <div className="flex items-center gap-3">
              <div className="w-10 h-10 rounded-xl bg-emerald/20 flex items-center justify-center">
                <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="#00E676" strokeWidth="2" strokeLinecap="round">
                  <path d="M22 11.08V12a10 10 0 11-5.93-9.14"/><polyline points="22 4 12 14.01 9 11.01"/>
                </svg>
              </div>
              <div>
                <p className="font-display font-bold text-emerald text-sm">Exact match found in cache</p>
                <p className="font-body text-xs text-ink-400">Same dataset + same config → same results</p>
              </div>
            </div>
            <div className="grid grid-cols-2 gap-3 text-xs">
              {[
                { label: 'Session',   value: cachedMeta.session_name },
                { label: 'Saved',     value: new Date(cachedMeta.created_at).toLocaleString() },
                { label: 'Models',    value: cachedMeta.models?.join(', ') },
                { label: 'Best SPD',  value: cachedMeta.best_spd?.toFixed(4) ?? '—' },
                { label: 'Best Acc',  value: cachedMeta.best_accuracy ? `${(cachedMeta.best_accuracy*100).toFixed(1)}%` : '—' },
                { label: 'Best Model',value: cachedMeta.best_model ?? '—' },
              ].map(item => (
                <div key={item.label} className="bg-ink-800/90 rounded-lg p-2.5">
                  <p className="text-[10px] font-display uppercase tracking-wider text-ink-500 mb-0.5">{item.label}</p>
                  <p className="font-mono text-emerald/90">{item.value}</p>
                </div>
              ))}
            </div>
            <button onClick={loadFromCache} className="btn-primary w-full">
              ⚡ Load Instantly (no retraining)
            </button>
          </div>

          <div className="card border-ink-700 text-center py-4 space-y-3">
            <p className="text-xs text-ink-400 font-body">Want different results? You can retrain from scratch.</p>
            <button onClick={() => setStatus('idle')} className="btn-ghost text-xs">
              ↺ Retrain with fresh models
            </button>
          </div>
        </div>
      )}

      {/* ── IDLE: ready to train ── */}
      {status === 'idle' && (
        <div className="space-y-4 fade-up">
          <div className="card-glow space-y-4">
            <p className="label">Configuration Summary</p>
            <div className="grid grid-cols-2 gap-3">
              {[
                { label: 'Target',     value: config.target_column },
                { label: 'Protected',  value: config.protected_attribute },
                { label: 'Privileged', value: String(config.privileged_value) },
                { label: 'Features',   value: `${config.feature_columns.length} selected` },
                { label: 'Models',     value: modelConfig.models.join(', ') },
                { label: 'Pipeline',   value: modelConfig.mitigation_steps.join(' → ') || 'None' },
              ].map(item => (
                <div key={item.label}>
                  <p className="text-[10px] font-display font-bold uppercase tracking-widest text-ink-500 mb-0.5">{item.label}</p>
                  <p className="text-sm font-mono text-cyan/90 truncate">{item.value}</p>
                </div>
              ))}
            </div>
          </div>

          {/* Session name */}
          <div className="card-glow space-y-2">
            <label className="label">Session Name <span className="text-ink-400 font-normal normal-case tracking-normal">(auto-generated if blank)</span></label>
            <input
              type="text"
              value={sessionName}
              onChange={e => setSessionName(e.target.value)}
              placeholder={suggestedName()}
              className="input-field"
            />
            <p className="text-[11px] text-ink-500 font-body">
              Auto-generated: <span className="text-ink-300 font-mono text-[10px]">{suggestedName()}</span>
            </p>
          </div>

          <div className="flex gap-3">
            <button onClick={onBack} className="btn-ghost flex-1">← Back</button>
            <button onClick={startTraining} className="btn-primary flex-[2]">⚡ Launch Training</button>
          </div>
        </div>
      )}

      {/* ── RUNNING ── */}
      {(status === 'running' || status === 'complete') && (
        <div className="card-glow space-y-6">
          <div className="space-y-2">
            <div className="flex justify-between items-center">
              <span className="font-display text-xs text-cyan">{step}</span>
              <span className="font-mono text-xs text-ink-400">{progress}%</span>
            </div>
            <div className="h-2 bg-ink-700 rounded-full overflow-hidden">
              <div className="h-full rounded-full transition-all duration-700"
                style={{ width: `${progress}%`, background: 'linear-gradient(90deg,#00bcd4,#00E5FF)', boxShadow:'0 0 12px rgba(0,229,255,0.5)' }}
              />
            </div>
          </div>
          <div className="space-y-2">
            {stages.map(stage => {
              const s = getStageStatus(stage.threshold)
              return (
                <div key={stage.label} className="flex items-center gap-3">
                  <div className={`w-5 h-5 rounded-full flex items-center justify-center flex-shrink-0 transition-all ${
                    s==='done' ? 'bg-emerald' : s==='active' ? 'bg-cyan pulse-glow' : 'bg-ink-700 border border-ink-600'
                  }`}>
                    {s==='done'   && <svg width="10" height="10" viewBox="0 0 10 10" fill="none"><path d="M1.5 5l2.5 2.5 5-5" stroke="#0A0E1A" strokeWidth="1.8" strokeLinecap="round"/></svg>}
                    {s==='active' && <div className="w-2 h-2 bg-ink-900 rounded-full"/>}
                  </div>
                  <span className={`font-body text-sm ${s==='done'?'text-emerald':s==='active'?'text-cyan':'text-ink-400'}`}>{stage.label}</span>
                </div>
              )
            })}
          </div>
        </div>
      )}

      {/* Live logs */}
      {logs.length > 0 && (
        <div className="card bg-ink-900 border-ink-700">
          <p className="label text-[10px] mb-2">Live Output</p>
          <div ref={logRef} className="h-36 overflow-y-auto font-mono text-[11px] text-ink-400 space-y-0.5 leading-relaxed">
            {logs.map((line, i) => (
              <div key={i} className="hover:text-ink-200 transition-colors">
                <span className="text-ink-500 select-none mr-2">&gt;</span>{line}
              </div>
            ))}
          </div>
        </div>
      )}

      {error && (
        <div className="card border-rose/30 bg-rose/5 text-rose text-sm font-body">
          <strong className="font-display">Error:</strong> {error}
        </div>
      )}

      {status === 'failed' && (
        <div className="flex gap-3">
          <button onClick={onBack} className="btn-ghost flex-1">← Back</button>
          <button onClick={startTraining} className="btn-primary flex-[2]">Retry</button>
        </div>
      )}
    </div>
  )
}
