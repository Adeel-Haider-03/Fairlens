import { useState, useMemo } from 'react'

export default function StepConfigure({ uploadData, onNext, onBack }) {
  const { columns } = uploadData

  const [target,    setTarget]    = useState('')
  const [favorable, setFavorable] = useState('')
  const [threshold, setThreshold]  = useState('')
  const [protected_, setProtected] = useState('')
  const [privileged, setPrivileged]= useState('')
  const [features,  setFeatures]  = useState([])
  const [error, setError]         = useState('')

  // Derived
  const targetValues = useMemo(() => {
    if (!target) return []
    const col = columns.find(c => c.name === target)
    return col ? col.sample_values : []
  }, [target, columns])

  const protectedCol   = useMemo(() => columns.find(c => c.name === protected_), [protected_, columns])
  const isNumericProt  = protectedCol?.is_numeric ?? false

  const privValues = useMemo(() => {
    if (!protected_) return []
    const col = columns.find(c => c.name === protected_)
    return col ? col.sample_values : []
  }, [protected_, columns])

  const availableFeatures = useMemo(() =>
    columns.filter(c => c.name !== target && c.name !== protected_),
    [columns, target, protected_]
  )

  const toggleFeature = (name) =>
    setFeatures(f => f.includes(name) ? f.filter(x => x !== name) : [...f, name])

  const selectAll = () => setFeatures(availableFeatures.map(c => c.name))
  const clearAll  = () => setFeatures([])

  const validate = () => {
    if (!target)     return 'Please select a target column'
    if (!protected_) return 'Please select a protected attribute'
    if (!privileged) return 'Please select the privileged group value'
    if (features.length === 0) return 'Please select at least one feature column'
    return null
  }

  const handleNext = () => {
    const err = validate()
    if (err) { setError(err); return }
    onNext({
      target_column:      target,
      protected_attribute: protected_,
      privileged_value:   privileged,
      favorable_value:    favorable || null,
      protected_threshold: threshold || null,
      feature_columns:    features,
    })
  }

  return (
    <div className="fade-up max-w-3xl mx-auto space-y-6">
      <div className="text-center mb-8">
        <h2 className="font-display font-bold text-2xl text-white mb-2">Configure Variables</h2>
        <p className="text-ink-400 font-body text-sm">Tell FairLens which columns to analyse for fairness.</p>
      </div>

      <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
        {/* Target column */}
        <div className="card-glow space-y-3">
          <div className="flex items-center gap-2 mb-1">
            <div className="w-2 h-2 rounded-full bg-cyan" />
            <p className="label mb-0">Target Column <span className="text-rose">*</span></p>
          </div>
          <p className="text-xs text-ink-400 font-body">The outcome you want to predict (e.g. income, loan approval)</p>
          <select value={target} onChange={e => { setTarget(e.target.value); setFavorable('') }} className="input-field">
            <option value="">— Select target —</option>
            {columns.map(c => <option key={c.name} value={c.name}>{c.name}</option>)}
          </select>
          {target && (
            <div className="tag border-cyan/20 text-cyan bg-cyan/10 text-[11px] w-fit">
              {columns.find(c=>c.name===target)?.unique_count} unique values
            </div>
          )}
        </div>

        {/* Positive class selector */}
        {target && (
          <div className="card-glow space-y-3 fade-up" style={{gridColumn:'1/-1'}}>
            <div className="flex items-center gap-2 mb-1">
              <div className="w-2 h-2 rounded-full bg-emerald" />
              <p className="label mb-0">Positive Class <span className="text-ink-500 font-normal text-[11px]">(optional — auto-detected if not set)</span></p>
            </div>
            <p className="text-xs text-ink-400 font-body">
              Which value means a <strong className="text-white">positive / favorable outcome</strong>?
              e.g. for credit risk: <code className="text-amber">2</code> (bad) or for income: <code className="text-amber">&gt;50K</code>
            </p>
            <div className="flex flex-wrap gap-2 items-center">
              <button onClick={() => setFavorable('')}
                className={`px-4 py-2 rounded-xl font-display text-xs font-bold border transition-all ${
                  favorable === '' ? 'bg-ink-600 border-ink-400 text-white' : 'bg-ink-800 border-ink-700 text-ink-400 hover:border-ink-500'
                }`}>
                Auto-detect
              </button>
              {targetValues.map(v => (
                <button key={v} onClick={() => setFavorable(v)}
                  className={`px-4 py-2 rounded-xl font-display text-xs font-bold border transition-all ${
                    favorable === v ? 'bg-emerald/15 border-emerald text-emerald' : 'bg-ink-800 border-ink-700 text-ink-400 hover:border-emerald/50'
                  }`}>
                  {v}
                </button>
              ))}
            </div>
            {favorable
              ? <p className="text-xs text-emerald">✓ <strong>{favorable}</strong> will be encoded as class 1 (positive)</p>
              : <p className="text-xs text-ink-500">Auto-detect will infer the positive class from data patterns</p>
            }
          </div>
        )}

        {/* Protected attribute */}
        <div className="card-glow space-y-3">
          <div className="flex items-center gap-2 mb-1">
            <div className="w-2 h-2 rounded-full bg-amber" />
            <p className="label mb-0">Protected Attribute <span className="text-rose">*</span></p>
          </div>
          <p className="text-xs text-ink-400 font-body">Sensitive attribute to check bias against (e.g. race, gender, age)</p>
          <select value={protected_} onChange={e => { setProtected(e.target.value); setPrivileged('') }} className="input-field">
            <option value="">— Select attribute —</option>
            {columns.filter(c => c.name !== target).map(c =>
              <option key={c.name} value={c.name}>{c.name}</option>
            )}
          </select>
          {protected_ && (
            <div className="tag border-amber/20 text-amber bg-amber/5 text-[11px] w-fit">
              {columns.find(c=>c.name===protected_)?.unique_count} groups
            </div>
          )}
        </div>
      </div>

      {/* Privileged value */}
      {protected_ && (
        <div className="card-glow space-y-3 fade-up">
          <div className="flex items-center gap-2 mb-1">
            <div className="w-2 h-2 rounded-full bg-rose" />
            <p className="label mb-0">Privileged Group Value <span className="text-rose">*</span></p>
          </div>
          <p className="text-xs text-ink-400 font-body">
            Which value represents the privileged group? The bias analysis will compare others against this group.
          </p>
          <div className="flex flex-wrap gap-2">
            {privValues.map(v => (
              <button key={v}
                onClick={() => setPrivileged(v)}
                className={`px-4 py-2 rounded-xl font-display text-xs font-bold border transition-all duration-200 ${
                  privileged === v
                    ? 'bg-rose/10 border-rose text-rose'
                    : 'bg-ink-800 border-ink-600 text-ink-300 hover:border-rose/50'
                }`}>
                {v}
              </button>
            ))}
          </div>
          {privileged && (
            <p className="text-xs text-ink-400">
              Non-<span className="text-rose font-bold">{privileged}</span> values will be treated as unprivileged groups.
            </p>
          )}
        </div>
      )}

      {/* Numeric protected attribute threshold */}
      {protected_ && isNumericProt && (
        <div className="card-glow space-y-3 fade-up">
          <div className="flex items-center gap-2 mb-1">
            <div className="w-2 h-2 rounded-full bg-purple-400" />
            <p className="label mb-0">Numeric Threshold <span className="text-ink-500 font-normal text-[11px]">(optional)</span></p>
          </div>
          <p className="text-xs text-ink-400 font-body">
            <strong className="text-white">{protected_}</strong> is numeric. Define a cut-off to split into two groups —
            values <strong className="text-white">≥ threshold</strong> = privileged, values <strong className="text-white">&lt; threshold</strong> = unprivileged.
            Leave blank to use the median automatically.
          </p>
          <div className="flex items-center gap-3">
            <input
              type="number"
              value={threshold}
              onChange={e => setThreshold(e.target.value)}
              placeholder={`e.g. 25  (median auto-used if blank)`}
              className="input-field flex-1"
            />
            {threshold && (
              <button onClick={() => setThreshold('')} className="btn-ghost text-xs py-2 px-3">Clear</button>
            )}
          </div>
          {threshold && (
            <p className="text-xs text-emerald">
              ✓ <strong>{protected_} ≥ {threshold}</strong> = privileged · <strong>&lt; {threshold}</strong> = unprivileged
            </p>
          )}
          {!threshold && (
            <p className="text-xs text-ink-500">Auto: median split will be used</p>
          )}
        </div>
      )}

      {/* Feature selection */}
      <div className="card-glow space-y-4">
        <div className="flex items-center justify-between">
          <div>
            <div className="flex items-center gap-2 mb-1">
              <div className="w-2 h-2 rounded-full bg-emerald" />
              <p className="label mb-0">Feature Columns <span className="text-rose">*</span></p>
            </div>
            <p className="text-xs text-ink-400 font-body">Select inputs the model will train on</p>
          </div>
          <div className="flex gap-2">
            <button onClick={selectAll} className="btn-ghost text-xs py-1.5 px-3">All</button>
            <button onClick={clearAll}  className="btn-ghost text-xs py-1.5 px-3">Clear</button>
          </div>
        </div>

        <div className="grid grid-cols-2 sm:grid-cols-3 gap-2">
          {availableFeatures.map(col => {
            const selected = features.includes(col.name)
            return (
              <button key={col.name} onClick={() => toggleFeature(col.name)}
                className={`flex items-center gap-2 px-3 py-2.5 rounded-xl border text-xs font-body text-left transition-all duration-200 ${
                  selected
                    ? 'bg-emerald/10 border-emerald/40 text-emerald'
                    : 'bg-ink-800 border-ink-700 text-ink-400 hover:border-ink-500'
                }`}>
                <div className={`w-3.5 h-3.5 rounded flex items-center justify-center flex-shrink-0 ${
                  selected ? 'bg-emerald' : 'bg-ink-700 border border-ink-600'
                }`}>
                  {selected && <svg width="8" height="8" viewBox="0 0 8 8" fill="none">
                    <path d="M1 4l2 2 4-4" stroke="#0A0E1A" strokeWidth="1.5" strokeLinecap="round"/>
                  </svg>}
                </div>
                <div className="min-w-0">
                  <div className="truncate font-medium text-[11px]">{col.name}</div>
                  <div className="text-[10px] opacity-60">{col.is_numeric ? 'numeric' : 'categorical'}</div>
                </div>
              </button>
            )
          })}
        </div>

        {features.length > 0 && (
          <p className="text-xs text-ink-400">{features.length} features selected</p>
        )}
      </div>

      {error && (
        <div className="card border-rose/30 bg-rose/5 text-rose text-sm font-body">{error}</div>
      )}

      <div className="flex gap-3">
        <button onClick={onBack} className="btn-ghost flex-1">← Back</button>
        <button onClick={handleNext} className="btn-primary flex-[2]">Continue to Models →</button>
      </div>
    </div>
  )
}
