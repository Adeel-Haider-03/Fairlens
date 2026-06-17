import { useState } from 'react'

const MODELS = [
  { id: 'RF',       name: 'Random Forest',   desc: 'Handles numeric + categorical features well',    badge: 'Classic ML' },
  { id: 'XGBoost',  name: 'XGBoost',         desc: 'High accuracy on structured data',               badge: 'Boosting' },
  { id: 'LightGBM', name: 'LightGBM',        desc: 'Fast training, great for larger datasets',       badge: 'Boosting' },
  { id: 'TabNet',   name: 'TabNet',           desc: 'Deep learning with attention for tabular data',  badge: 'Deep Learning' },
]

const MITIGATION = [
  {
    id: 'Reweighing',
    stage: 'Pre-Processing',
    stageColor: 'cyan',
    name: 'Reweighing',
    desc: 'Adjusts instance weights to ensure demographic groups are fairly represented before training.',
    paper: 'Calders & Verwer (2010)',
  },
  {
    id: 'ADB',
    stage: 'In-Processing',
    stageColor: 'amber',
    name: 'Adversarial Debiasing',
    desc: 'Uses an adversarial network to penalise the model if it relies on sensitive attributes during training.',
    paper: 'Zhang et al. (2018)',
    requires: 'Reweighing',
  },
  {
    id: 'CEO',
    stage: 'Post-Processing',
    stageColor: 'rose',
    name: 'Calibrated Equalised Odds',
    desc: 'Adjusts classification thresholds post-training to balance TPR and FPR across groups.',
    paper: 'Pleiss et al. (2017)',
    requires: 'ADB',
  },
]

const SMOTE = [
  { id: 'Standard',   desc: 'Classic SMOTE oversampling' },
  { id: 'Borderline', desc: 'Focus on borderline minority samples' },
  { id: 'ADASYN',     desc: 'Adaptive synthetic sampling' },
  { id: 'KMeans',     desc: 'Cluster-based oversampling' },
]

export default function StepModels({ onNext, onBack }) {
  const [models,       setModels]       = useState(['RF'])
  const [mitigation,   setMitigation]   = useState(['Reweighing', 'ADB', 'CEO'])
  const [smoteVars,    setSmoteVars]    = useState([])
  const [testSize,     setTestSize]     = useState(0.3)
  const [error,        setError]        = useState('')

  const toggleModel = (id) =>
    setModels(m => m.includes(id) ? m.filter(x => x!==id) : [...m, id])

  const toggleMit = (id) => {
    setMitigation(m => {
      if (m.includes(id)) {
        // Remove this and anything that requires it
        const toRemove = [id]
        if (id === 'Reweighing') toRemove.push('ADB', 'CEO')
        if (id === 'ADB')        toRemove.push('CEO')
        return m.filter(x => !toRemove.includes(x))
      } else {
        // Add prerequisites
        const toAdd = [id]
        if (id === 'CEO' && !m.includes('ADB'))        toAdd.unshift('ADB')
        if (id === 'ADB' && !m.includes('Reweighing')) toAdd.unshift('Reweighing')
        if (id === 'CEO' && !m.includes('Reweighing')) toAdd.unshift('Reweighing')
        return [...new Set([...m, ...toAdd])]
      }
    })
  }

  const toggleSmote = (id) =>
    setSmoteVars(s => s.includes(id) ? s.filter(x=>x!==id) : [...s, id])

  const handleNext = () => {
    if (models.length === 0)     { setError('Select at least one model'); return }
    if (mitigation.length === 0 && smoteVars.length === 0) {
      setError('Select at least one mitigation technique'); return
    }
    onNext({ models, mitigation_steps: mitigation, smote_variants: smoteVars, test_size: testSize })
  }

  const colorMap = { cyan: 'text-cyan border-cyan/30 bg-cyan/10', amber: 'text-amber border-amber/30 bg-amber/5', rose: 'text-rose border-rose/30 bg-rose/5' }
  const dotMap   = { cyan: 'bg-cyan', amber: 'bg-amber', rose: 'bg-rose' }

  return (
    <div className="fade-up max-w-3xl mx-auto space-y-6">
      <div className="text-center mb-8">
        <h2 className="font-display font-bold text-2xl text-white mb-2">Select Models & Mitigation</h2>
        <p className="text-ink-400 font-body text-sm">Choose which models to train and which bias mitigation pipeline to apply.</p>
      </div>

      {/* Models */}
      <div className="card-glow space-y-4">
        <p className="label">ML Models</p>
        <div className="grid grid-cols-2 gap-3">
          {MODELS.map(m => {
            const sel = models.includes(m.id)
            return (
              <button key={m.id} onClick={() => toggleModel(m.id)}
                className={`text-left p-4 rounded-xl border transition-all duration-200 ${
                  sel ? 'bg-cyan/10 border-cyan/40' : 'bg-ink-800 border-ink-700 hover:border-ink-500'
                }`}>
                <div className="flex items-start justify-between mb-2">
                  <span className={`font-display font-bold text-sm ${sel ? 'text-cyan' : 'text-white'}`}>{m.name}</span>
                  <div className={`w-5 h-5 rounded flex items-center justify-center flex-shrink-0 ml-2 ${sel ? 'bg-cyan' : 'bg-ink-700 border border-ink-600'}`}>
                    {sel && <svg width="10" height="10" viewBox="0 0 10 10" fill="none"><path d="M1.5 5l2.5 2.5 5-5" stroke="#0A0E1A" strokeWidth="1.8" strokeLinecap="round"/></svg>}
                  </div>
                </div>
                <p className="text-[11px] text-ink-400 font-body">{m.desc}</p>
                <span className="mt-2 inline-block tag border-ink-600 text-ink-400 text-[10px]">{m.badge}</span>
              </button>
            )
          })}
        </div>
      </div>

      {/* Mitigation pipeline */}
      <div className="card-glow space-y-4">
        <div>
          <p className="label">Bias Mitigation Pipeline</p>
          <p className="text-xs text-ink-400 font-body">Applied in sequence: Pre → In → Post processing. Selecting a stage auto-adds its required predecessors — e.g. selecting CEO also enables ADB and Reweighing.</p>
        </div>
        <div className="space-y-3">
          {MITIGATION.map(m => {
            const sel = mitigation.includes(m.id)
            return (
              <button key={m.id} onClick={() => toggleMit(m.id)}
                className={`w-full text-left p-4 rounded-xl border transition-all duration-200 flex gap-4 ${
                  sel ? `border-${m.stageColor}/30 bg-${m.stageColor}/5` : 'bg-ink-800 border-ink-700 hover:border-ink-600'
                }`}>
                <div className={`w-5 h-5 rounded flex items-center justify-center flex-shrink-0 mt-0.5 ${
                  sel ? `bg-${m.stageColor}` : 'bg-ink-700 border border-ink-600'
                }`}>
                  {sel && <svg width="10" height="10" viewBox="0 0 10 10" fill="none"><path d="M1.5 5l2.5 2.5 5-5" stroke="#0A0E1A" strokeWidth="1.8" strokeLinecap="round"/></svg>}
                </div>
                <div className="flex-1 min-w-0">
                  <div className="flex items-center gap-2 mb-1">
                    <span className={`tag text-[10px] ${colorMap[m.stageColor]}`}>
                      <span className={`w-1 h-1 rounded-full ${dotMap[m.stageColor]}`} />
                      {m.stage}
                    </span>
                    <span className="font-display font-bold text-sm text-white">{m.name}</span>
                  </div>
                  <p className="text-[11px] text-ink-400 font-body">{m.desc}</p>
                  <div className="flex items-center gap-3 mt-1">
                    <p className="text-[10px] text-ink-400 font-mono">{m.paper}</p>
                    {m.requires && (
                      <span className="text-[10px] text-ink-500 font-mono border border-ink-700 rounded px-1.5 py-0.5">
                        requires {m.requires}
                      </span>
                    )}
                  </div>
                </div>
              </button>
            )
          })}
        </div>
      </div>

      {/* SMOTE variants */}
      <div className="card-glow space-y-4">
        <div>
          <p className="label">SMOTE Variants <span className="text-ink-400 font-normal normal-case tracking-normal">(optional — for comparison only)</span></p>
          <p className="text-xs text-ink-400 font-body">The paper shows SMOTE often worsens fairness. Include to verify this finding.</p>
        </div>
        <div className="grid grid-cols-2 gap-2">
          {SMOTE.map(s => {
            const sel = smoteVars.includes(s.id)
            return (
              <button key={s.id} onClick={() => toggleSmote(s.id)}
                className={`text-left px-3 py-2.5 rounded-xl border text-xs transition-all duration-200 flex items-center gap-2 ${
                  sel ? 'bg-amber/5 border-amber/30 text-amber' : 'bg-ink-800 border-ink-700 text-ink-400 hover:border-ink-600'
                }`}>
                <div className={`w-3.5 h-3.5 rounded flex-shrink-0 flex items-center justify-center ${sel ? 'bg-amber' : 'bg-ink-700 border border-ink-600'}`}>
                  {sel && <svg width="8" height="8" viewBox="0 0 8 8" fill="none"><path d="M1 4l2 2 4-4" stroke="#0A0E1A" strokeWidth="1.5" strokeLinecap="round"/></svg>}
                </div>
                <div>
                  <div className="font-bold font-display">{s.id}</div>
                  <div className="text-[10px] opacity-60 font-body">{s.desc}</div>
                </div>
              </button>
            )
          })}
        </div>
      </div>

      {/* Test size */}
      <div className="card-glow space-y-3">
        <p className="label">Test Split Size</p>
        <div className="flex items-center gap-4">
          <input type="range" min="0.1" max="0.5" step="0.05" value={testSize}
                 onChange={e => setTestSize(parseFloat(e.target.value))}
                 className="flex-1 accent-cyan" />
          <span className="font-mono text-cyan font-bold text-sm w-12 text-right">{Math.round(testSize*100)}%</span>
        </div>
        <p className="text-[11px] text-ink-500 font-body">
          {Math.round((1-testSize)*100)}% train / {Math.round(testSize*100)}% test · Standard split is 70/30
        </p>
      </div>

      {error && <div className="card border-rose/30 bg-rose/5 text-rose text-sm font-body">{error}</div>}

      <div className="flex gap-3">
        <button onClick={onBack} className="btn-ghost flex-1">← Back</button>
        <button onClick={handleNext} className="btn-primary flex-[2]">Start Training →</button>
      </div>
    </div>
  )
}
