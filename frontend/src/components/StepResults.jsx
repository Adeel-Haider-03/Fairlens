import { useState, useEffect } from 'react'
import {
  RadarChart, Radar, PolarGrid, PolarAngleAxis, PolarRadiusAxis,
  BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, Legend,
  ResponsiveContainer, LineChart, Line, ReferenceLine, Cell,
} from 'recharts'

// ─── Constants ────────────────────────────────────────────────────────────────
const STAGE_KEY   = ['original','reweigh','reweigh_adb','reweigh_adb_ceo']
const STAGE_LABEL = { original:'Baseline', reweigh:'Reweighing', reweigh_adb:'Reweigh + ADB', reweigh_adb_ceo:'Full Pipeline' }
const STAGE_COLOR = { original:'#4a5f96', reweigh:'#FFB830', reweigh_adb:'#00E676', reweigh_adb_ceo:'#00E5FF' }
const STAGE_TAG   = { original:'No Mitigation', reweigh:'Pre-Processing', reweigh_adb:'In-Processing', reweigh_adb_ceo:'Post-Processing' }
const MC = { RF:'#00E5FF', XGBoost:'#00E676', LightGBM:'#FFB830', TabNet:'#FF4D6D' }
const MI = { RF:'🌲', XGBoost:'⚡', LightGBM:'💡', TabNet:'🧠' }
const TT = { background:'#0D1228', border:'1px solid rgba(0,229,255,0.15)', borderRadius:10, fontSize:11, fontFamily:'Space Mono,monospace', padding:'8px 12px' }

const fmt       = (v, d=4) => (v == null || isNaN(v)) ? '—' : Number(v).toFixed(d)
const fmtDI     = (v) => {
  if (v == null || isNaN(v)) return { text: '—', title: 'DI is undefined — one group had zero positive predictions in this split' }
  return { text: Number(v).toFixed(4), title: null }
}
// Render DI value with tooltip when undefined
const DiCell = ({ v, fair }) => {
  const d = fmtDI(v)
  return (
    <span
      title={d.title || undefined}
      className={`rounded px-1.5 py-0.5 text-[10px] font-bold inline-block cursor-help ${
        d.title ? 'bg-ink-700/70 text-ink-500 border border-dashed border-ink-600'
        : fair ? 'bg-emerald-500/15 text-emerald-400'
        : 'bg-red-500/15 text-red-400'
      }`}>
      {d.text}
    </span>
  )
}
const pct       = v => (v == null || isNaN(v)) ? '—' : `${(v * 100).toFixed(1)}%`
const isFairSPD = v => v != null && Math.abs(v) < 0.05
const isFairDI  = v => v != null && v >= 0.8 && v <= 1.25
const isFair    = (v, m) => m === 'spd' ? isFairSPD(v) : m === 'di' ? isFairDI(v) : (v != null && Math.abs(v) < 0.05)

// ─── Animated number ──────────────────────────────────────────────────────────
function AnimNum({ value, decimals=4, color='#00E5FF', size='text-3xl', prefix='' }) {
  const num = parseFloat(value)
  const [v, setV] = useState(0)
  useEffect(() => {
    if (isNaN(num)) return
    let s = null
    const tick = ts => { if (!s) s = ts; const p = Math.min((ts-s)/1200,1); setV(Math.abs(num)*p); if (p<1) requestAnimationFrame(tick) }
    requestAnimationFrame(tick)
  }, [num])
  if (isNaN(num)) return <span className={`font-mono font-black ${size}`} style={{color}}>{value}</span>
  return <span className={`font-mono font-black ${size}`} style={{color}}>{num < 0 ? '−' : prefix}{v.toFixed(decimals)}</span>
}

// ─── Hero Banner ──────────────────────────────────────────────────────────────
function HeroBanner({ results, config }) {
  const { summary, mitigation_results, baseline } = results
  const models = Object.keys(mitigation_results || {})

  const bestCEO = models.reduce((best, m) => {
    const ceo = mitigation_results[m]?.reweigh_adb_ceo
    if (!ceo) return best
    return (!best || Math.abs(ceo.spd) < Math.abs(best.spd)) ? { ...ceo, model: m } : best
  }, null)

  const spdBefore = Math.abs(baseline?.spd || 0)
  const spdAfter  = Math.abs(bestCEO?.spd || 0)
  const reduction = spdBefore > 0 ? ((spdBefore - spdAfter) / spdBefore * 100).toFixed(0) : 0
  const fullyFair = isFairSPD(bestCEO?.spd) && isFairDI(bestCEO?.di)

  return (
    <div className="relative rounded-2xl overflow-hidden border border-cyan/15 p-6"
      style={{background:'linear-gradient(135deg,#0D1228 0%,#0a1528 50%,#0D1228 100%)'}}>
      <div className="absolute inset-0 pointer-events-none overflow-hidden">
        <div className="absolute -top-20 -right-20 w-80 h-80 rounded-full opacity-5"
          style={{background:'radial-gradient(circle,#00E5FF,transparent 70%)'}}/>
        <div className="absolute -bottom-20 -left-20 w-60 h-60 rounded-full opacity-5"
          style={{background:'radial-gradient(circle,#00E676,transparent 70%)'}}/>
      </div>
      <div className="relative z-10 grid grid-cols-2 sm:grid-cols-4 gap-5">

        {/* Bias reduction */}
        <div className="col-span-2 sm:col-span-1 flex flex-col gap-1">
          <p className="label text-[10px] mb-1">Bias Reduction</p>
          <AnimNum value={reduction} decimals={0} color="#00E676" size="text-5xl"/>
          <p className="font-mono text-xs text-emerald-400/60">% |SPD| removed</p>
          <p className="text-[10px] text-ink-500 font-body mt-1">Full pipeline · best: {bestCEO?.model}</p>
          <div className={`inline-flex items-center gap-1.5 px-2.5 py-1.5 rounded-lg text-xs font-display font-bold self-start mt-2 ${
            fullyFair ? 'bg-emerald-500/15 text-emerald-400 border border-emerald-500/30'
            : isFairSPD(bestCEO?.spd) ? 'bg-cyan/10 text-cyan border border-cyan/30'
            : 'bg-amber-500/15 text-amber-400 border border-amber-500/30'
          }`}>
            {fullyFair ? '✓ Fully Fair' : isFairSPD(bestCEO?.spd) ? '✓ SPD Fair Zone' : '~ Bias Reduced'}
          </div>
        </div>

        {/* Initial SPD */}
        <div className="flex flex-col gap-1">
          <p className="label text-[10px] mb-1">Initial SPD</p>
          <AnimNum value={fmt(baseline?.spd)} decimals={4} color="#FF4D6D" size="text-2xl"/>
          <p className="text-[10px] text-ink-500 font-body mt-1">No mitigation applied</p>
          <div className="h-1 bg-ink-700 rounded-full mt-1 overflow-hidden">
            <div className="h-full bg-rose-400 rounded-full" style={{width:`${Math.min(spdBefore*600,100)}%`}}/>
          </div>
          <p className="text-[9px] text-ink-400 font-mono mt-1">
            {isFairSPD(baseline?.spd) ? '✓ Already in fair zone' : '✗ Outside fair zone |SPD|≥0.05'}
          </p>
        </div>

        {/* Final SPD */}
        <div className="flex flex-col gap-1">
          <p className="label text-[10px] mb-1">Final SPD</p>
          <AnimNum value={fmt(bestCEO?.spd)} decimals={4} color="#00E5FF" size="text-2xl"/>
          <p className="text-[10px] text-ink-500 font-body mt-1">{bestCEO?.model} · full pipeline</p>
          <div className="h-1 bg-ink-700 rounded-full mt-1 overflow-hidden">
            <div className="h-full bg-cyan rounded-full" style={{width:`${Math.min(spdAfter*600,100)}%`}}/>
          </div>
          <p className="text-[9px] text-ink-400 font-mono mt-1">
            {isFairSPD(bestCEO?.spd) ? '✓ Fair zone |SPD| < 0.05' : 'Target: |SPD| < 0.05'}
          </p>
        </div>

        {/* Accuracy */}
        <div className="flex flex-col gap-1">
          <p className="label text-[10px] mb-1">Best Accuracy</p>
          <AnimNum value={fmt(summary?.best_accuracy)} decimals={4} color="#FFB830" size="text-2xl"/>
          <p className="text-[10px] text-ink-500 font-body mt-1">{bestCEO?.model} after mitigation</p>
          <div className="h-1 bg-ink-700 rounded-full mt-1 overflow-hidden">
            <div className="h-full bg-amber-400 rounded-full" style={{width:`${(summary?.best_accuracy||0)*100}%`}}/>
          </div>
          <p className="text-[9px] text-ink-400 font-mono mt-1">{baseline?.total_rows?.toLocaleString()} rows analysed</p>
        </div>
      </div>
    </div>
  )
}

// ─── Model cards ──────────────────────────────────────────────────────────────
function ModelCards({ mitigation_results }) {
  const models = Object.keys(mitigation_results)
  return (
    <div className="space-y-3">
      <p className="label">Final Pipeline Results — Per Model</p>
      <div className="grid grid-cols-2 lg:grid-cols-4 gap-3">
        {models.map(m => {
          const keys  = Object.keys(mitigation_results[m])
          const last  = mitigation_results[m][keys[keys.length-1]] || {}
          const first = mitigation_results[m]['original'] || {}
          const color = MC[m] || '#00E5FF'
          const spdFair = isFairSPD(last.spd)
          const diFair  = isFairDI(last.di)
          const spdRed  = (first.spd && last.spd != null && first.spd !== 0)
            ? ((Math.abs(first.spd) - Math.abs(last.spd)) / Math.abs(first.spd) * 100).toFixed(0)
            : null
          return (
            <div key={m} className="rounded-2xl border p-4 relative overflow-hidden flex flex-col gap-3"
              style={{background:`${color}06`,borderColor:`${color}25`,boxShadow:`0 0 30px ${color}10`}}>
              <div className="absolute top-0 right-0 w-24 h-24 rounded-full pointer-events-none opacity-5"
                style={{background:`radial-gradient(circle,${color},transparent 70%)`,transform:'translate(30%,-30%)'}}/>
              <div className="flex items-center gap-2">
                <span className="text-2xl">{MI[m]||'●'}</span>
                <div>
                  <p className="font-display font-bold text-sm text-white">{m}</p>
                  <p className="text-[10px] text-ink-500 font-body">{keys.length} stages</p>
                </div>
              </div>
              <div className="space-y-2">
                <div className="flex justify-between items-center">
                  <span className="text-[10px] text-ink-500 font-mono">Accuracy</span>
                  <span className="font-mono font-bold text-sm" style={{color}}>{pct(last.accuracy)}</span>
                </div>
                <div className="flex justify-between items-center">
                  <span className="text-[10px] text-ink-500 font-mono">SPD</span>
                  <span className={`font-mono font-bold text-sm ${spdFair?'text-emerald-300':'text-rose-300'}`}>
                    {fmt(last.spd)} {spdFair?'✓':''}
                  </span>
                </div>
                <div className="flex justify-between items-center">
                  <span className="text-[10px] text-ink-500 font-mono">DI</span>
                  <span className={`font-mono font-bold text-sm ${diFair?'text-emerald-300':'text-amber-300'}`}>
                    {fmt(last.di)} {diFair?'✓':''}
                  </span>
                </div>
              </div>
              <div>
                <div className="flex justify-between text-[9px] text-ink-400 font-mono mb-1">
                  <span>bias before</span><span>after</span>
                </div>
                <div className="h-1.5 bg-ink-700 rounded-full overflow-hidden">
                  <div className="h-full bg-rose-400/50 rounded-full" style={{width:`${Math.min(Math.abs(first.spd||0)*600,100)}%`}}/>
                </div>
                <div className="h-1.5 bg-ink-700 rounded-full overflow-hidden mt-0.5">
                  <div className="h-full bg-emerald-400 rounded-full" style={{width:`${Math.min(Math.abs(last.spd||0)*600,100)}%`}}/>
                </div>
              </div>
              {spdRed !== null && (
                <p className="text-[9px] font-mono border-t border-ink-700 pt-2" style={{color}}>
                  {spdRed}% SPD reduction · {spdFair && diFair ? '✓ Fully fair' : spdFair ? '✓ SPD fair' : 'Bias reduced'}
                </p>
              )}
            </div>
          )
        })}
      </div>
    </div>
  )
}

// ─── Pipeline flow ────────────────────────────────────────────────────────────
function PipelineFlow({ mitigation_results }) {
  const models = Object.keys(mitigation_results)
  const [selModel, setSelModel] = useState(models[0])
  const m0 = selModel || models[0]

  const stages = STAGE_KEY
    .filter(k => mitigation_results[m0]?.[k])
    .map(k => ({
      key: k, label: STAGE_LABEL[k], tag: STAGE_TAG[k], color: STAGE_COLOR[k],
      spd: mitigation_results[m0][k]?.spd,
      di:  mitigation_results[m0][k]?.di,
    }))

  const before    = Math.abs(mitigation_results[m0]?.original?.spd || 0)
  const after     = Math.abs(mitigation_results[m0]?.reweigh_adb_ceo?.spd || 0)
  const reduction = before > 0 ? ((before - after) / before * 100).toFixed(0) : 0

  return (
    <div className="card-glow space-y-4">
      <div className="flex items-center justify-between flex-wrap gap-3">
        <div>
          <p className="label">3-Stage Mitigation Pipeline</p>
          <p className="text-[11px] text-ink-400 font-body">SPD at each stage — target: 0</p>
        </div>
        <div className="flex gap-1.5 flex-wrap">
          {models.map(m => (
            <button key={m} onClick={() => setSelModel(m)}
              className={`px-3 py-1 rounded-lg text-[10px] font-display font-bold transition-all border ${
                selModel===m ? 'text-ink-900 border-transparent' : 'bg-transparent border-ink-700 text-ink-400 hover:border-ink-500'
              }`}
              style={selModel===m ? {background:MC[m]||'#00E5FF'} : {}}>
              {MI[m]} {m}
            </button>
          ))}
        </div>
      </div>

      <div className="flex items-stretch overflow-x-auto pb-2">
        {stages.map((s, i) => {
          const fair = s.spd != null && isFairSPD(s.spd)
          return (
            <div key={s.key} className="flex items-center flex-shrink-0">
              <div style={{minWidth:130}}>
                <div className="relative rounded-xl p-3 w-full border"
                  style={{background:`${s.color}10`,borderColor:`${s.color}40`,boxShadow:`0 0 20px ${s.color}15`}}>
                  <p className="text-[9px] font-display uppercase tracking-widest mb-1" style={{color:s.color}}>{s.tag}</p>
                  <p className="font-display font-bold text-xs text-white truncate mb-2">{s.label}</p>
                  {s.spd != null && (
                    <div className="space-y-1">
                      <div className="flex items-center justify-between">
                        <span className="text-[9px] text-ink-500 font-mono">SPD</span>
                        <span className={`font-mono text-xs font-bold ${fair?'text-emerald-300':'text-rose-300'}`}>
                          {fmt(s.spd)} {fair?'✓':''}
                        </span>
                      </div>
                      <div className="flex items-center justify-between">
                        <span className="text-[9px] text-ink-500 font-mono">DI</span>
                        <span className={`font-mono text-xs font-bold ${isFairDI(s.di)?'text-emerald-300':'text-amber-300'}`}>
                          {fmt(s.di)}
                        </span>
                      </div>
                      <div className="h-1 bg-ink-700 rounded-full mt-1 overflow-hidden">
                        <div className={`h-full rounded-full ${fair?'bg-emerald-400':'bg-rose-400'}`}
                          style={{width:`${Math.min(Math.abs(s.spd)*600,100)}%`}}/>
                      </div>
                    </div>
                  )}
                </div>
              </div>
              {i < stages.length-1 && (
                <div className="flex flex-col items-center mx-1.5 flex-shrink-0 self-center">
                  <div className="flex items-center gap-0.5">
                    <div className="w-6 h-px bg-gradient-to-r from-ink-600 to-ink-500"/>
                    <svg width="8" height="8" viewBox="0 0 8 8" fill="none">
                      <path d="M0 4h7M4 1l3 3-3 3" stroke="#4a5f96" strokeWidth="1.5" strokeLinecap="round"/>
                    </svg>
                  </div>
                  <span className="text-[8px] text-ink-500 mt-0.5 font-mono">+layer</span>
                </div>
              )}
            </div>
          )
        })}
      </div>

      <div className="flex items-center gap-4 pt-2 border-t border-ink-700 flex-wrap">
        <div className="flex items-center gap-2">
          <span className="text-[10px] text-ink-500 font-mono">|SPD| before:</span>
          <span className="font-mono text-xs text-rose-400 font-bold">{before.toFixed(4)}</span>
        </div>
        <svg width="20" height="10" viewBox="0 0 20 10">
          <path d="M0 5h16M13 2l3 3-3 3" stroke="#4a5f96" strokeWidth="1.2" strokeLinecap="round"/>
        </svg>
        <div className="flex items-center gap-2">
          <span className="text-[10px] text-ink-500 font-mono">after:</span>
          <span className="font-mono text-xs text-emerald-400 font-bold">{after.toFixed(4)}</span>
        </div>
        <div className="ml-auto tag border-emerald-500/30 bg-emerald-500/15 text-emerald-400 font-mono text-xs">
          ↓ {reduction}% bias removed
        </div>
      </div>
    </div>
  )
}

// ─── Line charts ──────────────────────────────────────────────────────────────
function PipelineChart({ mitigation_results, metric='spd', label='SPD', optimal=0 }) {
  const models = Object.keys(mitigation_results)
  const data = STAGE_KEY.map(stage => {
    const pt = { stage: STAGE_LABEL[stage] }
    models.forEach(m => { const v = mitigation_results[m]?.[stage]; if (v) pt[m] = parseFloat((v[metric]??0).toFixed(4)) })
    return pt
  }).filter(d => models.some(m => d[m] !== undefined))

  return (
    <div className="card-glow space-y-3">
      <div>
        <p className="label">{label} Across Pipeline Stages</p>
        <p className="text-[11px] text-ink-400 font-body">
          {metric==='spd' ? 'Target: 0. Each stage reduces demographic disparity.' : 'Fair zone [0.8–1.25] marked in green.'}
        </p>
      </div>
      <ResponsiveContainer width="100%" height={240}>
        <LineChart data={data} margin={{left:8,right:28,top:8,bottom:0}}>
          <CartesianGrid strokeDasharray="3 3" stroke="rgba(42,58,110,0.4)"/>
          <XAxis dataKey="stage" tick={{fontSize:9}}/>
          <YAxis tick={{fontSize:9}} domain={['auto','auto']}
            label={{value:label,angle:-90,position:'insideLeft',fill:'#7b8db8',fontSize:9}}/>
          <Tooltip contentStyle={TT} formatter={(v,n)=>[v?.toFixed(4),n]}/>
          {metric==='di'&&<ReferenceLine y={0.8}  stroke="#00E676" strokeDasharray="3 3" label={{value:'0.8', fill:'#00E676',fontSize:8,position:'right'}}/>}
          {metric==='di'&&<ReferenceLine y={1.25} stroke="#00E676" strokeDasharray="3 3" label={{value:'1.25',fill:'#00E676',fontSize:8,position:'right'}}/>}
          <ReferenceLine y={optimal} stroke="#00E5FF" strokeDasharray="5 3"
            label={{value:metric==='spd'?'← Target':'1.0',fill:'#00E5FF',fontSize:8,position:'right'}}/>
          <Legend wrapperStyle={{fontSize:10}}/>
          {models.map(m => (
            <Line key={m} type="monotone" dataKey={m} stroke={MC[m]||'#888'} strokeWidth={2.5}
              dot={{r:5,strokeWidth:2,fill:'#0D1228'}} activeDot={{r:7}}/>
          ))}
        </LineChart>
      </ResponsiveContainer>
    </div>
  )
}

// ─── Accuracy vs Fairness bar ─────────────────────────────────────────────────
function AccFairChart({ mitigation_results }) {
  const models = Object.keys(mitigation_results)
  const data = models.map(m => {
    const sk   = Object.keys(mitigation_results[m])
    const last = mitigation_results[m][sk[sk.length-1]]
    return { model:m, Accuracy:parseFloat((last?.accuracy||0).toFixed(4)), '|SPD|':parseFloat(Math.abs(last?.spd||0).toFixed(4)) }
  })
  return (
    <div className="card-glow space-y-3">
      <div>
        <p className="label">Accuracy – Fairness Tradeoff</p>
        <p className="text-[11px] text-ink-400 font-body">High accuracy + low |SPD| = ideal outcome</p>
      </div>
      <ResponsiveContainer width="100%" height={240}>
        <BarChart data={data} margin={{left:0,right:8,top:8,bottom:0}}>
          <CartesianGrid strokeDasharray="3 3" stroke="rgba(42,58,110,0.4)"/>
          <XAxis dataKey="model" tick={{fontSize:10}}/>
          <YAxis tick={{fontSize:10}}/>
          <Tooltip contentStyle={TT} formatter={v=>[v?.toFixed(4)]}/>
          <Legend wrapperStyle={{fontSize:10}}/>
          <Bar dataKey="Accuracy" fill="#00E5FF" radius={[4,4,0,0]}/>
          <Bar dataKey="|SPD|" name="|SPD| ↓ lower=fairer" fill="#FF4D6D" radius={[4,4,0,0]}/>
        </BarChart>
      </ResponsiveContainer>
    </div>
  )
}

// ─── Radar ────────────────────────────────────────────────────────────────────
function ModelRadar({ model, stages }) {
  const sk    = Object.keys(stages)
  const first = stages['original'] || {}
  const last  = stages[sk[sk.length-1]] || {}
  const norm  = (v, m) => m==='di' ? Math.min(Math.abs(v-1)*200,100) : Math.min(Math.abs(v)*200,100)
  const data  = ['spd','di','aod','eod'].map(m => ({
    metric: m.toUpperCase(),
    Before: parseFloat(norm(first[m]??0, m).toFixed(1)),
    After:  parseFloat(norm(last[m]??0,  m).toFixed(1)),
  }))
  return (
    <div className="card-glow space-y-2">
      <p className="label">{model} — Bias Profile Before vs After</p>
      <p className="text-[11px] text-ink-400 font-body">Smaller blue = less bias across all metrics</p>
      <ResponsiveContainer width="100%" height={210}>
        <RadarChart data={data}>
          <PolarGrid stroke="rgba(42,58,110,0.6)"/>
          <PolarAngleAxis dataKey="metric" tick={{fontSize:11,fill:'#7b8db8'}}/>
          <PolarRadiusAxis angle={90} domain={[0,100]} tick={{fontSize:8}} tickCount={3}/>
          <Radar name="Before" dataKey="Before" stroke="#FF4D6D" fill="#FF4D6D" fillOpacity={0.15} strokeWidth={2}/>
          <Radar name="After"  dataKey="After"  stroke="#00E5FF" fill="#00E5FF" fillOpacity={0.25} strokeWidth={2}/>
          <Tooltip contentStyle={TT} formatter={v=>[`${v.toFixed(1)} bias`]}/>
          <Legend wrapperStyle={{fontSize:9}}/>
        </RadarChart>
      </ResponsiveContainer>
    </div>
  )
}

// ─── Stage bar per model ──────────────────────────────────────────────────────
function ModelStageChart({ model, stages }) {
  const data = Object.entries(stages).map(([s,v]) => ({
    stage: STAGE_LABEL[s]||s,
    '|SPD|':   parseFloat(Math.abs(v.spd??0).toFixed(4)),
    '|DI–1|':  parseFloat(Math.abs((v.di??1)-1).toFixed(4)),
    Accuracy:  parseFloat((v.accuracy||0).toFixed(4)),
  }))
  return (
    <div className="card-glow space-y-2">
      <p className="label">{model} — Metrics Per Stage</p>
      <ResponsiveContainer width="100%" height={210}>
        <BarChart data={data} margin={{left:0,right:8,top:8,bottom:0}}>
          <CartesianGrid strokeDasharray="3 3" stroke="rgba(42,58,110,0.4)"/>
          <XAxis dataKey="stage" tick={{fontSize:8}}/>
          <YAxis tick={{fontSize:9}}/>
          <Tooltip contentStyle={TT} formatter={v=>[v?.toFixed(4)]}/>
          <Legend wrapperStyle={{fontSize:9}}/>
          <Bar dataKey="|SPD|"    name="|SPD| ↓ less bias"  fill="#FF4D6D" radius={[3,3,0,0]}/>
          <Bar dataKey="|DI–1|"   name="|DI–1| ↓ less bias" fill="#FFB830" radius={[3,3,0,0]}/>
          <Bar dataKey="Accuracy" name="Accuracy ↑"          fill="#00E676" radius={[3,3,0,0]}/>
        </BarChart>
      </ResponsiveContainer>
    </div>
  )
}

// ─── Per-model deep dive ──────────────────────────────────────────────────────
function ModelDeepDive({ mitigation_results }) {
  const models = Object.keys(mitigation_results)
  const [sel, setSel] = useState(models[0])
  const stages = mitigation_results[sel] || {}
  const sk = Object.keys(stages)

  return (
    <div className="space-y-4">
      <div className="flex gap-2 flex-wrap">
        {models.map(m => (
          <button key={m} onClick={() => setSel(m)}
            className={`px-4 py-2 rounded-xl font-display text-sm font-bold transition-all border ${
              sel===m ? 'text-ink-900 border-transparent' : 'bg-transparent border-ink-700 text-ink-400 hover:border-ink-500'
            }`}
            style={sel===m ? {background:MC[m]||'#00E5FF',boxShadow:`0 0 24px ${MC[m]||'#00E5FF'}50`} : {}}>
            {MI[m]||'●'} {m}
          </button>
        ))}
      </div>

      {/* Stage summary cards */}
      <div className="grid grid-cols-2 sm:grid-cols-4 gap-3">
        {sk.map(stage => {
          const v = stages[stage]
          return (
            <div key={stage} className="rounded-xl p-3 border border-ink-700 bg-ink-800">
              <p className="text-[10px] font-display uppercase tracking-widest mb-2"
                style={{color:STAGE_COLOR[stage]||'#888'}}>
                {STAGE_LABEL[stage]||stage}
              </p>
              <p className="font-mono text-xl font-bold text-white">{pct(v.accuracy)}</p>
              <p className={`font-mono text-xs mt-1 ${isFairSPD(v.spd)?'text-emerald-300':'text-rose-300'}`}>
                SPD {fmt(v.spd)} {isFairSPD(v.spd)?'✓':''}
              </p>
              <p className={`font-mono text-xs ${isFairDI(v.di)?'text-emerald-300':'text-amber-300'}`}>
                DI {fmt(v.di)} {isFairDI(v.di)?'✓':''}
              </p>
            </div>
          )
        })}
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
        <ModelRadar model={sel} stages={stages}/>
        <ModelStageChart model={sel} stages={stages}/>
      </div>

      {/* Full metrics table */}
      <div className="card-glow overflow-x-auto">
        <div className="flex items-center justify-between mb-3 flex-wrap gap-2">
          <p className="label mb-0">{sel} — Full Metrics Breakdown</p>
          <div className="flex gap-2 text-[10px] font-mono">
            <span className="tag border-emerald-400/20 text-emerald-400 bg-emerald-400/5">■ In fair range</span>
            <span className="tag border-rose-400/20 text-rose-400 bg-rose-400/5">■ Outside range</span>
          </div>
        </div>
        <table className="w-full text-[11px] font-mono">
          <thead>
            <tr style={{background:'rgba(0,229,255,0.06)'}}>
              {['Stage','Accuracy','Bal. Acc','SPD','DI','AOD','EOD'].map(h => (
                <th key={h} className="px-3 py-2.5 text-left text-ink-300 font-bold border-b border-ink-700 whitespace-nowrap">{h}</th>
              ))}
            </tr>
          </thead>
          <tbody>
            {sk.map(stage => {
              const v    = stages[stage]
              const isFull = stage === 'reweigh_adb_ceo'
              return (
                <tr key={stage} className={`border-b border-ink-800 hover:bg-ink-700/40 ${isFull?'bg-cyan/10':''}`}>
                  <td className="px-3 py-2.5 whitespace-nowrap font-bold text-white" style={{borderLeft:`3px solid ${STAGE_COLOR[stage]||'#888'}`,paddingLeft:'12px'}}>
                    {isFull ? '★ ' : ''}{STAGE_LABEL[stage]||stage}
                  </td>
                  {['accuracy','balanced_accuracy','spd','di','aod','eod'].map(metric => (
                    <td key={metric} className="px-3 py-2.5 text-center">
                      {metric === 'di'
                        ? <DiCell v={v[metric]} fair={isFairDI(v[metric])} />
                        : <span className={`rounded px-1.5 py-0.5 text-[10px] font-bold inline-block border ${
                            isFair(v[metric],metric) || metric==='accuracy' || metric==='balanced_accuracy'
                              ? 'bg-emerald-500/20 text-emerald-300 border-emerald-500/40'
                              : 'bg-red-500/20 text-red-300 border-red-500/40'
                          }`}>
                            {fmt(v[metric])}
                          </span>
                      }
                    </td>
                  ))}
                </tr>
              )
            })}
          </tbody>
        </table>
        <p className="text-[10px] text-ink-400 font-body mt-3">
          ★ = Full pipeline result · Fair: |SPD| &lt; 0.05 · DI ∈ [0.8,1.25] · |AOD| &lt; 0.05 · |EOD| &lt; 0.05
          · <span title="DI shows — when one group had zero positive predictions in this test split">DI = — means undefined (zero predictions in one group)</span>
        </p>
      </div>
    </div>
  )
}

// ─── SMOTE section ────────────────────────────────────────────────────────────
function SmoteSection({ smote_results, baseline }) {
  if (!smote_results || Object.keys(smote_results).length === 0)
    return <div className="card text-ink-400 text-sm text-center py-12">No SMOTE variants were run in this session.</div>

  const valid    = Object.entries(smote_results).filter(([,v]) => !v.error)
  const baseSPD  = baseline?.spd || 0
  const allWorse = valid.length > 0 && valid.every(([,v]) => Math.abs(v.spd) >= Math.abs(baseSPD))
  const someWorse = valid.some(([,v]) => Math.abs(v.spd) > Math.abs(baseSPD))

  const spdData   = valid.map(([k,v]) => ({ variant:k, SPD:parseFloat(v.spd.toFixed(4)) }))
  const multiData = valid.map(([k,v]) => ({
    variant:k,
    '|SPD|':  parseFloat(Math.abs(v.spd).toFixed(4)),
    '|DI–1|': parseFloat(Math.abs((v.di??1)-1).toFixed(4)),
    '|AOD|':  parseFloat(Math.abs(v.aod??0).toFixed(4)),
    '|EOD|':  parseFloat(Math.abs(v.eod??0).toFixed(4)),
  }))

  const vColor  = allWorse ? '#FF4D6D' : someWorse ? '#FFB830' : '#00E676'
  const vBorder = allWorse ? 'border-rose-500/35 bg-rose-500/5' : someWorse ? 'border-amber-500/30 bg-amber-500/5' : 'border-emerald-500/30 bg-emerald-500/5'

  return (
    <div className="space-y-4">
      {/* Verdict */}
      <div className={`rounded-2xl p-5 border ${vBorder}`}>
        <div className="flex items-start gap-4">
          <div className="w-12 h-12 rounded-xl flex items-center justify-center flex-shrink-0 text-2xl"
            style={{background:`${vColor}18`}}>
            {allWorse ? '⚠️' : someWorse ? 'ℹ️' : '✅'}
          </div>
          <div>
            <p className="font-display font-black text-lg mb-1" style={{color:vColor}}>
              {allWorse ? 'SMOTE Worsens Fairness on This Dataset'
                : someWorse ? 'SMOTE Has Mixed Fairness Effects'
                : 'SMOTE Does Not Hurt Fairness Here'}
            </p>
            <p className="text-sm text-ink-300 font-body leading-relaxed">
              {allWorse
                ? `All ${valid.length} SMOTE variants increased |SPD| above the dataset baseline of ${Math.abs(baseSPD).toFixed(4)}. Oversampling balances class distribution but reinforces demographic disparities rather than resolving them.`
                : someWorse
                ? 'Some SMOTE variants increased bias. Oversampling has inconsistent fairness effects on this dataset.'
                : `SMOTE variants maintained or improved fairness metrics on this dataset. Baseline |SPD| = ${Math.abs(baseSPD).toFixed(4)}.`}
            </p>
            <div className="flex items-center gap-2 mt-3">
              <span className="tag border-ink-600 text-ink-400 text-[10px]">{valid.length} variants tested</span>
              <span className="tag border-ink-600 text-ink-400 text-[10px]">Baseline |SPD| = {Math.abs(baseSPD).toFixed(4)}</span>
            </div>
          </div>
        </div>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
        <div className="card-glow space-y-3">
          <p className="label">SPD Per SMOTE Variant</p>
          <p className="text-[11px] text-ink-400 font-body">Amber = baseline · Red bar = worse than baseline</p>
          <ResponsiveContainer width="100%" height={220}>
            <BarChart data={spdData} margin={{left:0,right:20,top:8,bottom:0}}>
              <CartesianGrid strokeDasharray="3 3" stroke="rgba(42,58,110,0.4)"/>
              <XAxis dataKey="variant" tick={{fontSize:8}}/>
              <YAxis tick={{fontSize:9}}/>
              <Tooltip contentStyle={TT} formatter={v=>[v?.toFixed(4)]}/>
              <ReferenceLine y={0}      stroke="#00E5FF" strokeDasharray="4 3" label={{value:'Fair',     fill:'#00E5FF',fontSize:8,position:'right'}}/>
              <ReferenceLine y={baseSPD} stroke="#FFB830" strokeDasharray="4 3" label={{value:'Baseline',fill:'#FFB830',fontSize:8,position:'right'}}/>
              <Bar dataKey="SPD" radius={[4,4,0,0]}>
                {spdData.map((d,i) => <Cell key={i} fill={Math.abs(d.SPD)>Math.abs(baseSPD)?'#FF4D6D':'#00E676'}/>)}
              </Bar>
            </BarChart>
          </ResponsiveContainer>
        </div>

        <div className="card-glow space-y-3">
          <p className="label">All Fairness Metrics by Variant</p>
          <p className="text-[11px] text-ink-400 font-body">Lower = less bias across all dimensions</p>
          <ResponsiveContainer width="100%" height={220}>
            <BarChart data={multiData} margin={{left:0,right:8,top:8,bottom:0}}>
              <CartesianGrid strokeDasharray="3 3" stroke="rgba(42,58,110,0.4)"/>
              <XAxis dataKey="variant" tick={{fontSize:8}}/>
              <YAxis tick={{fontSize:9}}/>
              <Tooltip contentStyle={TT} formatter={v=>[v?.toFixed(4)]}/>
              <Legend wrapperStyle={{fontSize:9}}/>
              <Bar dataKey="|SPD|"  fill="#FF4D6D" radius={[3,3,0,0]}/>
              <Bar dataKey="|DI–1|" fill="#FFB830" radius={[3,3,0,0]}/>
              <Bar dataKey="|AOD|"  fill="#a855f7" radius={[3,3,0,0]}/>
              <Bar dataKey="|EOD|"  fill="#00E5FF" radius={[3,3,0,0]}/>
            </BarChart>
          </ResponsiveContainer>
        </div>
      </div>

      {/* SMOTE table */}
      <div className="card-glow overflow-x-auto">
        <p className="label mb-3">SMOTE Variants — Full Metric Table</p>
        <table className="w-full text-[11px] font-mono">
          <thead>
            <tr style={{background:'rgba(0,229,255,0.05)'}}>
              {['Variant','Accuracy','Balanced Acc','SPD','DI','AOD','EOD'].map(h => (
                <th key={h} className="px-3 py-2 text-left text-ink-300 font-bold border-b border-ink-700 whitespace-nowrap">{h}</th>
              ))}
            </tr>
          </thead>
          <tbody>
            <tr className="border-b border-ink-800 bg-ink-700/40">
              <td className="px-3 py-2 font-bold text-amber-400">No SMOTE (Baseline)</td>
              <td className="px-3 py-2 text-ink-500">—</td>
              <td className="px-3 py-2 text-ink-500">—</td>
              <td className="px-3 py-2 text-amber-400 font-bold">{fmt(baseline?.spd)}</td>
              <td className="px-3 py-2 text-amber-400 font-bold">{fmt(baseline?.di)}</td>
              <td className="px-3 py-2 text-ink-500">—</td>
              <td className="px-3 py-2 text-ink-500">—</td>
            </tr>
            {Object.entries(smote_results).map(([variant, vals]) => (
              <tr key={variant} className="border-b border-ink-800 hover:bg-ink-700/40">
                <td className="px-3 py-2 font-bold text-ink-200">{variant}</td>
                {vals.error
                  ? <td colSpan={6} className="px-3 py-2 text-rose-400 text-xs">{vals.error}</td>
                  : ['accuracy','balanced_accuracy','spd','di','aod','eod'].map(m => (
                    <td key={m} className="px-3 py-2">
                      <span className={`rounded px-1.5 py-0.5 text-[10px] font-bold ${
                        (m==='spd' && Math.abs(vals[m])>Math.abs(baseSPD)) ||
                        (m==='di'  && Math.abs((vals[m]??1)-1)>Math.abs((baseline?.di??1)-1))
                          ? 'bg-red-500/20 text-red-300 border border-red-500/30' : 'bg-ink-700/50 text-ink-100'
                      }`}>
                        {fmt(vals[m])}
                      </span>
                    </td>
                  ))
                }
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  )
}

// ─── Analysis Insights ────────────────────────────────────────────────────────
function AnalysisInsights({ results }) {
  const { mitigation_results, smote_results, baseline } = results
  const models = Object.keys(mitigation_results)

  const bestCEO = models.reduce((b, m) => {
    const ceo = mitigation_results[m]?.reweigh_adb_ceo
    if (!ceo) return b
    return (!b || Math.abs(ceo.spd) < Math.abs(b.spd)) ? { ...ceo, model:m } : b
  }, null)

  const spdBefore = Math.abs(baseline?.spd || 0)
  const spdAfter  = Math.abs(bestCEO?.spd  || 0)
  const spdRed    = spdBefore > 0 ? ((spdBefore-spdAfter)/spdBefore*100).toFixed(1) : '0'

  const allSmoteWorse = smote_results &&
    Object.values(smote_results).filter(v=>!v.error).length > 0 &&
    Object.values(smote_results).filter(v=>!v.error).every(v=>Math.abs(v.spd)>=spdBefore)

  const bestAccM = models.reduce((b,m) => {
    const a = mitigation_results[m]?.original?.accuracy || 0
    return a > (b.acc||0) ? {model:m,acc:a} : b
  }, {})

  const costs = models.map(m => {
    const sk   = Object.keys(mitigation_results[m])
    const orig = mitigation_results[m]['original']?.accuracy || 0
    const last = mitigation_results[m][sk[sk.length-1]]?.accuracy || 0
    return orig - last
  })
  const avgCost = (costs.reduce((a,b)=>a+b,0)/costs.length*100).toFixed(2)

  const insights = [
    {
      icon:'📉', color:'#00E5FF',
      title:`${spdRed}% Statistical Bias Reduced`,
      body:`SPD reduced from ${fmt(baseline?.spd)} → ${fmt(bestCEO?.spd)} using the full 3-stage pipeline. Best model: ${bestCEO?.model}.`,
    },
    {
      icon: isFairSPD(bestCEO?.spd)?'✅':'🎯', color: isFairSPD(bestCEO?.spd)?'#00E676':'#FFB830',
      title: isFairSPD(bestCEO?.spd) ? 'Fair Zone Achieved' : 'Significant Bias Reduction',
      body: isFairSPD(bestCEO?.spd)
        ? `${bestCEO?.model} SPD = ${fmt(bestCEO?.spd)} — within the accepted fair zone (|SPD| < 0.05). ${isFairDI(bestCEO?.di)?'DI also within [0.8,1.25].':''}`
        : `Bias significantly reduced but fair zone threshold not yet reached. SPD = ${fmt(bestCEO?.spd)}.`,
    },
    {
      icon:'💡', color:'#FFB830',
      title:`${bestAccM.model} Achieves Highest Accuracy`,
      body:`${bestAccM.model} reaches ${pct(bestAccM.acc)} baseline accuracy. Average accuracy cost across all models after full mitigation: ${avgCost}%.`,
    },
    {
      icon:'🔗', color:'#a855f7',
      title:'Multi-Stage Pipeline Outperforms Any Single Technique',
      body:'Each stage contributes incrementally — Reweighing handles pre-processing bias, ADB applies adversarial in-processing, and CEO provides post-hoc threshold calibration.',
    },
    allSmoteWorse && {
      icon:'⚠️', color:'#FF4D6D',
      title:'SMOTE Does Not Improve Fairness on This Dataset',
      body:`All ${Object.keys(smote_results).filter(k=>!smote_results[k].error).length} SMOTE variants increased |SPD| above the baseline. Class oversampling addresses imbalance but does not mitigate demographic bias.`,
    },
    {
      icon:'📊', color:'#00E676',
      title:`Dataset: ${baseline?.total_rows?.toLocaleString()} Rows Analysed`,
      body:`Class split: ${baseline?.class_distribution?.[0]?.toLocaleString()} (0) / ${baseline?.class_distribution?.[1]?.toLocaleString()} (1). ${models.length} models evaluated across ${STAGE_KEY.length} pipeline stages.`,
    },
  ].filter(Boolean)

  return (
    <div className="space-y-4">
      <p className="label">Analysis Insights</p>
      <div className="grid grid-cols-1 sm:grid-cols-2 gap-3">
        {insights.map((f, i) => (
          <div key={i} className="rounded-xl p-4 border flex gap-3"
            style={{background:`${f.color}08`,borderColor:`${f.color}25`}}>
            <span className="text-xl mt-0.5 shrink-0">{f.icon}</span>
            <div>
              <p className="font-display font-bold text-sm mb-1" style={{color:f.color}}>{f.title}</p>
              <p className="text-xs text-ink-400 font-body leading-relaxed">{f.body}</p>
            </div>
          </div>
        ))}
      </div>
    </div>
  )
}

// ─── Per-group confusion matrix ───────────────────────────────────────────────
function GroupConfusionMatrix({ model, stages, config }) {
  const sk   = Object.keys(stages)
  const last = stages[sk[sk.length - 1]] || {}
  const priv_label   = `${config.protected_attribute}=${config.privileged_value} (privileged)`
  const unpriv_label = `${config.protected_attribute}≠${config.privileged_value} (unprivileged)`

  if (!last.cm_privileged) return null

  const Cell = ({ label, value, color }) => (
    <div className={`rounded-lg p-2 text-center border ${color}`}>
      <p className="text-[9px] text-ink-500 font-mono mb-1">{label}</p>
      <p className="font-mono font-bold text-sm text-white">{value}</p>
    </div>
  )

  const GroupCM = ({ cm, label }) => {
    if (!cm || cm.n === 0) return null
    const tpr = cm.tp + cm.fn > 0 ? (cm.tp / (cm.tp + cm.fn) * 100).toFixed(1) : '—'
    const fpr = cm.fp + cm.tn > 0 ? (cm.fp / (cm.fp + cm.tn) * 100).toFixed(1) : '—'
    return (
      <div className="space-y-2">
        <p className="text-[11px] font-mono text-ink-300 truncate">{label}</p>
        <p className="text-[10px] text-ink-500 font-body">n = {cm.n} samples</p>
        <div className="grid grid-cols-2 gap-1.5">
          <Cell label="True Pos"  value={cm.tp} color="border-emerald-500/30 bg-emerald-500/15"/>
          <Cell label="False Pos" value={cm.fp} color="border-rose-500/30 bg-rose-500/15"/>
          <Cell label="False Neg" value={cm.fn} color="border-amber-500/30 bg-amber-500/15"/>
          <Cell label="True Neg"  value={cm.tn} color="border-cyan/20 bg-cyan/10"/>
        </div>
        <div className="flex gap-3 text-[10px] font-mono">
          <span className="text-emerald-400">TPR {tpr}%</span>
          <span className="text-rose-400">FPR {fpr}%</span>
        </div>
      </div>
    )
  }

  return (
    <div className="card-glow space-y-4">
      <div>
        <p className="label">{model} — Per-Group Confusion Matrix (Full Pipeline)</p>
        <p className="text-[11px] text-ink-400 font-body">
          Breakdown of predictions for each demographic group · Equal TPR and FPR = fair model
        </p>
      </div>
      <div className="grid grid-cols-1 sm:grid-cols-2 gap-6">
        <GroupCM cm={last.cm_privileged}   label={priv_label}/>
        <GroupCM cm={last.cm_unprivileged} label={unpriv_label}/>
      </div>
    </div>
  )
}

// ─── Feature Importance chart ─────────────────────────────────────────────────
function FeatureImportanceChart({ feature_importance, protected_attribute }) {
  if (!feature_importance || feature_importance.length === 0) return null
  const top = feature_importance.slice(0, 12)
  return (
    <div className="card-glow space-y-3">
      <div>
        <p className="label">Feature Importance</p>
        <p className="text-[11px] text-ink-400 font-body">
          Which features drive predictions most. Protected attribute highlighted in amber.
        </p>
      </div>
      <div className="space-y-1.5">
        {top.map(({ feature, importance }) => {
          const isProt = feature === protected_attribute
          const pct    = (importance * 100).toFixed(1)
          const width  = `${Math.min(importance / top[0].importance * 100, 100)}%`
          return (
            <div key={feature} className="flex items-center gap-3">
              <div className={`text-[11px] font-mono w-32 truncate flex-shrink-0 ${isProt ? 'text-amber font-bold' : 'text-ink-300'}`}>
                {isProt ? '⚠ ' : ''}{feature}
              </div>
              <div className="flex-1 h-4 bg-ink-700 rounded-full overflow-hidden">
                <div
                  className={`h-full rounded-full transition-all duration-700 ${isProt ? 'bg-amber' : 'bg-cyan'}`}
                  style={{ width }}
                />
              </div>
              <div className="text-[10px] font-mono text-ink-400 w-10 text-right flex-shrink-0">{pct}%</div>
            </div>
          )
        })}
      </div>
      <p className="text-[10px] text-ink-400 font-body">
        Computed using Random Forest on training data · Top {top.length} of {feature_importance.length} features shown
      </p>
    </div>
  )
}

// ─── Main export ──────────────────────────────────────────────────────────────
// ─── PDF Report Generator ─────────────────────────────────────────────────────
function generatePDF(results, config) {
  const { summary, mitigation_results, smote_results, baseline } = results
  const models  = Object.keys(mitigation_results || {})
  const date    = new Date().toLocaleDateString('en-GB', { day:'2-digit', month:'short', year:'numeric' })
  const spdBefore = Math.abs(baseline?.spd || 0)

  const bestCEO = models.reduce((best, m) => {
    const ceo = mitigation_results[m]?.reweigh_adb_ceo
    if (!ceo || ceo.spd == null) return best
    return (!best || Math.abs(ceo.spd) < Math.abs(best.spd)) ? { ...ceo, model: m } : best
  }, null)
  const spdAfter  = Math.abs(bestCEO?.spd || 0)
  const reduction = spdBefore > 0 ? ((spdBefore - spdAfter) / spdBefore * 100).toFixed(1) : '0'

  // Use results.config for full config data (has models/steps)
  // config prop only has target/protected/features
  const fullConfig = results.config || {}
  const fmt = (v, d=4) => (v == null || isNaN(v)) ? 'N/A' : Number(v).toFixed(d)
  const fairSPD = v => v != null && Math.abs(v) < 0.05 ? '✓ Fair' : '✗ Unfair'
  const fairDI  = v => v != null && v >= 0.8 && v <= 1.25 ? '✓ Fair' : '✗ Unfair'

  // Build HTML report
  const html = `<!DOCTYPE html>
<html>
<head>
<meta charset="UTF-8">
<title>FairLens Report</title>
<style>
  * { margin:0; padding:0; box-sizing:border-box; }
  body { font-family: Georgia, serif; color: #1a1a2e; background: #fff; padding: 40px; max-width: 900px; margin: 0 auto; }
  h1 { font-size: 26px; color: #0a0e1a; border-bottom: 3px solid #00b4d8; padding-bottom: 10px; margin-bottom: 6px; }
  h2 { font-size: 16px; color: #00b4d8; margin: 28px 0 10px; text-transform: uppercase; letter-spacing: 1px; }
  h3 { font-size: 13px; color: #333; margin: 14px 0 6px; }
  .meta { color: #666; font-size: 12px; margin-bottom: 24px; }
  .meta span { margin-right: 18px; }
  .kpi-grid { display: grid; grid-template-columns: repeat(4, 1fr); gap: 12px; margin-bottom: 20px; }
  .kpi { background: #f0faff; border: 1px solid #b8e6f5; border-radius: 8px; padding: 14px; text-align: center; }
  .kpi .val { font-size: 22px; font-weight: bold; color: #006d9e; font-family: monospace; }
  .kpi .lbl { font-size: 11px; color: #666; margin-top: 4px; }
  table { width: 100%; border-collapse: collapse; font-size: 12px; margin-bottom: 16px; }
  th { background: #0a0e1a; color: #00b4d8; padding: 8px 10px; text-align: left; font-size: 11px; }
  td { padding: 7px 10px; border-bottom: 1px solid #e8e8e8; }
  tr:nth-child(even) { background: #f9f9f9; }
  .fair   { color: #16a34a; font-weight: bold; }
  .unfair { color: #dc2626; font-weight: bold; }
  .best-row { background: #fffbeb !important; font-weight: bold; }
  .warn { background: #fff7ed; border: 1px solid #fed7aa; border-radius: 6px; padding: 10px 14px; font-size: 12px; color: #9a3412; margin-bottom: 12px; }
  .footer { margin-top: 40px; padding-top: 14px; border-top: 1px solid #ddd; font-size: 11px; color: #999; }
  @media print { body { padding: 20px; } }
</style>
</head>
<body>

<h1>FairLens — Bias Mitigation Report</h1>
<div class="meta">
  <span>📅 ${date}</span>
  <span>🎯 Target: <strong>${fullConfig.target_column || config.target_column}</strong></span>
  <span>🔒 Protected: <strong>${fullConfig.protected_attribute || config.protected_attribute}</strong></span>
  <span>⭐ Privileged: <strong>${fullConfig.privileged_value || config.privileged_value}</strong></span>
  <span>📊 Rows: <strong>${baseline?.total_rows?.toLocaleString()}</strong></span>
</div>

${(baseline?.warnings||[]).map(w => `<div class="warn">⚠ ${w.msg}</div>`).join('')}

<h2>Executive Summary</h2>
<div class="kpi-grid">
  <div class="kpi"><div class="val">${fmt(baseline?.spd)}</div><div class="lbl">Baseline SPD</div></div>
  <div class="kpi"><div class="val">${fmt(bestCEO?.spd)}</div><div class="lbl">Final SPD (${bestCEO?.model||'—'})</div></div>
  <div class="kpi"><div class="val">${reduction}%</div><div class="lbl">Bias Reduction</div></div>
  <div class="kpi"><div class="val">${fmt(summary?.best_accuracy,4)}</div><div class="lbl">Best Accuracy</div></div>
</div>

<h2>Full Pipeline Results</h2>
<table>
  <tr><th>Model</th><th>Stage</th><th>Accuracy</th><th>SPD</th><th>SPD Status</th><th>DI</th><th>DI Status</th><th>AOD</th><th>EOD</th></tr>
  ${models.map(m => {
    const stages = mitigation_results[m] || {}
    const sk = Object.keys(stages)
    return sk.map((stage, si) => {
      const v = stages[stage]
      const isBest = stage === 'reweigh_adb_ceo'
      const stageLabel = {original:'Baseline',reweigh:'Reweighing',reweigh_adb:'Reweigh+ADB',reweigh_adb_ceo:'Full Pipeline'}[stage]||stage
      return `<tr class="${isBest?'best-row':''}">
        ${si===0?`<td rowspan="${sk.length}" style="font-weight:bold;vertical-align:top;padding-top:10px">${m}</td>`:''}
        <td>${isBest?'★ ':''} ${stageLabel}</td>
        <td>${fmt(v.accuracy,4)}</td>
        <td style="font-family:monospace">${fmt(v.spd)}</td>
        <td class="${v.spd!=null&&Math.abs(v.spd)<0.05?'fair':'unfair'}">${fairSPD(v.spd)}</td>
        <td style="font-family:monospace">${fmt(v.di)}</td>
        <td class="${v.di!=null&&v.di>=0.8&&v.di<=1.25?'fair':'unfair'}">${fairDI(v.di)}</td>
        <td style="font-family:monospace">${fmt(v.aod)}</td>
        <td style="font-family:monospace">${fmt(v.eod)}</td>
      </tr>`
    }).join('')
  }).join('')}
</table>

<h2>SMOTE Oversampling Results</h2>
${Object.keys(smote_results||{}).length === 0
  ? '<p style="color:#666;font-size:12px">No SMOTE variants were run.</p>'
  : `<table>
  <tr><th>Variant</th><th>Accuracy</th><th>SPD</th><th>DI</th><th>AOD</th><th>EOD</th></tr>
  <tr style="background:#fff7ed"><td><em>Baseline (no SMOTE)</em></td><td>—</td>
    <td>${fmt(baseline?.spd)}</td><td>${fmt(baseline?.di)}</td><td>—</td><td>—</td></tr>
  ${Object.entries(smote_results||{}).map(([k,v]) => v.error
    ? `<tr><td>${k}</td><td colspan="5" style="color:#dc2626">${v.error}</td></tr>`
    : `<tr><td>${k}</td><td>${fmt(v.accuracy,4)}</td><td>${fmt(v.spd)}</td><td>${fmt(v.di)}</td><td>${fmt(v.aod)}</td><td>${fmt(v.eod)}</td></tr>`
  ).join('')}
</table>`}

${results.feature_importance?.length > 0 ? `
<h2>Feature Importance (Top 10)</h2>
<table>
  <tr><th>Feature</th><th>Importance</th><th>Visual</th></tr>
  ${results.feature_importance.slice(0,10).map(({feature,importance}) => `
  <tr ${feature===config.protected_attribute?'style="background:#fffbeb"':''}>
    <td style="font-family:monospace">${feature===config.protected_attribute?'⚠ ':''}${feature}</td>
    <td>${(importance*100).toFixed(2)}%</td>
    <td><div style="background:#00b4d8;height:10px;border-radius:4px;width:${Math.min(importance/results.feature_importance[0].importance*100,100)}%"></div></td>
  </tr>`).join('')}
</table>` : ''}

<h2>Configuration</h2>
<table>
  <tr><th>Parameter</th><th>Value</th></tr>
  <tr><td>Models</td><td>${(fullConfig.models || []).join(', ')}</td></tr>
  <tr><td>Mitigation Steps</td><td>${(fullConfig.mitigation_steps || []).join(' → ')}</td></tr>
  <tr><td>SMOTE Variants</td><td>${(fullConfig.smote_variants || []).join(', ') || 'None'}</td></tr>
  <tr><td>Test Split</td><td>${(((fullConfig.test_size||config.test_size)||0.3)*100).toFixed(0)}%</td></tr>
  <tr><td>Feature Count</td><td>${(fullConfig.feature_columns||config.feature_columns||[]).length}</td></tr>
  <tr><td>Total Rows</td><td>${baseline?.total_rows?.toLocaleString()}</td></tr>
  <tr><td>Class Distribution</td><td>0: ${baseline?.class_distribution?.[0]?.toLocaleString()} · 1: ${baseline?.class_distribution?.[1]?.toLocaleString()}</td></tr>
</table>

<div class="footer">
  Generated by FairLens · Fairness detection and mitigation framework ·
  Fair zones: |SPD| &lt; 0.05 · DI ∈ [0.8, 1.25] · |AOD| &lt; 0.05 · |EOD| &lt; 0.05
</div>

</body>
</html>`

  // Open in new window and trigger print dialog
  const win = window.open('', '_blank')
  win.document.write(html)
  win.document.close()
  win.focus()
  setTimeout(() => win.print(), 500)
}

export default function StepResults({ results, config, onReset }) {
  const [tab, setTab] = useState('overview')
  const { summary, mitigation_results, smote_results, baseline } = results
  const models = Object.keys(mitigation_results || {})

  const tabs = [
    { id:'overview', label:'📊 Overview'  },
    { id:'pipeline', label:'⚙️ Pipeline'  },
    { id:'models',   label:'🤖 Per Model' },
    { id:'insights', label:'🔬 Insights'  },
    { id:'smote',    label:'🧬 SMOTE'     },
  ]

  return (
    <div className="fade-up max-w-6xl mx-auto space-y-5 pb-16">

      {/* Top bar */}
      <div className="flex items-start justify-between gap-4 flex-wrap">
        <div>
          <h2 className="font-display font-bold text-2xl text-white">Fairness Analysis Results</h2>
          <p className="text-ink-400 font-body text-sm mt-1">
            Protected: <span className="text-cyan font-mono">{config.protected_attribute}</span>
            <span className="text-ink-400 mx-2">·</span>
            Privileged: <span className="text-cyan font-mono">{config.privileged_value}</span>
            <span className="text-ink-400 mx-2">·</span>
            Target: <span className="text-cyan font-mono">{config.target_column}</span>
            <span className="text-ink-400 mx-2">·</span>
            <span className="text-ink-400">{models.length} models · {baseline?.total_rows?.toLocaleString()} rows</span>
          </p>
        </div>
        <div className="flex gap-2">
          <button onClick={() => {
            const a = Object.assign(document.createElement('a'), {
              href: URL.createObjectURL(new Blob([JSON.stringify(results,null,2)],{type:'application/json'})),
              download: 'fairlens_results.json'
            }); a.click()
          }} className="btn-ghost text-xs">↓ Export JSON</button>
          <button onClick={() => generatePDF(results, config)}
            className="btn-ghost text-xs">📄 Download Report</button>
          <button onClick={onReset} className="btn-ghost text-xs">↺ New Run</button>
        </div>
      </div>

      <HeroBanner results={results} config={config}/>

      {/* Null metrics banner — all SPD null means protected attribute encoding issue */}
      {results.baseline && results.baseline.spd == null && (
        <div className="flex items-start gap-3 rounded-xl p-4 border border-amber-500/50 bg-amber-500/15 text-amber-200 text-sm font-body">
          <span className="text-xl flex-shrink-0">⚠️</span>
          <div>
            <p className="font-bold mb-1">Fairness metrics could not be computed</p>
            <p className="text-xs opacity-90">
              The privileged group had 0 members in the dataset — this usually means the protected attribute
              was pre-encoded (e.g. White=0 in a numeric column). The engine auto-corrects this.
              If you still see this message, try selecting a different privileged value or check that
              your protected attribute column is correctly configured.
            </p>
          </div>
        </div>
      )}

      {/* Runtime warnings from engine */}
      {results.baseline?.warnings?.length > 0 && (
        <div className="space-y-2">
          {results.baseline.warnings.map((w, i) => (
            <div key={i} className={`flex items-start gap-3 rounded-xl p-4 border text-sm font-body ${
              w.level === 'error'
                ? 'border-rose-500/40 bg-rose-500/15 text-rose-300'
                : 'border-amber-500/40 bg-amber-500/15 text-amber-300'
            }`}>
              <span className="text-xl flex-shrink-0">{w.level === 'error' ? '🔴' : '⚠️'}</span>
              <div>
                <p className="font-bold mb-0.5">{w.level === 'error' ? 'Error' : 'Warning'}</p>
                <p className="text-xs opacity-90">{w.msg}</p>
              </div>
            </div>
          ))}
        </div>
      )}

      {/* Tabs */}
      <div className="flex gap-1 bg-ink-800 p-1 rounded-xl overflow-x-auto">
        {tabs.map(t => (
          <button key={t.id} onClick={() => setTab(t.id)}
            className={`px-4 py-2 rounded-lg font-display text-xs font-bold transition-all whitespace-nowrap ${
              tab===t.id ? 'text-white shadow-sm' : 'text-ink-500 hover:text-ink-300'
            }`}
            style={tab===t.id ? {background:'rgba(0,229,255,0.12)',borderBottom:'2px solid #00E5FF'} : {}}>
            {t.label}
          </button>
        ))}
      </div>

      {tab==='overview' && (
        <div className="space-y-4 fade-up">
          <ModelCards mitigation_results={mitigation_results}/>
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
            <PipelineChart mitigation_results={mitigation_results} metric="spd" label="SPD" optimal={0}/>
            <AccFairChart  mitigation_results={mitigation_results}/>
          </div>
          <FeatureImportanceChart
            feature_importance={results.feature_importance}
            protected_attribute={config.protected_attribute}
          />
        </div>
      )}

      {tab==='pipeline' && (
        <div className="space-y-4 fade-up">
          <PipelineFlow mitigation_results={mitigation_results}/>
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
            <PipelineChart mitigation_results={mitigation_results} metric="spd" label="SPD" optimal={0}/>
            <PipelineChart mitigation_results={mitigation_results} metric="di"  label="DI"  optimal={1}/>
          </div>
        </div>
      )}

      {tab==='models' && (
        <div className="fade-up space-y-4">
          <ModelDeepDive mitigation_results={mitigation_results}/>
          {Object.keys(mitigation_results).map(m => (
            <GroupConfusionMatrix
              key={m}
              model={m}
              stages={mitigation_results[m]}
              config={config}
            />
          ))}
        </div>
      )}

      {tab==='insights' && (
        <div className="fade-up">
          <AnalysisInsights results={results}/>
        </div>
      )}

      {tab==='smote' && (
        <div className="fade-up">
          <SmoteSection smote_results={smote_results} baseline={baseline}/>
        </div>
      )}

    </div>
  )
}
