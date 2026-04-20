import React, { useMemo } from 'react'
import {
  RadarChart, Radar, PolarGrid, PolarAngleAxis, PolarRadiusAxis,
  ResponsiveContainer, Legend, Tooltip,
} from 'recharts'
import { X, Plus } from 'lucide-react'
import { getMetricColor } from '../data/modelData'

const CHART_COLORS = ['#667eea', '#f093fb', '#10b981', '#f59e0b', '#00E5FF']

export default function ComparisonModal({ rankedData, selectedModels, detailModel, onClose, onToggleModel }) {
  const compareData = useMemo(() => {
    if (detailModel) return [detailModel]
    return Array.from(selectedModels)
      .map((name) => rankedData.find((m) => m.model === name))
      .filter(Boolean)
  }, [selectedModels, rankedData, detailModel])

  const radarData = useMemo(() => {
    if (compareData.length === 0) return []
    const metrics = ['Accuracy', 'O-Score', 'Instruction Following', 'Low Overthinking', 'Token Efficiency']
    return metrics.map((metric, i) => {
      const row = { metric }
      compareData.forEach((model) => {
        const key = model.model.split('/').pop()
        switch (i) {
          case 0: row[key] = model.accuracy; break
          case 1: row[key] = model.efficiency * 100; break
          case 2: row[key] = model.instruction; break
          case 3: row[key] = Math.max(0, 100 - Math.log10(Math.max(model.overthinking, 1)) * 10); break
          case 4: row[key] = Math.max(0, 100 - model.tokens / 10); break
        }
      })
      return row
    })
  }, [compareData])

  if (compareData.length === 0) return null

  const isDetail = !!detailModel

  return (
    <div
      className="fixed inset-0 z-[2000] flex items-center justify-center p-4"
      onClick={(e) => {
        if (e.target === e.currentTarget) onClose()
      }}
    >
      {/* Backdrop with animated gradient edges */}
      <div className="absolute inset-0 bg-black/70 backdrop-enter">
        <div className="absolute inset-0 backdrop-blur-xl" />
        {/* Glowing edges */}
        <div className="absolute top-0 left-0 right-0 h-px bg-gradient-to-r from-transparent via-neon-indigo/20 to-transparent" />
        <div className="absolute bottom-0 left-0 right-0 h-px bg-gradient-to-r from-transparent via-neon-purple/20 to-transparent" />
        <div className="absolute top-0 bottom-0 left-0 w-px bg-gradient-to-b from-transparent via-neon-indigo/10 to-transparent" />
        <div className="absolute top-0 bottom-0 right-0 w-px bg-gradient-to-b from-transparent via-neon-purple/10 to-transparent" />
      </div>

      {/* Modal */}
      <div className="relative w-full max-w-5xl max-h-[90vh] overflow-y-auto glass-strong rounded-2xl border border-neon-indigo/15 shadow-2xl modal-enter">
        {/* Top gradient border */}
        <div className="absolute top-0 left-0 right-0 h-[2px] bg-gradient-to-r from-transparent via-neon-indigo/50 to-transparent" />

        {/* Corner decorations */}
        <div className="absolute top-0 left-0 w-12 h-12 border-t-2 border-l-2 border-neon-indigo/20 rounded-tl-2xl" />
        <div className="absolute top-0 right-0 w-12 h-12 border-t-2 border-r-2 border-neon-purple/20 rounded-tr-2xl" />
        <div className="absolute bottom-0 left-0 w-12 h-12 border-b-2 border-l-2 border-neon-indigo/10 rounded-bl-2xl" />
        <div className="absolute bottom-0 right-0 w-12 h-12 border-b-2 border-r-2 border-neon-purple/10 rounded-br-2xl" />

        {/* Close button */}
        <button
          onClick={onClose}
          className="absolute top-5 right-5 z-10 w-10 h-10 rounded-xl bg-white/5 flex items-center justify-center text-zinc-500 hover:text-red-400 hover:bg-red-500/10 transition-all duration-300 btn-press border border-transparent hover:border-red-500/20"
        >
          <X size={18} />
        </button>

        <div className="p-8 md:p-10">
          <h2 className="text-2xl md:text-3xl font-black gradient-text mb-2 tracking-tight">
            {isDetail ? 'Model Details' : 'Model Comparison'}
          </h2>
          <p className="text-xs text-zinc-600 uppercase tracking-[0.2em] mb-8">
            {isDetail ? 'In-depth performance analysis' : `Comparing ${compareData.length} models`}
          </p>

          {isDetail ? (
            // Single model detail view
            <div>
              <div className="text-center mb-10">
                <h3 className="text-xl md:text-2xl font-bold text-white mb-2">{detailModel.model}</h3>
                <div className="flex items-center justify-center gap-3 text-sm text-zinc-500 flex-wrap">
                  <span className="px-3 py-1 rounded-lg bg-neon-indigo/10 border border-neon-indigo/15 text-neon-indigo font-mono text-xs">
                    Rank #{detailModel.rank}
                  </span>
                  <span className="px-3 py-1 rounded-lg bg-white/5 border border-white/10 font-mono text-xs">
                    {detailModel.params} Params
                  </span>
                  {detailModel.quantization !== 'None' && (
                    <span className="px-3 py-1 rounded-lg bg-neon-purple/10 border border-neon-purple/15 text-neon-purple font-mono text-xs">
                      {detailModel.quantization}
                    </span>
                  )}
                </div>
              </div>

              {/* Key metrics */}
              <div className="grid grid-cols-1 sm:grid-cols-3 gap-4 mb-8">
                {[
                  { label: 'Accuracy', value: `${detailModel.accuracy.toFixed(2)}%`, metric: 'accuracy', val: detailModel.accuracy, gradient: 'from-neon-indigo/10 to-neon-purple/5', border: 'border-neon-indigo/10' },
                  { label: 'O-Score', value: detailModel.efficiency.toFixed(3), metric: 'efficiency', val: detailModel.efficiency * 100, gradient: 'from-neon-pink/10 to-neon-magenta/5', border: 'border-neon-pink/10' },
                  { label: 'Instruction Following', value: `${detailModel.instruction.toFixed(2)}%`, metric: 'instruction', val: detailModel.instruction, gradient: 'from-emerald-500/10 to-emerald-600/5', border: 'border-emerald-500/10' },
                ].map((item) => (
                  <div key={item.label} className={`bg-gradient-to-br ${item.gradient} rounded-2xl p-6 text-center border ${item.border} backdrop-blur-sm`}>
                    <div className="text-[10px] font-bold text-zinc-500 uppercase tracking-[0.2em] mb-3">{item.label}</div>
                    <div className="text-3xl font-black font-mono" style={{ color: getMetricColor(item.val, item.metric) }}>
                      {item.value}
                    </div>
                  </div>
                ))}
              </div>

              {/* Secondary metrics */}
              <div className="grid grid-cols-2 sm:grid-cols-4 gap-3 mb-10">
                {[
                  { label: 'Verbose Ratio', value: detailModel.overthinking.toFixed(1), metric: 'overthinking', val: detailModel.overthinking },
                  { label: 'Avg Tokens', value: detailModel.tokens.toFixed(1), metric: 'tokens', val: detailModel.tokens },
                  { label: 'Avg Words', value: detailModel.words.toFixed(1), metric: 'words', val: detailModel.words },
                  { label: 'Avg Characters', value: detailModel.chars.toFixed(1), metric: 'chars', val: detailModel.chars },
                ].map((item) => (
                  <div key={item.label} className="bg-white/[0.02] rounded-xl p-5 border border-white/[0.04] hover:border-white/[0.08] transition-all">
                    <div className="text-[9px] font-bold text-zinc-600 uppercase tracking-[0.2em] mb-2">{item.label}</div>
                    <div className="text-xl font-bold font-mono" style={{ color: getMetricColor(item.val, item.metric) }}>
                      {item.value}
                    </div>
                  </div>
                ))}
              </div>

              <div className="text-center">
                <button
                  onClick={() => {
                    onToggleModel(detailModel.model)
                    onClose()
                  }}
                  className="relative inline-flex items-center gap-2 px-8 py-3.5 rounded-xl bg-gradient-to-r from-neon-indigo to-neon-purple text-white font-semibold hover:shadow-neon-indigo transition-all btn-press overflow-hidden group"
                >
                  <div className="absolute inset-0 bg-gradient-to-r from-neon-purple to-neon-indigo opacity-0 group-hover:opacity-100 transition-opacity duration-500" />
                  <Plus size={16} className="relative z-10" />
                  <span className="relative z-10">Add to Comparison</span>
                </button>
              </div>
            </div>
          ) : (
            // Multi-model comparison
            <div>
              {/* Comparison table */}
              <div className="overflow-x-auto mb-10 rounded-xl border border-white/[0.06]">
                <table className="w-full text-sm">
                  <thead>
                    <tr className="bg-gradient-to-r from-neon-indigo/[0.08] to-neon-purple/[0.05]">
                      <th className="px-4 py-4 text-left text-[10px] font-bold text-neon-indigo uppercase tracking-[0.2em] border-b border-white/[0.06]">Metric</th>
                      {compareData.map((m) => (
                        <th key={m.model} className="px-4 py-4 text-center text-xs font-bold text-white border-b border-white/[0.06]">
                          {m.model.split('/').pop()}
                        </th>
                      ))}
                    </tr>
                  </thead>
                  <tbody>
                    {[
                      { label: 'Rank', render: (m) => `#${m.rank}` },
                      { label: 'Parameters', render: (m) => m.params },
                      { label: 'Accuracy', render: (m) => `${m.accuracy.toFixed(2)}%`, metric: 'accuracy', val: (m) => m.accuracy },
                      { label: 'O-Score', render: (m) => m.efficiency.toFixed(3), metric: 'efficiency', val: (m) => m.efficiency * 100 },
                      { label: 'Instruction Following', render: (m) => `${m.instruction.toFixed(2)}%`, metric: 'instruction', val: (m) => m.instruction },
                      { label: 'Verbose Ratio', render: (m) => m.overthinking.toFixed(1), metric: 'overthinking', val: (m) => m.overthinking },
                      { label: 'Avg Tokens', render: (m) => m.tokens.toFixed(1), metric: 'tokens', val: (m) => m.tokens },
                    ].map((row, rowIdx) => (
                      <tr key={row.label} className={`border-b border-white/[0.03] ${rowIdx % 2 === 0 ? 'bg-white/[0.01]' : ''}`}>
                        <td className="px-4 py-3.5 font-semibold text-zinc-300 text-xs uppercase tracking-wider">{row.label}</td>
                        {compareData.map((m) => (
                          <td
                            key={m.model}
                            className="px-4 py-3.5 text-center font-mono font-semibold"
                            style={{ color: row.metric ? getMetricColor(row.val(m), row.metric) : '#e4e4e7' }}
                          >
                            {row.render(m)}
                          </td>
                        ))}
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>

              {/* Radar chart */}
              <div className="h-[400px] glass-card rounded-2xl p-6">
                <h4 className="text-xs font-bold text-neon-indigo uppercase tracking-[0.2em] mb-4">Performance Radar</h4>
                <ResponsiveContainer width="100%" height="90%">
                  <RadarChart data={radarData}>
                    <PolarGrid stroke="rgba(255,255,255,0.06)" />
                    <PolarAngleAxis dataKey="metric" tick={{ fill: '#a1a1aa', fontSize: 11, fontWeight: 500 }} />
                    <PolarRadiusAxis
                      angle={90}
                      domain={[0, 100]}
                      tick={{ fill: '#52525b', fontSize: 9 }}
                      axisLine={false}
                    />
                    {compareData.map((model, i) => (
                      <Radar
                        key={model.model}
                        name={model.model.split('/').pop()}
                        dataKey={model.model.split('/').pop()}
                        stroke={CHART_COLORS[i]}
                        fill={CHART_COLORS[i]}
                        fillOpacity={0.12}
                        strokeWidth={2}
                      />
                    ))}
                    <Legend wrapperStyle={{ fontSize: 12, fontWeight: 500 }} />
                    <Tooltip
                      contentStyle={{
                        background: 'rgba(12, 12, 30, 0.95)',
                        border: '1px solid rgba(102, 126, 234, 0.3)',
                        borderRadius: '16px',
                        backdropFilter: 'blur(20px)',
                      }}
                    />
                  </RadarChart>
                </ResponsiveContainer>
              </div>
            </div>
          )}
        </div>
      </div>
    </div>
  )
}
