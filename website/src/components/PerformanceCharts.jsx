import React, { useMemo, memo } from 'react'
import {
  ScatterChart, Scatter, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer,
  BarChart, Bar, LineChart, Line, AreaChart, Area, RadarChart, Radar, PolarGrid,
  PolarAngleAxis, PolarRadiusAxis, Legend, Cell, PieChart, Pie,
  ComposedChart,
} from 'recharts'
import { BarChart3 } from 'lucide-react'

const COLORS = {
  indigo: '#667eea',
  purple: '#7C4DFF',
  pink: '#f093fb',
  magenta: '#ff6b9d',
  cyan: '#00E5FF',
  teal: '#00BFA5',
  emerald: '#10b981',
  amber: '#f59e0b',
  red: '#ef4444',
}

const CustomTooltip = ({ active, payload, label, formatter }) => {
  if (!active || !payload?.length) return null
  return (
    <div className="recharts-custom-tooltip">
      {label && <p className="text-xs text-zinc-400 mb-1.5 font-medium">{label}</p>}
      {payload.map((entry, i) => (
        <p key={i} className="text-sm font-semibold" style={{ color: entry.color || entry.fill }}>
          {entry.name}: {formatter ? formatter(entry.value, entry.name) : entry.value}
        </p>
      ))}
    </div>
  )
}

const ScatterTooltip = ({ active, payload }) => {
  if (!active || !payload?.length) return null
  const data = payload[0]?.payload
  if (!data) return null
  return (
    <div className="recharts-custom-tooltip">
      <p className="text-sm font-bold text-white mb-1.5">{data.name}</p>
      <p className="text-xs text-neon-indigo font-mono">Accuracy: {data.x?.toFixed(2)}%</p>
      <p className="text-xs text-neon-pink font-mono">O-Score: {(data.y / 100)?.toFixed(3)}</p>
    </div>
  )
}

const BubbleTooltip = ({ active, payload }) => {
  if (!active || !payload?.length) return null
  const data = payload[0]?.payload
  if (!data) return null
  return (
    <div className="recharts-custom-tooltip">
      <p className="text-sm font-bold text-white mb-1.5">{data.name}</p>
      <p className="text-xs text-red-400 font-mono">Verbose Ratio: {data.overthinking?.toFixed(1)}</p>
      <p className="text-xs text-neon-indigo font-mono">Accuracy: {data.y?.toFixed(2)}%</p>
      <p className="text-xs text-neon-cyan font-mono">O-Score: {data.efficiency?.toFixed(3)}</p>
    </div>
  )
}

const ChartCard = ({ title, children }) => (
  <div className="glass-card rounded-2xl p-6 md:p-8 reveal-scale chart-card-bg group/chart hover:border-neon-indigo/20 transition-all duration-500">
    <h3 className="text-[10px] font-bold text-neon-indigo uppercase tracking-[0.2em] mb-5 flex items-center gap-2 relative z-10">
      <span className="w-1.5 h-1.5 rounded-full bg-neon-indigo animate-pulse" />
      {title}
    </h3>
    <div className="h-[320px] relative z-10">{children}</div>
  </div>
)

const PerformanceCharts = memo(function PerformanceCharts({ rankedData }) {
  // 1. Accuracy vs Efficiency scatter
  const scatterData = useMemo(
    () =>
      rankedData.map((m) => ({
        x: m.accuracy,
        y: m.efficiency * 100,
        name: m.model.split('/').pop(),
        z: 80,
      })),
    [rankedData]
  )

  // 2. Model Size vs Performance (Top 15)
  const sizeData = useMemo(() => {
    const sorted = [...rankedData].sort((a, b) => b.accuracy - a.accuracy)
    return sorted.slice(0, 15).map((m) => ({
      name: m.model.split('/').pop(),
      accuracy: m.accuracy,
      params: parseFloat(m.params.replace(/[^0-9.]/g, '')) || 0,
    }))
  }, [rankedData])

  // 3. Overthinking vs Accuracy bubble
  const bubbleData = useMemo(
    () =>
      rankedData.map((m) => ({
        x: Math.log10(Math.max(m.overthinking, 1)),
        y: m.accuracy,
        z: m.efficiency * 40 + 5,
        name: m.model.split('/').pop(),
        overthinking: m.overthinking,
        efficiency: m.efficiency,
      })),
    [rankedData]
  )

  // 4. Radar chart by family
  const radarData = useMemo(() => {
    const families = {
      Qwen: rankedData.filter((m) => m.model.includes('Qwen')),
      Phi: rankedData.filter((m) => m.model.includes('Phi') || m.model.includes('phi')),
      Llama: rankedData.filter((m) => m.model.includes('Llama') || m.model.includes('llama')),
      Mistral: rankedData.filter((m) => m.model.includes('Mistral') || m.model.includes('mistral')),
      GPT: rankedData.filter((m) => m.model.includes('GPT')),
    }

    const metrics = ['Accuracy', 'O-Score', 'Instruction', 'Token Eff.', 'Low Tokens']
    return metrics.map((metric, i) => {
      const row = { metric }
      Object.entries(families).forEach(([family, models]) => {
        if (models.length === 0) {
          row[family] = 0
          return
        }
        const avg = (fn) => models.reduce((s, m) => s + fn(m), 0) / models.length
        switch (i) {
          case 0: row[family] = avg((m) => m.accuracy); break
          case 1: row[family] = avg((m) => m.efficiency * 100); break
          case 2: row[family] = avg((m) => m.instruction); break
          case 3: row[family] = avg((m) => (m.accuracy / m.tokens) * 100); break
          case 4: row[family] = Math.max(0, 100 - avg((m) => m.tokens) / 10); break
        }
      })
      return row
    })
  }, [rankedData])

  // 5. Instruction following distribution (doughnut -> pie)
  const pieData = useMemo(() => {
    const ranges = [
      { min: 0, max: 30, label: 'Low (0-30%)', color: COLORS.red },
      { min: 30, max: 50, label: 'Fair (30-50%)', color: COLORS.amber },
      { min: 50, max: 70, label: 'Good (50-70%)', color: COLORS.emerald },
      { min: 70, max: 100, label: 'Excellent (70%+)', color: COLORS.indigo },
    ]
    return ranges
      .map((r) => {
        const models = rankedData.filter((m) => m.accuracy >= r.min && m.accuracy < r.max)
        if (models.length === 0) return null
        const avgInstr = models.reduce((s, m) => s + m.instruction, 0) / models.length
        return { name: `${r.label} (${models.length})`, value: avgInstr, fill: r.color, count: models.length }
      })
      .filter(Boolean)
  }, [rankedData])

  // 6. Top 20 performance area chart
  const areaData = useMemo(() => {
    const sorted = [...rankedData].sort((a, b) => a.rank - b.rank)
    return sorted.slice(0, 20).map((m) => ({
      name: `#${m.rank}`,
      fullName: m.model.split('/').pop(),
      accuracy: m.accuracy,
      instruction: m.instruction,
      efficiency: m.efficiency * 100,
    }))
  }, [rankedData])

  const radarFamilyColors = {
    Qwen: COLORS.emerald,
    Phi: COLORS.indigo,
    Llama: COLORS.pink,
    Mistral: COLORS.amber,
    GPT: COLORS.cyan,
  }

  const tooltipStyle = {
    background: 'rgba(12, 12, 30, 0.95)',
    border: '1px solid rgba(102, 126, 234, 0.25)',
    borderRadius: '16px',
  }

  return (
    <section className="max-w-7xl mx-auto px-6 mb-16 pt-8">
      <h2 className="text-3xl md:text-5xl font-black text-center mb-12 reveal tracking-tight">
        <BarChart3 className="inline-block mr-3 text-neon-indigo animate-pulse-soft" size={36} />
        <span className="gradient-text">Performance Insights</span>
      </h2>

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6 md:gap-8">
        {/* 1. Accuracy vs O-Score */}
        <ChartCard title="Accuracy vs O-Score Trade-off">
          <ResponsiveContainer width="100%" height="100%">
            <ScatterChart margin={{ top: 10, right: 20, bottom: 20, left: 10 }}>
              <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.04)" />
              <XAxis
                dataKey="x"
                type="number"
                name="Accuracy"
                unit="%"
                tick={{ fill: '#71717a', fontSize: 11 }}
                axisLine={{ stroke: 'rgba(255,255,255,0.06)' }}
                label={{ value: 'Accuracy (%)', position: 'insideBottom', offset: -10, fill: '#52525b', fontSize: 11 }}
              />
              <YAxis
                dataKey="y"
                type="number"
                name="O-Score"
                tick={{ fill: '#71717a', fontSize: 11 }}
                axisLine={{ stroke: 'rgba(255,255,255,0.06)' }}
                label={{ value: 'O-Score (%)', angle: -90, position: 'insideLeft', offset: 0, fill: '#52525b', fontSize: 11 }}
              />
              <Tooltip content={<ScatterTooltip />} />
              <Scatter data={scatterData} fill={COLORS.indigo} fillOpacity={0.7} stroke={COLORS.indigo} strokeWidth={1} />
            </ScatterChart>
          </ResponsiveContainer>
        </ChartCard>

        {/* 2. Model Size vs Performance */}
        <ChartCard title="Model Size vs Performance (Top 15)">
          <ResponsiveContainer width="100%" height="100%">
            <ComposedChart data={sizeData} margin={{ top: 10, right: 20, bottom: 60, left: 10 }}>
              <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.04)" />
              <XAxis
                dataKey="name"
                tick={{ fill: '#71717a', fontSize: 9, angle: -45, textAnchor: 'end' }}
                axisLine={{ stroke: 'rgba(255,255,255,0.06)' }}
                height={80}
              />
              <YAxis
                yAxisId="left"
                tick={{ fill: '#71717a', fontSize: 11 }}
                axisLine={{ stroke: 'rgba(255,255,255,0.06)' }}
                label={{ value: 'Accuracy (%)', angle: -90, position: 'insideLeft', fill: '#52525b', fontSize: 11 }}
              />
              <YAxis
                yAxisId="right"
                orientation="right"
                tick={{ fill: '#71717a', fontSize: 11 }}
                axisLine={{ stroke: 'rgba(255,255,255,0.06)' }}
                label={{ value: 'Params (B)', angle: 90, position: 'insideRight', fill: '#52525b', fontSize: 11 }}
              />
              <Tooltip contentStyle={tooltipStyle} labelStyle={{ color: '#e4e4e7' }} />
              <Legend wrapperStyle={{ fontSize: 11, fontWeight: 500 }} />
              <Bar yAxisId="left" dataKey="accuracy" name="Accuracy (%)" fill={COLORS.indigo} fillOpacity={0.8} radius={[6, 6, 0, 0]} />
              <Line yAxisId="right" dataKey="params" name="Parameters (B)" stroke={COLORS.pink} strokeWidth={2} dot={{ fill: COLORS.pink, r: 4, strokeWidth: 0 }} />
            </ComposedChart>
          </ResponsiveContainer>
        </ChartCard>

        {/* 3. Overthinking vs Accuracy (Bubble) */}
        <ChartCard title="Overthinking vs Accuracy">
          <ResponsiveContainer width="100%" height="100%">
            <ScatterChart margin={{ top: 10, right: 20, bottom: 20, left: 10 }}>
              <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.04)" />
              <XAxis
                dataKey="x"
                type="number"
                name="Overthinking (log)"
                tick={{ fill: '#71717a', fontSize: 11 }}
                axisLine={{ stroke: 'rgba(255,255,255,0.06)' }}
                label={{ value: 'Overthinking (log scale)', position: 'insideBottom', offset: -10, fill: '#52525b', fontSize: 11 }}
              />
              <YAxis
                dataKey="y"
                type="number"
                name="Accuracy"
                tick={{ fill: '#71717a', fontSize: 11 }}
                axisLine={{ stroke: 'rgba(255,255,255,0.06)' }}
                label={{ value: 'Accuracy (%)', angle: -90, position: 'insideLeft', fill: '#52525b', fontSize: 11 }}
              />
              <Tooltip content={<BubbleTooltip />} />
              <Scatter data={bubbleData} fill={COLORS.red} fillOpacity={0.5} stroke={COLORS.red} strokeWidth={1}>
                {bubbleData.map((_, i) => (
                  <Cell key={i} r={bubbleData[i].z} />
                ))}
              </Scatter>
            </ScatterChart>
          </ResponsiveContainer>
        </ChartCard>

        {/* 4. O-Score Radar by Family */}
        <ChartCard title="O-Score by Model Family">
          <ResponsiveContainer width="100%" height="100%">
            <RadarChart data={radarData} margin={{ top: 10, right: 30, bottom: 10, left: 30 }}>
              <PolarGrid stroke="rgba(255,255,255,0.06)" />
              <PolarAngleAxis dataKey="metric" tick={{ fill: '#a1a1aa', fontSize: 10, fontWeight: 500 }} />
              <PolarRadiusAxis
                angle={90}
                domain={[0, 100]}
                tick={{ fill: '#52525b', fontSize: 9 }}
                axisLine={false}
              />
              {Object.entries(radarFamilyColors).map(([family, color]) => (
                <Radar
                  key={family}
                  name={family}
                  dataKey={family}
                  stroke={color}
                  fill={color}
                  fillOpacity={0.12}
                  strokeWidth={2}
                />
              ))}
              <Legend wrapperStyle={{ fontSize: 11, fontWeight: 500 }} />
              <Tooltip contentStyle={tooltipStyle} />
            </RadarChart>
          </ResponsiveContainer>
        </ChartCard>

        {/* 5. Instruction Following Distribution */}
        <ChartCard title="Instruction Following by Accuracy Range">
          <ResponsiveContainer width="100%" height="100%">
            <PieChart>
              <Pie
                data={pieData}
                cx="50%"
                cy="50%"
                innerRadius="50%"
                outerRadius="80%"
                paddingAngle={3}
                dataKey="value"
                nameKey="name"
                label={({ name, value }) => `${value.toFixed(1)}%`}
                labelLine={{ stroke: 'rgba(255,255,255,0.15)' }}
              >
                {pieData.map((entry, i) => (
                  <Cell key={i} fill={entry.fill} stroke={entry.fill} strokeWidth={1} />
                ))}
              </Pie>
              <Tooltip
                contentStyle={tooltipStyle}
                formatter={(val) => `${val.toFixed(1)}% avg instruction following`}
              />
              <Legend wrapperStyle={{ fontSize: 11, fontWeight: 500 }} />
            </PieChart>
          </ResponsiveContainer>
        </ChartCard>

        {/* 6. Top 20 Performance Trends (Area) */}
        <ChartCard title="Top 20 Models Performance Trends">
          <ResponsiveContainer width="100%" height="100%">
            <AreaChart data={areaData} margin={{ top: 10, right: 20, bottom: 20, left: 10 }}>
              <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.04)" />
              <XAxis
                dataKey="name"
                tick={{ fill: '#71717a', fontSize: 10 }}
                axisLine={{ stroke: 'rgba(255,255,255,0.06)' }}
                label={{ value: 'Model Rank', position: 'insideBottom', offset: -10, fill: '#52525b', fontSize: 11 }}
              />
              <YAxis
                domain={[0, 100]}
                tick={{ fill: '#71717a', fontSize: 11 }}
                axisLine={{ stroke: 'rgba(255,255,255,0.06)' }}
                label={{ value: 'Score', angle: -90, position: 'insideLeft', fill: '#52525b', fontSize: 11 }}
              />
              <Tooltip
                contentStyle={tooltipStyle}
                labelFormatter={(label, payload) => {
                  const item = payload?.[0]?.payload
                  return item?.fullName || label
                }}
                formatter={(value, name) => {
                  if (name === 'O-Score') return [(value / 100).toFixed(3), name]
                  return [`${value.toFixed(2)}%`, name]
                }}
              />
              <Legend wrapperStyle={{ fontSize: 11, fontWeight: 500 }} />
              <defs>
                <linearGradient id="gradientAccuracy" x1="0" y1="0" x2="0" y2="1">
                  <stop offset="0%" stopColor={COLORS.indigo} stopOpacity={0.3} />
                  <stop offset="100%" stopColor={COLORS.indigo} stopOpacity={0.02} />
                </linearGradient>
                <linearGradient id="gradientInstruction" x1="0" y1="0" x2="0" y2="1">
                  <stop offset="0%" stopColor={COLORS.emerald} stopOpacity={0.25} />
                  <stop offset="100%" stopColor={COLORS.emerald} stopOpacity={0.02} />
                </linearGradient>
                <linearGradient id="gradientEfficiency" x1="0" y1="0" x2="0" y2="1">
                  <stop offset="0%" stopColor={COLORS.pink} stopOpacity={0.2} />
                  <stop offset="100%" stopColor={COLORS.pink} stopOpacity={0.02} />
                </linearGradient>
              </defs>
              <Area
                type="monotone"
                dataKey="accuracy"
                name="Accuracy"
                stroke={COLORS.indigo}
                fill="url(#gradientAccuracy)"
                strokeWidth={2}
              />
              <Area
                type="monotone"
                dataKey="instruction"
                name="Instruction Following"
                stroke={COLORS.emerald}
                fill="url(#gradientInstruction)"
                strokeWidth={2}
              />
              <Area
                type="monotone"
                dataKey="efficiency"
                name="O-Score"
                stroke={COLORS.pink}
                fill="url(#gradientEfficiency)"
                strokeWidth={2}
              />
            </AreaChart>
          </ResponsiveContainer>
        </ChartCard>
      </div>
    </section>
  )
})

export default PerformanceCharts
