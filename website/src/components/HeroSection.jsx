import React, { useMemo, useState } from 'react'
import { Github, Package, FileText, Sparkles, Zap, Target, Copy, Check, Terminal, Award } from 'lucide-react'
import AnimatedCounter from './AnimatedCounter'
import { usePyPIVersion } from '../hooks/usePyPIVersion'

export default function HeroSection({ rankedData }) {
  const pypiVersion = usePyPIVersion()
  const [copied, setCopied] = useState(false)

  const handleCopyPip = () => {
    navigator.clipboard.writeText('pip install llmthinkbench').then(() => {
      setCopied(true)
      setTimeout(() => setCopied(false), 2000)
    })
  }
  const stats = useMemo(() => {
    const maxAcc = Math.max(...rankedData.map((m) => m.accuracy))
    const minAcc = Math.min(...rankedData.map((m) => m.accuracy))
    const avgEff = rankedData.reduce((s, m) => s + m.efficiency, 0) / rankedData.length
    return { count: rankedData.length, maxAcc, minAcc, avgEff }
  }, [rankedData])

  return (
    <section className="relative py-24 md:py-36 px-6 overflow-hidden">
      {/* Scanline overlay */}
      <div className="absolute inset-0 pointer-events-none z-[2] opacity-30"
        style={{
          background: 'repeating-linear-gradient(0deg, transparent, transparent 2px, rgba(102, 126, 234, 0.008) 2px, rgba(102, 126, 234, 0.008) 4px)',
        }}
      />

      {/* Animated wireframe shapes */}
      <div className="absolute inset-0 pointer-events-none overflow-hidden">
        {/* Large rotating wireframe hexagon */}
        <div
          className="absolute top-16 left-[8%] w-28 h-28 wireframe-shape border-neon-indigo/15 rounded-lg opacity-30"
          style={{ animationDuration: '25s' }}
        />
        {/* Orbiting circle */}
        <div
          className="absolute top-32 right-[12%] w-20 h-20 wireframe-shape border-neon-purple/15 rounded-full opacity-25"
          style={{ animationDuration: '18s', animationDirection: 'reverse' }}
        />
        {/* Smaller shapes with depth */}
        <div
          className="absolute bottom-24 left-[15%] w-14 h-14 wireframe-shape border-neon-pink/10 rounded-lg opacity-20"
          style={{ animationDuration: '22s' }}
        />
        <div
          className="absolute top-24 left-[55%] w-32 h-32 wireframe-shape border-neon-cyan/8 rounded-full opacity-15"
          style={{ animationDuration: '30s', animationDirection: 'reverse' }}
        />
        <div
          className="absolute bottom-20 right-[20%] w-16 h-16 wireframe-shape border-neon-indigo/12 rounded-lg opacity-20"
          style={{ animationDuration: '15s' }}
        />
        {/* Orbit ring decorations */}
        <div className="absolute top-1/2 left-1/2 -translate-x-1/2 -translate-y-1/2 w-[600px] h-[600px] border border-neon-indigo/[0.04] rounded-full animate-spin-slower" />
        <div className="absolute top-1/2 left-1/2 -translate-x-1/2 -translate-y-1/2 w-[800px] h-[800px] border border-neon-purple/[0.03] rounded-full animate-counter-spin" />
      </div>

      <div className="max-w-5xl mx-auto text-center relative z-10">
        {/* Main title */}
        <div className="mb-8 reveal">
          {/* ACL 2026 acceptance badge */}
          <a
            href="https://www.arxiv.org/abs/2507.04023"
            target="_blank"
            rel="noopener noreferrer"
            className="group inline-flex items-center gap-2.5 px-5 py-2 rounded-full border border-neon-pink/30 bg-gradient-to-r from-neon-pink/10 via-neon-purple/10 to-neon-indigo/10 mb-5 backdrop-blur-sm hover:border-neon-pink/50 transition-all duration-500 hover:shadow-[0_0_30px_rgba(240,147,251,0.2)]"
          >
            <Award size={14} className="text-neon-pink animate-pulse" />
            <span className="text-xs font-semibold tracking-[0.2em] uppercase">
              <span className="text-neon-pink">Accepted</span>
              <span className="text-zinc-500 mx-2">·</span>
              <span className="text-neon-indigo">ACL 2026</span>
            </span>
          </a>

          {/* Framework badge */}
          <div className="inline-flex items-center gap-2 px-5 py-2 rounded-full border border-neon-indigo/20 bg-neon-indigo/5 mb-8 ml-2 backdrop-blur-sm">
            <Sparkles size={14} className="text-neon-indigo animate-pulse" />
            <span className="text-xs font-semibold text-neon-indigo tracking-[0.2em] uppercase">
              AI Benchmark Framework
            </span>
          </div>

          {/* Title with glow effect */}
          <h2 className="text-6xl md:text-8xl lg:text-9xl font-black tracking-tighter mb-8 leading-[0.9]">
            <span className="gradient-text hero-title-glow inline-block">LLMThink</span>
            <span className="text-white hero-title-glow inline-block">Bench</span>
          </h2>

          <p className="text-lg md:text-xl text-zinc-400 max-w-2xl mx-auto leading-relaxed tracking-wide">
            An open-source framework for benchmarking{' '}
            <span className="text-neon-indigo font-semibold">math reasoning</span>
            {' '}&{' '}
            <span className="text-neon-pink font-semibold">overthinking</span>
            {' '}in LLMs.
          </p>
        </div>

        {/* Pip install command - modern terminal style */}
        <div className="flex justify-center mb-8 reveal" style={{ transitionDelay: '0.15s' }}>
          <button
            onClick={handleCopyPip}
            className="group relative flex items-center gap-3 px-6 py-3.5 rounded-2xl border border-zinc-800/80 bg-[#0c0c1e]/80 backdrop-blur-xl hover:border-neon-indigo/30 transition-all duration-500 hover:shadow-[0_0_30px_rgba(102,126,234,0.1)] overflow-hidden"
          >
            {/* Shimmer sweep on hover */}
            <div className="absolute inset-0 bg-gradient-to-r from-transparent via-neon-indigo/[0.04] to-transparent opacity-0 group-hover:opacity-100 transition-opacity duration-500 group-hover:animate-[shimmer_2s_ease-in-out]" />

            <Terminal size={16} className="text-neon-indigo/60 relative z-10 shrink-0" />
            <code className="font-mono text-sm relative z-10 select-all">
              <span className="text-neon-indigo/50">$</span>{' '}
              <span className="text-zinc-400 group-hover:text-zinc-200 transition-colors">pip install</span>{' '}
              <span className="text-neon-indigo font-semibold">llmthinkbench</span>
            </code>
            <div className="relative z-10 flex items-center justify-center w-8 h-8 rounded-lg bg-zinc-800/50 group-hover:bg-neon-indigo/10 transition-all duration-300 shrink-0">
              {copied ? (
                <Check size={14} className="text-emerald-400" />
              ) : (
                <Copy size={14} className="text-zinc-500 group-hover:text-neon-indigo transition-colors" />
              )}
            </div>
            {copied && (
              <span className="absolute -top-8 left-1/2 -translate-x-1/2 text-[11px] font-medium text-emerald-400 bg-emerald-400/10 border border-emerald-400/20 px-2.5 py-1 rounded-lg whitespace-nowrap animate-[slideUp_0.2s_ease-out]">
                Copied to clipboard
              </span>
            )}
          </button>
        </div>

        {/* Badge links with premium hover */}
        <div className="flex flex-wrap justify-center gap-4 mb-14 reveal" style={{ transitionDelay: '0.3s' }}>
          <a
            href="https://pypi.org/project/llmthinkbench/"
            target="_blank"
            rel="noopener noreferrer"
            className="group relative flex items-center gap-2.5 px-6 py-3 rounded-full border border-neon-indigo/20 bg-neon-indigo/5 hover:bg-neon-indigo/15 hover:border-neon-indigo/50 transition-all duration-500 hover:shadow-neon-indigo hover:-translate-y-1 btn-press overflow-hidden"
          >
            <div className="absolute inset-0 bg-gradient-to-r from-neon-indigo/0 via-neon-indigo/5 to-neon-indigo/0 opacity-0 group-hover:opacity-100 transition-opacity duration-500" />
            <Package size={18} className="text-neon-indigo relative z-10 group-hover:scale-110 transition-transform" />
            <span className="text-sm font-semibold text-zinc-300 group-hover:text-white relative z-10 tracking-wide">
              PyPI {pypiVersion ? `v${pypiVersion}` : ''}
            </span>
          </a>
          <a
            href="https://github.com/ctrl-gaurav/LLMThinkBench"
            target="_blank"
            rel="noopener noreferrer"
            className="group relative flex items-center gap-2.5 px-6 py-3 rounded-full border border-neon-purple/20 bg-neon-purple/5 hover:bg-neon-purple/15 hover:border-neon-purple/50 transition-all duration-500 hover:shadow-neon-purple hover:-translate-y-1 btn-press overflow-hidden"
          >
            <div className="absolute inset-0 bg-gradient-to-r from-neon-purple/0 via-neon-purple/5 to-neon-purple/0 opacity-0 group-hover:opacity-100 transition-opacity duration-500" />
            <Github size={18} className="text-neon-purple relative z-10 group-hover:scale-110 group-hover:rotate-12 transition-all duration-300" />
            <span className="text-sm font-semibold text-zinc-300 group-hover:text-white relative z-10 tracking-wide">GitHub</span>
          </a>
          <a
            href="https://www.arxiv.org/abs/2507.04023"
            target="_blank"
            rel="noopener noreferrer"
            className="group relative flex items-center gap-2.5 px-6 py-3 rounded-full border border-neon-pink/20 bg-neon-pink/5 hover:bg-neon-pink/15 hover:border-neon-pink/50 transition-all duration-500 hover:shadow-neon-pink hover:-translate-y-1 btn-press overflow-hidden"
          >
            <div className="absolute inset-0 bg-gradient-to-r from-neon-pink/0 via-neon-pink/5 to-neon-pink/0 opacity-0 group-hover:opacity-100 transition-opacity duration-500" />
            <FileText size={18} className="text-neon-pink relative z-10 group-hover:scale-110 transition-transform" />
            <span className="text-sm font-semibold text-zinc-300 group-hover:text-white relative z-10 tracking-wide">Preprint (arXiv)</span>
          </a>
        </div>

        {/* Stats grid - HUD style */}
        <div
          className="grid grid-cols-2 md:grid-cols-4 gap-4 md:gap-6 max-w-4xl mx-auto reveal"
          style={{ transitionDelay: '0.4s' }}
        >
          {[
            { icon: Target, color: 'neon-indigo', value: stats.count, suffix: '+', decimals: 0, label: 'Models Tested', iconBg: 'bg-neon-indigo/10', iconHover: 'group-hover:bg-neon-indigo/20' },
            { icon: Zap, color: 'emerald', value: stats.maxAcc, suffix: '%', decimals: 1, label: 'Best Accuracy', iconBg: 'bg-emerald-500/10', iconHover: 'group-hover:bg-emerald-500/20', iconColor: 'text-emerald-400' },
            { icon: Sparkles, color: 'neon-pink', value: stats.minAcc, suffix: '%', decimals: 1, label: 'Min Accuracy', iconBg: 'bg-neon-pink/10', iconHover: 'group-hover:bg-neon-pink/20' },
            { icon: Zap, color: 'neon-cyan', value: stats.avgEff, suffix: '', decimals: 3, label: 'Avg Efficiency', iconBg: 'bg-neon-cyan/10', iconHover: 'group-hover:bg-neon-cyan/20' },
          ].map((stat, i) => {
            const Icon = stat.icon
            return (
              <div
                key={stat.label}
                className="glass-card rounded-2xl p-5 md:p-6 card-hover-3d group hud-stat"
                style={{ animationDelay: `${i * 100}ms` }}
              >
                <div className={`flex items-center justify-center w-10 h-10 md:w-12 md:h-12 rounded-xl ${stat.iconBg} ${stat.iconHover} mx-auto mb-3 transition-all duration-300 group-hover:scale-110`}>
                  <Icon size={22} className={stat.iconColor || `text-${stat.color}`} />
                </div>
                <div className="text-2xl md:text-3xl lg:text-4xl font-black text-white tracking-tight">
                  <AnimatedCounter end={stat.value} suffix={stat.suffix} decimals={stat.decimals} />
                </div>
                <div className="text-[10px] md:text-xs text-zinc-500 mt-1.5 uppercase tracking-[0.15em] font-medium">{stat.label}</div>
              </div>
            )
          })}
        </div>
      </div>
    </section>
  )
}
