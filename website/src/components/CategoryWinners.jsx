import React, { useMemo } from 'react'
import { Trophy, Crown, Shield, Brain } from 'lucide-react'

export default function CategoryWinners({ rankedData }) {
  const winners = useMemo(() => {
    const bestAccuracy = rankedData.reduce((a, b) => (a.accuracy > b.accuracy ? a : b))
    const bestEfficiency = rankedData.reduce((a, b) => (a.efficiency > b.efficiency ? a : b))
    const bestInstruction = rankedData.reduce((a, b) => (a.instruction > b.instruction ? a : b))
    return { bestAccuracy, bestEfficiency, bestInstruction }
  }, [rankedData])

  const cards = [
    {
      title: 'Best Overall Accuracy',
      model: winners.bestAccuracy,
      metric: `${winners.bestAccuracy.accuracy.toFixed(2)}%`,
      icon: Crown,
      gradient: 'from-amber-400 via-yellow-500 to-amber-600',
      textColor: 'text-amber-400',
      bgGlow: 'rgba(251, 191, 36, 0.15)',
      borderColor: 'border-amber-500/20',
      accentColor: '#fbbf24',
      medalEmoji: '1',
      isGold: true,
    },
    {
      title: 'Highest O-Score',
      model: winners.bestEfficiency,
      metric: winners.bestEfficiency.efficiency.toFixed(3),
      icon: Shield,
      gradient: 'from-zinc-300 via-zinc-400 to-zinc-500',
      textColor: 'text-zinc-300',
      bgGlow: 'rgba(161, 161, 170, 0.12)',
      borderColor: 'border-zinc-400/20',
      accentColor: '#a1a1aa',
      medalEmoji: '2',
      isGold: false,
    },
    {
      title: 'Best Instruction Following',
      model: winners.bestInstruction,
      metric: `${winners.bestInstruction.instruction.toFixed(2)}%`,
      icon: Brain,
      gradient: 'from-amber-600 via-orange-600 to-amber-800',
      textColor: 'text-amber-600',
      bgGlow: 'rgba(180, 83, 9, 0.12)',
      borderColor: 'border-amber-700/20',
      accentColor: '#d97706',
      medalEmoji: '3',
      isGold: false,
    },
  ]

  return (
    <section className="max-w-7xl mx-auto px-6 mb-16 pt-8">
      <h2 className="text-3xl md:text-5xl font-black text-center mb-12 reveal tracking-tight">
        <Trophy className="inline-block mr-3 text-amber-400 animate-pulse-soft" size={36} />
        <span className="gradient-text">Category Winners</span>
      </h2>

      <div className="grid grid-cols-1 md:grid-cols-3 gap-6 md:gap-8">
        {cards.map((card, index) => {
          const Icon = card.icon
          return (
            <div
              key={card.title}
              className="reveal-scale"
              style={{ transitionDelay: `${index * 0.15}s` }}
            >
              <div
                className={`group relative rounded-2xl overflow-hidden ${card.borderColor} bg-space-600/30 backdrop-blur-md transition-all duration-700 hover:scale-[1.04] hover:-translate-y-3 cursor-pointer perspective-card`}
                style={{
                  boxShadow: `0 0 0 rgba(0,0,0,0)`,
                }}
                onMouseEnter={(e) => {
                  e.currentTarget.style.boxShadow = `0 25px 80px ${card.bgGlow}, 0 0 60px ${card.bgGlow}`
                }}
                onMouseLeave={(e) => {
                  e.currentTarget.style.boxShadow = `0 0 0 rgba(0,0,0,0)`
                }}
              >
                {/* Animated gradient border top */}
                <div className={`h-1 bg-gradient-to-r ${card.gradient} relative overflow-hidden`}>
                  <div className="absolute inset-0 bg-gradient-to-r from-transparent via-white/30 to-transparent animate-shimmer-slow" />
                </div>

                {/* Animated corner accents */}
                <div className="absolute top-0 left-0 w-8 h-8 border-t border-l transition-all duration-500 rounded-tl-2xl opacity-0 group-hover:opacity-100" style={{ borderColor: card.accentColor + '40' }} />
                <div className="absolute top-0 right-0 w-8 h-8 border-t border-r transition-all duration-500 rounded-tr-2xl opacity-0 group-hover:opacity-100" style={{ borderColor: card.accentColor + '40' }} />
                <div className="absolute bottom-0 left-0 w-8 h-8 border-b border-l transition-all duration-500 rounded-bl-2xl opacity-0 group-hover:opacity-100" style={{ borderColor: card.accentColor + '40' }} />
                <div className="absolute bottom-0 right-0 w-8 h-8 border-b border-r transition-all duration-500 rounded-br-2xl opacity-0 group-hover:opacity-100" style={{ borderColor: card.accentColor + '40' }} />

                {/* Shimmer overlay on hover */}
                <div className="absolute inset-0 overflow-hidden pointer-events-none">
                  <div className="absolute -top-1/2 -left-1/2 w-[200%] h-[200%] opacity-0 group-hover:opacity-100 transition-opacity duration-700">
                    <div
                      className="w-full h-full"
                      style={{
                        background: `radial-gradient(circle at center, ${card.bgGlow}, transparent 60%)`,
                      }}
                    />
                  </div>
                </div>

                <div className="relative p-8 md:p-10 text-center perspective-card-inner">
                  {/* Medal / Icon */}
                  <div className={`relative inline-block mb-5 ${card.isGold ? 'sparkle-effect' : ''}`}>
                    <div
                      className={`w-18 h-18 md:w-20 md:h-20 rounded-2xl bg-gradient-to-br ${card.gradient} flex items-center justify-center mx-auto transform group-hover:scale-110 group-hover:rotate-[8deg] transition-all duration-700`}
                      style={{
                        boxShadow: `0 0 30px ${card.bgGlow}, 0 10px 40px ${card.bgGlow}`,
                        width: '5rem',
                        height: '5rem',
                      }}
                    >
                      <Icon size={32} className="text-white drop-shadow-lg" />
                    </div>
                    {/* Glow ring behind icon */}
                    <div
                      className="absolute inset-0 rounded-2xl animate-pulse-ring"
                      style={{
                        background: `radial-gradient(circle, ${card.bgGlow}, transparent 70%)`,
                        width: '5rem',
                        height: '5rem',
                      }}
                    />
                    {/* Rank badge */}
                    <div
                      className={`absolute -top-2 -right-2 w-8 h-8 rounded-full bg-gradient-to-br ${card.gradient} flex items-center justify-center text-xs font-black text-white shadow-xl`}
                      style={{ boxShadow: `0 0 15px ${card.bgGlow}` }}
                    >
                      #{card.medalEmoji}
                    </div>
                  </div>

                  {/* Title */}
                  <h3 className={`text-xs font-bold ${card.textColor} mb-4 uppercase tracking-[0.2em]`}>
                    {card.title}
                  </h3>

                  {/* Model name */}
                  <p className="text-lg md:text-xl font-bold text-white mb-1 group-hover:text-zinc-100 transition-colors truncate px-2">
                    {card.model.model.split('/').pop()}
                  </p>
                  <p className="text-[11px] text-zinc-600 mb-5 truncate font-mono">
                    {card.model.model}
                  </p>

                  {/* Metric value - slot machine style display */}
                  <div
                    className="text-4xl md:text-5xl font-black inline-block font-mono tracking-tight"
                    style={{
                      background: `linear-gradient(135deg, ${card.accentColor}, white, ${card.accentColor})`,
                      backgroundSize: '200% 200%',
                      WebkitBackgroundClip: 'text',
                      WebkitTextFillColor: 'transparent',
                      backgroundClip: 'text',
                      animation: 'gradientText 3s ease infinite',
                    }}
                  >
                    {card.metric}
                  </div>

                  {/* Progress bar with glow animation */}
                  <div className="mt-5 mx-auto max-w-[220px]">
                    <div className="h-2 rounded-full bg-white/5 overflow-hidden relative">
                      <div
                        className={`h-full rounded-full bg-gradient-to-r ${card.gradient} progress-bar-fill relative`}
                        style={{
                          width: `${card.medalEmoji === '2' ? card.model.efficiency * 100 : parseFloat(card.metric)}%`,
                        }}
                      />
                      {/* Glow at the tip */}
                      <div
                        className="absolute top-0 bottom-0 w-2 rounded-full"
                        style={{
                          right: `${100 - (card.medalEmoji === '2' ? card.model.efficiency * 100 : parseFloat(card.metric))}%`,
                          background: card.accentColor,
                          boxShadow: `0 0 10px ${card.accentColor}, 0 0 20px ${card.accentColor}80`,
                        }}
                      />
                    </div>
                  </div>
                </div>
              </div>
            </div>
          )
        })}
      </div>
    </section>
  )
}
