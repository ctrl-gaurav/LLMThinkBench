import React, { useEffect, useRef, memo } from 'react'

const ParticleBackground = memo(function ParticleBackground() {
  const containerRef = useRef(null)

  useEffect(() => {
    const container = containerRef.current
    if (!container) return

    const particleCount = 80
    const particles = []

    const colors = [
      'rgba(102, 126, 234, 0.5)',
      'rgba(102, 126, 234, 0.3)',
      'rgba(124, 77, 255, 0.4)',
      'rgba(124, 77, 255, 0.25)',
      'rgba(240, 147, 251, 0.3)',
      'rgba(240, 147, 251, 0.2)',
      'rgba(0, 229, 255, 0.25)',
      'rgba(0, 229, 255, 0.15)',
    ]

    for (let i = 0; i < particleCount; i++) {
      const particle = document.createElement('div')
      particle.className = 'particle'

      // Vary sizes for depth effect
      const sizeGroup = Math.random()
      let size, opacity, duration, glowSize

      if (sizeGroup < 0.5) {
        // Small distant particles (more common)
        size = Math.random() * 1.5 + 0.5
        opacity = 0.3 + Math.random() * 0.3
        duration = Math.random() * 15 + 18
        glowSize = size * 2
      } else if (sizeGroup < 0.85) {
        // Medium particles
        size = Math.random() * 2.5 + 1.5
        opacity = 0.4 + Math.random() * 0.3
        duration = Math.random() * 12 + 14
        glowSize = size * 3
      } else {
        // Large close particles (rare)
        size = Math.random() * 3 + 2.5
        opacity = 0.5 + Math.random() * 0.3
        duration = Math.random() * 8 + 10
        glowSize = size * 4
      }

      const color = colors[Math.floor(Math.random() * colors.length)]
      const xDrift = Math.random() * 100 - 50 // Random X drift

      particle.style.cssText = `
        width: ${size}px;
        height: ${size}px;
        left: ${Math.random() * 100}%;
        background: ${color};
        opacity: ${opacity};
        animation: particleFloat ${duration}s infinite linear;
        animation-delay: ${Math.random() * duration}s;
        box-shadow: 0 0 ${glowSize}px ${color};
        --x-drift: ${xDrift}px;
      `

      container.appendChild(particle)
      particles.push(particle)
    }

    return () => {
      particles.forEach((p) => p.remove())
    }
  }, [])

  return (
    <div
      ref={containerRef}
      className="fixed inset-0 z-[-1] pointer-events-none overflow-hidden"
      aria-hidden="true"
    />
  )
})

export default ParticleBackground
