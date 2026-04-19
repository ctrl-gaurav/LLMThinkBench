import React, { useState, useEffect, useRef } from 'react'

export default function AnimatedCounter({ end, duration = 2000, suffix = '', decimals = 0 }) {
  const [count, setCount] = useState(0)
  const ref = useRef(null)
  const hasAnimated = useRef(false)

  useEffect(() => {
    const observer = new IntersectionObserver(
      ([entry]) => {
        if (entry.isIntersecting && !hasAnimated.current) {
          hasAnimated.current = true
          const startTime = performance.now()
          const animate = (currentTime) => {
            const elapsed = currentTime - startTime
            const progress = Math.min(elapsed / duration, 1)
            // easeOutQuart
            const eased = 1 - Math.pow(1 - progress, 4)
            setCount(eased * end)
            if (progress < 1) {
              requestAnimationFrame(animate)
            }
          }
          requestAnimationFrame(animate)
        }
      },
      { threshold: 0.5 }
    )

    if (ref.current) observer.observe(ref.current)
    return () => observer.disconnect()
  }, [end, duration])

  return (
    <span ref={ref} className="font-mono font-black tabular-nums tracking-tight" style={{ fontVariantNumeric: 'tabular-nums' }}>
      {count.toFixed(decimals)}{suffix}
    </span>
  )
}
