'use client'


import { motion, useReducedMotion } from 'framer-motion'

import Hero from '@/components/Hero'
import Features from '@/components/Features'
import TechSpecs from '@/components/TechSpecs'
import Demo from '@/components/Demo'
import Comparison from '@/components/Comparison'
import CTA from '@/components/CTA'
import Navigation from '@/components/Navigation'
import Footer from '@/components/Footer'

export default function Home() {
  const shouldReduceMotion = useReducedMotion()
  
  // Respect user's motion preferences
  const animationProps = shouldReduceMotion 
    ? { initial: {}, animate: {}, transition: {} }
    : {
        initial: { opacity: 0, y: 20 },
        animate: { opacity: 1, y: 0 },
        transition: { duration: 0.4, ease: 'easeOut' }
      }

  return (
    <main className="min-h-screen bg-gradient-to-br from-dental-50 via-white to-mint-50">
      <Navigation />
      
      {/* Hero Section */}
      <motion.section {...animationProps}>
        <Hero />
      </motion.section>

      {/* Features Section */}
      <motion.section 
        {...animationProps}
        transition={{ ...animationProps.transition, delay: 0.1 }}
      >
        <Features />
      </motion.section>

      {/* Interactive Demo */}
      <motion.section 
        {...animationProps}
        transition={{ ...animationProps.transition, delay: 0.2 }}
      >
        <Demo />
      </motion.section>

      {/* Technical Specifications */}
      <motion.section 
        {...animationProps}
        transition={{ ...animationProps.transition, delay: 0.3 }}
      >
        <TechSpecs />
      </motion.section>

      {/* Comparison with Commercial Solutions */}
      <motion.section 
        {...animationProps}
        transition={{ ...animationProps.transition, delay: 0.4 }}
      >
        <Comparison />
      </motion.section>

      {/* Call to Action */}
      <motion.section 
        {...animationProps}
        transition={{ ...animationProps.transition, delay: 0.5 }}
      >
        <CTA />
      </motion.section>

      <Footer />
    </main>
  )
}