'use client'

import { motion, useReducedMotion } from 'framer-motion'
import { 
  PlayIcon,
  ArrowRightIcon,
  CheckCircleIcon,
  CubeIcon,
  CpuChipIcon,
  EyeIcon
} from '@heroicons/react/24/outline'
import { useState } from 'react'

const Hero = () => {
  const [isVideoPlaying, setIsVideoPlaying] = useState(false)
  const shouldReduceMotion = useReducedMotion()

  const stats = [
    { label: 'Cost Reduction', value: '100x', description: '$500 vs $50,000' },
    { label: 'Accuracy', value: '0.1mm', description: 'Sub-millimeter precision' },
    { label: 'Processing Speed', value: '30 FPS', description: 'Real-time reconstruction' },
    { label: 'Technology Stack', value: 'Modern', description: 'Latest innovations' },
  ]

  const features = [
    'Real-time 3D reconstruction',
    'AI-powered dental analysis', 
    'Multiple hardware support',
    'Clinical workflow integration',
    'Cost-effective & scalable'
  ]

  const containerVariants = shouldReduceMotion ? undefined : {
    hidden: { opacity: 0 },
    visible: {
      opacity: 1,
      transition: {
        staggerChildren: 0.1,
        delayChildren: 0.2
      }
    }
  }

  const itemVariants = shouldReduceMotion ? undefined : {
    hidden: { opacity: 0, y: 20 },
    visible: {
      opacity: 1,
      y: 0,
      transition: { duration: 0.5, ease: 'easeOut' }
    }
  }

  return (
    <section className="relative pt-24 pb-16 overflow-hidden">
      {/* Background Elements */}
      <div className="absolute inset-0 bg-gradient-to-br from-dental-50 via-white to-mint-50" />
      <div className="absolute top-0 right-0 w-1/3 h-1/3 bg-gradient-to-bl from-dental-100/50 to-transparent rounded-full blur-3xl" />
      <div className="absolute bottom-0 left-0 w-1/4 h-1/4 bg-gradient-to-tr from-mint-100/50 to-transparent rounded-full blur-3xl" />

      <div className="relative container-max section-padding">
        <motion.div 
          className="grid lg:grid-cols-2 gap-12 items-center"
          variants={containerVariants}
          initial="hidden"
          animate="visible"
        >
          {/* Left Column - Content */}
          <div className="space-y-8">
            <motion.div variants={itemVariants} className="space-y-6">
              {/* Badge */}
              <div className="inline-flex items-center space-x-2 bg-dental-100 text-dental-700 px-4 py-2 rounded-full text-sm font-medium">
                <CheckCircleIcon className="w-4 h-4" />
                <span>Professional Grade • Advanced Technology</span>
              </div>

              {/* Headline */}
              <h1 className="text-4xl md:text-5xl lg:text-6xl font-bold text-neutral-900 leading-tight">
                Professional{' '}
                <span className="bg-gradient-to-r from-dental-600 to-mint-500 bg-clip-text text-transparent">
                  Dental Scanning
                </span>{' '}
                Made Accessible
              </h1>

              {/* Subheadline */}
              <p className="text-xl text-neutral-600 leading-relaxed max-w-2xl">
                Advanced dental scanner delivering professional-grade 3D reconstruction, 
                AI-powered analysis, and clinical workflow integration using cutting-edge open-source technologies.
              </p>
            </motion.div>

            {/* Features List */}
            <motion.div variants={itemVariants} className="space-y-3">
              {features.map((feature, index) => (
                <div key={index} className="flex items-center space-x-3">
                  <CheckCircleIcon className="w-5 h-5 text-mint-500 flex-shrink-0" />
                  <span className="text-neutral-700">{feature}</span>
                </div>
              ))}
            </motion.div>

            {/* CTA Buttons */}
            <motion.div variants={itemVariants} className="flex flex-col sm:flex-row gap-4">
              <motion.button
                className="flex items-center justify-center space-x-2 bg-dental-600 hover:bg-dental-700 text-white px-8 py-4 rounded-xl font-semibold text-lg shadow-dental transition-all duration-300"
                whileHover={shouldReduceMotion ? {} : { scale: 1.02, y: -2 }}
                whileTap={shouldReduceMotion ? {} : { scale: 0.98 }}
              >
                <span>Get Started Free</span>
                <ArrowRightIcon className="w-5 h-5" />
              </motion.button>

              <motion.button
                className="flex items-center justify-center space-x-2 bg-white hover:bg-neutral-50 text-dental-600 px-8 py-4 rounded-xl font-semibold text-lg border-2 border-dental-200 hover:border-dental-300 transition-all duration-300"
                whileHover={shouldReduceMotion ? {} : { scale: 1.02, y: -2 }}
                whileTap={shouldReduceMotion ? {} : { scale: 0.98 }}
                onClick={() => setIsVideoPlaying(true)}
              >
                <PlayIcon className="w-5 h-5" />
                <span>Watch Demo</span>
              </motion.button>
            </motion.div>

            {/* Stats */}
            <motion.div variants={itemVariants} className="grid grid-cols-2 md:grid-cols-4 gap-6 pt-8">
              {stats.map((stat, index) => (
                <div key={index} className="text-center">
                  <div className="text-2xl md:text-3xl font-bold text-dental-600">{stat.value}</div>
                  <div className="text-sm font-medium text-neutral-700">{stat.label}</div>
                  <div className="text-xs text-neutral-500">{stat.description}</div>
                </div>
              ))}
            </motion.div>
          </div>

          {/* Right Column - Visual */}
          <motion.div variants={itemVariants} className="relative">
            {/* 3D Scanner Visualization */}
            <div className="relative bg-gradient-to-br from-white to-dental-50 rounded-2xl p-8 shadow-gentle border border-dental-100">
              {/* Scanner Interface Mockup */}
              <div className="space-y-6">
                {/* Header */}
                <div className="flex items-center justify-between">
                  <div className="flex items-center space-x-3">
                    <div className="w-3 h-3 bg-mint-400 rounded-full"></div>
                    <div className="w-3 h-3 bg-yellow-400 rounded-full"></div>
                    <div className="w-3 h-3 bg-red-400 rounded-full"></div>
                  </div>
                  <div className="text-sm text-neutral-500">OpenDentalScan v2.0</div>
                </div>

                {/* 3D Viewport */}
                <div className="aspect-video bg-gradient-to-br from-dental-900 to-neutral-800 rounded-xl relative overflow-hidden">
                  {/* 3D Mesh Visualization */}
                  <div className="absolute inset-0 flex items-center justify-center">
                    <motion.div
                      className="w-32 h-32 bg-gradient-to-br from-dental-400 to-mint-400 rounded-full opacity-80"
                      animate={shouldReduceMotion ? {} : {
                        scale: [1, 1.1, 1],
                        rotate: [0, 180, 360],
                      }}
                      transition={{
                        duration: 8,
                        repeat: Infinity,
                        ease: "linear"
                      }}
                    />
                  </div>
                  
                  {/* Scanning Grid Overlay */}
                  <div className="absolute inset-0 bg-gradient-to-r from-transparent via-dental-400/20 to-transparent animate-pulse-gentle" />
                  
                  {/* Status Indicators */}
                  <div className="absolute top-4 left-4 space-y-2">
                    <div className="flex items-center space-x-2 text-white text-sm">
                      <div className="w-2 h-2 bg-mint-400 rounded-full animate-pulse-gentle"></div>
                      <span>Real-time Scanning</span>
                    </div>
                    <div className="flex items-center space-x-2 text-white text-sm">
                      <CubeIcon className="w-4 h-4" />
                      <span>TSDF Fusion: Active</span>
                    </div>
                    <div className="flex items-center space-x-2 text-white text-sm">
                      <CpuChipIcon className="w-4 h-4" />
                      <span>AI Analysis: 95% Accuracy</span>
                    </div>
                  </div>

                  {/* Performance Metrics */}
                  <div className="absolute bottom-4 right-4 text-white text-sm">
                    <div>30 FPS • 0.1mm Precision</div>
                  </div>
                </div>

                {/* Control Panel */}
                <div className="grid grid-cols-3 gap-4">
                  <button className="flex flex-col items-center space-y-2 p-4 bg-dental-50 hover:bg-dental-100 rounded-lg transition-colors duration-200">
                    <EyeIcon className="w-6 h-6 text-dental-600" />
                    <span className="text-sm font-medium text-dental-700">Scan</span>
                  </button>
                  <button className="flex flex-col items-center space-y-2 p-4 bg-mint-50 hover:bg-mint-100 rounded-lg transition-colors duration-200">
                    <CpuChipIcon className="w-6 h-6 text-mint-600" />
                    <span className="text-sm font-medium text-mint-700">Analyze</span>
                  </button>
                  <button className="flex flex-col items-center space-y-2 p-4 bg-neutral-50 hover:bg-neutral-100 rounded-lg transition-colors duration-200">
                    <CubeIcon className="w-6 h-6 text-neutral-600" />
                    <span className="text-sm font-medium text-neutral-700">Export</span>
                  </button>
                </div>
              </div>

              {/* Floating Elements */}
              <motion.div
                className="absolute -top-4 -right-4 w-16 h-16 bg-gradient-to-br from-mint-400 to-mint-500 rounded-xl shadow-gentle flex items-center justify-center"
                animate={shouldReduceMotion ? {} : { y: [-5, 5, -5] }}
                transition={{ duration: 3, repeat: Infinity, ease: "easeInOut" }}
              >
                <CheckCircleIcon className="w-8 h-8 text-white" />
              </motion.div>

              <motion.div
                className="absolute -bottom-4 -left-4 w-12 h-12 bg-gradient-to-br from-dental-400 to-dental-500 rounded-lg shadow-gentle flex items-center justify-center"
                animate={shouldReduceMotion ? {} : { y: [5, -5, 5] }}
                transition={{ duration: 4, repeat: Infinity, ease: "easeInOut" }}
              >
                <CpuChipIcon className="w-6 h-6 text-white" />
              </motion.div>
            </div>
          </motion.div>
        </motion.div>
      </div>
    </section>
  )
}

export default Hero