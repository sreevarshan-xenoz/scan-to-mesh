'use client'

import { useState } from 'react'
import { motion, useReducedMotion } from 'framer-motion'
import { 
  PlayIcon,
  PauseIcon,
  ArrowPathIcon,
  EyeIcon,
  CpuChipIcon,
  DocumentArrowDownIcon,
  ChartBarIcon
} from '@heroicons/react/24/outline'

const Demo = () => {
  const [activeDemo, setActiveDemo] = useState('realtime')
  const [isPlaying, setIsPlaying] = useState(false)
  const shouldReduceMotion = useReducedMotion()

  const demoTabs = [
    {
      id: 'realtime',
      name: 'Real-Time Scanning',
      icon: EyeIcon,
      description: 'Live 3D reconstruction as you scan'
    },
    {
      id: 'ai',
      name: 'AI Analysis',
      icon: CpuChipIcon,
      description: 'Automated tooth segmentation and analysis'
    },
    {
      id: 'export',
      name: 'Professional Export',
      icon: DocumentArrowDownIcon,
      description: 'Clinical-grade output formats'
    },
    {
      id: 'performance',
      name: 'Performance Metrics',
      icon: ChartBarIcon,
      description: 'Real-time system monitoring'
    }
  ]

  const containerVariants = shouldReduceMotion ? {} : {
    hidden: { opacity: 0 },
    visible: {
      opacity: 1,
      transition: {
        staggerChildren: 0.1,
        delayChildren: 0.2
      }
    }
  }

  const itemVariants = shouldReduceMotion ? {} : {
    hidden: { opacity: 0, y: 30 },
    visible: {
      opacity: 1,
      y: 0,
      transition: { duration: 0.5, ease: 'easeOut' }
    }
  }

  return (
    <section id="demo" className="section-padding bg-gradient-to-br from-dental-50 to-mint-50">
      <div className="container-max">
        <motion.div
          variants={containerVariants}
          initial="hidden"
          whileInView="visible"
          viewport={{ once: true, margin: "-100px" }}
        >
          {/* Section Header */}
          <motion.div variants={itemVariants} className="text-center mb-16">
            <div className="inline-flex items-center space-x-2 bg-white text-dental-700 px-4 py-2 rounded-full text-sm font-medium mb-4 shadow-soft">
              <PlayIcon className="w-4 h-4" />
              <span>Interactive Demo</span>
            </div>
            
            <h2 className="text-3xl md:text-4xl lg:text-5xl font-bold text-neutral-900 mb-6">
              See{' '}
              <span className="bg-gradient-to-r from-dental-600 to-mint-500 bg-clip-text text-transparent">
                OpenDentalScan
              </span>{' '}
              in Action
            </h2>
            
            <p className="text-xl text-neutral-600 max-w-3xl mx-auto leading-relaxed">
              Experience the power of professional dental scanning with our interactive demonstration. 
              Real-time 3D reconstruction, AI analysis, and clinical workflow integration.
            </p>
          </motion.div>

          {/* Demo Interface */}
          <motion.div variants={itemVariants} className="bg-white rounded-3xl shadow-gentle border border-neutral-200 overflow-hidden">
            {/* Tab Navigation */}
            <div className="border-b border-neutral-200 p-6">
              <div className="flex flex-wrap gap-2">
                {demoTabs.map((tab) => (
                  <motion.button
                    key={tab.id}
                    onClick={() => setActiveDemo(tab.id)}
                    className={`flex items-center space-x-2 px-4 py-3 rounded-xl font-medium transition-all duration-200 ${
                      activeDemo === tab.id
                        ? 'bg-dental-600 text-white shadow-dental'
                        : 'bg-neutral-100 text-neutral-600 hover:bg-neutral-200'
                    }`}
                    whileHover={shouldReduceMotion ? {} : { scale: 1.02 }}
                    whileTap={shouldReduceMotion ? {} : { scale: 0.98 }}
                  >
                    <tab.icon className="w-4 h-4" />
                    <span className="hidden sm:inline">{tab.name}</span>
                    <span className="sm:hidden">{tab.name.split(' ')[0]}</span>
                  </motion.button>
                ))}
              </div>
            </div>

            {/* Demo Content */}
            <div className="p-6">
              <div className="grid lg:grid-cols-2 gap-8 items-center">
                {/* Demo Viewport */}
                <div className="relative">
                  <div className="aspect-video bg-gradient-to-br from-neutral-900 to-dental-900 rounded-2xl overflow-hidden relative">
                    {/* Video/Demo Content */}
                    <div className="absolute inset-0 flex items-center justify-center">
                      {activeDemo === 'realtime' && (
                        <motion.div
                          className="relative w-full h-full"
                          initial={shouldReduceMotion ? {} : { opacity: 0 }}
                          animate={shouldReduceMotion ? {} : { opacity: 1 }}
                          transition={{ duration: 0.3 }}
                        >
                          {/* 3D Scanning Simulation */}
                          <div className="absolute inset-0 bg-gradient-to-r from-transparent via-dental-400/30 to-transparent animate-pulse-gentle" />
                          <motion.div
                            className="absolute top-1/2 left-1/2 transform -translate-x-1/2 -translate-y-1/2 w-40 h-40 bg-gradient-to-br from-dental-400 to-mint-400 rounded-full opacity-80"
                            animate={shouldReduceMotion ? {} : {
                              scale: [1, 1.2, 1],
                              rotate: [0, 360],
                            }}
                            transition={{
                              duration: 4,
                              repeat: Infinity,
                              ease: "easeInOut"
                            }}
                          />
                          
                          {/* Scanning Grid */}
                          <div className="absolute inset-0 grid grid-cols-8 grid-rows-6 gap-1 p-4">
                            {Array.from({ length: 48 }).map((_, i) => (
                              <motion.div
                                key={i}
                                className="bg-dental-400/20 rounded-sm"
                                animate={shouldReduceMotion ? {} : {
                                  opacity: [0.2, 0.8, 0.2],
                                }}
                                transition={{
                                  duration: 2,
                                  repeat: Infinity,
                                  delay: i * 0.05,
                                }}
                              />
                            ))}
                          </div>
                        </motion.div>
                      )}

                      {activeDemo === 'ai' && (
                        <motion.div
                          className="relative w-full h-full flex items-center justify-center"
                          initial={shouldReduceMotion ? {} : { opacity: 0 }}
                          animate={shouldReduceMotion ? {} : { opacity: 1 }}
                          transition={{ duration: 0.3 }}
                        >
                          {/* AI Analysis Visualization */}
                          <div className="relative">
                            <motion.div
                              className="w-32 h-32 bg-gradient-to-br from-mint-400 to-dental-400 rounded-full"
                              animate={shouldReduceMotion ? {} : {
                                scale: [1, 1.1, 1],
                              }}
                              transition={{
                                duration: 2,
                                repeat: Infinity,
                                ease: "easeInOut"
                              }}
                            />
                            
                            {/* AI Detection Boxes */}
                            {[...Array(6)].map((_, i) => (
                              <motion.div
                                key={i}
                                className="absolute w-8 h-8 border-2 border-mint-400 rounded"
                                style={{
                                  top: `${20 + i * 15}%`,
                                  left: `${10 + i * 12}%`,
                                }}
                                animate={shouldReduceMotion ? {} : {
                                  opacity: [0, 1, 0],
                                }}
                                transition={{
                                  duration: 1.5,
                                  repeat: Infinity,
                                  delay: i * 0.2,
                                }}
                              />
                            ))}
                          </div>
                        </motion.div>
                      )}

                      {activeDemo === 'export' && (
                        <motion.div
                          className="relative w-full h-full flex items-center justify-center"
                          initial={shouldReduceMotion ? {} : { opacity: 0 }}
                          animate={shouldReduceMotion ? {} : { opacity: 1 }}
                          transition={{ duration: 0.3 }}
                        >
                          {/* Export Process Visualization */}
                          <div className="space-y-4 text-center">
                            <motion.div
                              className="w-24 h-24 bg-gradient-to-br from-dental-400 to-mint-400 rounded-xl mx-auto"
                              animate={shouldReduceMotion ? {} : {
                                rotateY: [0, 360],
                              }}
                              transition={{
                                duration: 3,
                                repeat: Infinity,
                                ease: "linear"
                              }}
                            />
                            <div className="text-white text-sm space-y-1">
                              <div>Exporting STL...</div>
                              <div className="w-32 h-2 bg-neutral-700 rounded-full mx-auto overflow-hidden">
                                <motion.div
                                  className="h-full bg-gradient-to-r from-dental-400 to-mint-400"
                                  animate={shouldReduceMotion ? {} : {
                                    width: ['0%', '100%'],
                                  }}
                                  transition={{
                                    duration: 2,
                                    repeat: Infinity,
                                    ease: "easeInOut"
                                  }}
                                />
                              </div>
                            </div>
                          </div>
                        </motion.div>
                      )}

                      {activeDemo === 'performance' && (
                        <motion.div
                          className="relative w-full h-full p-6"
                          initial={shouldReduceMotion ? {} : { opacity: 0 }}
                          animate={shouldReduceMotion ? {} : { opacity: 1 }}
                          transition={{ duration: 0.3 }}
                        >
                          {/* Performance Charts */}
                          <div className="grid grid-cols-2 gap-4 h-full">
                            <div className="space-y-2">
                              <div className="text-white text-sm">FPS</div>
                              <div className="flex items-end space-x-1 h-16">
                                {[...Array(8)].map((_, i) => (
                                  <motion.div
                                    key={i}
                                    className="w-3 bg-gradient-to-t from-dental-400 to-mint-400 rounded-t"
                                    animate={shouldReduceMotion ? {} : {
                                      height: [`${20 + Math.random() * 40}%`, `${30 + Math.random() * 50}%`],
                                    }}
                                    transition={{
                                      duration: 1,
                                      repeat: Infinity,
                                      repeatType: "reverse",
                                      delay: i * 0.1,
                                    }}
                                  />
                                ))}
                              </div>
                            </div>
                            
                            <div className="space-y-2">
                              <div className="text-white text-sm">GPU Usage</div>
                              <div className="flex items-end space-x-1 h-16">
                                {[...Array(8)].map((_, i) => (
                                  <motion.div
                                    key={i}
                                    className="w-3 bg-gradient-to-t from-mint-400 to-dental-400 rounded-t"
                                    animate={shouldReduceMotion ? {} : {
                                      height: [`${40 + Math.random() * 30}%`, `${50 + Math.random() * 40}%`],
                                    }}
                                    transition={{
                                      duration: 1.5,
                                      repeat: Infinity,
                                      repeatType: "reverse",
                                      delay: i * 0.15,
                                    }}
                                  />
                                ))}
                              </div>
                            </div>
                          </div>
                        </motion.div>
                      )}
                    </div>

                    {/* Play/Pause Button */}
                    <motion.button
                      className="absolute bottom-4 right-4 w-12 h-12 bg-white/20 backdrop-blur-sm rounded-full flex items-center justify-center text-white hover:bg-white/30 transition-colors duration-200"
                      onClick={() => setIsPlaying(!isPlaying)}
                      whileHover={shouldReduceMotion ? {} : { scale: 1.1 }}
                      whileTap={shouldReduceMotion ? {} : { scale: 0.9 }}
                    >
                      {isPlaying ? (
                        <PauseIcon className="w-6 h-6" />
                      ) : (
                        <PlayIcon className="w-6 h-6 ml-1" />
                      )}
                    </motion.button>

                    {/* Status Indicators */}
                    <div className="absolute top-4 left-4 space-y-2">
                      <div className="flex items-center space-x-2 text-white text-sm bg-black/20 backdrop-blur-sm rounded-lg px-3 py-1">
                        <div className="w-2 h-2 bg-mint-400 rounded-full animate-pulse-gentle"></div>
                        <span>Live Demo</span>
                      </div>
                    </div>
                  </div>
                </div>

                {/* Demo Description */}
                <div className="space-y-6">
                  <div>
                    <h3 className="text-2xl font-bold text-neutral-900 mb-4">
                      {demoTabs.find(tab => tab.id === activeDemo)?.name}
                    </h3>
                    <p className="text-neutral-600 leading-relaxed mb-6">
                      {demoTabs.find(tab => tab.id === activeDemo)?.description}
                    </p>
                  </div>

                  {/* Feature Details */}
                  <div className="space-y-4">
                    {activeDemo === 'realtime' && (
                      <div className="space-y-3">
                        <div className="flex items-center space-x-3">
                          <div className="w-2 h-2 bg-dental-500 rounded-full"></div>
                          <span className="text-neutral-700">30+ FPS real-time processing</span>
                        </div>
                        <div className="flex items-center space-x-3">
                          <div className="w-2 h-2 bg-mint-500 rounded-full"></div>
                          <span className="text-neutral-700">Sub-millimeter accuracy</span>
                        </div>
                        <div className="flex items-center space-x-3">
                          <div className="w-2 h-2 bg-dental-500 rounded-full"></div>
                          <span className="text-neutral-700">GPU-accelerated TSDF fusion</span>
                        </div>
                      </div>
                    )}

                    {activeDemo === 'ai' && (
                      <div className="space-y-3">
                        <div className="flex items-center space-x-3">
                          <div className="w-2 h-2 bg-mint-500 rounded-full"></div>
                          <span className="text-neutral-700">95%+ tooth segmentation accuracy</span>
                        </div>
                        <div className="flex items-center space-x-3">
                          <div className="w-2 h-2 bg-dental-500 rounded-full"></div>
                          <span className="text-neutral-700">Automated pathology detection</span>
                        </div>
                        <div className="flex items-center space-x-3">
                          <div className="w-2 h-2 bg-mint-500 rounded-full"></div>
                          <span className="text-neutral-700">Real-time quality assessment</span>
                        </div>
                      </div>
                    )}

                    {activeDemo === 'export' && (
                      <div className="space-y-3">
                        <div className="flex items-center space-x-3">
                          <div className="w-2 h-2 bg-dental-500 rounded-full"></div>
                          <span className="text-neutral-700">STL, OBJ, PLY, DICOM formats</span>
                        </div>
                        <div className="flex items-center space-x-3">
                          <div className="w-2 h-2 bg-mint-500 rounded-full"></div>
                          <span className="text-neutral-700">CAD/CAM workflow integration</span>
                        </div>
                        <div className="flex items-center space-x-3">
                          <div className="w-2 h-2 bg-dental-500 rounded-full"></div>
                          <span className="text-neutral-700">Clinical report generation</span>
                        </div>
                      </div>
                    )}

                    {activeDemo === 'performance' && (
                      <div className="space-y-3">
                        <div className="flex items-center space-x-3">
                          <div className="w-2 h-2 bg-mint-500 rounded-full"></div>
                          <span className="text-neutral-700">Real-time system monitoring</span>
                        </div>
                        <div className="flex items-center space-x-3">
                          <div className="w-2 h-2 bg-dental-500 rounded-full"></div>
                          <span className="text-neutral-700">Adaptive quality control</span>
                        </div>
                        <div className="flex items-center space-x-3">
                          <div className="w-2 h-2 bg-mint-500 rounded-full"></div>
                          <span className="text-neutral-700">Performance optimization</span>
                        </div>
                      </div>
                    )}
                  </div>

                  {/* CTA */}
                  <motion.button
                    className="flex items-center space-x-2 bg-dental-600 hover:bg-dental-700 text-white px-6 py-3 rounded-xl font-semibold transition-colors duration-200 shadow-dental"
                    whileHover={shouldReduceMotion ? {} : { scale: 1.02 }}
                    whileTap={shouldReduceMotion ? {} : { scale: 0.98 }}
                  >
                    <ArrowPathIcon className="w-5 h-5" />
                    <span>Try Interactive Demo</span>
                  </motion.button>
                </div>
              </div>
            </div>
          </motion.div>
        </motion.div>
      </div>
    </section>
  )
}

export default Demo