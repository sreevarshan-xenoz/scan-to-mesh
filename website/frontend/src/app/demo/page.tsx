'use client'

import { useState, useEffect, useRef, Suspense } from 'react'
import { motion, useReducedMotion } from 'framer-motion'
import Link from 'next/link'
import { Canvas, useFrame, useThree } from '@react-three/fiber'
import * as THREE from 'three'
import { 
  ArrowLeftIcon,
  PlayIcon,
  PauseIcon,
  ArrowPathIcon,
  EyeIcon,
  CpuChipIcon,
  DocumentArrowDownIcon,
  ChartBarIcon,
  CameraIcon,
  AdjustmentsHorizontalIcon,
  InformationCircleIcon,
  CheckCircleIcon,
  ClockIcon,
  CubeIcon,
  SparklesIcon,
  XMarkIcon,
  CurrencyDollarIcon,
  AcademicCapIcon,
  EyeSlashIcon,
  Cog6ToothIcon
} from '@heroicons/react/24/outline'

// Mock 3D Models - In production, these would be actual GLTF files
const mockScanSessions = [
  {
    id: 'upper-arch',
    name: 'Upper Dental Arch',
    description: 'Complete upper jaw scan with 14 teeth',
    duration: '2.3 seconds',
    points: 45000,
    meshes: 'High resolution',
    formats: ['GLTF', 'STL', 'OBJ']
  },
  {
    id: 'lower-arch', 
    name: 'Lower Dental Arch',
    description: 'Lower jaw with crown preparation',
    duration: '1.8 seconds',
    points: 38000,
    meshes: 'Ultra high resolution',
    formats: ['GLTF', 'STL', 'OBJ', 'DICOM']
  },
  {
    id: 'full-mouth',
    name: 'Full Mouth Scan',
    description: 'Complete oral cavity reconstruction',
    duration: '4.1 seconds', 
    points: 89000,
    meshes: 'Professional grade',
    formats: ['GLTF', 'STL', 'OBJ', 'DICOM', 'PLY']
  }
]

// Simple camera rotation component
function CameraRotation({ autoRotate }: { autoRotate: boolean }) {
  const { camera } = useThree()
  
  useFrame((state) => {
    if (autoRotate) {
      camera.position.x = Math.cos(state.clock.elapsedTime * 0.2) * 4
      camera.position.z = Math.sin(state.clock.elapsedTime * 0.2) * 4
      camera.lookAt(0, 0, 0)
    }
  })
  
  return null
}

// Point Cloud Animation Component
function AnimatedPointCloud({ progress, totalPoints = 5000 }: { progress: number, totalPoints?: number }) {
  const pointsRef = useRef<THREE.Points>(null)
  const [geometry, setGeometry] = useState<THREE.BufferGeometry>()

  useEffect(() => {
    const points = []
    const currentPoints = Math.floor((progress / 100) * totalPoints)
    
    for (let i = 0; i < currentPoints; i++) {
      // Create realistic dental arch point distribution
      const t = i / totalPoints
      const angle = t * Math.PI * 1.8 - Math.PI * 0.9 // U-shaped arch
      const radius = 1.2 + Math.sin(angle * 8) * 0.15 // Tooth variations
      
      const x = Math.cos(angle) * radius + (Math.random() - 0.5) * 0.05
      const z = Math.sin(angle) * radius * 0.7 + (Math.random() - 0.5) * 0.05
      const y = Math.sin(angle * 16) * 0.1 + (Math.random() - 0.5) * 0.03
      
      points.push(x, y, z)
    }
    
    const geo = new THREE.BufferGeometry()
    geo.setAttribute('position', new THREE.Float32BufferAttribute(points, 3))
    setGeometry(geo)
  }, [progress, totalPoints])

  useFrame((state) => {
    if (pointsRef.current) {
      pointsRef.current.rotation.y = Math.sin(state.clock.elapsedTime * 0.1) * 0.05
    }
  })

  if (!geometry) return null

  return (
    <points ref={pointsRef} geometry={geometry} frustumCulled={false}>
      <pointsMaterial 
        transparent 
        color="#22c55e" 
        size={0.015} 
        sizeAttenuation={true}
        opacity={0.8}
      />
    </points>
  )
}

// Dental Mesh Component with Layer Control
function DentalMesh({ 
  progress, 
  showTeeth = true, 
  showGums = true, 
  wireframe = false 
}: { 
  progress: number
  showTeeth?: boolean
  showGums?: boolean
  wireframe?: boolean
}) {
  const meshRef = useRef<THREE.Group>(null)
  const opacity = Math.min(progress / 100, 1)

  useFrame((state) => {
    if (meshRef.current) {
      meshRef.current.rotation.y = Math.sin(state.clock.elapsedTime * 0.08) * 0.02
    }
  })

  return (
    <group ref={meshRef}>
      {/* Gum base */}
      {showGums && (
        <mesh position={[0, -0.2, 0]}>
          <cylinderGeometry args={[1.4, 1.2, 0.3, 32]} />
          <meshStandardMaterial 
            color="#ffb3ba" 
            transparent 
            opacity={opacity}
            wireframe={wireframe}
          />
        </mesh>
      )}
      
      {/* Individual teeth */}
      {showTeeth && Array.from({ length: 14 }).map((_, i) => {
        const angle = (i / 14) * Math.PI * 1.6 - Math.PI * 0.8
        const x = Math.cos(angle) * 1.1
        const z = Math.sin(angle) * 0.8
        const visible = progress > (i / 14) * 100
        const toothHeight = 0.4 + Math.random() * 0.2
        
        return visible ? (
          <mesh
            key={i}
            position={[x, toothHeight/2 - 0.1, z]}
            rotation={[0, -angle, 0]}
          >
            <boxGeometry args={[0.12, toothHeight, 0.15]} />
            <meshStandardMaterial 
              color="#ffffff" 
              transparent 
              opacity={opacity}
              wireframe={wireframe}
            />
          </mesh>
        ) : null
      })}
      
      {progress >= 100 && (
        <mesh position={[0, -0.8, 0]}>
          <planeGeometry args={[2, 0.3]} />
          <meshBasicMaterial color="#22c55e" transparent opacity={0.8} />
        </mesh>
      )}
    </group>
  )
}

// Main 3D Scene
function DentalScene({ 
  stage, 
  progress, 
  showTeeth, 
  showGums, 
  wireframe,
  selectedSession 
}: {
  stage: 'pointcloud' | 'mesh' | 'polish' | 'complete'
  progress: number
  showTeeth: boolean
  showGums: boolean
  wireframe: boolean
  selectedSession: any
}) {
  return (
    <>
      <ambientLight intensity={0.4} />
      <directionalLight position={[5, 5, 5]} intensity={0.6} />
      <directionalLight position={[-5, 2, -5]} intensity={0.3} />
      <pointLight position={[0, 2, 0]} intensity={0.4} color="#ffffff" />
      
      {stage === 'pointcloud' && (
        <AnimatedPointCloud 
          progress={progress} 
          totalPoints={selectedSession?.points || 5000}
        />
      )}
      
      {(stage === 'mesh' || stage === 'polish' || stage === 'complete') && (
        <DentalMesh 
          progress={progress}
          showTeeth={showTeeth}
          showGums={showGums}
          wireframe={wireframe}
        />
      )}
      
      <CameraRotation autoRotate={stage === 'complete'} />
    </>
  )
}

// Loading Component
function SceneLoader() {
  return (
    <div className="w-full h-full flex items-center justify-center bg-gradient-to-br from-dental-900 to-neutral-800 rounded-2xl">
      <div className="text-center text-white">
        <div className="animate-spin w-8 h-8 border-2 border-white border-t-transparent rounded-full mx-auto mb-4"></div>
        <p>Loading 3D Scene...</p>
      </div>
    </div>
  )
}

const DemoPage = () => {
  const shouldReduceMotion = useReducedMotion()
  
  // Demo state
  const [selectedSession, setSelectedSession] = useState(mockScanSessions[0])
  const [isPlaying, setIsPlaying] = useState(false)
  const [progress, setProgress] = useState(0)
  const [stage, setStage] = useState<'pointcloud' | 'mesh' | 'polish' | 'complete'>('pointcloud')
  
  // 3D viewer controls
  const [showTeeth, setShowTeeth] = useState(true)
  const [showGums, setShowGums] = useState(true)
  const [wireframe, setWireframe] = useState(false)
  
  // Animation stages
  const stages = [
    { id: 'pointcloud', name: 'Point Cloud Capture', duration: 30, color: 'mint' },
    { id: 'mesh', name: 'Mesh Reconstruction', duration: 40, color: 'dental' },
    { id: 'polish', name: 'AI Enhancement', duration: 20, color: 'mint' },
    { id: 'complete', name: 'Export Ready', duration: 10, color: 'dental' }
  ]

  // Export formats
  const exportFormats = [
    { name: 'GLTF', description: 'Web-optimized 3D format', size: '1.2 MB', icon: CubeIcon },
    { name: 'STL', description: '3D printing ready', size: '2.4 MB', icon: CubeIcon },
    { name: 'OBJ', description: 'Universal 3D format', size: '1.8 MB', icon: CubeIcon },
    { name: 'DICOM', description: 'Medical standard', size: '4.2 MB', icon: AcademicCapIcon }
  ]

  // Commercial comparison
  const comparison = [
    {
      name: 'Commercial Scanners',
      price: '$50,000+',
      demo: 'Static Images Only',
      features: [
        'Limited customization',
        'Vendor lock-in',
        'Expensive support',
        'Proprietary formats',
        'No live demo'
      ],
      color: 'neutral'
    },
    {
      name: 'OpenDentalScan',
      price: '$500',
      demo: 'Live Interactive 3D',
      features: [
        'Full customization',
        'Open technology stack',
        'Professional support',
        'Multiple export formats',
        'Real-time preview'
      ],
      color: 'dental'
    }
  ]

  // Start animation
  const startAnimation = () => {
    setIsPlaying(true)
    setProgress(0)
    setStage('pointcloud')
    
    const interval = setInterval(() => {
      setProgress(prev => {
        const newProgress = prev + 1
        
        // Update stage based on progress
        if (newProgress <= 30) {
          setStage('pointcloud')
        } else if (newProgress <= 70) {
          setStage('mesh')
        } else if (newProgress <= 90) {
          setStage('polish')
        } else {
          setStage('complete')
        }
        
        if (newProgress >= 100) {
          setIsPlaying(false)
          clearInterval(interval)
          return 100
        }
        return newProgress
      })
    }, 60) // Smooth animation
  }

  const resetAnimation = () => {
    setIsPlaying(false)
    setProgress(0)
    setStage('pointcloud')
  }

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
    hidden: { opacity: 0, y: 30 },
    visible: {
      opacity: 1,
      y: 0,
      transition: { duration: 0.5, ease: 'easeOut' }
    }
  }

  return (
    <div className="min-h-screen bg-gradient-to-br from-dental-50 via-white to-mint-50">
      {/* Header */}
      <section className="pt-24 pb-16">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <motion.div
            variants={containerVariants}
            initial="hidden"
            animate="visible"
          >
            <motion.div variants={itemVariants} className="mb-8">
              <Link
                href="/"
                className="inline-flex items-center space-x-2 text-dental-600 hover:text-dental-700 font-medium transition-colors duration-200"
              >
                <ArrowLeftIcon className="w-4 h-4" />
                <span>Back to Home</span>
              </Link>
            </motion.div>
            
            <motion.div variants={itemVariants} className="text-center">
              <div className="inline-flex items-center space-x-2 bg-mint-100 text-mint-700 px-4 py-2 rounded-full text-sm font-medium mb-6">
                <PlayIcon className="w-4 h-4" />
                <span>Interactive 3D Demo</span>
              </div>
              
              <h1 className="text-4xl md:text-5xl lg:text-6xl font-bold text-neutral-900 mb-6">
                Experience{' '}
                <span className="bg-gradient-to-r from-dental-600 to-mint-500 bg-clip-text text-transparent">
                  Real-Time
                </span>{' '}
                3D Scanning
              </h1>
              
              <p className="text-xl text-neutral-600 max-w-3xl mx-auto mb-8">
                Try our professional dental scanning technology with pre-recorded scan sessions. 
                See step-by-step point cloud capture, mesh reconstruction, and interactive 3D models.
              </p>
            </motion.div>
          </motion.div>
        </div>
      </section>

      {/* Main Demo Interface */}
      <section className="py-16">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="grid lg:grid-cols-3 gap-8">
            
            {/* Left Sidebar - Session Selection & Controls */}
            <motion.div
              variants={containerVariants}
              initial="hidden"
              whileInView="visible"
              viewport={{ once: true }}
              className="space-y-6"
            >
              {/* Scan Session Selection */}
              <motion.div variants={itemVariants} className="bg-white rounded-2xl p-6 border border-neutral-200 shadow-soft">
                <h3 className="text-lg font-bold text-neutral-900 mb-4">Pre-recorded Scan Sessions</h3>
                <div className="space-y-3">
                  {mockScanSessions.map((session) => (
                    <button
                      key={session.id}
                      onClick={() => setSelectedSession(session)}
                      className={`w-full text-left p-4 rounded-xl border-2 transition-all duration-200 ${
                        selectedSession.id === session.id
                          ? 'border-dental-500 bg-dental-50'
                          : 'border-neutral-200 hover:border-dental-200 hover:bg-neutral-50'
                      }`}
                    >
                      <div className="font-semibold text-neutral-900">{session.name}</div>
                      <div className="text-sm text-neutral-600 mb-2">{session.description}</div>
                      <div className="flex justify-between text-xs text-neutral-500">
                        <span>{session.duration}</span>
                        <span>{session.points.toLocaleString()} points</span>
                      </div>
                    </button>
                  ))}
                </div>
              </motion.div>

              {/* Animation Controls */}
              <motion.div variants={itemVariants} className="bg-white rounded-2xl p-6 border border-neutral-200 shadow-soft">
                <h3 className="text-lg font-bold text-neutral-900 mb-4">Animation Controls</h3>
                
                <div className="space-y-4">
                  <div className="flex space-x-3">
                    <button
                      onClick={startAnimation}
                      disabled={isPlaying}
                      className="flex-1 flex items-center justify-center space-x-2 bg-mint-600 hover:bg-mint-700 disabled:bg-mint-400 text-white px-4 py-3 rounded-xl font-semibold transition-colors duration-200"
                    >
                      <PlayIcon className="w-5 h-5" />
                      <span>{isPlaying ? 'Playing...' : 'Start Demo'}</span>
                    </button>
                    
                    <button
                      onClick={resetAnimation}
                      className="flex items-center justify-center bg-neutral-600 hover:bg-neutral-700 text-white px-4 py-3 rounded-xl transition-colors duration-200"
                    >
                      <ArrowPathIcon className="w-5 h-5" />
                    </button>
                  </div>

                  {/* Progress Bar */}
                  <div>
                    <div className="flex justify-between text-sm mb-2">
                      <span className="text-neutral-600">Progress</span>
                      <span className="text-dental-600 font-medium">{Math.round(progress)}%</span>
                    </div>
                    <div className="w-full bg-neutral-200 rounded-full h-2">
                      <div 
                        className="bg-gradient-to-r from-dental-500 to-mint-500 h-2 rounded-full transition-all duration-300"
                        style={{ width: `${progress}%` }}
                      />
                    </div>
                  </div>

                  {/* Current Stage */}
                  <div className="bg-mint-50 rounded-lg p-3">
                    <div className="text-sm font-medium text-mint-900">
                      Current Stage: {stages.find(s => s.id === stage)?.name}
                    </div>
                  </div>
                </div>
              </motion.div>

              {/* 3D Viewer Controls */}
              <motion.div variants={itemVariants} className="bg-white rounded-2xl p-6 border border-neutral-200 shadow-soft">
                <h3 className="text-lg font-bold text-neutral-900 mb-4">3D Viewer Controls</h3>
                
                <div className="space-y-4">
                  <div className="flex items-center justify-between">
                    <span className="text-neutral-700">Show Teeth</span>
                    <button
                      onClick={() => setShowTeeth(!showTeeth)}
                      className={`flex items-center space-x-2 px-3 py-1 rounded-lg transition-colors duration-200 ${
                        showTeeth ? 'bg-dental-100 text-dental-700' : 'bg-neutral-100 text-neutral-600'
                      }`}
                    >
                      {showTeeth ? <EyeIcon className="w-4 h-4" /> : <EyeSlashIcon className="w-4 h-4" />}
                      <span>{showTeeth ? 'Visible' : 'Hidden'}</span>
                    </button>
                  </div>

                  <div className="flex items-center justify-between">
                    <span className="text-neutral-700">Show Gums</span>
                    <button
                      onClick={() => setShowGums(!showGums)}
                      className={`flex items-center space-x-2 px-3 py-1 rounded-lg transition-colors duration-200 ${
                        showGums ? 'bg-dental-100 text-dental-700' : 'bg-neutral-100 text-neutral-600'
                      }`}
                    >
                      {showGums ? <EyeIcon className="w-4 h-4" /> : <EyeSlashIcon className="w-4 h-4" />}
                      <span>{showGums ? 'Visible' : 'Hidden'}</span>
                    </button>
                  </div>

                  <div className="flex items-center justify-between">
                    <span className="text-neutral-700">Wireframe</span>
                    <button
                      onClick={() => setWireframe(!wireframe)}
                      className={`flex items-center space-x-2 px-3 py-1 rounded-lg transition-colors duration-200 ${
                        wireframe ? 'bg-mint-100 text-mint-700' : 'bg-neutral-100 text-neutral-600'
                      }`}
                    >
                      <Cog6ToothIcon className="w-4 h-4" />
                      <span>{wireframe ? 'On' : 'Off'}</span>
                    </button>
                  </div>
                </div>
              </motion.div>
            </motion.div>

            {/* Center - 3D Viewer */}
            <motion.div
              variants={itemVariants}
              initial="hidden"
              whileInView="visible"
              viewport={{ once: true }}
              className="lg:col-span-2"
            >
              <div className="bg-white rounded-3xl p-8 border border-neutral-200 shadow-gentle">
                <div className="flex items-center justify-between mb-6">
                  <h3 className="text-2xl font-bold text-neutral-900">Interactive 3D Viewer</h3>
                  <div className="flex items-center space-x-2 text-sm text-neutral-600">
                    <div className={`w-2 h-2 rounded-full ${isPlaying ? 'bg-mint-500 animate-pulse' : 'bg-neutral-400'}`} />
                    <span>{isPlaying ? 'Live Demo' : 'Ready'}</span>
                  </div>
                </div>

                {/* 3D Canvas */}
                <div className="aspect-video bg-gradient-to-br from-dental-900 to-neutral-800 rounded-2xl overflow-hidden">
                  <Suspense fallback={<SceneLoader />}>
                    <Canvas camera={{ position: [0, 1, 4], fov: 50 }}>
                      <DentalScene
                        stage={stage}
                        progress={progress}
                        showTeeth={showTeeth}
                        showGums={showGums}
                        wireframe={wireframe}
                        selectedSession={selectedSession}
                      />
                    </Canvas>
                  </Suspense>
                </div>

                {/* Viewer Info */}
                <div className="mt-6 grid grid-cols-2 md:grid-cols-4 gap-4 text-center">
                  <div>
                    <div className="text-2xl font-bold text-dental-600">{selectedSession.points.toLocaleString()}</div>
                    <div className="text-sm text-neutral-600">Points</div>
                  </div>
                  <div>
                    <div className="text-2xl font-bold text-mint-600">{selectedSession.duration}</div>
                    <div className="text-sm text-neutral-600">Scan Time</div>
                  </div>
                  <div>
                    <div className="text-2xl font-bold text-dental-600">{selectedSession.meshes}</div>
                    <div className="text-sm text-neutral-600">Quality</div>
                  </div>
                  <div>
                    <div className="text-2xl font-bold text-mint-600">{selectedSession.formats.length}</div>
                    <div className="text-sm text-neutral-600">Formats</div>
                  </div>
                </div>
              </div>
            </motion.div>
          </div>
        </div>
      </section>

      {/* Export Options */}
      <section className="py-16">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <motion.div
            variants={containerVariants}
            initial="hidden"
            whileInView="visible"
            viewport={{ once: true }}
          >
            <motion.div variants={itemVariants} className="text-center mb-12">
              <h2 className="text-3xl md:text-4xl font-bold text-neutral-900 mb-4">
                Export in Multiple Formats
              </h2>
              <p className="text-xl text-neutral-600 max-w-2xl mx-auto">
                Professional-grade 3D models ready for CAD/CAM workflows, 3D printing, and medical analysis
              </p>
            </motion.div>

            <motion.div variants={itemVariants} className="bg-white rounded-3xl p-8 border border-neutral-200 shadow-gentle">
              <div className="grid md:grid-cols-2 lg:grid-cols-4 gap-6">
                {exportFormats.map((format, index) => (
                  <div key={index} className="text-center p-6 border border-neutral-200 rounded-2xl hover:border-dental-200 transition-colors duration-200">
                    <format.icon className="w-12 h-12 text-dental-600 mx-auto mb-4" />
                    <h3 className="text-lg font-bold text-neutral-900 mb-2">{format.name}</h3>
                    <p className="text-neutral-600 text-sm mb-4">{format.description}</p>
                    <div className="text-xs text-neutral-500 mb-4">{format.size}</div>
                    <button className="w-full bg-dental-600 hover:bg-dental-700 text-white px-4 py-2 rounded-lg font-medium transition-colors duration-200">
                      Download
                    </button>
                  </div>
                ))}
              </div>
            </motion.div>
          </motion.div>
        </div>
      </section>

      {/* Comparison Section */}
      <section className="py-16 bg-neutral-50">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <motion.div
            variants={containerVariants}
            initial="hidden"
            whileInView="visible"
            viewport={{ once: true }}
          >
            <motion.div variants={itemVariants} className="text-center mb-12">
              <h2 className="text-3xl md:text-4xl font-bold text-neutral-900 mb-4">
                Why Choose OpenDentalScan?
              </h2>
              <p className="text-xl text-neutral-600 max-w-2xl mx-auto">
                See the difference between traditional commercial scanners and our open-source approach
              </p>
            </motion.div>

            <motion.div variants={itemVariants} className="grid md:grid-cols-2 gap-8">
              {comparison.map((item, index) => (
                <div
                  key={index}
                  className={`p-8 rounded-3xl border-2 ${
                    item.color === 'dental'
                      ? 'border-dental-200 bg-dental-50'
                      : 'border-neutral-200 bg-white'
                  }`}
                >
                  <div className="text-center mb-6">
                    <h3 className="text-2xl font-bold text-neutral-900 mb-2">{item.name}</h3>
                    <div className="text-3xl font-bold text-dental-600 mb-2">{item.price}</div>
                    <div className="text-neutral-600">{item.demo}</div>
                  </div>

                  <ul className="space-y-3">
                    {item.features.map((feature, featureIndex) => (
                      <li key={featureIndex} className="flex items-center space-x-3">
                        <CheckCircleIcon className={`w-5 h-5 ${
                          item.color === 'dental' ? 'text-dental-600' : 'text-neutral-400'
                        }`} />
                        <span className="text-neutral-700">{feature}</span>
                      </li>
                    ))}
                  </ul>

                  {item.color === 'dental' && (
                    <div className="mt-6 text-center">
                      <Link
                        href="/pricing"
                        className="inline-flex items-center space-x-2 bg-dental-600 hover:bg-dental-700 text-white px-6 py-3 rounded-xl font-semibold transition-colors duration-200"
                      >
                        <span>Get Started</span>
                        <ArrowPathIcon className="w-4 h-4" />
                      </Link>
                    </div>
                  )}
                </div>
              ))}
            </motion.div>
          </motion.div>
        </div>
      </section>

      {/* Call to Action */}
      <section className="py-16">
        <div className="max-w-4xl mx-auto px-4 sm:px-6 lg:px-8 text-center">
          <motion.div
            variants={containerVariants}
            initial="hidden"
            whileInView="visible"
            viewport={{ once: true }}
          >
            <motion.div variants={itemVariants} className="bg-gradient-to-r from-dental-600 to-mint-500 rounded-3xl p-12 text-white">
              <h2 className="text-3xl md:text-4xl font-bold mb-6">
                Ready to Transform Your Dental Practice?
              </h2>
              <p className="text-xl mb-8 opacity-90">
                Join hundreds of dental professionals using OpenDentalScan for precise, affordable 3D scanning
              </p>
              <div className="flex flex-col sm:flex-row gap-4 justify-center">
                <Link
                  href="/pricing"
                  className="bg-white text-dental-600 hover:bg-neutral-100 px-8 py-4 rounded-xl font-bold transition-colors duration-200"
                >
                  View Pricing
                </Link>
                <Link
                  href="/contact"
                  className="border-2 border-white text-white hover:bg-white hover:text-dental-600 px-8 py-4 rounded-xl font-bold transition-colors duration-200"
                >
                  Contact Sales
                </Link>
              </div>
            </motion.div>
          </motion.div>
        </div>
      </section>
    </div>
  )
}

export default DemoPage